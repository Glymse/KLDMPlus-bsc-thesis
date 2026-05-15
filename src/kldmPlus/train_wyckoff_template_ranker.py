from __future__ import annotations

import argparse
import ctypes
import gc
from dataclasses import dataclass
from pathlib import Path
import random
import sys
import time
from typing import Any
import warnings

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import yaml

from kldmPlus.data import CSPTask, resolve_data_root
from kldmPlus.data.csp import validate_lattice_configuration
from kldmPlus.sample_evaluation.sample_evaluation import build_structure_from_sample
from kldmPlus.symmetry.pyxtal_backend import build_pyxtal_wyckoff_result
from kldmPlus.symmetry.template_cache import load_template_cache
from kldmPlus.symmetry.template_ranker import (
    WyckoffTemplateRanker,
    save_template_ranker,
    templates_to_ranker_batch,
)
from kldmPlus.symmetry.wyckoff_templates import (
    WyckoffTemplate,
    extract_wyckoff_templates,
    flatten_site_signature,
    requested_conventional_atomic_numbers,
)


@dataclass(frozen=True)
class RankerExample:
    requested_sg: int
    positive: WyckoffTemplate
    negatives: tuple[WyckoffTemplate, ...]
    weight: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a saved Wyckoff template ranker for KLDM-DPnP-SG.")
    parser.add_argument("--config", default=None, help="KLDM YAML config used to resolve dataset/lattice settings.")
    parser.add_argument("--output", required=True, help="Path to save the ranker checkpoint.")
    parser.add_argument(
        "--template-cache",
        default=None,
        help="Offline template cache. When set, training does not call PyXtal or read the dataset.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=8000)
    parser.add_argument("--max-templates", type=int, default=512)
    parser.add_argument(
        "--template-nmax",
        type=int,
        default=20000,
        help="PyXtal list_wyckoff_combinations Nmax cap. Lower this if template extraction is killed.",
    )
    parser.add_argument("--negatives-per-positive", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--quick-templates", action="store_true")
    parser.add_argument("--symprec", type=float, default=1.0e-2)
    parser.add_argument("--pyxtal-tol", type=float, default=1.0e-2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument(
        "--checkpoint-every-examples",
        type=int,
        default=50,
        help="Save the ranker every N accepted examples so OOM kills do not lose all progress.",
    )
    parser.add_argument(
        "--gc-every-samples",
        type=int,
        default=25,
        help="Run Python GC and libc malloc_trim every N visited samples.",
    )
    parser.add_argument(
        "--max-runtime-s",
        type=float,
        default=0.0,
        help="Gracefully save and exit after this many seconds. Use 0 to disable.",
    )
    parser.add_argument("--resume", action="store_true", help="Initialize model weights from --output if it exists.")
    return parser.parse_args()


def load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if "sampler" not in config and "sampler_config" in config:
        sampler_path = (config_path.parent / str(config["sampler_config"])).expanduser().resolve()
        with sampler_path.open("r", encoding="utf-8") as handle:
            config["sampler"] = yaml.safe_load(handle) or {}
    return config_path, config


def build_ranker_examples_from_cache(
    *,
    cache_path: str | Path,
    negatives_per_positive: int,
    seed: int,
) -> list[RankerExample]:
    rng = random.Random(int(seed))
    cache = load_template_cache(cache_path)
    examples: list[RankerExample] = []
    skipped = 0
    for entry in cache.get("entries", {}).values():
        templates = list(entry.get("templates", []))
        if len(templates) < 2:
            skipped += 1
            continue
        templates_by_signature = {
            flatten_site_signature(template): template
            for template in templates
        }
        true_counts = dict(entry.get("true_signature_counts", {}) or {})
        for signature, count in true_counts.items():
            signature = tuple(signature)
            positive = templates_by_signature.get(signature)
            if positive is None:
                skipped += 1
                continue
            negatives = [
                template
                for template in templates
                if flatten_site_signature(template) != signature
            ]
            if not negatives:
                skipped += 1
                continue
            rng.shuffle(negatives)
            examples.append(
                RankerExample(
                    requested_sg=int(entry["space_group"]),
                    positive=positive,
                    negatives=tuple(negatives[: max(1, int(negatives_per_positive))]),
                    weight=max(1, int(count)),
                )
            )
    print(
        f"template_ranker_cache_load path={Path(cache_path).expanduser().resolve()} "
        f"entries={len(cache.get('entries', {}))} examples={len(examples)} skipped={skipped}",
        flush=True,
    )
    return examples


def build_dataset_and_lattice_transform(config: dict[str, Any], *, split: str):
    dataset_cfg = dict(config["dataset"])
    model_cfg = dict(config["model"])
    validate_lattice_configuration(
        lattice_representation=str(dataset_cfg.get("lattice_representation", "kldm")),
        lattice_parameterization=str(model_cfg["lattice_parameterization"]),
        lattice_diffusion_type=str(model_cfg.get("lattice_diffusion_type", "VP")),
    )
    task = CSPTask(
        dataset_name=str(dataset_cfg["name"]),
        lattice_parameterization=str(model_cfg["lattice_parameterization"]),
        lattice_representation=str(dataset_cfg.get("lattice_representation", "kldm")),
    )
    root = resolve_data_root(dataset_cfg.get("root"))
    dataset = task.fit_dataset(root=root, split=str(split), download=True)
    lattice_transform = task.make_lattice_transform(
        root=root,
        download=True,
        mattergen_limit_var_scaling_constant=model_cfg.get("mattergen_limit_var_scaling_constant"),
    )
    return dataset, lattice_transform


def trim_process_memory() -> None:
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        return
    try:
        libc.malloc_trim(0)
    except Exception:
        return


def maybe_load_ranker_for_resume(
    *,
    output_path: str | Path,
    hidden: int,
    device: torch.device,
    resume: bool,
) -> WyckoffTemplateRanker:
    model = WyckoffTemplateRanker(hidden=int(hidden)).to(device)
    checkpoint = Path(output_path).expanduser().resolve()
    if not bool(resume) or not checkpoint.exists():
        return model
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(checkpoint, map_location=device)
    model_kwargs = dict(payload.get("model_kwargs", {}))
    model = WyckoffTemplateRanker(**model_kwargs).to(device)
    model.load_state_dict(payload["state_dict"])
    print(f"template_ranker_resume path={checkpoint}", flush=True)
    return model


def true_signature_from_sample(
    *,
    sample: Any,
    lattice_transform: Any,
    symprec: float,
    pyxtal_tol: float,
) -> tuple[int, tuple[tuple[int, str], ...]]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"No Pauling electronegativity.*",
            category=UserWarning,
        )
        structure = build_structure_from_sample(
            f=sample.pos,
            l=sample.l,
            a=sample.atomic_numbers,
            lattice_transform=lattice_transform,
        )
        result = build_pyxtal_wyckoff_result(
            structure,
            symprec=float(symprec),
            pyxtal_tol=float(pyxtal_tol),
        )
    signature = tuple(
        sorted(
            (
                (int(atomic_number), str(label))
                for atomic_number, label in zip(result.anchor_atomic_numbers.tolist(), result.site_labels)
            ),
            key=lambda item: (item[0], item[1]),
        )
    )
    return int(result.space_group), signature


def build_ranker_examples(
    *,
    dataset: Any,
    lattice_transform: Any,
    max_samples: int,
    max_templates: int,
    template_nmax: int,
    negatives_per_positive: int,
    quick_templates: bool,
    symprec: float,
    pyxtal_tol: float,
    seed: int,
) -> list[RankerExample]:
    rng = random.Random(int(seed))
    limit = len(dataset) if int(max_samples) <= 0 else min(len(dataset), int(max_samples))
    examples: list[RankerExample] = []
    skipped = 0
    started = time.perf_counter()

    for sample_idx in range(limit):
        sample = dataset[sample_idx]
        try:
            requested_sg, true_signature = true_signature_from_sample(
                sample=sample,
                lattice_transform=lattice_transform,
                symprec=float(symprec),
                pyxtal_tol=float(pyxtal_tol),
            )
            conventional_atomic_numbers = requested_conventional_atomic_numbers(
                sample.atomic_numbers,
                space_group_number=int(requested_sg),
            )
            candidates = extract_wyckoff_templates(
                space_group_number=int(requested_sg),
                atomic_numbers=conventional_atomic_numbers,
                max_templates=int(max_templates),
                quick=bool(quick_templates),
                num_wp=(None, None),
                nmax=int(template_nmax),
            )
            positives = [
                template
                for template in candidates
                if flatten_site_signature(template) == true_signature
            ]
            negatives = [
                template
                for template in candidates
                if flatten_site_signature(template) != true_signature
            ]
            if not positives or not negatives:
                skipped += 1
                continue
            rng.shuffle(negatives)
            examples.append(
                RankerExample(
                    requested_sg=int(requested_sg),
                    positive=positives[0],
                    negatives=tuple(negatives[: max(1, int(negatives_per_positive))]),
                )
            )
        except Exception:
            skipped += 1
            continue

        if (sample_idx + 1) % 250 == 0:
            print(
                f"template_ranker_extract progress={sample_idx + 1}/{limit} "
                f"examples={len(examples)} skipped={skipped} elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )

    print(
        f"template_ranker_extract done samples={limit} examples={len(examples)} "
        f"skipped={skipped} elapsed_s={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return examples


def ranker_example_from_sample(
    *,
    sample: Any,
    lattice_transform: Any,
    max_templates: int,
    template_nmax: int,
    negatives_per_positive: int,
    quick_templates: bool,
    symprec: float,
    pyxtal_tol: float,
    rng: random.Random,
) -> RankerExample | None:
    requested_sg, true_signature = true_signature_from_sample(
        sample=sample,
        lattice_transform=lattice_transform,
        symprec=float(symprec),
        pyxtal_tol=float(pyxtal_tol),
    )
    conventional_atomic_numbers = requested_conventional_atomic_numbers(
        sample.atomic_numbers,
        space_group_number=int(requested_sg),
    )
    candidates = extract_wyckoff_templates(
        space_group_number=int(requested_sg),
        atomic_numbers=conventional_atomic_numbers,
        max_templates=int(max_templates),
        quick=bool(quick_templates),
        num_wp=(None, None),
        nmax=int(template_nmax),
    )
    positives = [
        template
        for template in candidates
        if flatten_site_signature(template) == true_signature
    ]
    negatives = [
        template
        for template in candidates
        if flatten_site_signature(template) != true_signature
    ]
    if not positives or not negatives:
        return None
    rng.shuffle(negatives)
    return RankerExample(
        requested_sg=int(requested_sg),
        positive=positives[0],
        negatives=tuple(negatives[: max(1, int(negatives_per_positive))]),
    )


def score_template_set(
    *,
    model: WyckoffTemplateRanker,
    requested_sg: int,
    templates: list[WyckoffTemplate],
    device: torch.device,
) -> torch.Tensor:
    payload = templates_to_ranker_batch(
        templates=templates,
        requested_sg=int(requested_sg),
        device=device,
        max_letter=int(model.max_letter),
        max_mult=int(model.max_mult),
    )
    return model(**payload)


def train(args: argparse.Namespace) -> None:
    if int(args.num_threads) > 0:
        torch.set_num_threads(int(args.num_threads))
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(str(args.device))
    config_ref = str(Path(args.config).expanduser().resolve()) if args.config else None
    cached_examples: list[RankerExample] | None = None
    dataset = None
    lattice_transform = None

    if args.template_cache:
        cached_examples = build_ranker_examples_from_cache(
            cache_path=args.template_cache,
            negatives_per_positive=int(args.negatives_per_positive),
            seed=int(args.seed),
        )
        limit = len(cached_examples)
        print(
            f"template_ranker_setup mode=cache examples={limit} "
            f"negatives={int(args.negatives_per_positive)} epochs={int(args.epochs)}",
            flush=True,
        )
    else:
        if not args.config:
            raise ValueError("--config is required when --template-cache is not provided.")
        _config_path, config = load_config(args.config)
        print(
            f"template_ranker_setup start split={args.split} max_samples={int(args.max_samples)} "
            f"max_templates={int(args.max_templates)} template_nmax={int(args.template_nmax)} "
            f"negatives={int(args.negatives_per_positive)} epochs={int(args.epochs)}",
            flush=True,
        )
        dataset, lattice_transform = build_dataset_and_lattice_transform(config, split=str(args.split))
        limit = len(dataset) if int(args.max_samples) <= 0 else min(len(dataset), int(args.max_samples))
        print(
            f"template_ranker_setup done dataset_size={len(dataset)} train_limit={limit}",
            flush=True,
        )

    model = maybe_load_ranker_for_resume(
        output_path=args.output,
        hidden=int(args.hidden),
        device=device,
        resume=bool(args.resume),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1.0e-4)
    rng = random.Random(int(args.seed))
    total_examples_seen = 0
    run_started = time.perf_counter()

    def save_progress(*, reason: str, epoch: int, examples_seen: int, pairs_seen: int) -> None:
        save_template_ranker(
            path=args.output,
            model=model,
            metadata={
                "config": config_ref,
                "template_cache": None if args.template_cache is None else str(Path(args.template_cache).expanduser().resolve()),
                "split": str(args.split),
                "max_samples": int(args.max_samples),
                "max_templates": int(args.max_templates),
                "template_nmax": int(args.template_nmax),
                "negatives_per_positive": int(args.negatives_per_positive),
                "epochs": int(args.epochs),
                "last_epoch": int(epoch),
                "lr": float(args.lr),
                "examples_seen": int(examples_seen),
                "pairs_seen": int(pairs_seen),
                "seed": int(args.seed),
                "streaming": True,
                "reason": str(reason),
            },
        )
        print(
            f"template_ranker_checkpoint reason={reason} path={Path(args.output).expanduser().resolve()} "
            f"epoch={epoch} examples_seen={examples_seen} pairs_seen={pairs_seen}",
            flush=True,
        )

    for epoch in range(1, int(args.epochs) + 1):
        order = list(range(limit))
        rng.shuffle(order)
        total_loss = 0.0
        total_pairs = 0
        examples_seen = 0
        skipped = 0
        started = time.perf_counter()
        model.train()
        for step_idx, sample_idx in enumerate(order, start=1):
            if cached_examples is not None:
                example = cached_examples[sample_idx]
            else:
                try:
                    example = ranker_example_from_sample(
                        sample=dataset[sample_idx],
                        lattice_transform=lattice_transform,
                        max_templates=int(args.max_templates),
                        template_nmax=int(args.template_nmax),
                        negatives_per_positive=int(args.negatives_per_positive),
                        quick_templates=bool(args.quick_templates),
                        symprec=float(args.symprec),
                        pyxtal_tol=float(args.pyxtal_tol),
                        rng=rng,
                    )
                except Exception:
                    example = None

            if example is None:
                skipped += 1
                continue

            templates = [example.positive, *example.negatives]
            scores = score_template_set(
                model=model,
                requested_sg=int(example.requested_sg),
                templates=templates,
                device=device,
            )
            pos_score = scores[0]
            neg_scores = scores[1:]
            loss = -F.logsigmoid(pos_score - neg_scores).mean()
            loss = loss * float(min(max(int(example.weight), 1), 32))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            pair_count = int(neg_scores.numel())
            total_loss += float(loss.detach().cpu().item()) * pair_count
            total_pairs += pair_count * max(1, int(example.weight))
            examples_seen += 1
            total_examples_seen += 1

            del example, templates, scores, loss
            if int(args.checkpoint_every_examples) > 0 and total_examples_seen % int(args.checkpoint_every_examples) == 0:
                save_progress(
                    reason="periodic",
                    epoch=epoch,
                    examples_seen=total_examples_seen,
                    pairs_seen=total_pairs,
                )
            if float(args.max_runtime_s) > 0.0 and time.perf_counter() - run_started >= float(args.max_runtime_s):
                save_progress(
                    reason="max_runtime",
                    epoch=epoch,
                    examples_seen=total_examples_seen,
                    pairs_seen=total_pairs,
                )
                return
            if step_idx % 250 == 0:
                print(
                    f"template_ranker_train_progress epoch={epoch}/{int(args.epochs)} "
                    f"items={step_idx}/{limit} examples={examples_seen} skipped={skipped} "
                    f"pairs={total_pairs} elapsed_s={time.perf_counter() - started:.1f}",
                    flush=True,
                )
            if int(args.gc_every_samples) > 0 and step_idx % int(args.gc_every_samples) == 0:
                trim_process_memory()

        mean_loss = total_loss / max(total_pairs, 1)
        print(
            f"template_ranker_train epoch={epoch}/{int(args.epochs)} "
            f"loss={mean_loss:.6f} examples={examples_seen} skipped={skipped} "
            f"pairs={total_pairs} elapsed_s={time.perf_counter() - started:.1f}",
            flush=True,
        )
        save_progress(
            reason="epoch",
            epoch=epoch,
            examples_seen=total_examples_seen,
            pairs_seen=total_pairs,
        )
        trim_process_memory()

    if total_examples_seen <= 0:
        raise RuntimeError("No ranker examples were extracted. Check PyXtal/spglib settings and max_templates.")

    save_template_ranker(
        path=args.output,
        model=model,
        metadata={
            "config": config_ref,
            "template_cache": None if args.template_cache is None else str(Path(args.template_cache).expanduser().resolve()),
            "split": str(args.split),
            "max_samples": int(args.max_samples),
            "max_templates": int(args.max_templates),
            "negatives_per_positive": int(args.negatives_per_positive),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "examples_seen": int(total_examples_seen),
            "seed": int(args.seed),
            "streaming": True,
        },
    )
    print(f"template_ranker_save path={Path(args.output).expanduser().resolve()}", flush=True)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
