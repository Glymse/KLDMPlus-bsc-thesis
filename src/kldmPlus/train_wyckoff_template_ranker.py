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
import yaml

from kldmPlus.data import CSPTask, resolve_data_root
from kldmPlus.data.csp import validate_lattice_configuration
from kldmPlus.sample_evaluation.sample_evaluation import build_structure_from_sample
from kldmPlus.symmetry.pyxtal_backend import build_pyxtal_wyckoff_result
from kldmPlus.symmetry.template_cache import load_template_cache
from kldmPlus.symmetry.template_ranker import (
    build_template_ranker,
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
    templates: tuple[WyckoffTemplate, ...]
    positive_indices: tuple[int, ...]
    weight: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Wyckoff template ranker for KLDM-DPnP-SG.")
    parser.add_argument("--config", default=None, help="KLDM YAML config used to resolve dataset/lattice settings.")
    parser.add_argument("--output", required=True, help="Path to save the ranker checkpoint.")
    parser.add_argument(
        "--template-cache",
        default=None,
        help="Offline template cache. When set, training reads PyXtal candidates from cache instead of recomputing them.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=8000)
    parser.add_argument("--max-templates", type=int, default=512)
    parser.add_argument(
        "--template-nmax",
        type=int,
        default=20000,
        help="PyXtal list_wyckoff_combinations Nmax cap for live extraction.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--ranker-model", default="mlp")
    parser.add_argument("--quick-templates", action="store_true")
    parser.add_argument("--symprec", type=float, default=1.0e-2)
    parser.add_argument("--pyxtal-tol", type=float, default=1.0e-2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument(
        "--checkpoint-every-examples",
        type=int,
        default=250,
        help="Save the ranker every N training examples.",
    )
    parser.add_argument(
        "--gc-every-examples",
        type=int,
        default=100,
        help="Run Python GC and libc malloc_trim every N training examples.",
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
    ranker_model: str,
    device: torch.device,
    resume: bool,
) -> torch.nn.Module:
    checkpoint = Path(output_path).expanduser().resolve()
    if bool(resume) and checkpoint.exists():
        try:
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
        except TypeError:  # pragma: no cover - older torch
            payload = torch.load(checkpoint, map_location=device)
        model_kwargs = dict(payload.get("model_kwargs", {}))
        model_name = str(payload.get("model_name", ranker_model))
        model = build_template_ranker(model_name=model_name, **model_kwargs).to(device)
        model.load_state_dict(payload["state_dict"])
        print(f"template_ranker_resume path={checkpoint}", flush=True)
        return model
    return build_template_ranker(model_name=ranker_model, hidden=int(hidden)).to(device)


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


def build_ranker_examples_from_cache(
    *,
    cache_path: str | Path,
) -> list[RankerExample]:
    cache = load_template_cache(cache_path)
    examples: list[RankerExample] = []
    skipped = 0
    for entry in cache.get("entries", {}).values():
        templates = tuple(entry.get("templates", []))
        if len(templates) < 2:
            skipped += 1
            continue
        signature_to_indices: dict[tuple[tuple[int, str], ...], list[int]] = {}
        for template_idx, template in enumerate(templates):
            signature_to_indices.setdefault(flatten_site_signature(template), []).append(int(template_idx))
        true_counts = dict(entry.get("true_signature_counts", {}) or {})
        for signature, count in true_counts.items():
            positives = tuple(signature_to_indices.get(tuple(signature), ()))
            if not positives:
                skipped += 1
                continue
            if len(positives) >= len(templates):
                skipped += 1
                continue
            examples.append(
                RankerExample(
                    requested_sg=int(entry["space_group"]),
                    templates=templates,
                    positive_indices=positives,
                    weight=max(1, int(count)),
                )
            )
    print(
        f"template_ranker_cache_load path={Path(cache_path).expanduser().resolve()} "
        f"entries={len(cache.get('entries', {}))} examples={len(examples)} skipped={skipped}",
        flush=True,
    )
    return examples


def build_ranker_examples(
    *,
    dataset: Any,
    lattice_transform: Any,
    max_samples: int,
    max_templates: int,
    template_nmax: int,
    quick_templates: bool,
    symprec: float,
    pyxtal_tol: float,
) -> list[RankerExample]:
    limit = len(dataset) if int(max_samples) <= 0 else min(len(dataset), int(max_samples))
    examples: list[RankerExample] = []
    skipped = 0
    started = time.perf_counter()

    for sample_idx in range(limit):
        sample_started = time.perf_counter()
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
            templates = tuple(
                extract_wyckoff_templates(
                    space_group_number=int(requested_sg),
                    atomic_numbers=conventional_atomic_numbers,
                    max_templates=int(max_templates),
                    quick=bool(quick_templates),
                    num_wp=(None, None),
                    nmax=int(template_nmax),
                )
            )
            if len(templates) < 2:
                skipped += 1
                continue
            positive_indices = tuple(
                template_idx
                for template_idx, template in enumerate(templates)
                if flatten_site_signature(template) == true_signature
            )
            if not positive_indices or len(positive_indices) >= len(templates):
                skipped += 1
                continue
            examples.append(
                RankerExample(
                    requested_sg=int(requested_sg),
                    templates=templates,
                    positive_indices=positive_indices,
                )
            )
        except Exception:
            skipped += 1
            continue

        if (sample_idx + 1) % 100 == 0:
            print(
                f"template_ranker_extract progress={sample_idx + 1}/{limit} "
                f"examples={len(examples)} skipped={skipped} "
                f"sample_elapsed_s={time.perf_counter() - sample_started:.1f} "
                f"elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )

    print(
        f"template_ranker_extract done samples={limit} examples={len(examples)} "
        f"skipped={skipped} elapsed_s={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return examples


def score_template_set(
    *,
    model: torch.nn.Module,
    requested_sg: int,
    templates: tuple[WyckoffTemplate, ...] | list[WyckoffTemplate],
    device: torch.device,
) -> torch.Tensor:
    payload = templates_to_ranker_batch(
        templates=list(templates),
        requested_sg=int(requested_sg),
        device=device,
        max_letter=int(getattr(model, "max_letter", 64)),
        max_mult=int(getattr(model, "max_mult", 256)),
    )
    return model(**payload)


def candidate_set_cross_entropy(
    *,
    scores: torch.Tensor,
    positive_indices: tuple[int, ...],
) -> torch.Tensor:
    positive_scores = scores[torch.as_tensor(positive_indices, device=scores.device, dtype=torch.long)]
    return torch.logsumexp(scores, dim=0) - torch.logsumexp(positive_scores, dim=0)


@torch.no_grad()
def evaluate_examples(
    *,
    model: torch.nn.Module,
    examples: list[RankerExample],
    device: torch.device,
) -> dict[str, float]:
    if not examples:
        return {
            "loss": 0.0,
            "top1_acc": 0.0,
            "positive_mass": 0.0,
            "examples": 0.0,
        }

    model.eval()
    total_weight = 0.0
    total_loss = 0.0
    top1_correct = 0.0
    positive_mass = 0.0
    for example in examples:
        scores = score_template_set(
            model=model,
            requested_sg=int(example.requested_sg),
            templates=example.templates,
            device=device,
        )
        loss = candidate_set_cross_entropy(scores=scores, positive_indices=example.positive_indices)
        probs = torch.softmax(scores, dim=0)
        positive_idx = torch.as_tensor(example.positive_indices, device=device, dtype=torch.long)
        weight = float(max(int(example.weight), 1))
        total_weight += weight
        total_loss += float(loss.detach().cpu().item()) * weight
        positive_mass += float(probs[positive_idx].sum().detach().cpu().item()) * weight
        top1_correct += float(int(torch.argmax(scores).item() in set(example.positive_indices))) * weight

    return {
        "loss": total_loss / max(total_weight, 1.0),
        "top1_acc": top1_correct / max(total_weight, 1.0),
        "positive_mass": positive_mass / max(total_weight, 1.0),
        "examples": float(len(examples)),
    }


def split_examples(
    *,
    examples: list[RankerExample],
    seed: int,
    val_fraction: float,
) -> tuple[list[RankerExample], list[RankerExample]]:
    items = list(examples)
    rng = random.Random(int(seed))
    rng.shuffle(items)
    if not items:
        return [], []
    val_count = int(round(len(items) * max(0.0, min(float(val_fraction), 0.9))))
    if len(items) > 1:
        val_count = min(max(val_count, 1), len(items) - 1)
    return items[val_count:], items[:val_count]


def train(args: argparse.Namespace) -> None:
    if int(args.num_threads) > 0:
        torch.set_num_threads(int(args.num_threads))
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(str(args.device))
    run_started = time.perf_counter()

    config_ref = str(Path(args.config).expanduser().resolve()) if args.config else None
    if args.template_cache:
        all_examples = build_ranker_examples_from_cache(cache_path=args.template_cache)
        mode = "cache"
    else:
        if not args.config:
            raise ValueError("--config is required when --template-cache is not provided.")
        _config_path, config = load_config(args.config)
        dataset, lattice_transform = build_dataset_and_lattice_transform(config, split=str(args.split))
        print(
            f"template_ranker_setup split={args.split} max_samples={int(args.max_samples)} "
            f"max_templates={int(args.max_templates)} template_nmax={int(args.template_nmax)}",
            flush=True,
        )
        all_examples = build_ranker_examples(
            dataset=dataset,
            lattice_transform=lattice_transform,
            max_samples=int(args.max_samples),
            max_templates=int(args.max_templates),
            template_nmax=int(args.template_nmax),
            quick_templates=bool(args.quick_templates),
            symprec=float(args.symprec),
            pyxtal_tol=float(args.pyxtal_tol),
        )
        mode = "dataset"
        trim_process_memory()

    train_examples, val_examples = split_examples(
        examples=all_examples,
        seed=int(args.seed),
        val_fraction=float(args.val_fraction),
    )
    print(
        f"template_ranker_setup mode={mode} train_examples={len(train_examples)} "
        f"val_examples={len(val_examples)} epochs={int(args.epochs)} ranker_model={args.ranker_model}",
        flush=True,
    )
    if not train_examples:
        raise RuntimeError("No ranker examples were extracted. Check the PyXtal cache/settings.")

    model = maybe_load_ranker_for_resume(
        output_path=args.output,
        hidden=int(args.hidden),
        ranker_model=str(args.ranker_model),
        device=device,
        resume=bool(args.resume),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1.0e-4)
    rng = random.Random(int(args.seed))
    total_examples_seen = 0

    def save_progress(*, reason: str, epoch: int, examples_seen: int) -> None:
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
                "epochs": int(args.epochs),
                "last_epoch": int(epoch),
                "lr": float(args.lr),
                "examples_seen": int(examples_seen),
                "seed": int(args.seed),
                "val_fraction": float(args.val_fraction),
                "ranker_model": str(args.ranker_model),
                "loss": "candidate_set_cross_entropy",
                "reason": str(reason),
            },
        )
        print(
            f"template_ranker_checkpoint reason={reason} path={Path(args.output).expanduser().resolve()} "
            f"epoch={epoch} examples_seen={examples_seen}",
            flush=True,
        )

    for epoch in range(1, int(args.epochs) + 1):
        order = list(range(len(train_examples)))
        rng.shuffle(order)
        total_weight = 0.0
        total_loss = 0.0
        top1_correct = 0.0
        positive_mass = 0.0
        started = time.perf_counter()
        model.train()

        for step_idx, example_idx in enumerate(order, start=1):
            example = train_examples[example_idx]
            scores = score_template_set(
                model=model,
                requested_sg=int(example.requested_sg),
                templates=example.templates,
                device=device,
            )
            loss = candidate_set_cross_entropy(
                scores=scores,
                positive_indices=example.positive_indices,
            )
            weight = float(max(int(example.weight), 1))
            weighted_loss = loss * weight

            optimizer.zero_grad(set_to_none=True)
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            with torch.no_grad():
                probs = torch.softmax(scores, dim=0)
                pos_idx = torch.as_tensor(example.positive_indices, device=device, dtype=torch.long)
                positive_mass += float(probs[pos_idx].sum().detach().cpu().item()) * weight
                top1_correct += float(int(torch.argmax(scores).item() in set(example.positive_indices))) * weight

            total_weight += weight
            total_loss += float(loss.detach().cpu().item()) * weight
            total_examples_seen += 1

            if int(args.checkpoint_every_examples) > 0 and total_examples_seen % int(args.checkpoint_every_examples) == 0:
                save_progress(
                    reason="periodic",
                    epoch=epoch,
                    examples_seen=total_examples_seen,
                )
            if float(args.max_runtime_s) > 0.0 and time.perf_counter() - run_started >= float(args.max_runtime_s):
                save_progress(
                    reason="max_runtime",
                    epoch=epoch,
                    examples_seen=total_examples_seen,
                )
                return
            if step_idx % 100 == 0:
                print(
                    f"template_ranker_train_progress epoch={epoch}/{int(args.epochs)} "
                    f"examples={step_idx}/{len(train_examples)} elapsed_s={time.perf_counter() - started:.1f}",
                    flush=True,
                )
            if int(args.gc_every_examples) > 0 and step_idx % int(args.gc_every_examples) == 0:
                trim_process_memory()

        val_metrics = evaluate_examples(model=model, examples=val_examples, device=device)
        train_top1 = top1_correct / max(total_weight, 1.0)
        train_positive_mass = positive_mass / max(total_weight, 1.0)
        train_loss = total_loss / max(total_weight, 1.0)
        print(
            f"template_ranker_train epoch={epoch}/{int(args.epochs)} "
            f"loss={train_loss:.6f} train_examples={len(train_examples)} "
            f"top1_acc={train_top1:.4f} positive_mass={train_positive_mass:.4f} "
            f"val_loss={val_metrics['loss']:.6f} val_examples={int(val_metrics['examples'])} "
            f"val_top1_acc={val_metrics['top1_acc']:.4f} val_positive_mass={val_metrics['positive_mass']:.4f} "
            f"elapsed_s={time.perf_counter() - started:.1f}",
            flush=True,
        )
        save_progress(
            reason="epoch",
            epoch=epoch,
            examples_seen=total_examples_seen,
        )
        trim_process_memory()

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
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "examples_seen": int(total_examples_seen),
            "seed": int(args.seed),
            "val_fraction": float(args.val_fraction),
            "ranker_model": str(args.ranker_model),
            "loss": "candidate_set_cross_entropy",
        },
    )
    print(f"template_ranker_save path={Path(args.output).expanduser().resolve()}", flush=True)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
