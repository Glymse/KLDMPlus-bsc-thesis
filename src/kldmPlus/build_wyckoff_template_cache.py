from __future__ import annotations

import argparse
import ctypes
import gc
import multiprocessing as mp
from pathlib import Path
import queue as queue_module
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
from kldmPlus.symmetry.template_cache import (
    add_true_signature,
    empty_template_cache,
    load_template_cache,
    put_cache_entry,
    save_template_cache,
    template_cache_key,
)
from kldmPlus.symmetry.wyckoff_templates import (
    extract_wyckoff_templates,
    requested_conventional_atomic_numbers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an offline Wyckoff template cache for KLDM-DPnP-SG.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=8000)
    parser.add_argument("--max-templates", type=int, default=512)
    parser.add_argument("--template-nmax", type=int, default=5000)
    parser.add_argument("--quick-templates", action="store_true")
    parser.add_argument("--symprec", type=float, default=1.0e-3)
    parser.add_argument("--pyxtal-tol", type=float, default=1.0e-3)
    parser.add_argument("--save-every-samples", type=int, default=50)
    parser.add_argument("--gc-every-samples", type=int, default=10)
    parser.add_argument("--max-runtime-s", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--worker-chunk-size",
        type=int,
        default=25,
        help="Process this many samples in a short-lived child process. Use 0 to run in-process.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if "sampler" not in config and "sampler_config" in config:
        sampler_path = (config_path.parent / str(config["sampler_config"])).expanduser().resolve()
        with sampler_path.open("r", encoding="utf-8") as handle:
            config["sampler"] = yaml.safe_load(handle) or {}
    return config


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


def _merge_chunk_cache(cache: dict[str, Any], chunk: dict[str, Any]) -> tuple[int, int]:
    built = 0
    reused = 0
    for key, chunk_entry in chunk.get("entries", {}).items():
        entry = cache.setdefault("entries", {}).get(key)
        if entry is None:
            if not chunk_entry.get("templates"):
                continue
            cache["entries"][key] = chunk_entry
            built += 1
            continue
        reused += 1
        for signature, count in dict(chunk_entry.get("true_signature_counts", {}) or {}).items():
            for _ in range(int(count)):
                add_true_signature(entry, signature=tuple(signature))
    return built, reused


def _known_key_token(key: Any) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    return (
        int(key[0]),
        tuple(int(v) for v in key[1]),
        tuple(int(v) for v in key[2]),
    )


def _process_chunk_worker(payload: dict[str, Any], queue: Any) -> None:
    try:
        config = load_config(payload["config"])
        dataset, lattice_transform = build_dataset_and_lattice_transform(config, split=str(payload["split"]))
        chunk_cache = empty_template_cache()
        known_keys = set(tuple(item) for item in payload.get("known_keys", []))
        skipped = 0
        processed: list[int] = []
        for sample_idx in range(int(payload["start"]), int(payload["end"])):
            sample_started = time.perf_counter()
            try:
                sample = dataset[sample_idx]
                requested_sg, true_signature = true_signature_from_sample(
                    sample=sample,
                    lattice_transform=lattice_transform,
                    symprec=float(payload["symprec"]),
                    pyxtal_tol=float(payload["pyxtal_tol"]),
                )
                conventional_atomic_numbers = requested_conventional_atomic_numbers(
                    sample.atomic_numbers,
                    space_group_number=int(requested_sg),
                )
                key = template_cache_key(
                    space_group_number=int(requested_sg),
                    atomic_numbers=conventional_atomic_numbers,
                )
                entry = chunk_cache.setdefault("entries", {}).get(key)
                if entry is None:
                    key_token = _known_key_token(key)
                    if key_token in known_keys:
                        templates = []
                    else:
                        templates = extract_wyckoff_templates(
                            space_group_number=int(requested_sg),
                            atomic_numbers=conventional_atomic_numbers,
                            max_templates=int(payload["max_templates"]),
                            quick=bool(payload["quick_templates"]),
                            num_wp=(None, None),
                            nmax=int(payload["template_nmax"]),
                        )
                    entry = put_cache_entry(chunk_cache, key=key, templates=templates)
                add_true_signature(entry, signature=true_signature)
                processed.append(sample_idx)
                queue.put(
                    {
                        "type": "sample",
                        "sample_idx": int(sample_idx),
                        "status": "ok",
                        "elapsed_s": time.perf_counter() - sample_started,
                        "entries": len(chunk_cache.get("entries", {})),
                    }
                )
            except Exception as exc:
                skipped += 1
                processed.append(sample_idx)
                queue.put(
                    {
                        "type": "sample",
                        "sample_idx": int(sample_idx),
                        "status": f"skip:{type(exc).__name__}",
                        "elapsed_s": time.perf_counter() - sample_started,
                        "entries": len(chunk_cache.get("entries", {})),
                    }
                )
        queue.put({"type": "done", "ok": True, "cache": chunk_cache, "processed": processed, "skipped": skipped})
    except Exception as exc:
        queue.put({"type": "done", "ok": False, "error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output = Path(args.output).expanduser().resolve()
    if bool(args.resume) and output.exists():
        try:
            cache = load_template_cache(output)
        except Exception as exc:
            backup = output.with_suffix(output.suffix + f".corrupt.{int(time.time())}")
            output.replace(backup)
            print(
                f"template_cache_resume_corrupt path={output} backup={backup} "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            cache = empty_template_cache()
    else:
        cache = empty_template_cache()
    processed = set(int(v) for v in cache.setdefault("metadata", {}).get("processed_indices", []))

    print(
        f"template_cache_setup split={args.split} max_samples={int(args.max_samples)} "
        f"max_templates={int(args.max_templates)} template_nmax={int(args.template_nmax)} "
        f"resume={int(bool(args.resume))} processed={len(processed)}",
        flush=True,
    )
    dataset, lattice_transform = build_dataset_and_lattice_transform(config, split=str(args.split))
    limit = len(dataset) if int(args.max_samples) <= 0 else min(len(dataset), int(args.max_samples))
    started = time.perf_counter()
    built = 0
    reused = 0
    skipped = 0

    if int(args.worker_chunk_size) > 0:
        ctx_name = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(ctx_name)
        chunk_size = max(1, int(args.worker_chunk_size))
        sample_idx = 0
        while sample_idx < limit:
            while sample_idx < limit and sample_idx in processed:
                sample_idx += 1
            if sample_idx >= limit:
                break
            chunk_start = sample_idx
            chunk_end = min(limit, chunk_start + chunk_size)
            payload = {
                "config": str(Path(args.config).expanduser().resolve()),
                "split": str(args.split),
                "start": int(chunk_start),
                "end": int(chunk_end),
                "known_keys": [
                    _known_key_token(key)
                    for key in cache.get("entries", {}).keys()
                ],
                "max_templates": int(args.max_templates),
                "template_nmax": int(args.template_nmax),
                "quick_templates": bool(args.quick_templates),
                "symprec": float(args.symprec),
                "pyxtal_tol": float(args.pyxtal_tol),
            }
            chunk_started = time.perf_counter()
            queue = ctx.Queue()
            process = ctx.Process(target=_process_chunk_worker, args=(payload, queue))
            process.start()
            result = None
            while process.is_alive() or not queue.empty():
                try:
                    message = queue.get(timeout=2.0)
                except queue_module.Empty:
                    continue
                if message.get("type") == "sample":
                    print(
                        f"template_cache_worker_progress chunk={chunk_start}:{chunk_end} "
                        f"sample={int(message['sample_idx']) + 1}/{limit} "
                        f"status={message['status']} sample_elapsed_s={float(message['elapsed_s']):.1f} "
                        f"chunk_entries={int(message['entries'])}",
                        flush=True,
                    )
                    continue
                result = message
                break
            process.join()
            chunk_elapsed = time.perf_counter() - chunk_started
            if result is None:
                result = {
                    "ok": False,
                    "error": f"worker_exitcode={process.exitcode}",
                }
            if not result.get("ok"):
                print(
                    f"template_cache_worker_failed chunk={chunk_start}:{chunk_end} "
                    f"error={result.get('error', 'unknown')}",
                    flush=True,
                )
                skipped += chunk_end - chunk_start
                processed.update(range(chunk_start, chunk_end))
            else:
                chunk_built, chunk_reused = _merge_chunk_cache(cache, result["cache"])
                built += chunk_built
                reused += chunk_reused
                skipped += int(result.get("skipped", 0))
                processed.update(int(v) for v in result.get("processed", []))

            cache["metadata"]["processed_indices"] = sorted(processed)
            save_template_cache(output, cache)
            print(
                f"template_cache_progress chunk={chunk_start}:{chunk_end} "
                f"chunk_elapsed_s={chunk_elapsed:.1f} samples={min(chunk_end, limit)}/{limit} "
                f"entries={len(cache.get('entries', {}))} built={built} reused={reused} "
                f"skipped={skipped} elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )
            trim_process_memory()
            sample_idx = chunk_end

        cache["metadata"]["processed_indices"] = sorted(processed)
        cache["metadata"]["config"] = str(Path(args.config).expanduser().resolve())
        cache["metadata"]["split"] = str(args.split)
        cache["metadata"]["max_templates"] = int(args.max_templates)
        cache["metadata"]["template_nmax"] = int(args.template_nmax)
        cache["metadata"]["worker_chunk_size"] = int(args.worker_chunk_size)
        save_template_cache(output, cache)
        print(
            f"template_cache_done path={output} samples={limit} entries={len(cache.get('entries', {}))} "
            f"built={built} reused={reused} skipped={skipped} elapsed_s={time.perf_counter() - started:.1f}",
            flush=True,
        )
        return

    for sample_idx in range(limit):
        if sample_idx in processed:
            continue
        try:
            sample = dataset[sample_idx]
            requested_sg, true_signature = true_signature_from_sample(
                sample=sample,
                lattice_transform=lattice_transform,
                symprec=float(args.symprec),
                pyxtal_tol=float(args.pyxtal_tol),
            )
            conventional_atomic_numbers = requested_conventional_atomic_numbers(
                sample.atomic_numbers,
                space_group_number=int(requested_sg),
            )
            key = template_cache_key(
                space_group_number=int(requested_sg),
                atomic_numbers=conventional_atomic_numbers,
            )
            entries = cache.setdefault("entries", {})
            entry = entries.get(key)
            if entry is None:
                templates = extract_wyckoff_templates(
                    space_group_number=int(requested_sg),
                    atomic_numbers=conventional_atomic_numbers,
                    max_templates=int(args.max_templates),
                    quick=bool(args.quick_templates),
                    num_wp=(None, None),
                    nmax=int(args.template_nmax),
                )
                entry = put_cache_entry(cache, key=key, templates=templates)
                built += 1
            else:
                reused += 1
            add_true_signature(entry, signature=true_signature)
            processed.add(sample_idx)
        except Exception:
            skipped += 1
            processed.add(sample_idx)

        if int(args.save_every_samples) > 0 and (sample_idx + 1) % int(args.save_every_samples) == 0:
            cache["metadata"]["processed_indices"] = sorted(processed)
            save_template_cache(output, cache)
            print(
                f"template_cache_progress samples={sample_idx + 1}/{limit} "
                f"entries={len(cache.get('entries', {}))} built={built} reused={reused} "
                f"skipped={skipped} elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )
        if int(args.gc_every_samples) > 0 and (sample_idx + 1) % int(args.gc_every_samples) == 0:
            trim_process_memory()
        if float(args.max_runtime_s) > 0.0 and time.perf_counter() - started >= float(args.max_runtime_s):
            cache["metadata"]["processed_indices"] = sorted(processed)
            save_template_cache(output, cache)
            print(f"template_cache_save reason=max_runtime path={output}", flush=True)
            return

    cache["metadata"]["processed_indices"] = sorted(processed)
    cache["metadata"]["config"] = str(Path(args.config).expanduser().resolve())
    cache["metadata"]["split"] = str(args.split)
    cache["metadata"]["max_templates"] = int(args.max_templates)
    cache["metadata"]["template_nmax"] = int(args.template_nmax)
    save_template_cache(output, cache)
    print(
        f"template_cache_done path={output} samples={limit} entries={len(cache.get('entries', {}))} "
        f"built={built} reused={reused} skipped={skipped} elapsed_s={time.perf_counter() - started:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
