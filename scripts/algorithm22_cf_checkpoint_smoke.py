from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe-input", default=None)
    args = parser.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_DISABLE_JIT", "true")
    os.environ.setdefault("KLDM_ALGO21_SAFE_MODE", "true")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    output_path = Path(args.output)
    result = {
        "ok": False,
        "python_executable": sys.executable,
        "imports": {},
    }
    try:
        repo_root = Path(args.repo_root).resolve()
        src_path = repo_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        for mod_name in ("jax", "haiku", "optax"):
            try:
                __import__(mod_name)
                result["imports"][mod_name] = True
            except Exception:
                result["imports"][mod_name] = False

        from kldmPlus.symmetry.crystalformer_backend import CrystalFormerLikelihood

        cf_like = CrystalFormerLikelihood(checkpoint_path=str(args.checkpoint))
        result["ok"] = True
        result["coordinate_only"] = bool(cf_like.coordinate_only)
        if args.probe_input:
            probe_path = Path(args.probe_input)
            with probe_path.open("rb") as handle:
                probe = pickle.load(handle)
            q = probe["q"]
            lattice_feature = probe["lattice_feature"]
            formula = probe.get("formula")
            payload = probe.get("symmetry_payload", probe.get("payload"))
            value1 = float(
                cf_like.nll_q(
                    payload=payload,
                    q=q,
                    lattice_feature=lattice_feature,
                    formula=formula,
                )
            )
            value2 = float(
                cf_like.nll_q(
                    payload=payload,
                    q=q,
                    lattice_feature=lattice_feature,
                    formula=formula,
                )
            )
            result["probe_nll_1"] = value1
            result["probe_nll_2"] = value2
            result["probe_repeat_delta_abs"] = abs(value2 - value1)
            result["probe_finite"] = bool(
                result["probe_nll_1"] == result["probe_nll_1"]
                and result["probe_nll_2"] == result["probe_nll_2"]
            )
        if hasattr(cf_like, "release_runtime"):
            cf_like.release_runtime()
    except BaseException as exc:
        tb = traceback.format_exc()
        result["ok"] = False
        result["error_type"] = type(exc).__name__
        result["error_message"] = f"{exc}\n{tb}"
        result["traceback"] = tb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle)
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
