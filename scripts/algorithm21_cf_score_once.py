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
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
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

    input_path = Path(args.input)
    output_path = Path(args.output)
    try:
        with input_path.open("rb") as handle:
            payload = pickle.load(handle)

        repo_root = Path(payload["repo_root"]).resolve()
        src_path = repo_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from kldmPlus.symmetry.crystalformer_backend import CrystalFormerLikelihood

        cf_like = CrystalFormerLikelihood(checkpoint_path=str(payload["checkpoint_path"]))
        value = float(
            cf_like.nll_q(
                payload=payload["symmetry_payload"],
                q=payload["q"],
                lattice_feature=payload["lattice_feature"],
                formula=payload.get("formula"),
            )
        )
        result = {"ok": True, "value": value}
    except BaseException as exc:
        tb = traceback.format_exc()
        result = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": f"{exc}\n{tb}",
            "traceback": tb,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle)
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
