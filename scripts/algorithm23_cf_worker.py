from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pickle
import sys
import traceback
from pathlib import Path


def _emit(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _load_request_payload(input_path: str):
    with Path(input_path).open("rb") as handle:
        return pickle.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    repo_root = Path(args.repo_root).resolve()
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    try:
        from kldmPlus.symmetry.crystalformer_backend import CrystalFormerLikelihood

        # The vendored CrystalFormer backend writes trace/debug text to stdout.
        # Keep the worker protocol on stdout pure-JSON by redirecting those
        # backend prints to stderr during runtime calls.
        with contextlib.redirect_stdout(sys.stderr):
            cf_like = CrystalFormerLikelihood(checkpoint_path=str(args.checkpoint))
        _emit(
            {
                "ok": True,
                "event": "started",
                "coordinate_only": bool(cf_like.coordinate_only),
                "pid": os.getpid(),
            }
        )
    except BaseException as exc:
        _emit(
            {
                "ok": False,
                "event": "startup_error",
                "error_type": type(exc).__name__,
                "error_message": f"{exc}\n{traceback.format_exc()}",
            }
        )
        return 2

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            request = json.loads(line)
            cmd = str(request.get("cmd", "")).strip().lower()
            if cmd == "shutdown":
                break
            if cmd == "ping":
                _emit({"ok": True, "event": "pong", "pid": os.getpid()})
                continue
            if cmd == "score_batch":
                payload = _load_request_payload(request["input"])
                values = []
                for item in payload["items"]:
                    symmetry_payload = item.get("symmetry_payload", item.get("payload"))
                    if symmetry_payload is None:
                        raise KeyError("symmetry_payload")
                    with contextlib.redirect_stdout(sys.stderr):
                        values.append(
                            float(
                                cf_like.nll_q(
                                    payload=symmetry_payload,
                                    q=item["q"],
                                    lattice_feature=item["lattice_feature"],
                                    formula=item.get("formula"),
                                )
                            )
                        )
                out = {"ok": True, "values": values}
                output_path = request.get("output")
                if output_path:
                    Path(output_path).write_text(json.dumps(out), encoding="utf-8")
                _emit({"ok": True, "event": "score_batch_done", "n": len(values)})
                continue
            if cmd == "probe":
                payload = _load_request_payload(request["input"])
                symmetry_payload = payload.get("symmetry_payload", payload.get("payload"))
                with contextlib.redirect_stdout(sys.stderr):
                    value1 = float(
                        cf_like.nll_q(
                            payload=symmetry_payload,
                            q=payload["q"],
                            lattice_feature=payload["lattice_feature"],
                            formula=payload.get("formula"),
                        )
                    )
                    value2 = float(
                        cf_like.nll_q(
                            payload=symmetry_payload,
                            q=payload["q"],
                            lattice_feature=payload["lattice_feature"],
                            formula=payload.get("formula"),
                        )
                    )
                out = {
                    "ok": True,
                    "value1": value1,
                    "value2": value2,
                    "repeat_delta_abs": abs(value2 - value1),
                }
                output_path = request.get("output")
                if output_path:
                    Path(output_path).write_text(json.dumps(out), encoding="utf-8")
                _emit({"ok": True, "event": "probe_done"})
                continue
            raise ValueError(f"Unsupported command {cmd!r}")
        except BaseException as exc:
            _emit(
                {
                    "ok": False,
                    "event": "request_error",
                    "error_type": type(exc).__name__,
                    "error_message": f"{exc}\n{traceback.format_exc()}",
                }
            )

    try:
        if "cf_like" in locals() and hasattr(cf_like, "release_runtime"):
            with contextlib.redirect_stdout(sys.stderr):
                cf_like.release_runtime()
    except Exception:
        pass
    _emit({"ok": True, "event": "shutdown_complete"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
