#!/usr/bin/env python3
"""R55-C deterministic canonical GPU contract evidence + SHA-256 sidecar.

Pure contract evidence only: no hardware probe, no timestamp, no float. Running
this tool twice on the same checkout must produce byte-identical JSON (the
two-build byte-identity acceptance gate).

Usage:
    python tools/zegrid_r55/gpu_contract_evidence.py --out DIR [--verify DIR]

Writes ``gpu_contract_evidence.json`` and ``gpu_contract_evidence.json.sha256``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from zemosaic.core.zegrid import gpu_readiness as gr  # noqa: E402

FILENAME = "gpu_contract_evidence.json"


def canonical_bytes() -> bytes:
    payload = gr.canonical_contract_dict()
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_evidence(out_dir: Path) -> tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes()
    digest = sha256_hex(data)
    json_path = out_dir / FILENAME
    json_path.write_bytes(data)
    (out_dir / (FILENAME + ".sha256")).write_text(
        f"{digest}  {FILENAME}\n", encoding="ascii"
    )
    return json_path, digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--verify", default=None, help="directory to compare against")
    args = parser.parse_args(argv)

    path, digest = write_evidence(Path(args.out))
    print(f"wrote {path}")
    print(f"sha256 {digest}")

    if args.verify:
        other = Path(args.verify) / FILENAME
        if not other.is_file():
            print(f"VERIFY_FAIL missing {other}")
            return 2
        if other.read_bytes() != path.read_bytes():
            print("VERIFY_FAIL evidence bytes differ")
            return 3
        expected = (Path(args.verify) / (FILENAME + ".sha256")).read_text(
            encoding="ascii"
        ).split()[0]
        if expected != digest:
            print("VERIFY_FAIL sha256 sidecar differs")
            return 4
        print("VERIFY_OK byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
