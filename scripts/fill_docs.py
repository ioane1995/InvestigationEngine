"""Write your REAL deployment measurements into the docs (README + docs/ARCHITECTURE.md) in one command.

Every value must come from a measurement you made; nothing is guessed. Example:

  python -m scripts.fill_docs --cosine 0.9998 --verdict COMPATIBLE --ram-mb 412 --cold-s 55 --qdrant "all 61 768 chunks, 38 MB used (Qdrant dashboard)"

where
  --cosine/--verdict  output of  python -m scripts.check_embedding_compat   (worst cosine + verdict)
  --ram-mb            Render dashboard > Metrics > Memory, after start-up
  --cold-s            python -m scripts.measure_deployment <url> --cold     (first /health after >15 min idle)
  --qdrant            Qdrant Cloud dashboard > your cluster > points / storage
A placeholder that is already filled (or missing) is reported, never silently skipped. Use --dry-run to preview."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cosine", required=True, help="worst cosine, e.g. 0.9998")
    ap.add_argument("--verdict", required=True, choices=["COMPATIBLE", "DRIFT"])
    ap.add_argument("--ram-mb", required=True, type=float)
    ap.add_argument("--cold-s", required=True, type=float)
    ap.add_argument("--qdrant", required=True, help="text after the word holds, e.g. all 61 768 chunks, 38 MB used (dashboard)")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    cold = f"{a.cold_s:.0f}"
    edits = {
        "docs/ARCHITECTURE.md": [
            ("<worst cosine = 0.xxxx, verdict COMPATIBLE / DRIFT>", f"worst cosine = {a.cosine}, verdict {a.verdict}"),
            ("<measured MB> of 512 MB", f"{a.ram_mb:.0f} MB of 512 MB"),
            ("<measured seconds>", f"{cold} s"),
            ("holds all 61 768 chunks <confirm usage in the Qdrant dashboard>", f"holds {a.qdrant}"),
        ],
        "README.md": [("<measured> seconds", f"{cold} seconds")],
    }
    problems = 0
    for rel, pairs in edits.items():
        path = root / rel
        text = path.read_text(encoding="utf-8")
        for old, new in pairs:
            if old not in text:
                print(f"  !! {rel}: placeholder not found (already filled?): {old}")
                problems += 1
                continue
            text = text.replace(old, new, 1)
            print(f"  ok {rel}: {old}  ->  {new}")
        if not a.dry_run:
            path.write_text(text, encoding="utf-8")
    print("\n(dry run: nothing written)" if a.dry_run else "\nDocs updated. Review with: git diff")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
