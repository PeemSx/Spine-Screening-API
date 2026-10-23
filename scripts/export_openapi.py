"""Export the FastAPI OpenAPI contract deterministically."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.main import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "openapi" / "openapi.json"


def rendered_openapi() -> str:
    """Return stable, reviewable JSON without starting model runtimes."""
    schema = create_app().openapi()
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail when the export is stale")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    expected = rendered_openapi()
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != expected:
            print(f"OpenAPI export is stale: {output}", file=sys.stderr)
            print("run from the workspace root: make export-contract", file=sys.stderr)
            return 1
        print(f"OpenAPI export is current: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(expected, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
