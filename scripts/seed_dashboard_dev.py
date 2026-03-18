from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MISSIONS = [
    {"id": 1, "name": "Bootstrap Dashboard", "status": "seeded", "progress": 100},
    {"id": 2, "name": "Verify Capabilities", "status": "ready", "progress": 25},
    {"id": 3, "name": "Exercise Fallback States", "status": "pending", "progress": 0},
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed explicit development data for dashboard APIs."
    )
    parser.add_argument(
        "--missions-path",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "dev"
        / "dashboard"
        / "missions.json",
    )
    args = parser.parse_args()

    args.missions_path.parent.mkdir(parents=True, exist_ok=True)
    args.missions_path.write_text(json.dumps(DEFAULT_MISSIONS, indent=2) + "\n", encoding="utf-8")
    print(f"Seeded missions fixture at {args.missions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
