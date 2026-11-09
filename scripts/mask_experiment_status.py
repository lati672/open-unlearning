"""Compact status for the reusable mask experiment and audit outputs."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "saves/experiments/mask_controls")
    parser.add_argument("--audit-root", type=Path, default=ROOT / "saves/audits/revision")
    args = parser.parse_args()
    completed = sorted(p.parent.name for p in args.output_root.glob("*/completed.json"))
    result = {"completed_count": len(completed), "completed": completed}
    pending = [p for p in args.output_root.glob("*.log") if p.stem not in completed]
    if pending:
        active = max(pending, key=lambda p: p.stat().st_mtime)
        text = active.read_text()[-10000:]
        steps = re.findall(r"(\d+)/(\d+) \[", text)
        evaluations = re.findall(r"Evaluate (\w+)", text)
        result["active"] = active.stem
        result["phase"] = f"evaluation {evaluations[-1]}" if evaluations else f"training {steps[-1][0]}/{steps[-1][1]}" if steps else "initializing"
        result["traceback_in_recent_log"] = "Traceback (most recent call last)" in text
    path = args.audit_root / "audit_report.json"
    if path.exists():
        result["audits_completed"] = list(json.loads(path.read_text())["groups"])
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
