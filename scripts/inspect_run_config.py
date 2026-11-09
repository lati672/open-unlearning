"""Inspect old experiment settings without unpickling arbitrary Python objects.

Portable usage (standard library only):
python3 scripts/inspect_run_config.py /path/to/old/run
Also accepts a training_args.bin, JSON, YAML, or log file directly.
"""
import argparse
import json
from pathlib import Path
import pickletools
import re
import zipfile

KEYS = {"per_device_train_batch_size", "per_gpu_train_batch_size", "gradient_accumulation_steps",
        "train_micro_batch_size_per_gpu", "train_batch_size", "effective_batch_size", "num_train_epochs",
        "num_processes", "world_size", "learning_rate", "weight_decay", "seed", "data_seed", "local_rank",
        "bf16", "fp16", "dataloader_drop_last", "global_step", "max_steps", "epoch"}
NAMES = {"training_args.bin", "training_args.json", "trainer_state.json", "config.yaml", "config.yml",
         "hydra.yaml", "overrides.yaml", "args.json", "resolved_config.json", "config.json"}


def primitive_pickle_fields(payload):
    """Read opcode literals; never execute pickle GLOBAL/REDUCE operations."""
    result, offset = {}, 0
    while offset < len(payload):
        pending = None
        last = None
        try:
            for opcode, arg, position in pickletools.genops(payload[offset:]):
                last = (opcode.name, position)
                if opcode.name in {"UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8", "STRING", "BINSTRING", "SHORT_BINSTRING"}:
                    value = arg.decode() if isinstance(arg, bytes) else arg
                    pending = value if value in KEYS else None
                elif pending and opcode.name in {"INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4", "FLOAT", "BINFLOAT", "NEWTRUE", "NEWFALSE", "NONE"}:
                    value = True if opcode.name == "NEWTRUE" else False if opcode.name == "NEWFALSE" else None if opcode.name == "NONE" else arg
                    result[pending] = value
                    pending = None
                elif opcode.name not in {"BINPUT", "LONG_BINPUT", "MEMOIZE", "FRAME"}:
                    pending = None
        except (ValueError, UnicodeDecodeError):
            break
        if last is None or last[0] != "STOP":
            break
        offset += last[1] + 1
    return result


def inspect_binary(path):
    if path.stat().st_size > 8 * 1024**2:
        return {"note": "Skipped unexpectedly large training_args.bin"}
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            member = next((m for m in archive.infolist() if m.filename.endswith("/data.pkl") or m.filename == "data.pkl"), None)
            if member is None or member.file_size > 8 * 1024**2:
                return {"note": "No bounded data.pkl member found"}
            payload = archive.read(member)
    else:
        payload = path.read_bytes()
    return primitive_pickle_fields(payload)


def flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "log_history":
                continue
            full = f"{prefix}.{key}" if prefix else key
            if key in KEYS and not isinstance(item, (dict, list)):
                result[full] = item
            result.update(flatten(item, full))
    return result


def inspect_text(path):
    if path.stat().st_size > 4 * 1024**2:
        return {"note": "Skipped large text file; use a small configuration file"}
    text = path.read_text(errors="replace")
    try:
        return flatten(json.loads(text))
    except json.JSONDecodeError:
        pattern = re.compile(r"(?:^|[.\s\"'])?(" + "|".join(sorted(KEYS, key=len, reverse=True)) + r")[\"']?\s*[:=]\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|true|false|null|None)", re.MULTILINE)
        result = {}
        for key, value in pattern.findall(text):
            value = {"None": None, "null": None, "true": True, "false": False}.get(value, value)
            if isinstance(value, str):
                value = float(value) if any(c in value for c in ".eE") else int(value)
            result[key] = value
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    root = args.path.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Path is not accessible on this machine: {root}")
    paths = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file() and (p.name in NAMES or p.suffix == ".log")]
    files = []
    for path in sorted(paths):
        settings = inspect_binary(path) if path.name == "training_args.bin" else inspect_text(path)
        if settings:
            files.append({"file": str(path), "settings": settings})
    print(json.dumps({"run": str(root), "files": files,
                      "note": "per-device micro-batch is distinct from gradient accumulation and effective batch size; directory names do not establish actual settings",
                      "top_level_entries": sorted(p.name for p in root.iterdir())[:80] if root.is_dir() else []}, indent=2))


if __name__ == "__main__":
    main()
