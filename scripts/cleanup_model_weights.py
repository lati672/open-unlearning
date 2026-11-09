"""Remove saved model weights from one run after its evaluation succeeds."""

import argparse
from pathlib import Path


def remove_model_weights(run_dir: Path) -> None:
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError(f"Expected a run directory, got {run_dir}")

    weight_suffixes = (".safetensors", ".safetensor", ".safetensors.index.json")
    checkpoints = [
        path for path in sorted(run_dir.glob("checkpoint-*"))
        if not path.is_symlink() and path.is_dir()
    ]
    for directory in [run_dir, *checkpoints]:
        for path in sorted(directory.iterdir()):
            if path.name.endswith(weight_suffixes) and not path.is_symlink() and path.is_file():
                path.unlink(missing_ok=True)
                print(f"Removed model weight file {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Output directory of the evaluated run")
    remove_model_weights(parser.parse_args().run_dir)
