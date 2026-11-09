"""Refresh pre-EOS-fix TOFU logprobs/masks without discarding valid controls.

Pause the experiment runner before invoking this command. A running Entity
control can finish: it does not depend on reference log probabilities.
python scripts/refresh_attention_cache.py --gpu 1
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="1")
    args = parser.parse_args()
    audit = ROOT / "saves/audits/revision"
    refreshed = ROOT / "saves/audits/tofu_attention_refreshed"
    experiments = ROOT / "saves/experiments/mask_controls"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = ROOT / "saves/audits/pre_attention_refresh" / stamp
    backup.mkdir(parents=True)
    for path in (ROOT / "saves/logprobs").glob("TOFU_*"):
        shutil.copytree(path, backup / "logprobs" / path.name)
    for path in audit.glob("*TOFU*"):
        (backup / "audit").mkdir(exist_ok=True)
        shutil.copy2(path, backup / "audit" / path.name)
    if (audit / "audit_report.json").exists():
        shutil.copy2(audit / "audit_report.json", backup / "audit_report.json")
    for path in experiments.glob("*adaptive*"):
        (backup / "experiments").mkdir(exist_ok=True)
        path.rename(backup / "experiments" / path.name)
    print("Archived old probability-dependent results:", backup, flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["TOKENIZERS_PARALLELISM"] = "false"
    subprocess.run([sys.executable, "setup_data.py", "--tofu-logprobs"], cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, "scripts/audit_masks.py", "--datasets", "TOFU", "--device", "cuda:0",
                    "--output-dir", str(refreshed)], cwd=ROOT, env=env, check=True)
    before = json.loads((backup / "audit_report.json").read_text())
    after = json.loads((refreshed / "audit_report.json").read_text())
    for group, row in after["groups"].items():
        previous = before["groups"][group]
        previous_commit = previous.get("target_model_commit", previous.get("model_commit"))
        assert row["target_model_commit"] == previous_commit, "Target checkpoint changed; all controls must be rerun"
    for path in refreshed.glob("masks_*_entity.jsonl"):
        previous = audit / path.name
        assert [json.loads(line) for line in previous.open()] == [json.loads(line) for line in path.open()], "Entity masks changed: Entity controls also need rerunning"
    for path in refreshed.glob("*TOFU*"):
        shutil.copy2(path, audit / path.name)
    # Check that old MUSE probabilities are mathematically reusable. In a causal
    # model, excluding only the terminal EOS key cannot affect any scored target.
    # Internal EOS/EOT changes WOULD invalidate later token probabilities.
    import torch
    from omegaconf import OmegaConf
    from data import get_datasets
    from model import get_tokenizer
    cfg = OmegaConf.load(ROOT / "configs/model/Llama-2-7b-hf.yaml")
    tokenizer = get_tokenizer(cfg.tokenizer_args)
    muse = {}
    for name in ["News", "Books"]:
        data_cfg = OmegaConf.load(ROOT / "configs/data/datasets/MUSE_forget.yaml")
        data_cfg.MUSE_forget.args.hf_args.path = f"muse-bench/MUSE-{name}"
        dataset = get_datasets(data_cfg, tokenizer=tokenizer, template_args=cfg.template_args)
        mismatches = 0
        for index in range(len(dataset)):
            sample = dataset[index]
            old = sample["input_ids"].ne(tokenizer.pad_token_id)
            new = sample["attention_mask"].bool()
            mismatches += int((old[:-1] != new[:-1]).any())
        assert mismatches == 0, f"MUSE {name} contains changed internal attention keys: regenerate MUSE logprobs/audits too"
        muse[name] = {"samples": len(dataset), "changed_internal_attention_samples": mismatches,
                      "legacy_logprobs_equivalent_for_all_scored_targets": True}
    report = json.loads((refreshed / "audit_report.json").read_text())
    info = {"backup": str(backup), "TOFU_groups": report["groups"], "MUSE_attention_equivalence": muse,
            "note": "TOFU pre-fix logprobs masked internal chat EOT keys and were regenerated. MUSE had only terminal EOS differences, proven harmless to causal next-token scores. Base/Entity controls are retained; old Adaptive-dependent results were archived."}
    (audit / "attention_refresh.json").write_text(json.dumps(info, indent=2) + "\n")
    # Another ongoing MUSE audit may still hold its old TOFU summaries in memory.
    # The final analyzer reapplies this independently stored TOFU refresh ledger.
    print("Attention refresh completed; run the experiment runner again to resume valid controls.", flush=True)


if __name__ == "__main__":
    main()
