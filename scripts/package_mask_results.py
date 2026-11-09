"""Package mask audit results, experiment outputs, and their source snapshot.

python scripts/package_mask_results.py
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "unlearning_results"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "saves/exports/unlearning_results.zip")
    args = parser.parse_args()
    audit = ROOT / "saves/audits/revision"
    experiments = ROOT / "saves/experiments/mask_controls"
    validation = json.loads((audit / "validation/final_checks.json").read_text())
    paths = {p for folder in [audit, experiments] for p in folder.rglob("*") if p.is_file()}
    paths.update(ROOT / name for name in validation["source_sha256"])
    paths.add(Path(__file__).resolve())
    paths = sorted(paths)
    completed = len(list(experiments.glob("*/completed.json")))
    if completed != validation["completed_experiments"]:
        raise ValueError("Completed experiment count differs from validation record")
    manifest = {"completed_experiments": completed, "files": [
        {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": digest(p)}
        for p in paths
    ]}
    readme = """Unlearning 实验结果包

建议从 saves/audits/revision/review_summary.txt 开始阅读。

主要报告：
- experiment_comparison.csv：15 组训练与评估结果。
- mask_audit_summary.csv：TOFU、MUSE News/Books mask 审计。
- token_categories.csv：启发式 token 分类统计。
- qualitative.html / qualitative.png：selected-token 高亮案例。
- selection_forgetting.png：选择比例与遗忘 NLL 变化。
- entity_alignment_*.jsonl：question、answer、tokens、labels、原始/有效 mask。
- tokens_*.jsonl：逐 token discrepancy、eligible、selected 等完整数据。
- analysis.json：详细分析、相关性和旧 RMU normalization 敏感性。
- validation/：测试日志与验证记录。

每组实验原始输出位于 saves/experiments/mask_controls/：
- <run>/evals/metrics.json：指标汇总。
- <run>/evals/forget05.json：逐样本生成答案及评分。
- <run>/evals/retain95.json、holdout05.json：逐样本概率与 NLL。
- <run>/resolved_config.json：运行配置。
- <run>/trainer_state.json：步数、epoch 和训练历史。
- <run>.log：训练和评估日志。
- baseline/：训练前基线评估。

src/、configs/、tests/、scripts/ 和 README.md 是相关实现及复用脚本的
源文件快照。PACKAGE_MANIFEST.json 记录每个文件的大小及 SHA256。

范围：训练比较为 TOFU forget05、seed=0；评估采用重点 QA 指标。
旧 TOFU forget01 E10 的 batch=4、累积=4 来源于用户确认，旧代码版本
尚未读取。旧 bug 影响为敏感性模拟，并非历史指标变化的精确重建。
Token 分类为启发式；MUSE 语义分类没有经过有效对齐验证。
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".zip.part")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, f"{PREFIX}/{path.relative_to(ROOT)}")
        archive.writestr(f"{PREFIX}/START_HERE.txt", readme)
        archive.writestr(f"{PREFIX}/PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    with zipfile.ZipFile(temporary) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP integrity check failed: {bad}")
        if len(archive.namelist()) != len(paths) + 2:
            raise ValueError("ZIP member count differs from packaged file count")
    temporary.replace(args.output)
    checksum = digest(args.output)
    args.output.with_suffix(".zip.sha256").write_text(f"{checksum}  {args.output.name}\n")
    print(json.dumps({"zip": str(args.output), "bytes": args.output.stat().st_size,
                      "files": len(paths) + 2, "experiments": completed,
                      "integrity_check": "passed", "sha256": checksum}, indent=2), flush=True)


if __name__ == "__main__":
    main()
