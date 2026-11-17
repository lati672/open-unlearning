"""Generate the portable TOFU Forget05 Adaptive mask relevance report and data bundle."""

import argparse
from base64 import b64encode
import csv
from hashlib import file_digest
from html import escape
import json
import math
from pathlib import Path
import shutil
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "saves/audits/adaptive_relevance"
FORMAL = ROOT / "saves/experiments/tofu_forget05_multiseed_v4_official"
TARGET = ROOT / "saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget05/TOFU_EVAL.json"
NAME = "tofu_forget05_adaptive_mask_relevance"
GROUPS = [("NPO", "summed", "Summed NPO"), ("NPO", "token_mean", "Token-normalized NPO"), ("RMU", "token_mean", "RMU")]
VARIANTS = {"base": "Base", "adaptive": "Adaptive", "random_adaptive": "Random-Adaptive"}
METRICS = {
    "ES": "ES ↓", "forget_probability": "F-Prob ↓", "forget_rouge": "F-ROUGE ↓",
    "forget_ratio": "F-Ratio ↑", "forget_quality": "F-Quality ↑",
    "utility": "Utility ↑", "privleak_abs": "|PrivLeak| ↓",
}
CATEGORIES = {
    "entity_name": "作者名", "factual_attribute": "答案属性（含描述性片段）",
    "other_content": "其他内容词", "function": "功能词和控制 token", "punctuation": "标点",
    "factual_attribute_direct_only": "具体事实值（答案属性的子集）",
}
RELEVANCE_METRICS = {"nll_increase": "NLL 增量", "probability_drop": "F-Prob 降幅",
                     "rougeL_recall_drop": "ROUGE-L recall 降幅", "extraction_strength_drop": "ES 降幅"}


def read(path):
    return json.loads(path.read_text())


def table_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def pct(value):
    return f"{float(value):.2%}"


def numeric(value):
    value = float(value)
    return f"{value:.2e}" if value and abs(value) < 0.0001 else f"{value:.4f}"


def table(headers, rows):
    header = "".join(f'<th scope="col">{escape(str(value))}</th>' for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def figure(path, alt, caption):
    data = b64encode(path.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{escape(alt)}"><figcaption>{escape(caption)}</figcaption></figure>'


def verify_inputs():
    summary = read(AUDIT / "summary.json")
    lock = read(FORMAL / "protocol_manifest.json")
    check(summary["training_git_commit"] == lock["code_commit"] and summary["protocol_sha256"] == lock["protocol_sha256"],
          "Adaptive relevance and official results belong to different protocols")
    check(summary["annotations_sha256"] == sha(AUDIT / "answer_spans.tsv")
          and summary["archived_target_eval_sha256"] == sha(TARGET), "Annotations or target baseline hash changed")
    data = {name: table_rows(AUDIT / f"{name}.csv") for name in [
        "tokens", "fact_coverage", "correlations", "forgetting_change_by_sample", "token_categories", "overlap_by_sample",
    ]}
    official = FORMAL / "multiseed_export"
    data.update({name: table_rows(official / f"{name}.csv") for name in [
        "multiseed_official_long", "paired_comparisons", "step60_mean_std", "step120_mean_std",
    ]})
    checks = {
        "tokens": 7165, "forgetting_change_by_sample": 14400, "correlations": 192, "overlap_by_sample": 200,
        "multiseed_official_long": 90, "paired_comparisons": 168, "step60_mean_std": 120, "step120_mean_std": 120,
    }
    for key, size in checks.items():
        check(len(data[key]) == size, f"Unexpected {key} row count")
    official_rows = data["multiseed_official_long"]
    check(len({(r["objective"], r["normalization"], r["variant"], r["seed"], r["step"]) for r in official_rows}) == 90,
          "Duplicate or missing official checkpoints")
    check(all(math.isfinite(float(row[key])) for row in official_rows for key in [*METRICS, "privleak_signed"]),
          "Nonfinite official metric")
    check(all(float(row["privleak_abs"]) == abs(float(row["privleak_signed"])) for row in official_rows),
          "PrivLeak absolute value differs")
    tokens = data["tokens"]
    check(len({int(r["doc_id"]) for r in tokens}) == 200 and len({r["author"] for r in tokens}) == 10,
          "Token audit cohort differs")
    check(sum(r["adaptive"] == "True" for r in tokens) == summary["overlap"]["adaptive_tokens"],
          "Adaptive selected count differs")
    check(all((float(r["delta_base_minus_target"]) < float(r["sequence_mean"])) == (r["adaptive"] == "True")
              for r in tokens), "Frozen Adaptive mask differs from strict below-mean rule")
    grouped = {}
    for r in tokens:
        key = int(r["doc_id"])
        count = grouped.setdefault(key, [0, 0])
        count[0] += r["adaptive"] == "True"
        count[1] += r["random_adaptive"] == "True"
    check(all(a == b for a, b in grouped.values()), "Frozen Random-Adaptive density differs")
    return summary, data, checks


def official_table(data, step):
    values = {(r["objective"], r["normalization"], r["variant"], r["metric"]): r
              for r in data[f"step{step}_mean_std"]}
    lines = []
    for obj, norm, label in GROUPS:
        for variant, shown in VARIANTS.items():
            stats = [values[obj, norm, variant, metric] for metric in METRICS]
            check(all(row["n"] == "3" for row in stats), "Incomplete official summary seeds")
            lines.append([label, shown, *[f"{numeric(r['mean'])} ± {numeric(r['std'])}" for r in stats]])
    return table(["Objective", "Variant", *METRICS.values()], lines)


def paired_table(data, step):
    values = {(r["objective"], r["normalization"], r["rhs"], r["metric"]): r
              for r in data["paired_comparisons"] if r["step"] == str(step) and r["lhs"] == "adaptive"}
    lines = []
    for obj, norm, label in GROUPS:
        for control in ("random_adaptive", "base"):
            cells = []
            for metric in METRICS:
                row = values[obj, norm, control, metric]
                mark = " *" if row["direction_consistent"] == "True" else ""
                cells.append(f"{float(row['mean_difference']):+.4g} ± {float(row['std_difference']):.3g}{mark}")
            lines.append([label, f"Adaptive − {VARIANTS[control]}", *cells])
    return table(["Objective", "配对差", *METRICS.values()], lines)


def correlation_table(data):
    values = {(r["objective"], r["metric"], r["contrast"]): r for r in data["correlations"]
              if r["step"] == "120" and r["seed"] == "mean_of_3"}
    lines = []
    for _, _, label in GROUPS:
        def text(metric, contrast, interval=False):
            row = values[label, metric, contrast]
            result = f"{float(row['pearson_r']):+.3f}"
            if interval:
                result += f" [{float(row['pearson_author_bootstrap_low']):+.3f}, {float(row['pearson_author_bootstrap_high']):+.3f}]"
            return result

        lines.append([label, *[text(metric, "adaptive_vs_target", metric == "nll_increase") for metric in RELEVANCE_METRICS],
                      text("nll_increase", "adaptive_minus_random", True)])
    return table(["Objective", *RELEVANCE_METRICS.values(), "额外 NLL 增量 vs Random-Adaptive"], lines)


def render(summary, data):
    dist = summary["discrepancy"]
    overlap = summary["overlap"]
    raw_categories = {(r["category"], r["mask"]): r for r in data["token_categories"]}
    categories = []
    for key, title in CATEGORIES.items():
        a, r = [raw_categories[key, mask] for mask in ["adaptive", "random_adaptive"]]
        categories.append([title, a["eligible"], a["selected"], pct(a["coverage"]),
                           pct(a["selected_composition"]), pct(r["coverage"])])
    coverage = [[
        title, f"{row['adaptive']}/{row['n']}", pct(row["adaptive_coverage"]), pct(row["random_coverage"]),
        pct(row["random_expected_coverage_given_counts"]), f"{100 * float(row['paired_gap']):+.2f}",
        f"[{100 * float(row['author_bootstrap_low']):+.2f}, {100 * float(row['author_bootstrap_high']):+.2f}]",
    ] for title, row in zip(["全部答案属性", "具体事实值", "描述性片段"], data["fact_coverage"], strict=True)]
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Adaptive Mask Relevance · TOFU Forget05</title>
<style>
:root{{color-scheme:light;--ink:#182a39;--muted:#486172;--accent:#0e6a69;--rule:#d8e4eb;--paper:#f5f8fb}}
*{{box-sizing:border-box}}body{{background:var(--paper);color:var(--ink);font:15px/1.66 system-ui,"PingFang SC","Microsoft Yahei",sans-serif;margin:0}}
main{{max-width:1180px;margin:auto;padding:36px 24px 56px}}header{{background:#e8f3f2;border:1px solid #cee3e2;border-radius:16px;padding:25px 31px 22px}}
h1{{font-size:clamp(1.8rem,5vw,2.65rem);letter-spacing:-.036em;line-height:1.12;margin:.22em 0}}h2{{font-size:1.36rem;margin:1.55em 0 .45em}}h3{{font-size:1.04rem}}
.kicker{{color:var(--accent);text-transform:uppercase;font-weight:750;letter-spacing:.09em;font-size:.83rem}}.muted,figcaption{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:18px 0}}
.card{{background:white;border:1px solid var(--rule);border-radius:10px;padding:12px 16px}}.card strong{{display:block;font-size:1.42rem;color:var(--accent)}}
.table-wrap{{overflow-x:auto;background:white;border:1px solid var(--rule);border-radius:9px;margin:.6em 0 1.1em}}
table{{width:100%;border-collapse:collapse;font-size:.86rem;font-variant-numeric:tabular-nums;white-space:nowrap}}
td,th{{padding:.58em .7em;border-bottom:1px solid #e6edf2;text-align:right}}td:first-child,th:first-child{{text-align:left}}
thead{{background:#eaf2f5}}tbody tr:nth-child(even){{background:#f8fafb}}tbody tr:last-child td{{border-bottom:0}}
figure{{margin:1em 0;background:white;border:1px solid var(--rule);border-radius:10px;padding:10px 14px}}figure img{{width:100%;height:auto;display:block}}
figcaption{{padding-top:5px;font-size:.87rem}}a{{color:#0b6564}}details{{padding:.7em 1em;border:1px solid var(--rule);border-radius:8px;background:white}}
summary{{cursor:pointer;color:var(--accent);font-weight:700}}code{{overflow-wrap:anywhere}}
@media print{{body{{background:white}}main{{max-width:none;padding:0}}header,figure,table{{break-inside:avoid}}.table-wrap{{overflow:visible}}table{{font-size:8pt}}*{{print-color-adjust:exact}}}}
</style>
</head>
<body><main>
<header><div class="kicker">Mask audit · Official three-seed evaluation</div>
<h1>Adaptive Mask Relevance</h1>
<p>TOFU Forget05 / Retain95 · Llama-3.2-1B-Instruct · 固定 mask 和已保存的逐样本评估</p>
</header>
<div class="cards">
<div class="card"><strong>{pct(summary["selection_ratio"])}</strong>Adaptive 有效 token 选择比例</div>
<div class="card"><strong>+23.04 pp</strong>具体事实值覆盖率相对等密度随机基线</div>
<div class="card"><strong>{pct(overlap["micro_jaccard"])}</strong>Adaptive 与 Entity 的 micro Jaccard</div>
<div class="card"><strong>3 seeds</strong>Step 120 为主要终点</div>
</div>

<h2>1. 范围与方法</h2>
<p>审计覆盖 {summary["authors"]} 位虚构作者的 {summary["samples"]} 条问答，共 {summary["eligible_tokens"]:,} 个有效答案目标。有效位置要求监督标签存在、当前 token 和前驱都被 attention 覆盖，并按位置 j 的目标对应位置 j−1 的 logits 对齐。真实受监督的 EOT 归入控制 token。</p>
<p>Adaptive 在每条样本内选择 Δ = log p(base) − log p(target) 严格低于该样本有效位置均值的 token。Random-Adaptive 从相同有效位置采样，逐条问答匹配数量。Entity 使用训练时冻结的作者名匹配 mask。</p>
<p>答案片段由单人按字符标注并投影到 tokenizer 位置。95 条有具体值标注，90 条有描述性片段，15 条没有非作者名答案片段。具体值包括书名、奖项、地点和职业。该标注反映问答中承载答案的片段，类别结果限于本次 Forget05 数据。</p>

<h2>2. Discrepancy 与 mask 重叠</h2>
{table(["组别", "Token 数", "均值", "P05", "中位数", "P95", "负值占比"],
        [["选中" if r["group"] == "selected" else "未选中", f"{r['n']:,}", numeric(r["mean"]),
          numeric(r["p05"]), numeric(r["p50"]), numeric(r["p95"]), pct(r["fraction_negative"])] for r in dist])}
<p>所有选中 token 的 Δ 均为负。差值分布反映了 below-mean 规则的行为。以 Entity 为正例，交集为 {overlap["intersection"]} 个 token，precision 为 <strong>{pct(overlap["micro_precision"])}</strong>，recall 为 <strong>{pct(overlap["micro_recall"])}</strong>，micro Jaccard 为 <strong>{pct(overlap["micro_jaccard"])}</strong>，macro Jaccard 为 {pct(overlap["macro_jaccard"])}。</p>
{figure(AUDIT / "discrepancy_and_coverage.png", "Discrepancy 累计分布及五类 token 覆盖率", "图 1　选中与未选中 token 的经验累积分布，以及各类别相对等密度随机 mask 的覆盖率。")}

<h2>3. 事实属性覆盖</h2>
{table(["类别", "有效 token", "Adaptive 选中", "Adaptive 类内覆盖", "占 Adaptive 选中", "Random 类内覆盖"], categories)}
{table(["标注范围", "Adaptive 选中/总量", "Adaptive", "冻结 Random", "等密度随机期望", "差值（pp）", "作者重采样 95% 区间（pp）"], coverage)}
<p>具体事实值的覆盖差为 <strong>+23.04 个百分点</strong>。全部答案属性占 Adaptive 选中 token 的 28.47%，其他内容词占 40.42%。覆盖率以该类别的有效 token 为分母，选中构成以 Adaptive 选中的全部 token 为分母。随机期望按每条问答的实际选择密度计算，区间对 10 个作者分组做 2,000 次重采样。</p>
<p><a href="examples/token_selection_examples.html">查看 8 个高亮案例</a>，包括事实全覆盖、完全漏选、部分覆盖及作者名重叠。</p>

<h2>4. 选择比例与逐样本遗忘变化</h2>
<p>每条问答的选择比例与相对初始 target 的遗忘变化配对。正值表示平均答案 NLL 增加，或 Probability、ROUGE-L recall、ES 下降。下表先在每条问答内平均三个训练 seed，再跨 200 条问答计算 Pearson r。方括号为作者重采样 95% 区间。</p>
{correlation_table(data)}
<p>选择比例与 NLL 增量在 Token-normalized NPO 和 RMU 中为正相关，在 Summed NPO 中接近零。以等密度随机对照为基准，NLL 额外变化的相关性分别为 −0.470、+0.025、+0.077。结果随 objective 和遗忘指标变化。数据包另含 Spearman、控制有效长度和初始 NLL 的调整结果，以及每个 seed 和 Step 60 的结果。</p>
{figure(AUDIT / "forgetting_correlations_step120.png", "Step 120 选择比例与逐样本 NLL 增量散点图", "图 2　每点是一条问答，NLL 增量先在三个 seed 内平均。拟合线描述关联。")}

<h2>5. 正式多种子指标</h2>
<p>Step 120 是主要终点。下表展示各方法的均值 ± 样本标准差（n=3）。PrivLeak 比较使用绝对值，CSV 同时保留原始 signed 值。</p>
{official_table(data, 120)}
<h3>Step 120 配对差</h3>
<p>差值按同一 seed 计算，方向是 Adaptive 减对照。星号表示三个 seed 的差值符号一致。与 Random-Adaptive 相比，Token-normalized NPO 的 F-Prob 降低 0.0552，Utility 也降低 0.0733。Summed NPO 的 F-Prob 升高 0.0158，RMU 则降低 0.0102。事实覆盖的富集与训练指标是两组不同的证据。</p>
{paired_table(data, 120)}
<details><summary>Step 60 训练动态和配对差</summary>
{official_table(data, 60)}
{paired_table(data, 60)}
</details>

<h2>6. 数据与复现</h2>
<ul>
<li><a href="data/token_audit/tokens.csv">逐 token 数据</a>、<a href="data/token_audit/samples.csv">200 条问答及 sample ID</a>、<a href="data/token_audit/answer_spans.tsv">事实片段标注</a>。</li>
<li><a href="data/token_audit/forgetting_change_by_sample.csv">14,400 行逐样本变化</a>、<a href="data/token_audit/correlations.csv">192 行相关性统计</a>、<a href="data/token_audit/fact_coverage.csv">事实覆盖与区间</a>。</li>
<li><a href="data/official/multiseed_official_long.csv">90 行正式 checkpoint 指标</a>、<a href="data/official/per_seed_results.csv">逐 seed 指标</a>、<a href="data/official/paired_comparisons.csv">全部配对差</a>。</li>
<li><a href="figures/discrepancy_and_coverage.pdf">图 1 PDF</a>、<a href="figures/forgetting_correlations_step120.pdf">图 2 PDF</a>、<a href="MANIFEST.json">文件清单与 SHA-256</a>。</li>
</ul>
<p class="muted">正式训练 Git commit <code>{escape(summary["training_git_commit"])}</code>。协议 SHA-256 <code>{escape(summary["protocol_sha256"])}</code>。复现脚本可在仓库根目录运行。相关性是同一数据集上的描述性分析。</p>
</main></body></html>
"""


def assemble(stage, summary, data, checks):
    def copy(source, relative):
        path = stage / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, path)

    for name in ["answer_spans.tsv", "correlations.csv", "discrepancy_quantiles.csv", "examples.csv", "fact_coverage.csv",
                 "forgetting_change_by_sample.csv", "overlap_by_sample.csv", "summary.json", "token_categories.csv", "tokens.csv"]:
        copy(AUDIT / name, f"data/token_audit/{name}")
    for name in ["multiseed_official_long.csv", "paired_comparisons.csv", "per_seed_results.csv", "step60_mean_std.csv",
                 "step120_mean_std.csv", "multiseed_verification.json"]:
        copy(FORMAL / "multiseed_export" / name, f"data/official/{name}")
    for name in ["protocol_manifest.json", "run_plan.json", "environment.json"]:
        copy(FORMAL / name, f"provenance/{name}")
    for variant in ["base", "adaptive", "entity", "random_adaptive"]:
        name = f"masks_TOFU_forget05_{variant}.jsonl"
        copy(FORMAL / "frozen_inputs" / name, f"data/frozen_masks/{name}")
    copy(TARGET, "data/official/target_before_unlearning_TOFU_EVAL.json")
    for name in ["discrepancy_and_coverage", "forgetting_correlations_step120"]:
        for suffix in ["png", "pdf"]:
            copy(AUDIT / f"{name}.{suffix}", f"figures/{name}.{suffix}")
    copy(AUDIT / "token_selection_examples.html", "examples/token_selection_examples.html")
    copy(AUDIT / "relevance_report.txt", "provenance/original_audit_summary.txt")
    for name in ["analyze_adaptive_relevance.py", Path(__file__).name]:
        copy(ROOT / "scripts" / name, f"reproduce/{name}")
    authors = {int(r["doc_id"]): r["author"] for r in data["tokens"]}
    fields = ["doc_id", "author", "question", "answer", "sequence_mean", "eligible_count", "selected_count", "selection_ratio"]
    with (stage / "data/token_audit/samples.csv").open("w", newline="") as out, (
        FORMAL / "frozen_inputs/tokens_TOFU_forget05.jsonl"
    ).open() as source:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for line in source:
            row = json.loads(line)
            writer.writerow({**{key: row[key] for key in fields if key != "author"}, "author": authors[row["doc_id"]]})
    (stage / "report.html").write_text(render(summary, data), encoding="utf-8")
    (stage / "README.txt").write_text(
        "Open report.html after extracting this ZIP. The charts in the report work offline.\n"
        "data/token_audit contains token annotations and paired sample changes.\n"
        "data/official contains all formal aggregate rows and the archived target baseline.\n"
        "The per-sample changes preserve the original metric values and derived contrasts.\n"
        "Run the scripts under reproduce/ from the open-unlearning checkout with the source artifacts.\n"
        "MANIFEST.json records the included file hashes.\n", encoding="utf-8"
    )
    manifest = {"dataset": "TOFU Forget05", "primary_step": 120, "seeds": [0, 1, 2],
                "counts": checks, "protocol_sha256": summary["protocol_sha256"],
                "files": {str(path.relative_to(stage)): sha(path) for path in sorted(stage.rglob("*")) if path.is_file()}}
    (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "saves/exports")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary, data, checks = verify_inputs()
    archive = args.output / f"{NAME}.zip"
    pending = archive.with_suffix(".zip.part")
    with tempfile.TemporaryDirectory(prefix=".adaptive-report-", dir=args.output) as directory:
        stage = Path(directory) / NAME
        stage.mkdir()
        manifest = assemble(stage, summary, data, checks)
        with ZipFile(pending, "w", compression=ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    bundle.write(path, f"{NAME}/{path.relative_to(stage)}")
        with ZipFile(pending) as bundle:
            check(len(bundle.namelist()) == len(manifest["files"]) + 1, "Archive member count differs")
            for name, digest in manifest["files"].items():
                with bundle.open(f"{NAME}/{name}") as stream:
                    check(file_digest(stream, "sha256").hexdigest() == digest, f"Archive member hash differs: {name}")
        report = args.output / NAME
        shutil.copytree(stage, report, dirs_exist_ok=True)
    pending.replace(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{sha(archive)}  {archive.name}\n")
    print(f"{archive} ({archive.stat().st_size:,} bytes, {len(manifest['files'])} hashed files)")
    print(report / "report.html")


if __name__ == "__main__":
    main()
