# MUSE Books Attribute-only vs. Random–Attribute：限定 pilot

**建议：暂不直接进入正式三 seed。** 标注/对齐/训练管线通过，只有很弱的文本 NLL 方向性变化，没有出现 forget QA 优势。

本轮仅 seed0，两组各 4 次 optimizer updates；全部结果均为 **pilot diagnostics**，不能用作 Table 5 正式结果，不能声称统计显著性、隐私优势或稳定提升。论文、Table 5、responses 和旧正式实验未修改；本轮保留模型与旧实验。

## 协议与输入

- 核对原 corrected token-mean RMU 的冻结源代码和配置：Llama-2-7B Books target，revision `c8dd3fb23a726762ec66d277933c7cff6767f3c2`；更新 layers5–7，在 layer7 计算 loss；steering=1、retain coefficient=3。
- 原模型的六个 FP32 存储分片在迁移后缺失；已补下载相同固定 revision，并用上游 LFS SHA-256 全部校验。加载为 BF16，不改变目标 checkpoint。
- 实际 raw forget/retain1 经原预处理分别得到 553/105 个块。forget 32 和 retain16 用 floor(i*(N-1)/(n-1)) 等间隔选取；ID、原文及 token 哈希保存于 inputs/。
- 保留原 max_length=2048 文本 token + BOS 的规则，最长实际输入2049，BOS不合格。32个 forget 输入均与原冻结文件一致。
- 本次只用 GPU0 L40S，microbatch2、accumulation8，有效 batch16。两组分别从原 target 初始化，同一64次 forget/retain访问顺序；LR1e-5、constant、无warmup/decay、BF16、PagedAdamW32bit、clip1。
- 保持原 RMU per-example token-mean activation MSE 和 retain项；逐样本误差和 token-count 都是 [B]，mask 直接对齐 hidden state h_j，不使用 NPO 的 j-1 位移。
- 为单卡运行，缓存原 target 的精确 retain microbatch layer7 激活，逐文件 hash 固定；两组初始化均与缓存做零容差复核。只让优化器范围内参数 requires_grad，采用 non-reentrant checkpointing 保持 layer-hook autograd；不宣称与原双卡布局数值等价。

## 标注与 mask

- 仅查看 forget 训练文本，由当前 Codex 助手生成非穷尽事实标注，每块最多4处：共 126 spans。没有使用 QA 答案指导选择，未根据结果修改标注。
- prompt、来源说明、原文、字符起止位置及可读预览已保存。当前接口不提供可确认的生成模型ID/版本/effort，记录为未知；没有借用旧TOFU会话的模型信息。自查和原文匹配不记为人工审核。
- 126个span均唯一原文匹配；取重叠eligible token并集，两组每块严格等数：各 1771/64260，**2.7560%**；空mask **0**。
- 单样本密度 1.660%–6.218%；span长度中位数50.0字符/13.0 tokens，范围5–33 tokens。
- Random seed20261122，逐doc ID派生NumPy RNG，无放回抽样。所有输入、标注、mask、顺序均在任何模型诊断前以FROZEN.json哈希冻结。

## 固定子集结果

| 模型 | Forget NLL | Retain NLL | FK-R (16题) | Ret-R (16题) |
|---|---:|---:|---:|---:|
| original | 0.972803320 | 0.047161275 | 0.559592 | 0.734051 |
| attribute | 0.973709283 | 0.047153167 | 0.559592 | 0.671551 |
| random_attribute | 0.973391052 | 0.047147141 | 0.559592 | 0.671551 |

Attribute-only − Random：Forget NLL **+0.000318231**，Retain NLL **+0.000006026**；两个QA差值均为0。NLL为固定训练文本上eligible标签的逐token加权均值，不是官方Utility。

两组 forget QA 的逐题生成和评分均与原始模型相同。Retain QA 的下降均来自同一道题从1分变0分，即 −1/16=−0.0625；两组产生相同错误回答。单题的贪心生成变化不构成稳定保留退化的统计结论，也没有体现 Attribute-only 相对优势。

## 选中/未选中位置的 log-probability 变化

| 训练组 | 共同 Attribute 位置 Δlogp | 共同非Attribute位置 Δlogp | 本组训练mask位置 Δlogp | 本组未选位置 Δlogp |
|---|---:|---:|---:|---:|
| attribute | -0.000632555 | -0.000913708 | -0.000632555 | -0.000913708 |
| random_attribute | +0.000063433 | -0.000606186 | -0.002070236 | -0.000545716 |

Attribute-only 在共同事实位置上有较弱的负向logp变化，而Random略为正；这是可继续观察的微弱信号。但Attribute-only对未选中位置的变化更大，且总体变化量很小、QA没有优势，不能声称事实层面的选择性遗忘已经成立。RMU直接优化隐藏表示，logp不是其直接优化目标。

## 数值与参数检查

- 两组恰好4次更新、64次样本曝光（32个forget块各两次）；loss/梯度均有限，顺序哈希一致，没有追加步数或seed。
- 优化器包含27个参数张量、607150080个参数，范围严格为layers5–7；各组有22个张量出现非零数值变化，范围外参数版本未变。原始reference缓存及steering文件未改变。
- attribute：优化循环32.81秒，梯度范数34.750–52.000；最后microbatch均值retain activation loss=0.001865033。PyTorch分配器峰值15.93GiB（包含诊断；不含部分bnb直接分配，未采样NVML峰值）。
- random_attribute：优化循环32.95秒，梯度范数19.875–43.000；最后microbatch均值retain activation loss=0.001714514。PyTorch分配器峰值15.93GiB（包含诊断；不含部分bnb直接分配，未采样NVML峰值）。

两个checkpoint保存所有更新层的绝对参数状态，需与原始target组合加载；不是独立完整7B模型。已从原target分别重载两个checkpoint，在固定forget/retain样本上复现保存的logp，差异见checkpoint_reload_verified.json。原始target、两个新checkpoint和旧实验均保留。

## 是否继续正式实验

**目前不建议直接投入正式三seed。** 这轮支持的是实现与输入管线可运行；方向性证据很弱，未在知识QA上得到相对Random的改善，且存在两组共有的单题retain生成损失。后续若继续，应先独立审查标注的事实性与上下文依赖、确认非穷尽四span/块的覆盖是否足以回答研究问题，再另行批准新的限定pilot。当前冻结标注不因结果回改，本轮不继续训练。

局限包括：单seed、少量训练/评估样本、很短训练、标注非穷尽且未独立人工验证；没有ES、隐私、攻击恢复或完整官方评估。轻量QA沿用官方模板、stopwords、max_new_tokens32和RougeL-F1实现，但batch4与FlashAttention2有别于原正式SDPA/batch16，不能写入Table5。

## 文件

- annotations.json / annotation_preview.md / annotation_prompt.txt / annotation_provenance.json：标注、字符位置、预览和来源。
- inputs/*_samples.jsonl、data_order.json、masks_*.jsonl、FROZEN.json、mask_audit.json：样本、顺序、冻结mask和审计。
- code/、plan.json、execution_commands.json、各组actual_config.json：源代码、版本、实际配置与命令。
- original/、attribute/、random_attribute/：逐token logp、训练记录、参数审计、16题QA输出及复核。
- retain_reference/：固定原target的参考激活；轻量归档仅含manifest，不含大张量。
- */checkpoint-4/updated_layers.safetensors：保留在服务器，轻量归档不含模型权重。
