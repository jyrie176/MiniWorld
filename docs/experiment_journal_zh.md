# MiniWorld V100 实验复盘

最后更新：2026-08-27

## 1. 项目总目标与主线

基于 MiniWorld 上游代码和公开权重，在 **4×Tesla V100-SXM2-32GB** 环境建立完整、可复现的
世界模型训练与评测体系，并围绕“**不确定性感知的自适应 rollout**”形成有实验支撑的项目贡献。

V100/FP16 兼容改造是工程基础，0.12B 训练是 sanity 阶段；两者都不是项目终点。完整主线为：

```text
V100 工程适配
    ↓
0.12B from-scratch sanity
    ↓
0.55B 公开预训练权重 continued training / domain adaptation
    ↓
不确定性估计与 uncertainty-error correlation
    ↓
adaptive rollout 核心创新
    ↓
horizon / threshold / K / OOD / 资源消融
    ↓
技术报告、可复现仓库与求职材料
```

项目最低成功标准：

1. 0.12B sanity 可复现，训练、评测、DDP 和 checkpoint 链路可靠；
2. 0.55B continued-training baseline 稳定，并与 pretrained zero-shot 公平比较；
3. 完成 uncertainty 与未来误差的相关性分析；
4. 完成 fixed horizon、单阈值和带平滑/迟滞 adaptive horizon 的消融；
5. 所有结论均可追溯到代码、配置、数据 manifest、run ID、checkpoint 和机器可读指标。

### 1.1 模型阶段

#### 0.12B（代码配置名 `B`）：学习与正确性验证

用途是 from-scratch sanity，不追求最终质量或 SOTA。它负责验证数据/动作对齐、损失、FP16、
attention、EMA、checkpoint、DDP、短 rollout 和评测方法。当前 64-episode、5k-step 训练属于这一阶段。

离开 0.12B 阶段前需要形成明确验收结论，但不得在小模型上无限追加训练，延误正式主线。

#### 0.55B（代码配置名 `0.5B`）：正式实验主模型

使用 MiniWorld 上游公开的 0.55B DROID checkpoint 做 continued training / domain adaptation，
不从头训练。必须先建立 zero-shot 固定评测，再进行继续训练；核心 adaptive rollout、OOD 和消融
均以此模型为主。

#### 1B：可选 stretch

只有在 0.55B baseline、adaptive rollout 和主要消融全部完成后才考虑。3B 不在项目范围内。

### 1.2 核心创新：不确定性感知的自适应 Rollout

固定 rollout horizon 会在高不确定区域继续积累误差。核心研究问题是：能否通过同一上下文的
`K` 次随机预测，使用 latent variance、特征 pairwise disagreement 等指标得到逐步不确定性 `u_t`，
并在 `u_t` 超过验证集选择的阈值 `τ` 时提前停止、缩短 horizon，或触发重新观测/重规划。

至少比较：

- fixed horizon；
- 单步阈值 adaptive horizon；
- 带平滑/迟滞的 adaptive horizon。

必须报告 uncertainty-error Pearson/Spearman、分桶误差、平均 horizon、覆盖率、提前停止率、
预测质量和额外推理成本。阈值只允许在 validation set 上选择，test set 只做最终报告。

### 1.3 任务前主线检查（强制）

开始任何开发、训练、下载或评估任务前，先回答：

1. 该任务属于哪一阶段：V100 基础、0.12B sanity、0.55B baseline、adaptive rollout、消融/OOD，
   还是最终报告？
2. 它要回答哪个明确问题，满足哪个验收标准？
3. 当前阶段是否存在优先级更高、尚未完成的 gate？
4. 产物如何支持 0.55B baseline 或 adaptive rollout，而不只是让当前视频“看起来更好”？
5. 如果无法映射到主线，是否应停止；若确需偏离，是否已明确记录原因、成本和返回主线的条件？

每项新实验在记录中增加一行：

```text
主线映射：Phase / 实验编号 / 对应验收项
```

未经这项检查，不启动新的长训练、大规模下载、模型扩展或旁支功能。

## 2. 硬件约束与技术选择

| 约束 | 影响 | 采用方案 |
| --- | --- | --- |
| V100 为 SM70 | 不能使用主流 FlashAttention-2 CUDA 后端 | 使用 PyTorch SDPA，即普通 attention 路径 |
| V100 没有原生 BF16 Tensor Core | 上游 BF16 配方不适用 | 使用 FP16 autocast + GradScaler |
| 单卡 32GB | batch 和上下文长度受限 | batch size 1，从 6 latent frames 开始 |
| 小规模数据 | 容易过拟合，泛化结论有限 | 先做单片段过拟合，再扩展到 64 episodes |

选择 SDPA 与 FP16 的目的不是声称它们与上游 BF16/FlashAttention 配方完全等价，
而是先验证 V100 上是否存在一条数值可控、可训练、可推理的兼容路径。

## 3. 工作时间线

### 阶段 A：V100 attention 与 FP16 兼容

#### 要解决的问题

原始代码默认依赖 FlashAttention 与 BF16，无法直接在 V100 上运行。

#### 为什么这样做

如果基础算子和混合精度路径不可靠，后面的 loss、视频质量和动作消融都没有解释价值。
因此先处理运行兼容性，再开始数据和模型实验。

#### 实际工作

- attention 后端支持自动选择与显式选择。
- V100 自动回退到 PyTorch SDPA。
- 增加 FP16 autocast 和 GradScaler。
- 仅在优化器步骤成功时更新 EMA 和 `global_step`。
- 记录 gradient norm、loss scale、跳步状态与累计跳步数。
- checkpoint 保存和恢复 optimizer、EMA、scaler 与全局步数状态。
- 增加 CPU 单元测试与 CUDA smoke test。
- 启动脚本允许覆盖 precision、attention backend 和训练配置。

#### 结果

- V100 上解析为 `precision=fp16`、`attention=sdpa`。
- 训练、checkpoint 恢复与推理路径可以运行。
- FlashAttention 变为可选依赖，不再阻塞 V100 环境。

#### 判断

V100 兼容路径成立。SDPA 的主要风险是速度和显存效率弱于 FlashAttention；FP16 的主要风险是
动态范围小于 BF16，但 GradScaler 能检测溢出并安全跳过异常优化器步骤。

---

### 阶段 B：准备 Wan2.2 VAE

#### 要解决的问题

MiniWorld 在 VAE latent 空间训练，缺少预训练 VAE 时无法编码和解码视频。

#### 实际工作

下载并校验 Wan2.2 VAE：

```text
/data/miniworld/checkpoints/wan2.2/Wan2.2_VAE.pth
size: 2,818,839,170 bytes
sha256: 20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36
```

#### 结果与判断

VAE 能在 V100 环境加载，参数量为 704,688,668。此后实验固定使用同一个 VAE，避免把
VAE 差异误认为 world model 差异。

---

### 阶段 C：16-episode、100-step MVP 与后续 1k-step 检查

#### 要解决的问题

验证完整的数据读取、VAE 编码、world model 训练、EMA、checkpoint、采样和 W&B 记录链路。

#### 配置

```text
dataset: DROID episodes 1000-1015，共 16 episodes
model: B
latent frames: 6
batch size: 1
precision: FP16
attention: SDPA
initial MVP steps: 100
subsequent comparison checkpoint: 1000 steps
```

输出目录：

```text
/data/miniworld/outputs/droid-v100-fp16-mvp-100
```

#### 结果

- 初始 100 个 optimizer steps 完成，未发生 FP16 跳步，loss scale 保持 `65536`。
- step-100 checkpoint 能完整恢复，并成功继续到 step 101。
- 后续训练/比较扩展到 step 1000。
- 可以从 checkpoint 生成视频。
- 初始 EMA decay 为 `0.9999`，在只有 1000 步的短训练中更新过慢，EMA 生成明显落后于普通权重。

初始 MVP 的独立技术记录见：

```text
docs/results/v100-real-data-mvp.md
```

#### 判断与下一步原因

链路已打通，但单看生成视频不能证明模型使用动作。下一步需要加入固定随机种子、权重来源选择和
动作消融，控制除动作之外的所有变量。

---

### 阶段 D：权重来源与动作消融工具

#### 要解决的问题

区分以下因素：

- 普通模型权重与 EMA 权重的差异。
- 真实动作、全零动作、时间顺序打乱动作的差异。
- 随机噪声变化与动作变化的差异。

#### 实际工作

采样入口增加：

```text
--seed
--weights_source model|ema
--action_variant real|zero|shuffle
```

所有动作对照使用同一 episode、同一初始帧和同一随机噪声。

#### 三种动作的含义

- `real`：使用数据集中的真实机器人动作。
- `zero`：将动作全部置零，检查模型是否忽略动作也能生成相似视频。
- `shuffle`：保留动作值分布，但打乱时间顺序，检查模型是否理解动作的时序关系。

#### 判断

固定 seed 是动作消融成立的前提，否则两个视频的不同可能只是扩散初始噪声不同。

---

### 阶段 E：单片段过拟合实验

#### 要解决的问题

在讨论泛化前，先验证模型是否有能力记住一个训练片段，并检查动作条件链路是否有效。

#### 为什么做单片段过拟合

这是一个诊断实验，不是最终训练方案。如果模型连一个固定片段都无法拟合，问题更可能来自公式、
数据对齐、条件注入或优化路径；如果单片段能拟合而多片段效果差，瓶颈更可能是数据规模与训练量。

#### 实际工作

- 增加 `--overfit_single_sample`。
- 固定 dataset index、视频片段、动作和训练输入。
- 训练 500 steps。
- EMA decay 改为 `0.99`，适应短训练。

输出目录：

```text
/data/miniworld/outputs/droid-v100-overfit-one
```

#### 结果

固定 episode 1000 的未来 RGB 帧 MAE（0-255 空间，越低越好）：

| 配置 | MAE |
| --- | ---: |
| 普通权重 + 真实动作 | 14.10 |
| EMA + 真实动作 | 17.62 |
| 普通权重 + 全零动作 | 18.23 |
| 普通权重 + 打乱动作 | 14.31 |

#### 判断

- 普通权重对固定片段有明显拟合能力。
- 全零动作比真实动作差，说明动作条件不是完全无效。
- 打乱动作与真实动作接近，尚不能证明模型准确学习了动作顺序。
- 短训练中 EMA 仍可能滞后，不能默认 EMA 一定优于普通权重。

#### 下一步原因

单片段结果只能证明容量和链路，不能证明跨 episode 泛化，因此扩展训练数据。

---

### 阶段 F：扩展到 64 episodes

#### 实际工作

构建数据集：

```text
/data/miniworld/datasets/droid-mini-1000-1063
episodes: 1000-1063，共 64 个成功 episode
```

校验结果：

- 64 个 parquet 文件。
- 64 个视频文件。
- 所有 64 个视频均能解码。
- 数据集体积约 48MB。

#### 为什么从 16 扩到 64

在仍可快速迭代的范围内增加场景和动作多样性，观察模型能否从“记住少数片段”转向学习更一般的
视频动力学。

---

### 阶段 G：64-episode、5k-step 训练

#### 配置

```text
dataset: DROID episodes 1000-1063
model: B
latent frames: 6
batch size: 1
precision: FP16
attention: SDPA
EMA decay: 0.99
learning rate: 2e-5
max steps: 5000
image log interval: 500 steps
checkpoint interval: 10 epochs
```

关键路径：

```text
checkpoint directory: /data/miniworld/outputs/droid-v100-64ep-5k
final checkpoint: epoch_0079_step_00005000.pt
training log: /data/miniworld/experiments/droid-v100-64ep-5k/train.log
W&B run: https://wandb.ai/irvingjyrie176-tencent/miniworld-v100/runs/wdslnxhb
```

#### 完成状态

- `global_step=5000`。
- 最终 checkpoint 约 1.98GB，内部包含 model、EMA、optimizer 和 scaler。
- W&B 完成同步，共 20 个媒体文件。
- 无 OOM、NaN 或异常堆栈。
- FP16 GradScaler 累计跳过 2 个溢出步骤，scale 从 `65536` 降至 `32768` 后稳定。

#### Loss 结果

| step 区间 | 记录点平均 loss | 中位数 |
| --- | ---: | ---: |
| 1-500 | 1.043 | 0.999 |
| 501-1000 | 0.482 | 0.441 |
| 1001-2000 | 0.343 | 0.291 |
| 2001-3000 | 0.289 | 0.243 |
| 3001-4000 | 0.280 | 0.231 |
| 4001-5000 | 0.245 | 0.209 |

最终记录点 loss 为 `0.18797`。

#### 为什么 W&B loss 有毛刺

当前 batch size 为 1，W&B 每 10 step 记录当前 batch 的即时 loss，而不是窗口均值。同时每步会改变：

- episode/片段；
- 扩散噪声；
- chunk timestep；
- 首帧干净或首个 chunk 干净的训练模式。

所以单点方差很大。分段均值持续下降、没有 NaN、GradScaler 仅安全跳过 2 步，说明当前现象是
高方差随机训练，而不是持续数值发散。

---

### 阶段 H：5k checkpoint 固定样本评估

#### 目的

确认训练完成不等于模型有效。使用同一 episode 1000、同一 seed `20260826`，分别评估普通权重、
EMA 和动作消融。

#### 结果

未来 RGB 帧 MAE：

| 配置 | MAE |
| --- | ---: |
| EMA + 真实动作 | **13.34** |
| 普通权重 + 打乱动作 | 17.55 |
| 普通权重 + 真实动作 | 17.68 |
| 普通权重 + 全零动作 | 17.96 |

输出间的未来帧 MAE：

| 对照 | 差异 MAE |
| --- | ---: |
| 普通真实动作 vs 全零动作 | 3.79 |
| 普通真实动作 vs 打乱动作 | 1.65 |
| 普通权重 vs EMA（真实动作） | 11.86 |

#### 视觉观察

- EMA 比普通权重更接近 GT。
- 主体布局可以保持。
- 机器人区域仍有融化、闪烁和细节破坏。
- 动作改变会影响输出，但真实动作没有稳定优于打乱动作。

#### 判断边界

该结果只来自一个固定 episode，不能用于声称模型已经学会通用动作控制。它说明：

1. EMA 在 5k 训练后已经追上并超过普通权重。
2. 模型对动作有响应。
3. 动作时序理解仍缺少证据，需要多 episode 统计。

评估产物：

```text
/data/miniworld/experiments/droid-v100-64ep-5k-eval
/data/miniworld/exports/miniworld-droid-64ep-5k-eval.tar.gz
archive sha256: 5a28d3d317a9110069ea821e0220ce8c9c9310d7505d8f339d635b02343a8480
```

---

### 阶段 I：recon_video 与 gen_video 诊断

#### 原始观察

W&B 中 `recon_video` 经常在约 1 秒后变成随机花点，而完整的 `gen_video` 质量反而较好。

这与“单步重建应该比完整生成简单”的直觉矛盾，因此不能直接归因于模型能力。

#### 数据流核对

训练前向加噪和 velocity target 为：

```python
z = (1 - t) * clean + t * noise
v_target = clean - noise
```

由此解析反推为：

```python
clean = z + t * v_target
```

但原始 recon 可视化使用了：

```python
x_pred = z + (1 - t) * v_pred
```

当 `t` 接近 1 时，输入几乎为纯噪声，错误公式又几乎不施加 velocity 修正，所以输出保留为花点；
只有 `t=0.5` 时两个系数碰巧相同。

#### 修复

修正为：

```python
x_pred = z + t * v_pred
```

同时新增解析回归测试：在模型返回完美 velocity 时，任意高 timestep 的预测都必须精确恢复 clean latent。

#### 验证结果

- 修复前测试按预期失败，2/2 元素不匹配。
- 修复后定向测试通过。
- 完整 CPU 测试集：`73 passed`。
- 提交：`0393514 fix: reconstruct clean latents with correct timestep`。

#### 重要判断

- 这是 `recon_video` 的可视化反推错误。
- 不影响训练 loss，因为 loss 正确比较 `v_pred` 与 `clean-noise`。
- 不影响 checkpoint 和 EMA。
- 不影响 `gen_video`；它使用从 `t=1` 到 `t=0` 的多步 velocity 积分。
- 已有 checkpoint 不需要重训，但旧 W&B recon 视频不可继续用于判断模型能力。

这是一次判断修正：最初曾把 recon 花点解释为单步去噪能力不足；核对公式后确认主要原因是可视化 bug。
复盘时应以后一个结论为准。

---

### 阶段 J：Windows 自动同步实验产物

#### 要解决的问题

训练在 Linux 服务器完成，而视频和对比图需要在 Windows 本地查看。手工执行 `scp`、解压和清理
容易出错，也降低每轮实验效率。

#### 实际工作

- 服务器统一将待下载压缩包放到 `/data/miniworld/exports`。
- Windows 使用 `D:\code\miniworld\sync-miniworld.ps1`。
- 脚本下载新压缩包、自动解压、删除本地压缩包，并用 marker 避免重复处理。
- 修复过目录名 `miniwold` → `miniworld`、PowerShell 编码和字符串解析问题。

#### 判断

后续每轮评估都应将小型结果包放入 exports；大 checkpoint 不默认同步，以免浪费带宽和本地空间。

---

### 阶段 K：固定 Train / Validation / Test 划分

#### 主线映射

Phase 3 / E1 / “0.12B sanity 有 episode 隔离、可审计且能复用于 0.55B 的评估划分”。

#### 为什么现在做

此前 episode 1000 的指标来自训练数据，只能诊断模型和动作链路，不能代表泛化。继续在训练 episode
上选择 checkpoint 或调整动作评估会产生数据泄漏，也无法为后续 0.55B zero-shot/continued training
提供公平基线。

#### 固定划分

| Split | Episodes | Count | 用途 |
| --- | --- | ---: | --- |
| train | 1000-1063 | 64 | 已完成的 0.12B 5k-step 训练 |
| validation | 1064-1079 | 16 | checkpoint、方法、阈值与诊断选择 |
| test | 1080-1095 | 16 | 最终报告，当前封存 |

本地数据路径：

```text
/data/miniworld/datasets/droid-mini-1000-1063
/data/miniworld/datasets/droid-validation-1064-1079
/data/miniworld/datasets/droid-test-1080-1095
```

只下载每个 episode 的 parquet 和 `observation.images.exterior_image_1_left` 视频，没有下载完整 DROID。

#### 校验结果

- 96 个 episode 互不重叠，当前新增 validation/test 各 16 个。
- 新增 32 个 episode 全部 `success=true`，最短 75 帧，满足 21 帧窗口。
- validation 与 test 均为 16 parquet + 16 video。
- 32/32 视频完整解码；视频张量 `(21, 240, 320, 3)`、动作张量 `(20, 7)`，全部 finite。
- 已记录各 split 的 episode manifest、数据内容 digest 与使用规则：

```text
manifests/droid-eval-splits.md
manifests/droid-train-1000-1063/episodes.jsonl
manifests/droid-validation-1064-1079/episodes.jsonl
manifests/droid-test-1080-1095/episodes.jsonl
```

#### 使用边界

validation 可以用于当前 0.12B 动作消融和后续 0.55B 方法选择。test 目前只做文件完整性检查，
不运行模型、不查看质量指标，直到最终方案冻结。

---

### 阶段 L：0.12B Held-out Validation 动作消融

#### 主线映射

Phase 3 / E1 / “判断 0.12B 是否在未训练 episode 上学到可靠动作控制”。

#### 配置

- checkpoint：64-episode、5k-step 最终 EMA；
- validation：episodes 1064-1079，共 16 个，未参与训练；
- test：保持封存，未运行模型；
- 条件：real、zero-valued normalized action、reverse；
- seed：`20260827 + sample_index`，三组完全一致；
- rollout：1 个观察 latent、共 6 latent / 21 RGB 帧；
- 指标：未来 20 个 RGB 帧的 0-255 MAE。

代码参数 `shuffle` 当前实际执行确定性时间倒序，因此报告统一称为 reverse。zero 是归一化空间数值零，
不等于 learned null/unconditional action。

#### 结果

| 条件 | 平均 MAE | 中位 MAE |
| --- | ---: | ---: |
| real | 12.700 | 12.181 |
| zero-valued | **12.293** | **12.143** |
| reverse | 12.701 | 12.228 |

- real 优于 zero：3/16；
- real 优于 reverse：10/16；
- real 同时优于两者：3/16；
- real 与 reverse 平均差仅 `0.002`，基本相同；
- real 与 zero 输出差异 MAE 为 `2.772`，real 与 reverse 为 `1.673`，说明动作会改变输出，但方向不可靠。

首帧 persistence baseline（重复第一帧）平均 MAE 为 `4.584`，并在 16/16 episode 上优于 real
生成。RGB MAE 会偏好静态/清晰输出，且部分 episode 的前 21 帧运动较弱，因此后续正式评测需要
多固定窗口与感知特征指标；但当前结果仍不能支持“0.12B 学会 held-out 动作控制”。

#### 判断与停止条件

0.12B 已完成工程 sanity：V100 FP16/SDPA、EMA、checkpoint、held-out 评估链路均可运行；但动作质量
gate 未通过。停止无目的追加 0.12B 训练。只诊断会同样阻塞 0.55B 的正确性问题（首先是动作/视频
时间对齐），完成 DDP/资源 gate 后进入 0.55B pretrained zero-shot baseline。

独立报告：

```text
docs/results/v100-012b-validation-action-ablation.md
```

### 阶段 M：官方 0.55B Zero-shot Validation 基线

#### 问题与主线位置

这一步回答两个问题：官方 0.55B DROID checkpoint 在固定 held-out validation 上是否真正使用正确
动作，以及 0.12B 的负面结果是动作链路错误还是小数据短训模型能力不足。它对应总主线 Phase 4 / E2，
不是继续扩展 0.12B 支线，也没有提前使用密封 test split。

#### 权重与兼容处理

- 官方文件：`MiniWorld_0_5b_droid.pt`，2,226,856,306 bytes；
- SHA-256：`e4b118befe88cee7338400c5510fdd497212b9b1988034290030b3ed351ced32`；
- 文件是 420 项裸 state dict，而不是本项目训练器写出的 `model/ema/meta` 封装；
- 共 556,675,120 参数，RoPE 表推断训练窗口为 64 latent frames；
- 为采样器增加裸 state dict 识别，严格加载报告 all keys matched；
- 固定 `av==12.3.0`，解决 torchvision 0.19 与新版 PyAV 写视频接口不兼容问题。

对应回归测试先失败再修复，`tests/test_cli.py` 26 项通过。另新增统一动作消融评测脚本及合成指标
测试，明确评分只覆盖未来 RGB frames 1-20，不把观测首帧计入误差。

#### 实验配置

```text
validation episodes: 1064-1079（16 episodes）
test episodes: 1080-1095（保持密封）
checkpoint weights: 官方唯一裸 state dict（无 model/EMA 二选一）
runtime: V100 + FP16 + SDPA
rollout: 6 latent frames / 21 RGB frames，首 latent 为观测历史
sampling steps: 20
CFG: 2.0
seed: 20260827 + sample index
conditions: real / zero-valued normalized action / time-reversed action
```

三张空闲 V100 分别运行一个条件；每组均生成并成功解码 16 个 MP4，且 timing JSONL 均为 16 行。
单样本平均生成时间为 real `6.34s`、zero `6.31s`、reverse `6.36s`。

#### 结果

| 条件 | 平均未来帧 MAE | 中位数 MAE |
| --- | ---: | ---: |
| real | **5.468** | **5.393** |
| zero | 9.368 | 8.442 |
| reverse | 9.241 | 8.068 |

- real 优于 zero：15/16；
- real 优于 reverse：15/16；
- real 同时优于两者：14/16；
- `zero - real = +3.900`，`reverse - real = +3.773`；
- real 与 zero 输出平均 MAE `6.982`，real 与 reverse 输出平均 MAE `6.936`。

首帧 persistence 平均 MAE 为 `4.468`，real 只在 4/16 episodes 上优于 persistence。该指标偏好静态
输出，不能替代感知和运动质量评测；但它仍表明当前短 horizon 像素精度尚未全面超过静态强基线。

#### 判断与下一步

官方 0.55B 在完全相同的验证/动作/seed 口径下稳定偏好正确动作，因此动作输入、VAE 时间压缩和
动作到 latent 的映射没有系统性错位证据。0.12B 的失败主要归因于 64 episodes、5k steps 的小规模
from-scratch checkpoint，而不是公共推理链路失效。

Phase 4 的 zero-shot gate 已建立。下一步应进行短时 continued-training 冒烟与资源/DDP 验证，并以
zero-shot 指标为不可回退基线；之后才进入 uncertainty-error correlation 与 adaptive rollout 创新主线。

独立报告：

```text
docs/results/v100-official-055b-validation-action-ablation.md
```

### 阶段 N：官方 0.55B Continued-training 单卡冒烟

#### 问题与主线位置

验证官方 0.55B checkpoint 能否正确初始化训练器的 model 与 EMA，并在 V100 FP16/SDPA 上完成优化、
checkpoint 保存/恢复及训练前后同口径 validation。对应 Phase 4 / E2、E3、E8；目标是关闭单卡工程
gate，而不是用 20 step 宣称质量提升。

#### 发现并修复的阻塞问题

训练器原来的 `load_pretrained()` 只读取封装 checkpoint 的 `model/ema_model`，面对官方 420 项裸
state dict 会静默加载零个参数。回归测试先复现 model/EMA 未被初始化，再加入裸 state dict 识别；
修复后 checkpoint 测试 3 项通过。

以 6 latent frames 构建 0.5B 后，官方权重加载仅报告：

```text
missing: net.feat_rope.freqs_cos, net.feat_rope.freqs_sin
unexpected: none
```

这两个是随 64→6 窗口变化重建的 RoPE buffer，不是学习参数；其余 learned weights 均继承，官方
裸权重同时初始化 model 与 EMA。

#### 配置与训练结果

```text
train episodes: 1000-1063
validation episodes: 1064-1079
test episodes: 1080-1095（未使用）
model/window: 0.5B / 6 latent frames / 21 RGB frames
batch: 1
optimizer: AdamW
lr / EMA / grad clip: 2e-5 / 0.9999 / 1.0
precision / attention: FP16 GradScaler / SDPA
effective steps: 20
seed: 20260827
W&B run: 9gwiazi7
```

- 20 个 loss：mean `0.1362`、median `0.1048`、min `0.0573`、max `0.3060`；
- final loss：`0.0573`；
- skipped steps：`0`；loss scale 全程 `65536`；
- 最大记录 grad norm：`4.4625`，由 clip=1.0 处理；
- 稳态吞吐：`0.925 sample/s`；
- 运行中显存观测：`17,557 MiB / 32,768 MiB`。

#### Checkpoint 与恢复

step-20 checkpoint 约 `8.87GB`，包含 420 项 model、420 项 EMA、optimizer、metadata 和 scaler。
恢复时 all keys matched，从 global step 20 和 scaler growth tracker 20 继续；step 21 正常完成，未跳步，
新 checkpoint 的 tracker 为 21。单卡保存/恢复 gate 通过。

#### 固定 Validation 前后对照

| 条件 | 官方 zero-shot MAE | step-20 EMA MAE | 差值 |
| --- | ---: | ---: | ---: |
| real | 5.468005 | 5.467836 | -0.000170 |
| zero | 9.367881 | 9.372440 | +0.004559 |
| reverse | 9.240826 | 9.243434 | +0.002608 |

real 对 zero/reverse/both 的胜率仍为 `15/16`、`15/16`、`14/16`，persistence 仍为 `4.468`。
EMA decay `0.9999` 下 20 次更新只产生极小变化，因此结论是“训练链路稳定且动作能力未退化”，不支持
“continued training 已提升质量”。

独立报告：`docs/results/v100-official-055b-continued-smoke.md`。

### 阶段 O：官方 0.55B 双卡 DDP / Resource Gate

#### 问题与主线位置

在等效 global batch=2 下，对比单卡 batch=2 与双卡每卡 batch=1，验证 DDP sampler、梯度同步、
FP16 稳定性、吞吐、每卡显存、checkpoint 和恢复。对应 Phase 4 / E2、E3、E8，是选择正式
continued-training 预算前的最后一个工程 gate。

#### 配置与 sampler

两组均从官方 0.55B 裸 state dict 初始化，使用 train episodes 1000-1063、6 latent frames、AdamW、
LR `2e-5`、EMA `0.9999`、FP16/SDPA、seed base `20260827`，各跑 10 个有效 step。

64-episode DistributedSampler 实测 rank 0/1 各 32 个索引，交集为 0，并集为 64；没有 rank 间
episode 重复或遗漏。

#### DDP loss 遥测修复

首次 DDP 训练的梯度同步正确，但打印/W&B loss 只取 rank 0 本地 batch，不能与单卡 global batch
严格比较。新增失败回归测试后，将日志 loss 改为所有 rank all-reduce 均值；修复不改变 backward、
optimizer 或 checkpoint。使用修复后的独立双卡 run 作为 loss 对照。

#### 结果

| 指标 | 单卡 GB2 | 双卡 GB2 |
| --- | ---: | ---: |
| 10 点 mean loss | 0.1154 | 0.1037 |
| median loss | 0.1015 | 0.1014 |
| loss range | 0.0519-0.2001 | 0.0535-0.1650 |
| steady step/s | 0.516 | 0.860 |
| steady sample/s | 1.031 | 1.720 |
| 训练态显存 | 18,169 MiB | 每卡 19,611 MiB |
| skipped step | 0 | 0 |
| loss scale | 65,536 | 65,536 |

双卡吞吐加速 `1.67x`，扩展效率约 `83.4%`。两组随机流与 sampler 顺序不相同，不能要求逐 step
loss 相等；但均值、中位数和范围一致，没有同步或优化异常迹象。双卡每卡额外显存来自 DDP
reducer/communication 状态，32GB V100 仍剩约 13GB。

#### 恢复验证

双卡 step-10 checkpoint 恢复显示 all keys matched，继续到 step 15；两 rank 训练态显存均为
`19,611 MiB`，采样时利用率为 100%/95%。最终 checkpoint 为 epoch 2、global step 15，包含 420
项 model、420 项 EMA、415 项 optimizer state；GradScaler scale `65536`、growth tracker `15`。

W&B：单卡 `fnqph9f6`；初次双卡/恢复 `kmta7zr2`；修复后 global-loss 双卡 `mmweo7q6`。

判断：双卡 DDP/resource gate 通过。下一步选择有边界的 continued-training 正式 baseline 预算与
validation/checkpoint 节奏；尚不进入 uncertainty-aware 创新阶段。

独立报告：`docs/results/v100-official-055b-ddp-resource-gate.md`。

### 阶段 P：官方 0.55B 双卡 1k-step Continued-Training Baseline

#### 问题与动机

在工程 gate 全部通过后，用有边界的 1000 个有效 step 回答：本地 64-episode DROID 续训是否能让
官方 0.55B 在固定 validation 上变好，并保持动作控制。这仍属于 Phase 4 baseline，不进入创新模块；
test episodes 1080-1095 全程密封。

#### 配置

```text
train / validation: episodes 1000-1063 / 1064-1079
model/window: official 0.55B / 6 latent frames / 21 RGB frames
runtime: 2x V100 DDP, per-rank batch 1, global batch 2
optimizer: AdamW, LR 2e-5, EMA 0.9999, grad clip 1.0
precision / attention: FP16 GradScaler / SDPA
effective steps: 1000
seed base: 20260827
W&B run: k2y8o0jh
```

保存点为 step 320、639、959、1000。attempted step 340 出现一次非有限梯度并被 GradScaler 安全
跳过，scale 从 `65536` 降为 `32768`，之后直到结束无第二次跳步。最终 loss `0.053203`，稳态约
`0.85 step/s` / `1.70 sample/s`。后段 step 640-959 的记录均值 `0.0908`，低于前段 1-320 的
`0.1149`，数值训练稳定。

#### 固定 Validation

使用相同 16 个 validation episodes、real action、seed、20 sampling steps 和 CFG 2.0：

| 权重 | step 320 | step 639 | step 1000 | 官方 zero-shot |
| --- | ---: | ---: | ---: | ---: |
| model | 6.7049 | 7.2835 | 6.4532 | 5.4680 |
| EMA | 5.4695 | 5.4742 | 5.4759 | 5.4680 |

model 相对官方逐 episode 改善数为 `0/16`、`0/16`、`1/16`；EMA 均值仅漂移 `+0.0015`、
`+0.0062`、`+0.0079`。最终 model 明显变化，因此按预定规则追加动作对照：real/zero/reverse
为 `6.4532/8.4448/9.2114`，real 胜率 `13/16`、`14/16`、同时最佳 `12/16`。动作通路仍有效，
但整体质量和动作优势均弱于官方 checkpoint。

#### Checkpoint、异常与判断

最终 checkpoint 为 epoch 32 / global step 1000，含 420 项 model、420 项 EMA、optimizer、metadata
和 scaler；大小 8,865,354,022 bytes，scaler scale `32768`、growth tracker `660`。评估首次沿用
官方 64 帧流式缓存，6 帧 checkpoint 的安全断言正确拒绝；将 active window 改为恰好 6 帧后重跑，
所有 96 条 real 视频及最终 32 条动作对照均成功。失败尝试没有形成指标，也未访问 test。

判断：不能把相同 `LR=2e-5 + 全参数 + 64 episodes` 配方直接延长到 5k。训练 loss 下降而
validation model MAE 恶化，符合小数据过拟合/灾难性遗忘；EMA 只压住退化，没有带来收益。官方
zero-shot 继续作为 Phase 4 质量基线。下一步只做一次更保守、边界清晰的 lower-LR/partial-freeze
适配对照；若仍无收益，就冻结官方 checkpoint 并进入 uncertainty-error correlation。

独立报告：`docs/results/v100-official-055b-continued-1k.md`。

### 阶段 Q：训练集扩到 1064 Episodes 的单变量对照

#### 问题与动机

阶段 P 在 64 episodes 上出现训练 loss 下降、validation 退化。为区分“数据反复过拟合”和“全参数
适配强度过高”，本阶段把训练集扩到 episodes 0-1063，其余官方初始化、双卡 global batch=2、
1000 steps、LR、EMA、FP16/SDPA 和 validation 全部不变。仍属于 Phase 4 baseline；test 1080-1095
继续密封。

#### 数据扩容与审计

从 `GEAR-Dreams/DreamZero-DROID-Data` 只下载 chunk-000 的 parquet 和
`exterior_image_1_left` 视频，再硬链接已有 chunk-001 的 1000-1063。最终 1064 个 episode 连续、
全部 success，长度 46-1481 帧。全量实际读取 1064/1064：视频 `(21,240,320,3)`、动作 `(20,7)`，
全部 finite。

```text
dataset: /data/miniworld/datasets/droid-expanded-0-1063
episodes.jsonl SHA: c57b4c83e1134be2a4a7ac23e6959ad1216270d6d5e085cd60090619eb938793
content digest: 2645485be04d479fa3571351462846ffaf801b3b7a16492e3cd71df5ec2737f1
manifest: manifests/droid-train-expanded-0-1063.md
```

下载过程中两次无效尝试均已诊断且未产生媒体文件：官方 snapshot API 枚举全仓库过慢；首次直接
下载因 `xargs -I %` 误替换 `printf` 格式符而请求错误文件名。修正为位置参数后 2000 个文件完整
下载，结构和内容审计通过。

#### 训练

```text
model/window: official 0.55B / 6 latent frames
runtime: 2x V100 DDP, global batch 2
optimizer: AdamW, LR 2e-5, EMA 0.9999, clip 1.0
budget: 1000 effective steps，约 1.88 次数据遍历
W&B: csp6jnw6
```

训练退出码 0。一次 overflow 被安全跳过，scale `65536→32768`，之后稳定；记录 loss 前半均值
`0.1169`、后半 `0.1088`，final `0.0847`，吞吐约 `0.85 step/s` / `1.70 sample/s`。checkpoint
为 step 531 和 1000。

#### Validation 与判断

| 权重 | step 531 | step 1000 | 官方 | 64-episode step 1000 |
| --- | ---: | ---: | ---: | ---: |
| model | 8.7416 | 6.3781 | **5.4680** | 6.4532 |
| EMA | 5.4804 | 5.5007 | **5.4680** | 5.4759 |

expanded final model 相对 64-episode 改善 `0.0751`，逐 episode 胜 `10/16`；但仍比官方差
`0.9101`，只在 `1/16` 上胜官方。final EMA 比官方差 `0.0326`。

final model 的 real/zero/reverse 为 `6.3781/9.2527/9.0868`；real 胜率恢复到
`15/16`、`15/16`、同时最佳 `14/16`，与官方完全一致，并优于 64-episode continued model 的
`13/16`、`14/16`、`12/16`。

结论：数据不足确实影响动作泛化，扩容恢复了动作控制一致性，也让 final model 小幅改善；但它没
解决整体质量退化。主要矛盾转为 `LR=2e-5` 的全参数适配强度过高，不能把本配方扩到 5k。下一项
应保持 expanded data 与 1k budget，只把 LR 降到 `2e-6`；若仍无收益，冻结官方 zero-shot 并进入
uncertainty-error correlation。

独立报告：`docs/results/v100-official-055b-expanded1064-continued-1k.md`。

### 阶段 R：Expanded Data + Lower LR 最终 Baseline 选择

#### 问题与单变量设计

阶段 Q 证明扩到 1064 episodes 能恢复动作泛化，但 `LR=2e-5` 仍破坏质量。本阶段保持数据、官方
初始化、双卡 global batch=2、1000 steps、EMA、FP16/SDPA、seed 和 validation 不变，只将 LR
降为 `2e-6`。这是 Phase 4 最后一组 continued-training 选择实验；test 1080-1095 保持密封。

#### 训练结果

```text
dataset: episodes 0-1063
LR / EMA / clip: 2e-6 / 0.9999 / 1.0
effective steps: 1000
W&B: d59p17p7
```

训练退出码 0。一次早期 overflow 被安全跳过，scale `65536→32768` 后稳定；100 个记录点的
mean/median loss 为 `0.1314/0.0911`，range `0.0485-0.5794`。毛刺对应有限梯度，是多任务轨迹
难度差异，不是数值发散。吞吐约 `0.85 step/s` / `1.70 sample/s`，checkpoint 为 step 531/1000。

#### 固定 Validation

| 权重 | step 531 | step 1000 | 官方 | expanded `2e-5` step 1000 |
| --- | ---: | ---: | ---: | ---: |
| model | 5.5340 | 5.5779 | **5.4680** | 6.3781 |
| EMA | 5.4670 | 5.4630 | 5.4680 | 5.5007 |

降 LR 显著减少遗忘：final model/EMA 相对 `2e-5` 改善 `0.8002/0.0376`。但 final model 仍比
官方差 `0.1099`，仅 `3/16` episode 改善。final EMA 数值上优 `0.0050`、`13/16` episode 为负
delta，但变化仅官方 MAE 的 `0.09%`，大部分逐 episode 差异只有 0.00x RGB 值；在 16 条选择集上
不构成有实际意义的质量收益。

final model 的 real/zero/reverse 为 `5.5779/9.4563/9.3373`；real 胜 `15/16`、`16/16`、同时
最佳 `15/16`，动作控制保持强健。persistence 仍为 `4.4681`，核心静态强基线限制未改变。

#### Phase 4 冻结决策

最终 checkpoint 完整包含 420 model、420 EMA、optimizer、metadata、scaler，epoch 2 / step 1000，
scale `32768`、tracker `982`。

Phase 4 正式冻结**官方 zero-shot 0.55B checkpoint**，不选 continued checkpoint：64 episodes
高 LR 明显遗忘；扩到 1064 episodes 恢复动作泛化但质量仍退化；lower LR 只把 EMA 变化压到近零，
没有可信收益。停止继续搜索 continued-training 超参数，下一主线进入 uncertainty-error correlation。
test 继续密封，留给最终冻结方法的一次性报告。

独立报告：`docs/results/v100-official-055b-expanded1064-lr2e6-continued-1k.md`。

## 4. 当前关键结论

1. **V100 路径可用：** SDPA + FP16 GradScaler 能完成 5k-step 训练与推理。
2. **训练整体收敛：** 即时 loss 毛刺明显，但分段均值持续下降，没有持续数值发散证据。
3. **当前首选 EMA：** 在 64-episode 5k checkpoint 的固定样本上，EMA MAE 明显低于普通权重。
4. **官方 0.55B 动作控制成立：** real 在 15/16 validation episodes 上分别优于 zero 和 reverse。
5. **0.12B 负面结果已定位：** 相同链路在官方 0.55B 上通过，问题主要是小模型短训能力不足。
6. **像素强基线仍未攻克：** 官方 0.55B real 平均 MAE 5.468，高于 persistence 的 4.468。
7. **旧 recon 不可信：** 反推公式错误已修复，训练与 gen 不受影响。
8. **validation 只用于选择：** test episodes 1080-1095 继续密封，不能提前反复查看。
9. **0.55B 单卡 continued-training gate 通过：** FP16、EMA、optimizer、scaler、保存与恢复均稳定。
10. **0.55B 双卡 DDP gate 通过：** 1.67x 吞吐、83.4% 效率、分片无重复、恢复稳定。
11. **0.55B 1k 全参数续训无收益：** model MAE 明显恶化，EMA 近似原点但没有改善，不能直接扩到 5k。
12. **扩到 1064 episodes 只解决部分问题：** 动作胜率恢复官方水平，但 model/EMA 质量仍差于官方；下一变量应是 LR。
13. **Phase 4 已冻结官方 0.55B：** LR `2e-6` 消除大部分遗忘但没有实际收益，停止 continued-training 搜索。

## 5. 当前产物索引

| 类型 | 路径/链接 |
| --- | --- |
| 64-episode 数据集 | `/data/miniworld/datasets/droid-mini-1000-1063` |
| Wan2.2 VAE | `/data/miniworld/checkpoints/wan2.2/Wan2.2_VAE.pth` |
| 单片段 checkpoint | `/data/miniworld/outputs/droid-v100-overfit-one/epoch_0500_step_00000500.pt` |
| 5k 最终 checkpoint | `/data/miniworld/outputs/droid-v100-64ep-5k/epoch_0079_step_00005000.pt` |
| 5k 训练日志 | `/data/miniworld/experiments/droid-v100-64ep-5k/train.log` |
| 5k 固定评估 | `/data/miniworld/experiments/droid-v100-64ep-5k-eval` |
| 官方 0.55B checkpoint | `/data/miniworld/checkpoints/miniworld-official-0.55b/MiniWorld_0_5b_droid.pt` |
| 官方 0.55B validation 基线 | `/data/miniworld/experiments/official-055b-validation-action-ablation` |
| 0.55B 单卡 continued checkpoint | `/data/miniworld/outputs/droid-official-055b-continued-smoke-lf6-20step` |
| 0.55B step-20 validation | `/data/miniworld/experiments/official-055b-continued-step20-validation-action-ablation` |
| 0.55B DDP gate | `/data/miniworld/experiments/droid-official-055b-ddp-gate-*` |
| 0.55B 1k continued checkpoints | `/data/miniworld/outputs/droid-official-055b-continued-1k-lf6-ddp2` |
| 0.55B 1k validation | `/data/miniworld/experiments/official-055b-continued-1k-validation` |
| Expanded 1064-episode 数据集 | `/data/miniworld/datasets/droid-expanded-0-1063` |
| Expanded 1k checkpoints | `/data/miniworld/outputs/droid-official-055b-expanded1064-continued-1k-lf6-ddp2` |
| Expanded 1k validation | `/data/miniworld/experiments/official-055b-expanded1064-continued-1k-validation` |
| Lower-LR 1k checkpoints | `/data/miniworld/outputs/droid-official-055b-expanded1064-lr2e6-1k-lf6-ddp2` |
| Lower-LR 1k validation | `/data/miniworld/experiments/official-055b-expanded1064-lr2e6-1k-validation` |
| 0.12B W&B | `https://wandb.ai/irvingjyrie176-tencent/miniworld-v100/runs/wdslnxhb` |
| Lower-LR 0.55B W&B | `https://wandb.ai/irvingjyrie176-tencent/miniworld-v100/runs/d59p17p7` |
| GitHub PR | `https://github.com/jyrie176/MiniWorld/pull/1` |

## 6. 下一步实验队列

当前主线位置：**Phase 4 已冻结官方 0.55B zero-shot baseline；下一阶段进入 uncertainty-error
correlation**。不继续增加 0.12B 或 continued-training 步数，不查看密封 test。

### P0：Phase 4 冻结结果

continued-training 选择已结束，官方 zero-shot 胜出，作为后续创新实验的统一 checkpoint。冻结依据：

- 官方 real/zero/reverse MAE 与动作胜率；
- 64 与 1064 episodes 的 `LR=2e-5` 负面对照；
- expanded data 的 `LR=2e-6` 保守适配对照；
- persistence、RGB MAE、训练成本、采样吞吐和显存。

下一项先建立 uncertainty 与未来预测误差的相关性，确认信号有效后才实现 adaptive rollout。
test split 继续保留到完整创新方法冻结后使用一次。

### P2：进入项目创新主线

Phase 4 gate 通过后按既定路线推进：uncertainty-error correlation → uncertainty-aware adaptive rollout →
horizon/threshold/K/OOD/resource ablations。若 continued training 无收益，仍以官方 zero-shot 作为创新
基线，不回到无目的的 0.12B 训练。

## 7. 后续记录规范

以后每完成一个步骤，在本文件追加一节，并至少填写：

```text
问题：这一步要回答什么？
动机：为什么现在做，而不是做其他实验？
改动/配置：代码、数据、seed、checkpoint 和关键超参数是什么？
结果：日志、指标、视频和产物在哪里？
判断：结果支持什么，不支持什么？
异常：失败、风险和被推翻的假设是什么？
下一步：基于什么证据进入下一项？
```

所有关键结论必须能追溯到日志、checkpoint、测试输出或评估产物；避免只记录“完成了”，而不记录
为什么做以及结果意味着什么。
