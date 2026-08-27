# MiniWorld V100 实验复盘

最后更新：2026-08-27

## 1. 项目目标

在单张 Tesla V100-SXM2-32GB 上建立一条可重复运行的 MiniWorld DROID
最小训练与评估路径，并回答以下问题：

1. V100 不支持 FlashAttention-2 和 BF16 时，能否使用 PyTorch SDPA 与 FP16
   完成训练和推理？
2. 模型能否在小规模 DROID 数据上学会视频动力学？
3. 模型是否真正使用机器人动作，而不只是从第一帧生成看似合理的视频？
4. 普通训练权重与 EMA 权重分别适合什么用途？

本轮不是论文规模复现。当前目标是先打通可靠的 MVP，再通过受控实验定位主要瓶颈。

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

## 4. 当前关键结论

1. **V100 路径可用：** SDPA + FP16 GradScaler 能完成 5k-step 训练与推理。
2. **训练整体收敛：** 即时 loss 毛刺明显，但分段均值持续下降，没有持续数值发散证据。
3. **当前首选 EMA：** 在 64-episode 5k checkpoint 的固定样本上，EMA MAE 明显低于普通权重。
4. **动作条件有影响但证据不足：** real 与 zero 输出不同，但 real 尚未稳定优于 shuffle。
5. **视频质量仍有限：** 结构保持尚可，局部细节和长时间一致性不足。
6. **旧 recon 不可信：** 反推公式错误已修复，训练与 gen 不受影响。
7. **单样本指标不是泛化指标：** 下一步必须做多 episode 评估。

## 5. 当前产物索引

| 类型 | 路径/链接 |
| --- | --- |
| 64-episode 数据集 | `/data/miniworld/datasets/droid-mini-1000-1063` |
| Wan2.2 VAE | `/data/miniworld/checkpoints/wan2.2/Wan2.2_VAE.pth` |
| 单片段 checkpoint | `/data/miniworld/outputs/droid-v100-overfit-one/epoch_0500_step_00000500.pt` |
| 5k 最终 checkpoint | `/data/miniworld/outputs/droid-v100-64ep-5k/epoch_0079_step_00005000.pt` |
| 5k 训练日志 | `/data/miniworld/experiments/droid-v100-64ep-5k/train.log` |
| 5k 固定评估 | `/data/miniworld/experiments/droid-v100-64ep-5k-eval` |
| W&B | `https://wandb.ai/irvingjyrie176-tencent/miniworld-v100/runs/wdslnxhb` |
| GitHub PR | `https://github.com/jyrie176/MiniWorld/pull/1` |

## 6. 下一步实验队列

### P0：重新验证正确 recon

使用最终 5k checkpoint，在固定样本上分别测试 `t=0.1/0.3/0.5/0.7/0.9` 的单步重建，确认：

- 修复后高 timestep 不再因为公式错误直接保留为噪声；
- 模型真实的单步去噪能力随 timestep 如何变化；
- recon 与 gen 的差异来自模型能力还是多步误差累积。

### P0：16-episode EMA 动作消融

对 16 个固定 episode 运行：

- EMA + real；
- EMA + zero；
- EMA + shuffle。

报告平均 MAE、中位数、逐 episode real 胜率和代表性视频。只有 real 在多数 episode 上优于两个对照，
才能初步认为动作条件学习有效。

### P1：根据评估结果选择训练方向

- real 稳定获胜：扩大数据量与训练步数。
- real 与 shuffle 接近：优先核对动作和视频时间对齐，并调整动作条件训练/CFG。
- recon 好、gen 差：处理多步误差累积和流式 rollout 设置。
- recon 本身差：继续优化 denoiser 训练，而不是先增加 rollout 长度。

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
