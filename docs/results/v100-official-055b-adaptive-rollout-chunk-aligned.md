# V100 官方 0.55B Chunk-Aligned Adaptive Rollout Stage A′

## 问题与更正原因

上一阶段在逐 latent-step 代理上得到 retained-quality 正面结果，但官方推理使用 `history_len=1`、
`df_chunk_size=2`。未来五步只能在 `(1,3,5)` 三个边界完成，step 4 触发时 step 5 已经随同一 chunk
生成。因此上一报告的 `generated coverage=0.9375` 不是可执行的 completed-chunk 成本；按真实边界重放
为 `1.0`。本阶段在查看 test 前公开更正并重新选择阈值。

本实验只复用官方 0.55B validation episodes 1064-1079 的冻结 `K=4`、80 行 CSV，不训练、不采样、
不解码、不修改 checkpoint。test episodes 1080-1095 未访问。

```text
evaluation commit: 6b37f589063b1b53b6efaaec88b6be5a3b5ccc8a
checkpoint SHA: e4b118befe88cee7338400c5510fdd497212b9b1988034290030b3ed351ced32
data manifest SHA: ac3e1dbff5a22732b54c47834751714f31302bf0aca859b4dc78a2809203ae70
source archive SHA: 4fe4b2fb4b59dce77199b46fb82109a3d1197378ee6fd1a1b1df1dca2a354d86
completion boundaries: 1, 3, 5
```

## 方法与门禁

策略仍为 latent `K=4` population variance，比较单阈值与 EMA `alpha=0.5`＋连续两步超阈值。每个
chunk 完成后，策略按顺序消费该 chunk 中新出现的 uncertainty；若其中触发，generated horizon 取当前
chunk 尾，retained horizon 取前一个完整边界。阈值继续用 16-fold LOEO 选择，RGB uncertainty 只作诊断。

Stage A′ 必须同时满足：retained coverage ≥0.80、MAE 严格低于匹配 fixed、至少赢 9/16 episodes、
p90 不恶化超过 0.10，以及 completed generated coverage <1.0。

## 正式结果

| policy | retained/generated coverage | retained/generated horizon | RGB MAE / matched fixed | wins | p90 / fixed p90 | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| threshold | 0.775 / 0.950 | 3.875 / 4.750 | 4.8328 / 4.8213 | 7/16 | 5.9564 / 6.2107 | fail |
| EMA + hysteresis | 0.850 / 1.000 | 4.250 / 5.000 | 4.8429 / 5.0616 | 6/16 | 5.7890 / 6.6042 | fail |

单阈值确实避开 4/80 个完整 future steps，`K=4` completed-member-step 代理为 `304/320=95%`，但
coverage 只有 77.5%、总体 MAE 还比匹配 fixed 高 0.0114，且只赢 7 个 episodes。它不能通过降低覆盖率
换取成本结果。全 validation deployment threshold 为 `0.03007829748094082`，但因门禁失败不授权部署。

EMA + hysteresis 在 retained coverage 和总体 MAE 上成立，却只赢 6 个 episodes，而且所有 80 个 future
steps 都完成生成，`K=4` 成本代理仍为 `320/320=100%`。其 deployment threshold
`0.022462889552116394` 同样不授权部署。

相对普通 fixed `K=1,H=5` 的 80 member-steps，单阈值和 EMA 策略仍分别是 3.8 倍和 4.0 倍。由于线上
`stream_inflight_chunks=8` 还会提前对后续 chunk 做部分去噪，completed-chunk 比例也不能称为墙钟加速。

## 独立复算与判断

独立标准库脚本只读取新 `policy_decisions.csv`、`chunk_costs.csv` 和冻结源 CSV，重新计算 numerator、
count、coverage、mean/median/p90/worst、request rate、解析 matched fixed、16 个 episode deltas、wins、
五项 gate 和 K=4/K=1 成本。比例在 `1e-12`、RGB MAE 在 `1e-9` 内与 JSON 完全一致；旧冻结阈值也
独立重放为 completed generated coverage `1.0`。

结论：**Stage A′ 失败，在线 early-stopping 不获授权。** uncertainty 仍具有 error-ranking 和输出信任
截断价值，但在当前五步短 rollout、两帧 chunk 和 K=4 成本下，没有策略同时满足质量、覆盖率、episode
稳定性与真实完整-chunk节省。下一步不修改在线推理；若继续探索，应把 `df_chunk_size=1` 作为需要重新
建立官方推理基线的独立子项目，而不是当前方法的自动 fallback。

正式产物：`/data/miniworld/experiments/official-055b-adaptive-rollout-chunk-aligned`。
轻量归档：`/data/miniworld/exports/miniworld-official-055b-adaptive-rollout-chunk-aligned.tar.gz`，
SHA256 `986462de7cee3df2183968d05cf5d2b74ccf6a7b131323c7d4be8440d7818616`，大小 40 KiB；
不重复包含源视频、latent 或 checkpoint。
