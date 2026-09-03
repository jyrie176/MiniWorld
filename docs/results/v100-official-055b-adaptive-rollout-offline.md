# V100 官方 0.55B 自适应 Rollout 离线评估

> **2026-08-27 成本语义更正：** 本报告的 retained-quality 结论仍有效，但
> `generated coverage=0.9375` 是逐 latent-step 理想化代理。官方 `history_len=1`、
> `df_chunk_size=2` 的真实完成边界为 `(1,3,5)`；冻结 EMA 策略按 chunk 重放后 generated coverage
> 为 `1.0`。因此本文的“在线授权”已被 Stage A′ 取代且撤销。详见
> `docs/results/v100-official-055b-adaptive-rollout-chunk-aligned.md`。

## 问题与主线位置

Phase 5 已证明 sampling disagreement 与未来 RGB error 相关。本实验回答下一层问题：在不重新训练、
不访问密封 test 的前提下，latent uncertainty 能否决定保留多少未来预测，并在至少 80% coverage 下优于
覆盖率严格匹配的固定 horizon。只有离线门禁通过，才允许修改在线推理。

本实验使用官方 zero-shot 0.55B、validation episodes 1064-1079 和冻结的 `K=4` correlation CSV。
test episodes 1080-1095 未访问。评估代码提交为 `6da0c14d1190a97a9c60dae9b15d83d747d82933`；
源归档 SHA256 为
`4fe4b2fb4b59dce77199b46fb82109a3d1197378ee6fd1a1b1df1dca2a354d86`。

## 方法

- 主信号：latent population variance；RGB disagreement 只画诊断曲线，不参与门禁。
- 策略一：当前步 uncertainty 超过阈值时请求新观测。
- 策略二：EMA（`alpha=0.5`）超过阈值连续两步时请求新观测。
- 请求发生在生成 step `t` 后，保留 horizon 为 `max(1,t-1)`，generated horizon 仍计为 `t`。
- 阈值选择：16 折 leave-one-episode-out；每折阈值只来自另外 15 个 episodes。
- 目标：coverage 至少 `0.80`；匹配基线在相邻两个 fixed horizons 之间作解析混合。
- 通过条件：coverage ≥ 0.80、总体 retained MAE 更低、至少赢 9/16 episodes、p90 不比匹配基线差
  0.10 以上，四项必须同时成立。

## 正式结果

| 策略 | retained coverage | generated coverage | retained horizon | generated horizon | retained RGB MAE | matched fixed MAE | wins | p90 / fixed p90 | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 单阈值 | 0.7750 | 0.8875 | 3.8750 | 4.4375 | 4.6512 | 4.8213 | 5/16 | 5.7890 / 6.2107 | fail |
| EMA + 连续两步 | **0.8125** | 0.9375 | 4.0625 | 4.6875 | **4.7402** | **4.9327** | **10/16** | **5.9564 / 6.4492** | **pass** |

EMA + 连续两步策略的 LOEO retained MAE 比匹配 fixed baseline 低 `0.1925`（约 `3.90%`），且四项
预注册条件全部通过。全 validation 拟合、供未来在线实现使用的 deployment threshold 为
`0.020729146897792816`。单阈值策略虽然总体误差更低，但 LOEO coverage 降到 0.775 且只赢 5 个
episodes，因此不能用它进入在线实现；这也说明仅优化总体均值会选出泛化不足的策略。

## 计算成本与边界

通过策略平均保留 4.0625 步，却平均生成 4.6875 步。相对固定 `K=4,H=5` 的 20 个 ensemble-member
step，离线解析成本代理从 20 降至 18.75，只节省 `6.25%`，不能把 retained coverage 18.75% 的下降
误报成等量算力收益。相对 `K=1,H=5`，该策略仍约为 3.75 倍 sample-step 成本；是否有墙钟收益必须由
下一阶段在线实现实测。

通过并不表示每条 episode 都改善。EMA + 连续两步在 1064、1066、1067、1070、1077、1078 上不如
匹配 fixed baseline，其中 1078 延续了上一阶段发现的低 uncertainty / 高 error 反例。当前策略也没有
真的从机器人取得新观测；`REQUEST_OBSERVATION` 只是确定 rollout 的信任边界。

## 独立复算与结论

独立脚本仅从 `policy_decisions.csv` 与冻结源 CSV 重新累计 retained numerator/count、generated horizon、
coverage、episode MAE、解析 fixed mixture、wins、p90 和四项 gate，没有调用评估器内部函数。全部数值与
JSON 在预注册的 `1e-12`（比例）和 `1e-9`（RGB MAE）容差内一致。

原始逐步代理结论为：EMA + 连续两步策略通过 retained-quality 门禁。后续 Stage A′ 已证明该策略在真实
chunk 边界下不节省完整生成 chunk，并撤销在线实现授权。本报告保留原始结果用于追溯，不再作为下一步
执行依据；test 仍未查看。

正式产物：`/data/miniworld/experiments/official-055b-adaptive-rollout-offline`。
结果归档：`/data/miniworld/exports/miniworld-official-055b-adaptive-rollout-offline.tar.gz`，SHA256
`7492b80ca0b14812e601a4aead8de0165d57c52c47b7a594c2e501e129c902c4`，大小 40 KiB；归档不重复包含
源 K=4 视频、latent 或 checkpoint。
