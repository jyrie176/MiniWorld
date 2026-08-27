# 官方 0.55B 选择性预测受控扰动压力测试

## 目标与边界

本实验检验冻结的 `K=2 + RGB disagreement + LOEO 80%` 策略面对两类输出扰动时的行为。它是
controlled output-corruption stress test，不是真实世界 OOD benchmark，也不代表机器人闭环结果。

继续复用 validation episodes 1064-1079 的四个 seed，并穷举全部六个 K=2 pair。test episodes
1080-1095 未读取。

## 扰动

- `common_brightness +16/+32`：两个成员受到完全相同的像素亮度偏移，模拟 ensemble 一致地犯错；
- `independent_noise σ=8/16`：两个成员使用确定性、不同的高斯噪声，模拟成员间不一致；
- 所有像素裁剪到 `[0,255]`，随机种子由 episode、pair、level 稳定派生；
- clean K=2 AURC 必须与上一阶段冻结结果在 `1e-5` 内一致后才接受 stress 结果。

跨 condition 的绝对 AURC 会随整体误差尺度变化，因此正式比较使用：

- uncertainty ratio：相对 clean 的平均 disagreement；
- AURC gain vs random：该 condition 自己的 random AURC 减实际 AURC；
- LOEO-80 reduction：该 condition 全量 mean MAE 减保留输出 mean MAE。

## 结果

| condition | level | uncertainty / clean | full error Δ | AURC gain vs random | LOEO-80 reduction | worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 0 | 1.0000 | +0.0000 | 1.5717 | 0.7847 | 7.7536 |
| common brightness | +16 | 0.9797 | +9.2076 | 0.3775 | 0.2647 | 15.8259 |
| common brightness | +32 | 0.9604 | +24.0177 | 0.1456 | 0.1258 | 31.0232 |
| independent noise | σ=8 | 3.0763 | +3.2788 | 1.0328 | 0.6004 | 10.3706 |
| independent noise | σ=16 | 5.5364 | +8.4527 | 0.7201 | 0.4886 | 15.6631 |

## 解释

成员独立扰动会显著提高 disagreement，门控仍能保留风险排序能力；严重噪声下虽然绝对误差升高，
uncertainty 响应达到 clean 的 `5.54x`，AURC gain 仍为 `0.7201`。

共模扰动揭示明确盲区：`+32` 亮度使 full error 增加 `24.02`，但 disagreement 反而因像素 clipping
略降至 `0.96x`。两个成员一致地犯错时，ensemble disagreement 无法可靠报警，LOEO-80 的风险降低也
从 clean 的 `0.7847` 缩小到 `0.1258`。

因此正式结论不是“uncertainty 对 OOD 鲁棒”，而是：**采样分歧能检测成员间不一致，但不能单独检测
ensemble 的共模错误。** 部署时需要额外的输入漂移检测、重建/感知检查或独立模型信号，不能把
disagreement 当成完整安全证明。

## 产物

```text
/data/miniworld/experiments/official-055b-selective-stress/
  stress_summary.json
  conditions.csv
  pair_metrics.csv
  report.md
```
