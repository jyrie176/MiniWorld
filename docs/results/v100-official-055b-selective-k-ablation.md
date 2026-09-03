# 官方 0.55B 选择性预测 K=1/2/4 成本—可靠性消融

## 问题

阶段 W 证明 K=4 RGB disagreement 可以拒绝高风险预测，但需要四次独立 rollout。本实验回答：减少到
K=2 后能否保留主要可靠性收益，以及 K=4 的额外两次采样换来了什么。

## 口径

- 复用 validation episodes 1064-1079 已保存的四个 seed 视频、latent 和误差，不重新采样；
- K=2 枚举四个 seed 的全部六种组合，不挑选最佳 pair；
- 每个 K=2 pair 的风险是该 pair 两个成员 RGB MAE 的均值；
- K=4 使用四成员正式结果；
- K=1 没有 disagreement，只作为不带 reliability gating 的 `1x` 成本下界；
- K=2/K=4 均使用 LOEO 80% policy；test episodes 1080-1095 保持密封；
- 原采样日志没有可靠起止时间，因此只报告实际 rollout 次数 `1x/2x/4x`，不声称 wall-time。

程序从视频/latent 重建 K=4 disagreement，并与冻结 CSV 在 `1e-5` 内一致后才计算子集。

## 结果

| K | 相对采样成本 | AURC | LOEO-80 mean MAE | LOEO-80 worst MAE |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1x | 不可用 | 5.4804（仅 100% coverage） | 不适用 |
| 2 | 2x | **3.9087** | **4.6958** | 7.7536 |
| 4 | 4x | **3.8979** | 4.7158 | **7.5638** |

K=2 六个 pair 的稳定性范围：

| 指标 | mean | min | max |
| --- | ---: | ---: | ---: |
| AURC | 3.9087 | 3.8939 | 3.9202 |
| LOEO-80 mean MAE | 4.6958 | 4.6588 | 4.7346 |
| LOEO-80 worst MAE | 7.7536 | 7.6633 | 7.8093 |

六个 pair 的 AURC 范围很窄，没有依赖某个幸运 seed pair。K=2 与 K=4 的 AURC 差值只有 `0.0108`
（约 K=4 的 `0.28%`），但少采样两次。K=2 的平均保留误差没有弱于 K=4；K=4 的明确优势是最坏
误差进一步降低约 `0.190`。

## 决策

冻结 **K=2 + RGB disagreement + LOEO 80% target coverage** 作为成本敏感的候选部署策略。它用 K=4
一半的 rollout 次数保留了几乎全部排序能力。K=4 定位为更看重尾部风险、允许更高计算成本的配置。

该结论是 validation 上的成本—可靠性选择，不是线上时延加速或闭环成功率结论。下一步可以做 OOD
stress test；密封 test 仍留到方法、K 和 coverage 全部冻结后只使用一次。

## 产物

```text
/data/miniworld/experiments/official-055b-selective-k-ablation/
  k_ablation_summary.json
  k2_pair_metrics.csv
  report.md
```
