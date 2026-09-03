# 官方 0.55B 不确定性感知选择性预测

## 目标

Chunk-aligned adaptive rollout 证明当前 `df_chunk_size=2`、五步未来窗口下不能可靠节省完整生成
chunk。本实验不再把 uncertainty 包装成推理加速，而是检验它能否作为可靠性门控：拒绝高不确定预测，
并把拒绝解释为“停止执行并请求新观测”。

实验复用冻结的官方 0.55B、validation episodes 1064-1079、K=4 采样产生的 80 条
episode-step 指标，不训练、不采样、不解码，也不读取 test episodes 1080-1095。

## 方法

- 预测单元：一个 episode 的一个 future latent step，共 16×5=80 个；
- 风险：四个 seed 平均的 future RGB MAE；
- 信号：latent disagreement 与 RGB disagreement；
- 按 uncertainty 从低到高保留预测，计算 risk-coverage curve 与离散 AURC；
- random baseline 的期望 AURC 等于全量平均风险；oracle 按真实误差排序，仅作为不可部署上界；
- coverage 100%、90%、80%、70% 均采用 LOEO：每次只用其余 15 个 episode 的 uncertainty
  分位数选择阈值，再应用到 held-out episode。100% 是不拒绝基线。

阈值选择从不使用 held-out error；四个 `error_seed_*` 列只用于上游构造平均 `error_rgb`，不作为
不确定性输入。

## 正式结果

| signal | AURC | random | oracle | 相对 random 改善 |
| --- | ---: | ---: | ---: | ---: |
| latent disagreement | 4.4025 | 5.4804 | 3.8187 | 1.0779 |
| RGB disagreement | **3.8979** | 5.4804 | 3.8187 | **1.5825** |

RGB disagreement 更接近 oracle 排序。LOEO 结果如下：

| signal | target coverage | realized coverage | mean MAE | P90 | worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.0 | 1.0000 | 5.4804 | 7.9138 | 14.0572 |
| latent | 0.9 | 0.9000 | 5.0494 | 7.4344 | 8.9689 |
| latent | 0.8 | 0.8125 | 4.7829 | 6.9615 | 7.9104 |
| latent | 0.7 | 0.7000 | 4.5039 | 6.6548 | 7.3926 |
| RGB | 0.9 | 0.9000 | 4.9795 | 7.3701 | 7.9439 |
| RGB | 0.8 | 0.8000 | **4.7158** | **6.9708** | **7.5638** |
| RGB | 0.7 | 0.7000 | **4.4238** | **6.4111** | **7.5638** |

在 RGB 80% operating point，平均 MAE 相对全量下降 `13.95%`，worst MAE 下降 `46.19%`。
这说明 disagreement 对高风险预测有较强排序能力，但结果不等于模型本身精度提升：质量改善来自
拒绝 20% 的输出。

## 结论与可陈述边界

正式结论是：**官方 0.55B 的多次采样 disagreement 可用作选择性预测置信门控，并在 held-out
episode 阈值选择下显著降低保留输出的平均和最坏误差。** 它支持“高不确定时重新观察”的系统设计，
不支持以下说法：

- 不支持在线推理已经提速；
- 不支持机器人闭环成功率已经提高；
- 不支持在完整 DROID 分布或 OOD 场景上已经泛化；
- 不支持 oracle，因为 oracle 使用真实未来误差，只是上界。

下一 gate 是冻结 RGB 80% policy 后，仅使用一次密封 test split 做最终确认；在此之前可先做 K=1/2/4
成本与可靠性消融，判断 K=4 的额外采样成本是否值得。

## 产物

```text
/data/miniworld/experiments/official-055b-selective-prediction/
  selective_prediction_summary.json
  risk_coverage_curve.csv
  loeo_folds.csv
  report.md
```
