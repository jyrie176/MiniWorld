# Official MiniWorld-0.55B lower-LR continued-training baseline

Result date: 2026-08-27

## Mainline decision

This is the final Phase 4 continued-training selection experiment. It keeps the
audited 1,064-episode train set and all settings from the expanded-data run,
changing only the learning rate from `2e-5` to `2e-6`. Its purpose is to decide
whether a conservative all-parameter adaptation can beat the released 0.55B
checkpoint before the uncertainty-error correlation phase. Validation remains
episodes 1064-1079; test episodes 1080-1095 remain sealed.

## Setup and training stability

- Initialization: official `MiniWorld_0_5b_droid.pt`
- Train data: DROID episodes 0-1063
- Runtime: two V100 GPUs, DDP, FP16 GradScaler, PyTorch SDPA
- Window/batch: six latent frames, one sample per rank, global batch two
- Optimization: 1,000 effective steps, AdamW, LR `2e-6`, EMA `0.9999`,
  gradient clip `1.0`
- Seed base: `20260827`
- W&B: `d59p17p7`

The run exited normally. One early overflow was skipped and reduced the loss
scale from 65,536 to 32,768; no later overflow occurred. Logged mean/median
loss were 0.1314/0.0911, with a 0.0485-0.5794 range caused by heterogeneous
episode difficulty. Final throughput was 0.85 step/s or 1.70 sample/s.
Checkpoints were saved at effective steps 531 and 1000.

## Fixed real-action validation

| Weights | Step 531 | Step 1000 | Official zero-shot | Expanded LR `2e-5`, step 1000 |
| --- | ---: | ---: | ---: | ---: |
| model | 5.5340 | 5.5779 | **5.4680** | 6.3781 |
| EMA | 5.4670 | 5.4630 | 5.4680 | 5.5007 |

Lowering LR substantially reduces forgetting: final model improves by 0.8002
MAE and final EMA by 0.0376 versus the `2e-5` expanded run. It does not provide
a meaningful quality gain over the released checkpoint. Final model is 0.1099
worse and beats the official checkpoint on only 3/16 paired episodes. Final
EMA is numerically 0.0050 better and improves 13/16 episodes, but the absolute
change is only 0.09% of the 5.468 baseline and individual deltas are mostly a
few thousandths of one RGB value. This is not a practically credible gain on
a 16-episode selection split.

Persistence remains 4.4681. Final model beats it on 4/16 episodes and final EMA
on 5/16, so the strong static-prediction limitation is unchanged.

## Final-model action ablation

| Condition | Mean MAE |
| --- | ---: |
| real | 5.5779 |
| zero-valued | 9.4563 |
| reversed | 9.3373 |

Real beats zero on 15/16 episodes, reverse on 16/16, and both on 15/16. The
lower-LR model therefore preserves strong action conditioning despite its small
real-action quality regression. EMA was not expanded to action ablations
because its real-action output changed only 0.005 from the official baseline.

## Checkpoint and conclusion

The final checkpoint contains 420 model tensors, 420 EMA tensors, optimizer
state, epoch 2, global step 1000, metadata, and GradScaler state. Its final
scale is 32,768 with growth tracker 982.

Phase 4 is frozen with the official zero-shot checkpoint, not a continued
checkpoint. Across the controlled sequence, 64-episode `2e-5` training caused
large forgetting; 1,064 episodes restored action consistency but not quality;
and `2e-6` reduced forgetting to a negligible EMA change without a meaningful
gain. More continued-training hyperparameter search is not justified before
the innovation work. The next mainline phase is uncertainty-error correlation
using the official checkpoint and validation split. The test split remains
sealed for the final frozen method.

## Artifacts

```text
/data/miniworld/outputs/droid-official-055b-expanded1064-lr2e6-1k-lf6-ddp2
/data/miniworld/experiments/droid-official-055b-expanded1064-lr2e6-1k-lf6-ddp2/train.log
/data/miniworld/experiments/official-055b-expanded1064-lr2e6-1k-validation
```
