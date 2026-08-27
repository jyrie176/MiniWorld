import torch

from scripts.evaluate_droid_action_ablation import compute_episode_metrics


def test_compute_episode_metrics_scores_only_future_frames():
    ground_truth = torch.tensor([0, 10, 20], dtype=torch.uint8).reshape(3, 1, 1, 1)
    real = torch.tensor([100, 12, 18], dtype=torch.uint8).reshape(3, 1, 1, 1)
    zero = torch.tensor([200, 9, 19], dtype=torch.uint8).reshape(3, 1, 1, 1)
    reverse = torch.tensor([50, 15, 25], dtype=torch.uint8).reshape(3, 1, 1, 1)

    metrics = compute_episode_metrics(ground_truth, real, zero, reverse)

    assert metrics == {
        "mae_real": 2.0,
        "final_mae_real": 2.0,
        "mae_zero": 1.0,
        "final_mae_zero": 1.0,
        "mae_reverse": 5.0,
        "final_mae_reverse": 5.0,
        "output_mae_real_zero": 2.0,
        "output_mae_real_reverse": 5.0,
        "mae_persistence": 15.0,
        "gt_final_motion_mae_from_frame0": 20.0,
    }
