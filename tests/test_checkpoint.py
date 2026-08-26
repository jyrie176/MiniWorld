from types import SimpleNamespace

import torch

from miniworld.train import load_pretrained, save_checkpoint


class _FakeScaler:
    def __init__(self, state):
        self._state = state
        self.loaded = None

    def state_dict(self):
        return self._state

    def load_state_dict(self, state):
        self.loaded = state


def _args(output_dir):
    return SimpleNamespace(
        output_dir=str(output_dir),
        wm_model="B",
        latent_frames=2,
        df_chunk_size=2,
        use_pose_cond=False,
        use_action_cond=True,
        cond_dim=4,
        cond_per_token=False,
        timestep_baseshift=2.667,
        mixed_precision="fp16",
        attention_backend="sdpa",
        max_grad_norm=1.0,
    )


def test_checkpoint_round_trips_scaler_state(tmp_path):
    model = torch.nn.Linear(2, 2)
    ema_model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    saved_scaler = _FakeScaler({"scale": 4096.0, "growth_tracker": 7})

    save_checkpoint(
        args=_args(tmp_path),
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        scaler=saved_scaler,
        epoch=3,
        global_step=11,
    )
    payload = torch.load(tmp_path / "last.pt", weights_only=False)
    assert payload["meta"]["mixed_precision"] == "fp16"
    assert payload["meta"]["attention_backend"] == "sdpa"
    assert payload["meta"]["max_grad_norm"] == 1.0

    restored_scaler = _FakeScaler({})
    epoch, step = load_pretrained(
        str(tmp_path / "last.pt"),
        model,
        ema_model,
        optimizer,
        scaler=restored_scaler,
    )

    assert (epoch, step) == (3, 11)
    assert restored_scaler.loaded == saved_scaler.state_dict()


def test_legacy_checkpoint_without_scaler_still_loads(tmp_path):
    model = torch.nn.Linear(2, 2)
    ema_model = torch.nn.Linear(2, 2)
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "epoch": 2,
            "global_step": 5,
        },
        path,
    )
    scaler = _FakeScaler({})

    epoch, step = load_pretrained(
        str(path), model, ema_model, scaler=scaler
    )

    assert (epoch, step) == (2, 5)
    assert scaler.loaded is None
