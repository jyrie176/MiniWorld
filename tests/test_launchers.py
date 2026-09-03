import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_command_stub(tmp_path: Path, name: str) -> Path:
    executable = tmp_path / name
    executable.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "${LAUNCH_LOG}"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _run_launcher(tmp_path: Path, script: str, command: str, extra_env):
    _write_command_stub(tmp_path, command)
    log_path = tmp_path / "launch.log"
    env = os.environ.copy()
    env.update(
        PATH=f"{tmp_path}:{env['PATH']}",
        LAUNCH_LOG=str(log_path),
        DATA_ROOT="/fake/data",
        VAE_CKPT="/fake/vae.pt",
        CKPT="/fake/model.pt",
        POSE_DIR="/fake/poses",
        MINIWORLD_GIT_COMMIT="test-commit",
        **extra_env,
    )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / script)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return log_path.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize(
    "script", ["scripts/train_droid.sh", "scripts/train_re10k.sh"]
)
def test_training_launchers_forward_v100_overrides_without_muon(tmp_path, script):
    lines = _run_launcher(
        tmp_path,
        script,
        "torchrun",
        {
            "MIXED_PRECISION": "fp16",
            "ATTENTION_BACKEND": "sdpa",
            "MAX_GRAD_NORM": "0.5",
            "USE_MUON": "0",
            "NPROC_PER_NODE": "1",
        },
    )

    assert len(lines) == 4
    assert all("--mixed_precision fp16" in line for line in lines)
    assert all("--attention_backend sdpa" in line for line in lines)
    assert all("--max_grad_norm 0.5" in line for line in lines)
    assert all("--use_muon" not in line for line in lines)
    assert all("--nproc_per_node=1" in line for line in lines)


def test_training_launcher_can_preserve_upstream_muon_option(tmp_path):
    lines = _run_launcher(
        tmp_path,
        "scripts/train_droid.sh",
        "torchrun",
        {"USE_MUON": "1"},
    )
    assert all("--use_muon" in line for line in lines)


@pytest.mark.parametrize(
    "script",
    [
        "scripts/sample_droid.sh",
        "scripts/sample_re10k.sh",
        "scripts/benchmark_droid_throughput.sh",
    ],
)
def test_sampling_launchers_forward_precision_and_backend(tmp_path, script):
    lines = _run_launcher(
        tmp_path,
        script,
        "python",
        {"PRECISION": "fp16", "ATTENTION_BACKEND": "sdpa"},
    )

    assert len(lines) == 1
    assert "--precision fp16" in lines[0]
    assert "--attention_backend sdpa" in lines[0]


def test_droid_sampling_launcher_forwards_uncertainty_export_controls(tmp_path):
    lines = _run_launcher(
        tmp_path,
        "scripts/sample_droid.sh",
        "python",
        {
            "SEED": "123",
            "ACTION_VARIANT": "real",
            "SAVE_LATENTS": "1",
            "TOTAL_LEN": "6",
            "SAMPLE_NUM_VIDEOS": "16",
        },
    )

    assert len(lines) == 1
    assert "--seed 123" in lines[0]
    assert "--action_variant real" in lines[0]
    assert "--save_latents" in lines[0]
    assert "--total_len 6" in lines[0]
    assert "--sample_num_videos 16" in lines[0]


def test_uncertainty_launcher_runs_four_seeds_then_evaluator(tmp_path):
    lines = _run_launcher(
        tmp_path,
        "scripts/evaluate_droid_uncertainty.sh",
        "python",
        {
            "OUTPUT_ROOT": str(tmp_path / "formal"),
            "GPU": "0",
            "SEEDS": "11,22,33,44",
        },
    )

    assert len(lines) == 5
    for index, seed in enumerate((11, 22, 33, 44)):
        assert f"--seed {seed}" in lines[index]
        assert "--wm_model 0.5B" in lines[index]
        assert "--precision fp16" in lines[index]
        assert "--attention_backend sdpa" in lines[index]
        assert "--action_variant real" in lines[index]
        assert "--save_latents" in lines[index]
        assert "--total_len 6" in lines[index]
        assert "--num_sampling_steps 20" in lines[index]
        assert "--sample_num_videos 16" in lines[index]
    assert "scripts/evaluate_droid_uncertainty.py" in lines[-1]
    assert lines[-1].count("--sample_root") == 4


def test_flash_attention_is_not_a_mandatory_requirement():
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    mandatory = {
        line.strip().lower()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "flash-attn" not in mandatory
