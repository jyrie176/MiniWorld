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


def test_flash_attention_is_not_a_mandatory_requirement():
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    mandatory = {
        line.strip().lower()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "flash-attn" not in mandatory
