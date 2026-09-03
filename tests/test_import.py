import builtins
import subprocess
import sys


def test_import_miniworld_without_flash_attention():
    code = """
import builtins

real_import = builtins.__import__

def block_flash(name, *args, **kwargs):
    if name == "flash_attn" or name.startswith("flash_attn."):
        raise ModuleNotFoundError("blocked flash_attn for import-safety test")
    return real_import(name, *args, **kwargs)

builtins.__import__ = block_flash
import miniworld
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_import_miniworld_with_broken_flash_binary():
    code = """
import builtins

real_import = builtins.__import__

def break_flash(name, *args, **kwargs):
    if name == "flash_attn" or name.startswith("flash_attn."):
        raise OSError("simulated incompatible flash-attn binary")
    return real_import(name, *args, **kwargs)

builtins.__import__ = break_flash
import miniworld
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
