"""End-to-end LDCast training on dgmr-rs radar data.

Runs the autoencoder, extracts its best state_dict, then runs the diffusion
model — in two separate subprocesses so the GPU is fully reset between stages.

Usage (from this directory):

    DGMR_RADAR_ROOT=/path/to/radar_data DGMR_RADAR_INDEX=/path/to/index.txt \\
        uv run python train_rust.py
    uv run python train_rust.py --height=128 --width=128  # smaller crops
    uv run python train_rust.py --skip_autoenc=True        # only stage 2
    uv run python train_rust.py --force_autoenc=True       # retrain stage 1

If autoencoder checkpoints already exist in --autoenc_dir you'll be prompted
whether to re-train; default (Enter or 'N') skips stage 1 and goes straight
to diffusion using the best existing checkpoint.
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
from pathlib import Path

from fire import Fire
from omegaconf import OmegaConf

SCRIPTS_DIR = Path(__file__).resolve().parent


def _check_bridge():
    try:
        import dgmr_py  # noqa: F401
    except ImportError as e:
        sys.exit(
            f"Cannot import dgmr_py ({e}). Build the bridge first:\n"
            "  cd /home/christian/github/ldcast\n"
            "  uv pip install 'maturin>=1.7,<2'\n"
            "  VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release "
            "--manifest-path ../dgmr-py/Cargo.toml"
        )


def _check_env(radar_root, index_path):
    # radar_root reaches the Rust loader only via DGMR_RADAR_ROOT (it has no API to pass it);
    # index_path is forwarded straight to the stage scripts. Require each only if not in the config.
    if radar_root is None and "DGMR_RADAR_ROOT" not in os.environ:
        sys.exit("Set radar_root in config/train_rust.yaml or export DGMR_RADAR_ROOT")
    if index_path is None and "DGMR_RADAR_INDEX" not in os.environ:
        sys.exit("Set index_path in config/train_rust.yaml or export DGMR_RADAR_INDEX")


_VAL_LOSS_RE = re.compile(r"val_rec_loss=(\d+\.\d+)")


def _best_autoenc_ckpt(autoenc_dir: Path) -> Path:
    """Return the .ckpt with the lowest val_rec_loss parsed from its filename.

    Lightning's ModelCheckpoint(save_top_k=3, monitor='val_rec_loss',
    filename='{epoch}-{val_rec_loss:.4f}') keeps the three best checkpoints.
    Alphabetical sort picks the highest epoch, not the best loss — parse it.
    """
    candidates = []
    for path in glob.glob(str(autoenc_dir / "*.ckpt")):
        m = _VAL_LOSS_RE.search(os.path.basename(path))
        if m:
            candidates.append((float(m.group(1)), path))
    if not candidates:
        sys.exit(f"No val_rec_loss checkpoints found in {autoenc_dir}")
    candidates.sort()
    return Path(candidates[0][1])


def _extract_state_dict(ckpt_path: Path, out_path: Path) -> Path:
    import torch

    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out_path)
    return out_path


def _existing_ckpts(autoenc_dir: Path) -> list[str]:
    return sorted(glob.glob(str(autoenc_dir / "*.ckpt")))


def _prompt_retrain(autoenc_dir: Path) -> bool:
    """Return True to re-train, False to skip stage 1. Default (Enter/N) = skip."""
    prompt = f"Autoenc checkpoint(s) found in {autoenc_dir}. Re-train? [y/N] "
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        answer = ""
    return answer in ("y", "yes")


def _run(cmd: list[str], stage_name: str) -> None:
    print(f"\n=== {stage_name} ===")
    print("$", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=SCRIPTS_DIR)
    if rc != 0:
        sys.exit(f"{stage_name} failed (exit code {rc})")


def run(
    radar_root: str | None = None,
    index_path: str | None = None,
    autoenc_dir: str = "../models/autoenc_rust",
    genforecast_dir: str = "../models/genforecast_rust",
    height: int = 256,
    width: int = 256,
    autoenc_batch_size: int = 16,
    genforecast_batch_size: int = 8,
    num_workers: int = 4,
    past_steps: int = 4,
    future_steps: int = 8,
    force_autoenc: bool = False,
    skip_autoenc: bool = False,
    precision: str = "bf16-mixed",
    optimizer_8bit: bool = False,
    max_epochs: int = 1000,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
    max_hours: float | None = None,
    early_stopping_patience: int = 6,
    sample_every_n_epochs: int = 1,
    autoenc_ckpt_path: str | None = None,
    genforecast_ckpt_path: str | None = None,
):
    """Train autoencoder then diffusion model in one shot."""
    if force_autoenc and skip_autoenc:
        sys.exit("--force_autoenc and --skip_autoenc are mutually exclusive")

    _check_bridge()
    _check_env(radar_root, index_path)
    # radar_root reaches the Rust loader only via DGMR_RADAR_ROOT; export it from the config so the
    # spawned stage subprocesses (which inherit os.environ) pick it up. index_path is forwarded
    # directly as --index_path below — no env round-trip.
    if radar_root is not None:
        os.environ["DGMR_RADAR_ROOT"] = radar_root

    autoenc_dir_p = (SCRIPTS_DIR / autoenc_dir).resolve()

    # Stage 1 decision
    if skip_autoenc:
        do_autoenc = False
    elif force_autoenc:
        do_autoenc = True
    elif _existing_ckpts(autoenc_dir_p):
        do_autoenc = _prompt_retrain(autoenc_dir_p)
    else:
        do_autoenc = True

    # Stage 1: autoencoder
    if do_autoenc:
        cmd = [
            sys.executable,
            "train_autoenc_rust.py",
            f"--model_dir={autoenc_dir}",
            f"--height={height}",
            f"--width={width}",
            f"--batch_size={autoenc_batch_size}",
            f"--num_workers={num_workers}",
            f"--past_steps={past_steps}",
            f"--future_steps={future_steps}",
            f"--precision={precision}",
            f"--max_epochs={max_epochs}",
            f"--sample_every_n_epochs={sample_every_n_epochs}",
            f"--early_stopping_patience={early_stopping_patience}",
        ]
        if limit_train_batches is not None:
            cmd.append(f"--limit_train_batches={limit_train_batches}")
        if limit_val_batches is not None:
            cmd.append(f"--limit_val_batches={limit_val_batches}")
        if max_hours is not None:
            cmd.append(f"--max_hours={max_hours}")
        if autoenc_ckpt_path is not None:
            cmd.append(f"--ckpt_path={autoenc_ckpt_path}")
        if index_path is not None:
            cmd.append(f"--index_path={index_path}")
        _run(cmd, stage_name="Stage 1: autoencoder")
    else:
        print(f"Skipping stage 1 (autoencoder); using existing checkpoints in {autoenc_dir_p}")

    # Extract state_dict from the best autoenc checkpoint
    print("\n=== Extracting autoencoder state_dict ===")
    best_ckpt = _best_autoenc_ckpt(autoenc_dir_p)
    state_dict_path = autoenc_dir_p / "state_dict.pt"
    _extract_state_dict(best_ckpt, state_dict_path)
    print(f"Picked {best_ckpt.name} -> wrote {state_dict_path}")

    # Stage 2: diffusion. autoenc_weights_fn must be a path resolvable from scripts/.
    autoenc_weights_arg = os.path.relpath(state_dict_path, SCRIPTS_DIR)
    cmd = [
        sys.executable,
        "train_genforecast_rust.py",
        f"--autoenc_weights_fn={autoenc_weights_arg}",
        f"--model_dir={genforecast_dir}",
        f"--height={height}",
        f"--width={width}",
        f"--batch_size={genforecast_batch_size}",
        f"--num_workers={num_workers}",
        f"--past_steps={past_steps}",
        f"--future_steps={future_steps}",
        f"--precision={precision}",
        f"--optimizer_8bit={optimizer_8bit}",
        f"--max_epochs={max_epochs}",
        f"--sample_every_n_epochs={sample_every_n_epochs}",
        f"--early_stopping_patience={early_stopping_patience}",
    ]
    if limit_train_batches is not None:
        cmd.append(f"--limit_train_batches={limit_train_batches}")
    if limit_val_batches is not None:
        cmd.append(f"--limit_val_batches={limit_val_batches}")
    if max_hours is not None:
        cmd.append(f"--max_hours={max_hours}")
    if genforecast_ckpt_path is not None:
        cmd.append(f"--ckpt_path={genforecast_ckpt_path}")
    if index_path is not None:
        cmd.append(f"--index_path={index_path}")
    _run(cmd, stage_name="Stage 2: diffusion model")

    print("\nAll stages complete.")


def main(config=None, **kwargs):
    cfg = OmegaConf.load(config) if config else {}
    cfg.update(kwargs)
    run(**cfg)


if __name__ == "__main__":
    Fire(main)
