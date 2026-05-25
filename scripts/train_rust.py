"""End-to-end LDCast training on dgmr-rs radar data.

Runs the autoencoder, extracts its best state_dict, then runs the diffusion
model — in two separate subprocesses so the GPU is fully reset between stages.

Everything is driven by the config; you shouldn't need CLI flags. The shipped
config/train_rust.yaml is loaded by default, so this is the whole command:

    uv run python train_rust.py

Two config knobs control the workflow:

    stages: both | autoenc | diffusion   # which stage(s) to run
    resume: false | true                  # false=restart; true=continue last.ckpt if any, else fresh

  both      = stage 1 (autoencoder) then stage 2 (diffusion)
  autoenc   = stage 1 only (extend the autoencoder, stop before diffusion)
  diffusion = stage 2 only (reuse the existing autoencoder)

  resume=false always starts the executed stage(s) from scratch; resume=true
  continues each from <stage_dir>/last.ckpt when present (Lightning restores
  epoch/optimizer/LR/EMA) and starts fresh when it's missing -- so a single
  resume=true survives stop/restart cycles.

Any field can still be overridden on the CLI, e.g. --stages=diffusion --resume=True.
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
DEFAULT_CONFIG = SCRIPTS_DIR.parent / "config" / "train_rust.yaml"


def _check_bridge():
    try:
        import dgmr_py  # noqa: F401
    except ImportError as e:
        sys.exit(
            f"Cannot import dgmr_py ({e}). Build the bridge first (from the ldcast repo root):\n"
            "  uv pip install 'maturin>=1.7,<2'\n"
            "  # CARGO_NET_GIT_FETCH_WITH_CLI=true is required: dgmr-py pulls private hdf5_* deps\n"
            "  # over SSH, and cargo's built-in libgit2 auth fails where the git CLI succeeds.\n"
            "  CARGO_NET_GIT_FETCH_WITH_CLI=true VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop \\\n"
            "    --release --manifest-path ../dgmr-py/Cargo.toml"
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


def _resume_ckpt(stage_dir: Path) -> str | None:
    """Path to the checkpoint to resume from, or None to start fresh.

    The autoencoder stage keeps a last.ckpt; the diffusion stage keeps a single
    rolling 'epoch=..-step=..ckpt' (save_top_k=1, no last.ckpt), so prefer
    last.ckpt when present and otherwise fall back to the newest *.ckpt.

    resume=true means "continue if there's a checkpoint, else start fresh", so a
    first run (no checkpoint yet) starts from scratch instead of erroring. One
    resume=true setting then does the right thing across stop/restart cycles
    (e.g. time-boxed training that stops and resumes each run)."""
    last = stage_dir / "last.ckpt"
    if last.exists():
        return str(last)
    ckpts = glob.glob(str(stage_dir / "*.ckpt"))
    return max(ckpts, key=os.path.getmtime) if ckpts else None


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
    genforecast_lr: float = 1e-4,
    genforecast_accumulate_grad_batches: int = 1,
    num_workers: int = 4,
    past_steps: int = 4,
    future_steps: int = 8,
    stages: str = "both",
    resume: bool = False,
    precision: str = "bf16-mixed",
    optimizer_8bit: bool = False,
    max_epochs: int = 1000,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
    max_hours: float | None = None,
    early_stopping_patience: int = 6,
    sample_every_n_epochs: int = 1,
):
    """Run the autoencoder and/or diffusion stages per the config (see module docstring)."""
    valid_stages = ("both", "autoenc", "diffusion")
    if stages not in valid_stages:
        sys.exit(f"stages must be one of {valid_stages}, got {stages!r}")

    _check_bridge()
    _check_env(radar_root, index_path)
    # radar_root reaches the Rust loader only via DGMR_RADAR_ROOT; export it from the config so the
    # spawned stage subprocesses (which inherit os.environ) pick it up. index_path is forwarded
    # directly as --index_path below — no env round-trip.
    if radar_root is not None:
        os.environ["DGMR_RADAR_ROOT"] = radar_root

    autoenc_dir_p = (SCRIPTS_DIR / autoenc_dir).resolve()
    genforecast_dir_p = (SCRIPTS_DIR / genforecast_dir).resolve()
    do_autoenc = stages in ("both", "autoenc")
    do_genforecast = stages in ("both", "diffusion")

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
        if index_path is not None:
            cmd.append(f"--index_path={index_path}")
        ckpt = _resume_ckpt(autoenc_dir_p) if resume else None
        if ckpt:
            cmd.append(f"--ckpt_path={ckpt}")
        elif resume:
            print(f"NOTE: resume=true but no last.ckpt in {autoenc_dir_p} yet — starting stage 1 fresh.")
        elif list(autoenc_dir_p.glob("*.ckpt")):
            print(f"NOTE: restarting stage 1 from scratch into non-empty {autoenc_dir_p}; old "
                  "checkpoints are kept and could be picked as 'best'. Clear the dir for a clean restart.")
        _run(cmd, stage_name="Stage 1: autoencoder")

    # Stage 2: diffusion (needs the trained autoencoder's weights as input)
    if do_genforecast:
        print("\n=== Extracting autoencoder state_dict ===")
        best_ckpt = _best_autoenc_ckpt(autoenc_dir_p)
        state_dict_path = autoenc_dir_p / "state_dict.pt"
        _extract_state_dict(best_ckpt, state_dict_path)
        print(f"Picked {best_ckpt.name} -> wrote {state_dict_path}")

        # autoenc_weights_fn must be a path resolvable from scripts/.
        autoenc_weights_arg = os.path.relpath(state_dict_path, SCRIPTS_DIR)
        cmd = [
            sys.executable,
            "train_genforecast_rust.py",
            f"--autoenc_weights_fn={autoenc_weights_arg}",
            f"--model_dir={genforecast_dir}",
            f"--height={height}",
            f"--width={width}",
            f"--batch_size={genforecast_batch_size}",
            f"--lr={genforecast_lr}",
            f"--accumulate_grad_batches={genforecast_accumulate_grad_batches}",
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
        if index_path is not None:
            cmd.append(f"--index_path={index_path}")
        ckpt = _resume_ckpt(genforecast_dir_p) if resume else None
        if ckpt:
            cmd.append(f"--ckpt_path={ckpt}")
        elif resume:
            print(f"NOTE: resume=true but no checkpoint in {genforecast_dir_p} yet — starting stage 2 fresh.")
        _run(cmd, stage_name="Stage 2: diffusion model")

    print("\nAll requested stages complete.")


def main(config=None, **kwargs):
    # Default to the shipped config so plain `train_rust.py` works; --config=<path> overrides.
    if config is None and DEFAULT_CONFIG.exists():
        config = str(DEFAULT_CONFIG)
    if config:
        print(f"Loading config: {config}")
    cfg = OmegaConf.load(config) if config else {}
    cfg.update(kwargs)
    run(**cfg)


if __name__ == "__main__":
    Fire(main)
