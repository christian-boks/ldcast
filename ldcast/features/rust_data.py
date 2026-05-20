"""LDCast data module backed by dgmr-py (the Rust .img radar loader).

Bypasses the NetCDF/PatchIndex pipeline: a `torch.utils.data.Dataset` calls
into the PyO3 bridge per sample, and the LightningDataModule emits batches
matching the existing LDCast contracts:

  mode='autoenc'   -> (x, y) where x = [(tensor[B,1,T,H,W], t_rel[B,T])]
                     and y = tensor[B,1,T,H,W] (predictors == targets).
                     Consumed by AutoencoderKL._loss (autoenc.py:48).
  mode='diffusion' -> (pred_batch, target_batch) where
                     pred_batch  = [(past[B,1,T_past,H,W], t_past[B,T_past])]
                     target_batch = future[B,1,T_future,H,W]
                     Consumed by LatentDiffusion.shared_step (diffusion.py:153).
"""
from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import dgmr_py

from ldcast.features.transform import Antialiasing


# ---- Transform: mm/hr → log10 → z-norm → Antialiasing -------------------
# dgmr-rs delivers float32 mm/hr already (Marshall-Palmer applied in Rust),
# so we skip default_rainrate_transform's scale-lookup step. Stats match the
# published LDCast values.
_RAIN_MEAN = -0.051
_RAIN_STD = 0.528
_RAIN_THRESH = 0.1   # mm/hr
_RAIN_FILL = 0.02    # mm/hr below-threshold fill (log10 of this = -1.7)
_MAX_LOAD_RETRIES = 1024


def mmhr_rainrate_transform():
    aa = Antialiasing()

    def transform(raw_mmhr: np.ndarray) -> np.ndarray:
        x = np.where(raw_mmhr >= _RAIN_THRESH, raw_mmhr, _RAIN_FILL).astype(
            np.float32, copy=False
        )
        np.log10(x, out=x)
        x -= _RAIN_MEAN
        x /= _RAIN_STD
        return aa(x)

    return transform


# ---- Dataset ----------------------------------------------------------------
class RustRadarDataset(Dataset):
    """One sample = (past[1,T_past,H,W], future[1,T_future,H,W]) post-transform."""

    def __init__(
        self,
        entries: list,
        past_steps: int,
        future_steps: int,
        height: int,
        width: int,
        full_frame: bool,
        cache_capacity: int,
    ):
        self.entries = entries
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.height = height
        self.width = width
        self.full_frame = full_frame
        self.cache_capacity = cache_capacity
        # Worker-local state, allocated on first __getitem__ inside the worker
        # process (each DataLoader worker gets its own cache after fork).
        self._cache = None
        self._transform = None

    def _ensure_worker_state(self):
        if self._cache is None:
            self._cache = dgmr_py.FrameCache(self.cache_capacity)
            self._transform = mmhr_rainrate_transform()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        self._ensure_worker_state()
        n = len(self.entries)
        # Some entries point at timestamps whose 16-frame window crosses a
        # gap in the .img archive; dgmr-rs's own trainer blacklists those.
        # Skip forward through neighbours until we find a loadable sample.
        last_err = None
        for attempt in range(_MAX_LOAD_RETRIES):
            i = (idx + attempt) % n
            try:
                past, future = dgmr_py.load_sample(
                    self.entries[i],
                    self._cache,
                    self.past_steps,
                    self.future_steps,
                    self.height,
                    self.width,
                    self.full_frame,
                )
                break
            except RuntimeError as e:
                last_err = e
        else:
            raise RuntimeError(
                f"all {_MAX_LOAD_RETRIES} retry attempts failed starting at idx={idx}; last error: {last_err}"
            )
        full = np.concatenate([past, future], axis=1)
        full = self._transform(full).astype(np.float32, copy=True)
        past_t = torch.from_numpy(full[:, : self.past_steps])
        future_t = torch.from_numpy(full[:, self.past_steps :])
        return past_t, future_t


# ---- DataModule -------------------------------------------------------------
def _split_entries(entries: list, valid_frac: float, seed: int):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(entries))
    rng.shuffle(idx)
    n_valid = int(round(len(entries) * valid_frac))
    valid = sorted(idx[:n_valid].tolist())
    train = sorted(idx[n_valid:].tolist())
    return [entries[i] for i in train], [entries[i] for i in valid]


class RustRadarDataModule(pl.LightningDataModule):
    def __init__(
        self,
        index_path: str,
        mode: str = "autoenc",
        past_steps: int = 4,
        future_steps: int = 12,
        height: int = 256,
        width: int = 256,
        full_frame: bool = False,
        batch_size: int = 16,
        num_workers: int = 4,
        cache_capacity: int = 64,
        valid_frac: float = 0.1,
        seed: int = 42,
        use_weighted_sampler: bool = True,
    ):
        super().__init__()
        assert mode in ("autoenc", "diffusion"), f"unknown mode {mode!r}"
        assert height % 32 == 0 and width % 32 == 0, "H,W must be divisible by 32"
        assert (past_steps + future_steps) % 4 == 0, (
            f"past+future={past_steps + future_steps} not divisible by autoenc time ratio 4"
        )
        self.mode = mode
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.height = height
        self.width = width
        self.full_frame = full_frame
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_capacity = cache_capacity
        self.valid_frac = valid_frac
        self.seed = seed
        self.use_weighted_sampler = use_weighted_sampler

        all_entries = dgmr_py.parse_index(index_path)
        if not all_entries:
            raise RuntimeError(f"no entries parsed from {index_path}")
        self.train_entries, self.valid_entries = _split_entries(
            all_entries, valid_frac, seed
        )

    def _ds(self, entries):
        return RustRadarDataset(
            entries,
            self.past_steps,
            self.future_steps,
            self.height,
            self.width,
            self.full_frame,
            self.cache_capacity,
        )

    def setup(self, stage=None):
        self.train_ds = self._ds(self.train_entries)
        self.valid_ds = self._ds(self.valid_entries)

    def _collate(self, samples):
        past = torch.stack([s[0] for s in samples], dim=0)
        future = torch.stack([s[1] for s in samples], dim=0)
        B = past.shape[0]
        if self.mode == "autoenc":
            full = torch.cat([past, future], dim=2)
            t_rel = (
                torch.arange(
                    -self.past_steps + 1,
                    self.future_steps + 1,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .expand(B, -1)
                .contiguous()
            )
            return [(full, t_rel)], full
        t_past = (
            torch.arange(-self.past_steps + 1, 1, dtype=torch.float32)
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )
        return [(past, t_past)], future

    def _loader(self, ds, entries, shuffle):
        sampler = None
        if shuffle and self.use_weighted_sampler:
            weights = torch.tensor(
                [max(e.weight, 0.0) for e in entries], dtype=torch.double
            )
            if weights.sum() > 0:
                sampler = WeightedRandomSampler(
                    weights, num_samples=len(entries), replacement=True
                )
                shuffle = False
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
            pin_memory=True,
            collate_fn=self._collate,
            drop_last=shuffle or sampler is not None,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, self.train_entries, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.valid_ds, self.valid_entries, shuffle=False)
