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
from torch.utils.data import DataLoader, Dataset, Sampler

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
    """One sample = (past[1,T_past,H,W], future[1,T_future,H,W]) post-transform.

    The index is held as four parallel NumPy arrays (ts:i64 unix seconds,
    x:u16, y:u16, weight:f32) instead of a list of dgmr_py.IndexEntry PyO3
    objects -- the new LDCast index is ~300M rows, and PyO3 wrapper overhead
    (~120 B/object) would cost ~36 GB vs ~16 B/row for the compact arrays.
    A fresh IndexEntry is constructed per __getitem__ via dgmr_py.make_entry.
    """

    def __init__(
        self,
        ts: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        past_steps: int,
        future_steps: int,
        height: int,
        width: int,
        full_frame: bool,
        cache_capacity: int,
        max_nocoverage_frac: float = 1.0,
    ):
        assert ts.shape == x.shape == y.shape, "ts/x/y arrays must share shape"
        self.ts = ts
        self.x = x
        self.y = y
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.height = height
        self.width = width
        self.full_frame = full_frame
        self.cache_capacity = cache_capacity
        self.max_nocoverage_frac = max_nocoverage_frac
        # Worker-local state, allocated on first __getitem__ inside the worker
        # process (each DataLoader worker gets its own cache after fork).
        self._cache = None
        self._transform = None

    def _ensure_worker_state(self):
        if self._cache is None:
            self._cache = dgmr_py.FrameCache(self.cache_capacity)
            self._transform = mmhr_rainrate_transform()

    def __len__(self):
        return int(self.ts.shape[0])

    def __getitem__(self, idx):
        self._ensure_worker_state()
        n = len(self)
        # Two reasons to skip an entry and try a neighbour:
        #  (1) its window crosses a gap in the .img archive (load raises;
        #      dgmr-rs's own trainer blacklists these), or
        #  (2) the crop sits largely outside the radar's coverage. Off-radar
        #      pixels arrive as the -1/32 mm/hr sentinel, which the transform
        #      would otherwise turn into fake "dry" 0.02 mm/hr. The original
        #      LDCast pipeline excludes any patch that is not fully finite
        #      (patches.py: `np.isfinite(patch).all()`); we mirror that with a
        #      tolerance for DMI's circular coverage (max_nocoverage_frac).
        last_err = None
        for attempt in range(_MAX_LOAD_RETRIES):
            i = (idx + attempt) % n
            entry = dgmr_py.make_entry(int(self.ts[i]), int(self.x[i]), int(self.y[i]))
            try:
                past, future = dgmr_py.load_sample(
                    entry,
                    self._cache,
                    self.past_steps,
                    self.future_steps,
                    self.height,
                    self.width,
                    self.full_frame,
                )
            except RuntimeError as e:
                last_err = e
                continue
            full = np.concatenate([past, future], axis=1)
            # off-radar is the only source of negatives (sentinel -1/32 mm/hr);
            # skipped in full_frame mode where padding is no-coverage by design.
            if (not self.full_frame) and self.max_nocoverage_frac < 1.0:
                nocov = float(np.mean(full < 0.0))
                if nocov > self.max_nocoverage_frac:
                    last_err = RuntimeError(
                        f"crop at idx={i} is {nocov:.0%} off-radar "
                        f"(> {self.max_nocoverage_frac:.0%} allowed)"
                    )
                    continue
            break
        else:
            raise RuntimeError(
                f"all {_MAX_LOAD_RETRIES} retry attempts failed starting at idx={idx}; last error: {last_err}"
            )
        full = self._transform(full).astype(np.float32, copy=True)
        past_t = torch.from_numpy(full[:, : self.past_steps])
        future_t = torch.from_numpy(full[:, self.past_steps :])
        return past_t, future_t


# ---- Sampler ----------------------------------------------------------------
class LDCastEqualFrequencySampler(Sampler[int]):
    """Uniform across LDCast intensity bins, uniform within each bin.

    Equivalent in distribution to feeding the bin-equal-frequency weights to
    `WeightedRandomSampler`, but does not call `torch.multinomial` (which has
    a hard cap of 2^24 categories) -- so it scales to the ~100M-row LDCast
    index. Also faster: an O(num_samples) draw vs multinomial's O(N) per draw.

    Each unique weight value identifies one LDCast bin (the indexer emits one
    weight per bin; all entries in a bin share its f32 value to 6 sig figs).
    Per epoch: draw `num_samples` bin indices uniformly, then for each bin
    draw row indices uniformly within that bin.
    """

    def __init__(self, weights: np.ndarray, num_samples: int, generator=None):
        super().__init__()
        unique_w, inverse = np.unique(weights, return_inverse=True)
        self.bin_indices = [
            np.flatnonzero(inverse == b).astype(np.int64)
            for b in range(len(unique_w))
        ]
        self.num_bins = len(unique_w)
        assert self.num_bins > 0, "no bins found in weights"
        self.num_samples = int(num_samples)
        self.generator = generator

    def __iter__(self):
        gen = self.generator
        if gen is None:
            gen = torch.Generator()
            gen.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        bin_picks = torch.randint(self.num_bins, (self.num_samples,), generator=gen)
        out = torch.empty(self.num_samples, dtype=torch.int64)
        order = torch.argsort(bin_picks)
        counts = torch.bincount(bin_picks, minlength=self.num_bins).tolist()
        offset = 0
        for b in range(self.num_bins):
            n = counts[b]
            if n == 0:
                continue
            bin_arr = self.bin_indices[b]
            pick_positions = torch.randint(len(bin_arr), (n,), generator=gen)
            out[order[offset : offset + n]] = torch.from_numpy(
                bin_arr[pick_positions.numpy()]
            )
            offset += n
        return iter(out.tolist())

    def __len__(self):
        return self.num_samples


# ---- DataModule -------------------------------------------------------------
def _load_ldcast_index(path: str, valid_frac: float, seed: int,
                       test_frac: float = 0.0, split_mode: str = "random"):
    """Parse + split + bin-0-dedup the LDCast index file.

    Returns three (ts, x, y, w) tuples: (train, val, test). `test` is empty
    unless test_frac > 0.

    split_mode:
      "temporal" -- hold out whole UTC days, so no timestamp is shared across
        train/val/test. A random per-crop split LEAKS: crops at the same
        timestamp (overlapping 128^2 windows) and at +-10 min (highly
        correlated radar) would straddle the boundary, making val/test
        optimistic. The LDCast paper splits by whole days for this reason.
      "random" -- legacy per-crop permutation split (fast, leaky; kept for
        reproducing earlier runs).

    Bin-0 entries (essentially-empty 128x128 crops; rain99 < 0.2 mm/hr) are the
    most populous bucket -- ~65% of the new index -- and are functionally
    identical for training (no rain anywhere in the 12-frame window). We keep
    a single representative bin-0 row in TRAIN and scale its weight by the bin's
    original population so the LDCast bin-equal-frequency invariant
    (sum(weight) per bin == N_total / num_nonempty_bins) is preserved exactly.
    VAL is left as a uniform random sample of the raw index -- val_loss_ema
    just needs a representative slice, not LDCast's sampling distribution.

    Identifying bin 0 from the weight column alone: all entries in the same
    LDCast bin share a single f32 weight value (the indexer writes weights at
    6 sig figs via `{:.6e}`, which round identically to f32). The bin with
    the largest population has the smallest weight by the indexer's formula
    `N_total / (num_nonempty_bins * N_in_bin)`. So `bin 0 == min(weights)`.
    """
    ts, x, y, w = dgmr_py.parse_ldcast_index(path)
    n_total = int(ts.shape[0])
    if n_total == 0:
        raise RuntimeError(f"no entries parsed from {path}")

    rng = np.random.default_rng(seed)
    # In both modes: train_idx is kept ASCENDING (the index is emitted in
    # timestamp order, so ascending train stays ~time-sorted, which the LDCast
    # sampler doesn't care about and linear scans prefer). val_idx/test_idx are
    # SHUFFLED, never sorted: the indexer emits all spatial positions per
    # timestamp consecutively, so a sorted val/test slice clusters into a few
    # adjacent timestamps -- val_dataloader's first ~200 samples once all came
    # from a single 10-min snapshot, badly skewing val_loss_ema and the cached
    # eval cases. Shuffling spreads them across the held-out set.
    if split_mode == "temporal":
        day = ts // 86400                       # whole UTC day per crop
        uniq = np.unique(day)
        rng.shuffle(uniq)
        n_test_d = int(round(uniq.size * test_frac))
        n_val_d = int(round(uniq.size * valid_frac))
        test_days = uniq[:n_test_d]
        val_days = uniq[n_test_d:n_test_d + n_val_d]
        test_mask = np.isin(day, test_days)
        val_mask = np.isin(day, val_days)
        train_idx = np.flatnonzero(~(test_mask | val_mask))
        val_idx = np.flatnonzero(val_mask); rng.shuffle(val_idx)
        test_idx = np.flatnonzero(test_mask); rng.shuffle(test_idx)
        del day, uniq, test_mask, val_mask
    elif split_mode == "random":
        perm = rng.permutation(n_total)
        n_test = int(round(n_total * test_frac))
        n_valid = int(round(n_total * valid_frac))
        test_idx = perm[:n_test]
        val_idx = perm[n_test:n_test + n_valid]
        train_idx = np.sort(perm[n_test + n_valid:])
        del perm
    else:
        raise ValueError(
            f"unknown split_mode {split_mode!r}; use 'temporal' or 'random'"
        )

    train_ts = ts[train_idx]
    train_x = x[train_idx]
    train_y = y[train_idx]
    train_w = w[train_idx]
    val_ts = ts[val_idx]
    val_x = x[val_idx]
    val_y = y[val_idx]
    val_w = w[val_idx]
    test_ts = ts[test_idx]
    test_x = x[test_idx]
    test_y = y[test_idx]
    test_w = w[test_idx]
    del ts, x, y, w, train_idx, val_idx, test_idx

    w_min = train_w.min()
    bin0_mask = train_w == w_min
    bin0_pop = int(bin0_mask.sum())
    if bin0_pop > 1:
        keep = ~bin0_mask
        rep = int(np.flatnonzero(bin0_mask)[0])
        train_ts = np.concatenate([train_ts[keep], train_ts[rep : rep + 1]])
        train_x = np.concatenate([train_x[keep], train_x[rep : rep + 1]])
        train_y = np.concatenate([train_y[keep], train_y[rep : rep + 1]])
        train_w = np.concatenate(
            [train_w[keep], np.array([w_min * bin0_pop], dtype=train_w.dtype)]
        )

    return (
        (train_ts, train_x, train_y, train_w),
        (val_ts, val_x, val_y, val_w),
        (test_ts, test_x, test_y, test_w),
    )


class RustRadarDataModule(pl.LightningDataModule):
    def __init__(
        self,
        index_path: str,
        mode: str = "autoenc",
        past_steps: int = 4,
        future_steps: int = 8,
        height: int = 256,
        width: int = 256,
        full_frame: bool = False,
        batch_size: int = 16,
        num_workers: int = 4,
        cache_capacity: int = 64,
        valid_frac: float = 0.1,
        test_frac: float = 0.0,
        split_mode: str = "random",
        seed: int = 42,
        use_weighted_sampler: bool = False,
        max_nocoverage_frac: float = 0.05,
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
        self.test_frac = test_frac
        self.split_mode = split_mode
        self.seed = seed
        self.use_weighted_sampler = use_weighted_sampler
        self.max_nocoverage_frac = max_nocoverage_frac

        (
            (self.train_ts, self.train_x, self.train_y, self.train_w),
            (self.val_ts, self.val_x, self.val_y, self.val_w),
            (self.test_ts, self.test_x, self.test_y, self.test_w),
        ) = _load_ldcast_index(index_path, valid_frac, seed, test_frac, split_mode)

    def _ds(self, ts, x, y):
        return RustRadarDataset(
            ts,
            x,
            y,
            self.past_steps,
            self.future_steps,
            self.height,
            self.width,
            self.full_frame,
            self.cache_capacity,
            self.max_nocoverage_frac,
        )

    def setup(self, stage=None):
        self.train_ds = self._ds(self.train_ts, self.train_x, self.train_y)
        self.valid_ds = self._ds(self.val_ts, self.val_x, self.val_y)
        self.test_ds = (self._ds(self.test_ts, self.test_x, self.test_y)
                        if self.test_ts.size else None)

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

    def _loader(self, ds, weights, shuffle):
        sampler = None
        # The LDCast index file (`index_ldcast_*.txt`, produced by
        # batch_hdf5_to_img/src/ld_indexer.rs) gives each crop a
        # bin-equal-frequency importance weight
        # `N_total / (num_nonempty_bins * N_in_bin)`. Sampling proportional
        # to weight makes every LDCast intensity bin equally likely per draw,
        # reproducing the original `sampling.EqualFrequencySampler`
        # (`scripts/train_nowcaster.py:122`) -- heavy rain ends up oversampled
        # ~10-100x vs the natural distribution. Required for LDCast-faithful
        # training; without it the index's raw row frequency wins (which is
        # heavily skewed toward empty crops) and heavy-rain skill plateaus.
        #
        # We use a custom LDCastEqualFrequencySampler (uniform across bins +
        # uniform within bin) rather than torch's WeightedRandomSampler because
        # the latter calls torch.multinomial, which is capped at 2^24 ~ 16.7M
        # categories -- the train arrays have ~90M rows post-dedup. Our sampler
        # is equivalent in distribution and not multinomial-bound.
        if shuffle and self.use_weighted_sampler and weights is not None:
            # Lightning consumes at most limit_train_batches * batch_size per
            # epoch (typically ~32k). 1M is a comfortable upper bound and
            # never exhausted in practice.
            n_eff = min(int(weights.size), 1_000_000)
            sampler = LDCastEqualFrequencySampler(weights, num_samples=n_eff)
            shuffle = False
        # Python 3.14 changed the default multiprocessing start method on Linux
        # from 'fork' to 'forkserver', which pickles the dataset to hand it to
        # each worker. RustRadarDataset's NumPy arrays pickle fine, but each
        # worker still constructs a dgmr_py.FrameCache lazily after fork (see
        # _ensure_worker_state), so we keep `fork` to preserve the pre-3.14
        # cache lifetime semantics. Workers do CPU-only work, so forking after
        # CUDA init in the parent is safe.
        mp_context = "fork" if self.num_workers > 0 else None
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
            multiprocessing_context=mp_context,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, self.train_w, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.valid_ds, None, shuffle=False)

    def test_dataloader(self):
        if getattr(self, "test_ds", None) is None:
            raise RuntimeError(
                "no test split; set split_mode='temporal' and test_frac>0"
            )
        return self._loader(self.test_ds, None, shuffle=False)
