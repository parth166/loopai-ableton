"""
Stable Audio 3 inference wrapper for the track-selection backend.

Loads ``StableAudioModel.from_pretrained(...)`` once on the first request and
serves subsequent generations from a single dedicated thread (mirrors the
Magenta/MLX backend's thread-discipline so we don't re-bind torch tensors
across worker threads).

Importing torch + stable_audio_3 is **lazy**: the FastAPI server boots even
when these heavy deps aren't installed, and the dialog grays out the SA3
engine via /sa3/health.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

# Default to the small-music checkpoint (CPU-friendly, 433M, 120s max).
DEFAULT_SA3_MODEL = os.environ.get("SA3_MODEL", "small-music")
SA3_SAMPLE_RATE = 44_100  # SAME-Small autoencoder native rate.


def _ensure_sa3_on_path() -> Path | None:
    """Add ``track-selection/stable-audio-3`` to sys.path if it isn't already.

    Returns the path that was added (or already present), or None if no
    in-tree copy is found.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "stable-audio-3",         # track-selection/stable-audio-3
        here.parent.parent / "stable-audio-3",  # hackathon/stable-audio-3
    ]
    for c in candidates:
        if (c / "stable_audio_3" / "__init__.py").exists():
            s = str(c)
            if s not in sys.path:
                sys.path.insert(0, s)
            return c
    return None


class SA3Engine:
    """Thin wrapper around StableAudioModel that mirrors Magenta's Backend."""

    def __init__(self, model_name: str = DEFAULT_SA3_MODEL):
        self._model_name = model_name
        self._model = None        # lazy
        self._import_error: str | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sa3")
        self._load_lock = threading.Lock()
        self._loaded = False

        # Probe import availability without actually loading the model.
        sa3_root = _ensure_sa3_on_path()
        self._sa3_root = sa3_root
        try:
            import torch  # noqa: F401
            import stable_audio_3  # noqa: F401
            self._available = True
        except Exception as e:  # pragma: no cover  (env-specific)
            self._available = False
            self._import_error = (
                f"{type(e).__name__}: {e}. "
                f"Install with: cd {sa3_root or '<stable-audio-3>'} && uv sync"
            )

    # ─── Public state ────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True if torch + stable_audio_3 imports succeed at all."""
        return self._available

    @property
    def loaded(self) -> bool:
        """True once the actual model weights are in memory."""
        return self._loaded

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def sample_rate(self) -> int:
        return SA3_SAMPLE_RATE

    @property
    def import_error(self) -> str | None:
        return self._import_error

    # ─── Loading ────────────────────────────────────────────────────────

    def _load_blocking(self) -> None:
        """Runs on the dedicated executor thread."""
        with self._load_lock:
            if self._loaded:
                return
            from stable_audio_3 import StableAudioModel

            print(
                f"[sa3] loading model={self._model_name!r}…",
                flush=True,
            )
            t0 = time.time()
            self._model = StableAudioModel.from_pretrained(self._model_name)
            self._loaded = True
            print(
                f"[sa3] loaded in {time.time() - t0:.1f}s",
                flush=True,
            )

    async def ensure_loaded_async(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load_blocking)

    # ─── Generation ─────────────────────────────────────────────────────

    def _load_init_audio(self, path: str) -> tuple[int, "object"] | None:
        """Load a WAV from disk and return (sample_rate, torch.Tensor) ready
        for ``model.generate(init_audio=...)``.

        - Resamples to the model's native sample rate so SA3 doesn't have
          to handle rate conversion internally.
        - Coerces to the model's channel count (mono <-> stereo as needed).
        - Returns ``None`` and prints a warning if the file is missing or
          unreadable; SA3 will then fall back to text-only.
        """
        if not path:
            return None
        try:
            import torch
            import torchaudio
        except Exception as e:
            print(f"[sa3] init_audio: torchaudio missing ({e})", flush=True)
            return None

        p = Path(path)
        if not p.is_file():
            print(f"[sa3] init_audio: file not found at {p}", flush=True)
            return None

        try:
            waveform, src_sr = torchaudio.load(str(p))  # [channels, samples]
        except Exception as e:
            print(f"[sa3] init_audio: failed to read {p}: {e}", flush=True)
            return None

        target_sr = SA3_SAMPLE_RATE
        if src_sr != target_sr:
            try:
                waveform = torchaudio.functional.resample(waveform, src_sr, target_sr)
            except Exception as e:
                print(
                    f"[sa3] init_audio: resample {src_sr}->{target_sr} failed ({e}); "
                    f"sending native rate",
                    flush=True,
                )
                target_sr = src_sr

        # Match the model's channel count (default stereo).
        cfg = getattr(self._model, "model_config", {}) or {}
        target_channels = int(cfg.get("io_channels", 2))
        ch = waveform.shape[0]
        if ch != target_channels:
            if target_channels == 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            elif target_channels == 2 and ch == 1:
                waveform = waveform.repeat(2, 1)
            else:
                # Unexpected layout — let SA3 surface the error.
                print(
                    f"[sa3] init_audio: channel mismatch ({ch} -> {target_channels}); "
                    f"forwarding as-is",
                    flush=True,
                )

        # SA3's tests dtype-match to the model. Half-precision models accept
        # float32 init audio (verified upstream), so we don't force a cast here.
        return int(target_sr), waveform

    def _generate_blocking(
        self,
        prompt: str,
        duration: float,
        cfg_scale: float,
        steps: int,
        seed: int,
        negative_prompt: str | None,
        init_audio_path: str | None,
        init_noise_level: float,
    ) -> tuple[np.ndarray, int, float]:
        """Returns (samples_channels_last_float32, sample_rate, elapsed_s)."""
        self._load_blocking()
        assert self._model is not None
        import torch  # noqa: F401  (imported here so it lives on this thread)

        t0 = time.time()
        kwargs: dict = {
            "prompt": prompt,
            "duration": float(duration),
            "cfg_scale": float(cfg_scale),
            "steps": int(steps),
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if seed is not None and seed >= 0:
            kwargs["seed"] = int(seed)

        init = self._load_init_audio(init_audio_path) if init_audio_path else None
        if init is not None:
            kwargs["init_audio"] = init
            # Clamp to the (0,1] range the upstream API expects. 0 would
            # just regurgitate the input, which is never useful here.
            lvl = float(init_noise_level)
            kwargs["init_noise_level"] = max(0.05, min(1.0, lvl))
            sr_in, wav_in = init
            print(
                f"[sa3] init_audio: {Path(init_audio_path).name} "
                f"sr={sr_in} shape={tuple(wav_in.shape)} noise={kwargs['init_noise_level']:.2f}",
                flush=True,
            )

        audio = self._model.generate(**kwargs)
        # `audio` shape is [batch, channels, samples]; we take batch[0],
        # squeeze nothing, then transpose to channels-last for soundfile.
        clip = audio[0].detach().to("cpu").float().numpy()
        if clip.ndim == 3:  # safety net
            clip = clip[0]
        if clip.ndim == 2:
            clip = clip.T  # [channels, time] -> [time, channels]
        elif clip.ndim == 1:
            clip = clip[:, None]
        samples = clip.astype(np.float32, copy=False)
        elapsed = time.time() - t0
        return samples, SA3_SAMPLE_RATE, elapsed

    async def generate_async(
        self,
        prompt: str,
        *,
        duration: float = 8.0,
        cfg_scale: float = 1.0,
        steps: int = 8,
        seed: int = -1,
        negative_prompt: Optional[str] = None,
        init_audio_path: Optional[str] = None,
        init_noise_level: float = 0.8,
    ) -> tuple[np.ndarray, int, float]:
        if not self._available:
            raise RuntimeError(
                f"Stable Audio 3 not importable: {self._import_error}"
            )
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                self._executor,
                self._generate_blocking,
                prompt,
                duration,
                cfg_scale,
                steps,
                seed,
                negative_prompt,
                init_audio_path,
                init_noise_level,
            )
        except Exception as e:
            print(
                f"[sa3] generate failed: {e}\n{traceback.format_exc()}",
                file=sys.stderr,
                flush=True,
            )
            raise

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
