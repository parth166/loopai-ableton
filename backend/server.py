#!/usr/bin/env python3
"""
Magenta-RT FastAPI server for the Ableton track-selection extension.

Loads MagentaRT2Mlxfn once and exposes:

  POST /generate { prompt, seconds, temperature, top_k, cfg_musiccoca }
       -> { wav_path, sample_rate, seconds, duration_ms }

  POST /reset    -> clears the autoregressive state

  GET  /health   -> { ready: bool, sample_rate: int }

Generated WAVs are written to OUT_DIR (default /tmp/magenta_track_selection).
The extension then imports that path via context.resources.importIntoProject.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from sa3_engine import SA3Engine, DEFAULT_SA3_MODEL

FPS = 25
DEFAULT_SECONDS = 4.0
DEFAULT_TEMPERATURE = 1.1
DEFAULT_TOP_K = 40
DEFAULT_CFG_MUSICCOCA = 4.0

# Stable Audio 3 defaults
DEFAULT_SA3_DURATION = 8.0
DEFAULT_SA3_CFG_SCALE = 1.0
DEFAULT_SA3_STEPS = 8

OUT_DIR = Path(os.environ.get("MAGENTA_OUT_DIR", "/tmp/magenta_track_selection"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    seconds: float = Field(DEFAULT_SECONDS, gt=0.1, le=30.0)
    temperature: float = Field(DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=512)
    cfg_musiccoca: float = Field(DEFAULT_CFG_MUSICCOCA, ge=0.0, le=10.0)
    reset_state: bool = False
    # When set, Magenta conditions on this WAV file instead of (or blended
    # with) the text prompt.  Path must exist on the local filesystem.
    audio_path: Optional[str] = None


class GenerateResponse(BaseModel):
    wav_path: str
    sample_rate: int
    seconds: float
    duration_ms: int
    elapsed_s: float
    realtime_ratio: float


class SA3GenerateRequest(BaseModel):
    """Stable Audio 3 generation request.

    The 2D-pad dialog maps:
      x (simple → complex) → keyword mutation on `prompt` (done client-side)
      y (high → low creativity) → `cfg_scale` (lower = more creative)
    """

    prompt: str = Field(..., min_length=1)
    negative_prompt: Optional[str] = None
    duration: float = Field(DEFAULT_SA3_DURATION, gt=0.5, le=120.0)
    cfg_scale: float = Field(DEFAULT_SA3_CFG_SCALE, ge=0.0, le=15.0)
    steps: int = Field(DEFAULT_SA3_STEPS, ge=1, le=200)
    seed: int = Field(-1, ge=-1)


class SA3GenerateResponse(GenerateResponse):
    cfg_scale: float
    steps: int
    seed: int


class Backend:
    """Wraps the MagentaRT2Mlxfn model.

    MLX GPU streams are thread-local: the stream is created on the thread
    that first instantiates a model. If a different thread later runs MLX
    ops, you get `There is no Stream(gpu, 1) in current thread`. FastAPI
    runs sync endpoints on a thread-pool worker, so every request would
    land on a different thread.

    To guarantee thread affinity we pin both the model load and every
    generate/reset call to a single dedicated worker thread via a
    ThreadPoolExecutor(max_workers=1).
    """

    def __init__(self, model_name: str, dry_run: bool):
        self.model_name = model_name
        self.dry_run = dry_run
        self._mrt = None
        self._sample_rate = 48_000
        self._state = None
        # All MLX work happens on this single thread.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mrt"
        )
        # Serialise concurrent /generate calls (the model isn't reentrant).
        self._lock = Lock()

        if not dry_run:
            # Run the load on the dedicated thread so the GPU stream is
            # registered there (and only there).
            self._executor.submit(self._load_model_blocking).result()

    def _load_model_blocking(self):
        from magenta_rt import MagentaRT2Mlxfn

        print(f"[backend] loading {self.model_name} ...", flush=True)
        t0 = time.time()
        self._mrt = MagentaRT2Mlxfn(
            size=self.model_name,
            temperature=DEFAULT_TEMPERATURE,
            top_k=DEFAULT_TOP_K,
            cfg_musiccoca=DEFAULT_CFG_MUSICCOCA,
        )
        self._sample_rate = int(self._mrt._sample_rate)
        print(
            f"[backend] {self.model_name} ready in {time.time() - t0:.1f}s "
            f"(sr={self._sample_rate})",
            flush=True,
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def ready(self) -> bool:
        return self.dry_run or self._mrt is not None

    def reset(self):
        # Touch state on the MLX thread to avoid races with in-flight gen.
        self._executor.submit(self._reset_blocking).result()

    def _reset_blocking(self):
        with self._lock:
            self._state = None

    def _build_style_embedding(self, req: GenerateRequest):
        """Return a style embedding from audio (preferred) or text.

        If audio_path is given and the file exists we load it as a Waveform
        and pass it directly to embed_style.  The text prompt is then used
        as a secondary / override description that can further steer the
        model by blending the two embeddings (average).  If audio_path is
        absent or unreadable we fall back to text-only.
        """
        from magenta_rt import audio as mrt_audio

        audio_embedding = None
        if req.audio_path:
            audio_path = Path(req.audio_path)
            if audio_path.exists():
                try:
                    waveform = mrt_audio.Waveform.from_file(str(audio_path))
                    audio_embedding = self._mrt.embed_style(
                        waveform, pool_across_time=True, use_mapper=True
                    )
                    print(
                        f"[backend] audio conditioning: {audio_path.name} "
                        f"({waveform.samples.shape[0] / waveform.sample_rate:.2f}s)",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[backend] audio embed failed ({e}), falling back to text",
                        flush=True,
                    )
            else:
                print(
                    f"[backend] audio_path not found: {req.audio_path}",
                    flush=True,
                )

        text_embedding = self._mrt.embed_style(
            req.prompt, use_mapper=True
        )

        if audio_embedding is not None:
            # Blend: 60 % audio + 40 % text — keeps tonal/rhythmic character
            # of the original loop while respecting the text direction.
            # StyleEmbedding is a TypeAlias for np.ndarray, so just return
            # an ndarray directly (do NOT wrap with `np.ndarray(...)`, which
            # would treat the input as a *shape* spec, not data).
            audio_arr = np.asarray(audio_embedding, dtype=np.float32)
            text_arr = np.asarray(text_embedding, dtype=np.float32)
            blended = 0.6 * audio_arr + 0.4 * text_arr
            # Re-normalise to unit sphere (MusicCoCa embeddings are
            # L2-normalised by convention).
            norm = float(np.linalg.norm(blended))
            if norm > 1e-8:
                blended = blended / norm
            print(
                f"[backend] blended embedding: shape={blended.shape} "
                f"(audio.shape={audio_arr.shape}, text.shape={text_arr.shape})",
                flush=True,
            )
            return blended.astype(np.float32)

        return text_embedding

    def _generate_blocking(
        self, req: GenerateRequest
    ) -> tuple[np.ndarray, int, float]:
        with self._lock:
            if req.reset_state:
                self._state = None

            t0 = time.time()
            if self.dry_run:
                samples = self._dry_run_generate(req)
                sr = self._sample_rate
            else:
                # Re-apply sampling params at every call so the dialog
                # sliders take effect without restarting the server.
                self._mrt.temperature = req.temperature
                self._mrt.top_k = req.top_k
                self._mrt.cfg_musiccoca = req.cfg_musiccoca

                embedding = self._build_style_embedding(req)
                frames = max(1, int(req.seconds * FPS))
                wav, self._state = self._mrt.generate(
                    style=embedding, frames=frames, state=self._state,
                )
                samples = wav.samples
                sr = int(wav.sample_rate)

            return samples, sr, time.time() - t0

    async def generate_async(
        self, req: GenerateRequest
    ) -> tuple[np.ndarray, int, float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._generate_blocking, req
        )

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _dry_run_generate(self, req: GenerateRequest) -> np.ndarray:
        """Sine + kick stub so the dialog is testable without MLX."""
        n = int(req.seconds * self._sample_rate)
        bpm = 120
        for tok in req.prompt.replace(",", " ").split():
            if tok.isdigit() and 60 <= int(tok) <= 200:
                bpm = int(tok)
                break
        t = np.arange(n) / self._sample_rate
        sine = 0.2 * np.sin(2 * np.pi * (110 + 0.5 * bpm) * t)
        beat = 60.0 / bpm
        kicks = np.zeros(n, dtype=np.float32)
        env_n = int(0.12 * self._sample_rate)
        env = np.exp(-np.arange(env_n) / (0.02 * self._sample_rate))
        for k in range(int(req.seconds / beat) + 1):
            i0 = int(k * beat * self._sample_rate)
            kick = (
                0.6
                * np.sin(2 * np.pi * 60 * np.arange(env_n) / self._sample_rate)
                * env
            )
            j = min(i0 + env_n, n)
            kicks[i0:j] += kick[: j - i0]
        mono = (sine + kicks).astype(np.float32)
        return np.stack([mono, mono], axis=-1)


class DiskFullError(RuntimeError):
    """Raised when we can't write a generated WAV because the FS is full
    or otherwise refuses the write. Server turns this into HTTP 507."""


# Keep at most this many generated clips in OUT_DIR. We delete the oldest
# files (by mtime) once the directory grows past the cap. This guards
# against unbounded /tmp growth across long sessions.
MAX_KEPT_CLIPS = int(os.environ.get("MAX_KEPT_CLIPS", "50"))


def _prune_out_dir(out_dir: Path, max_keep: int = MAX_KEPT_CLIPS) -> int:
    """Trim OUT_DIR to ``max_keep`` most-recent clips. Returns deleted count."""
    try:
        entries = [
            p for p in out_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".m4a", ".ogg"}
        ]
    except Exception:
        return 0
    if len(entries) <= max_keep:
        return 0
    entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    deleted = 0
    for stale in entries[max_keep:]:
        try:
            stale.unlink()
            deleted += 1
        except Exception:
            pass
    return deleted


def _disk_free_bytes(path: Path) -> int | None:
    try:
        st = os.statvfs(str(path))
        return int(st.f_bavail) * int(st.f_frsize)
    except Exception:
        return None


def write_wav(samples: np.ndarray, sample_rate: int, path: Path) -> None:
    """Write a 1- or 2-D float numpy array to ``path`` as 32-bit float WAV.

    Resilient to transient FS errors: retries once after a short delay
    (Metal shader cache + tempfile contention on a near-full disk can
    momentarily wedge `soundfile`). Gives up cleanly with `DiskFullError`
    if the second attempt also fails, so the caller can map this to a
    user-friendly HTTP 507 instead of a stack trace.
    """
    import soundfile as sf

    if samples.ndim == 1:
        samples = samples[:, None]
    payload = samples.astype(np.float32, copy=False)

    # Pre-flight: if we have <~10MB free, the write almost certainly fails.
    free = _disk_free_bytes(path.parent)
    if free is not None and free < 10 * 1024 * 1024:
        free_mib = free / (1024 * 1024)
        raise DiskFullError(
            f"Only {free_mib:.0f} MiB free on the volume hosting {path.parent}. "
            f"Free up disk space (try emptying ~/.cache, "
            f"~/Library/Caches, or the Trash) and try again."
        )

    last_err: Exception | None = None
    for attempt in range(2):
        try:
            sf.write(str(path), payload, sample_rate)
            return
        except (OSError, sf.LibsndfileError) as e:  # type: ignore[attr-defined]
            last_err = e
            if attempt == 0:
                # Best-effort cleanup of any half-written file + brief backoff.
                try:
                    if path.exists():
                        path.unlink()
                except Exception:
                    pass
                time.sleep(0.4)
                continue
            break
    raise DiskFullError(
        f"Failed to write {path.name}: {last_err}. "
        f"This usually means the volume is full — check `df -h /`."
    )


# ─── App factory ──────────────────────────────────────────────────────────────


def create_app(
    model_name: str,
    dry_run: bool,
    sa3_model: str = DEFAULT_SA3_MODEL,
) -> FastAPI:
    backend = Backend(model_name, dry_run)
    sa3 = SA3Engine(sa3_model)
    if sa3.available:
        print(
            f"[sa3] importable; model={sa3.model_name!r} (load deferred to first request)",
            flush=True,
        )
    else:
        print(
            f"[sa3] disabled — {sa3.import_error}",
            flush=True,
        )

    app = FastAPI(title="Magenta-RT + Stable Audio 3 Track Selection Backend")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {
            "ready": backend.ready,
            "sample_rate": backend.sample_rate,
            "model": "dry-run" if backend.dry_run else backend.model_name,
            "engines": {
                "magenta": {
                    "ready": backend.ready,
                    "sample_rate": backend.sample_rate,
                    "model": "dry-run" if backend.dry_run else backend.model_name,
                },
                "sa3": {
                    "available": sa3.available,
                    "loaded": sa3.loaded,
                    "model": sa3.model_name,
                    "sample_rate": sa3.sample_rate,
                    "error": sa3.import_error,
                },
            },
        }

    @app.post("/reset")
    def reset():
        backend.reset()
        return {"ok": True}

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        if not backend.ready:
            raise HTTPException(503, "Model not loaded")

        try:
            samples, sr, elapsed = await backend.generate_async(req)
        except Exception as e:
            print(f"[backend] generate failed: {e}", file=sys.stderr, flush=True)
            raise HTTPException(500, f"generate failed: {e}") from e

        wav_path = OUT_DIR / f"clip_{uuid.uuid4().hex[:8]}_{int(time.time())}.wav"
        try:
            write_wav(samples, sr, wav_path)
        except DiskFullError as e:
            print(f"[backend] write_wav blocked: {e}", file=sys.stderr, flush=True)
            raise HTTPException(507, str(e)) from e
        _prune_out_dir(OUT_DIR)

        duration_ms = int(samples.shape[0] / sr * 1000)
        rt_ratio = req.seconds / max(elapsed, 1e-6)
        print(
            f"[backend] generated {wav_path.name}  "
            f"prompt={req.prompt!r}  {duration_ms}ms  "
            f"elapsed={elapsed:.2f}s  ({rt_ratio:.2f}x rt)",
            flush=True,
        )
        return GenerateResponse(
            wav_path=str(wav_path),
            sample_rate=sr,
            seconds=req.seconds,
            duration_ms=duration_ms,
            elapsed_s=round(elapsed, 3),
            realtime_ratio=round(rt_ratio, 3),
        )

    @app.get("/audio/{name}")
    def audio(name: str):
        # Serve generated WAV back to the dialog for in-browser preview.
        path = OUT_DIR / name
        if not path.exists() or path.parent.resolve() != OUT_DIR.resolve():
            raise HTTPException(404, "not found")
        return FileResponse(path, media_type="audio/wav")

    # ─── Stable Audio 3 endpoints ────────────────────────────────────────

    @app.get("/sa3/health")
    def sa3_health():
        return {
            "available": sa3.available,
            "loaded": sa3.loaded,
            "model": sa3.model_name,
            "sample_rate": sa3.sample_rate,
            "error": sa3.import_error,
        }

    @app.post("/sa3/generate", response_model=SA3GenerateResponse)
    async def sa3_generate(req: SA3GenerateRequest):
        if not sa3.available:
            raise HTTPException(
                503,
                "Stable Audio 3 not available: " + (sa3.import_error or "unknown error"),
            )

        try:
            samples, sr, elapsed = await sa3.generate_async(
                req.prompt,
                duration=req.duration,
                cfg_scale=req.cfg_scale,
                steps=req.steps,
                seed=req.seed,
                negative_prompt=req.negative_prompt,
            )
        except Exception as e:
            raise HTTPException(500, f"sa3 generate failed: {e}") from e

        wav_path = OUT_DIR / f"sa3_{uuid.uuid4().hex[:8]}_{int(time.time())}.wav"
        try:
            write_wav(samples, sr, wav_path)
        except DiskFullError as e:
            print(f"[sa3] write_wav blocked: {e}", file=sys.stderr, flush=True)
            raise HTTPException(507, str(e)) from e
        _prune_out_dir(OUT_DIR)

        duration_ms = int(samples.shape[0] / sr * 1000)
        rt_ratio = req.duration / max(elapsed, 1e-6)
        print(
            f"[sa3] generated {wav_path.name}  prompt={req.prompt!r}  "
            f"cfg={req.cfg_scale} steps={req.steps} "
            f"{duration_ms}ms  elapsed={elapsed:.2f}s ({rt_ratio:.2f}x rt)",
            flush=True,
        )
        return SA3GenerateResponse(
            wav_path=str(wav_path),
            sample_rate=sr,
            seconds=req.duration,
            duration_ms=duration_ms,
            elapsed_s=round(elapsed, 3),
            realtime_ratio=round(rt_ratio, 3),
            cfg_scale=req.cfg_scale,
            steps=req.steps,
            seed=req.seed,
        )

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="mrt2_small")
    parser.add_argument(
        "--sa3-model",
        default=DEFAULT_SA3_MODEL,
        help="Stable Audio 3 checkpoint name (small-music, small-sfx, medium, ...).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip MLX model load, use sine+kick stub (for plumbing tests).",
    )
    args = parser.parse_args()

    import uvicorn

    app = create_app(args.model, args.dry_run, sa3_model=args.sa3_model)
    print(
        f"[backend] serving on http://{args.host}:{args.port}  "
        f"(out_dir={OUT_DIR})",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
