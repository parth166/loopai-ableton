# Unfold

An Ableton Live 12 extension that drops AI-generated audio clips straight into
your arrangement from a 2-D **exploration map**. Each click on the map fires
off a generation; each dot it leaves behind is an audition-ready clip you can
preview, regenerate, batch-select, and insert as new tracks side-by-side with
your originals.

Two engines are wired in:

| Engine | Backed by | Y-axis ("creativity") | When to use |
|---|---|---|---|
| **Magenta** | `magenta-realtime` (MLX, Apple Silicon) | `temperature` 1.6 → 0.5 | Continuous, audio-conditioned, vibey takes that respect a selected reference clip |
| **Stable Audio 3** | `stable-audio-3` (PyTorch) | `cfg_scale` 0.5 → 5.0 | Text-driven full mixes that obey BPM/key directives lifted from your Live set |

The X-axis ("complexity") rewrites the prompt sliding-scale on the way out —
left-of-center prepends `minimal: <prompt>`, right-of-center appends `…,
layered, polyrhythmic, dense arrangement, …`. Y-axis controls the engine's
sampling knob. Every dot is one prompt × one knob value × one seed.

Highlights:

- 2-D pad UI with ghost cursor, gridlines, per-engine accent palette
- **AI x4** button: drops one dot at the centre of each quadrant and runs
  them sequentially so you get four contrasting takes in one click
- **Shift-click** dots to multi-select, then **Insert N selected** to
  splice them all in as separate new audio tracks
- All inserts go on **brand-new tracks** — your source track is never
  overwritten
- For Stable Audio 3, the project's BPM and key (read live from Ableton's
  `Song.tempo` / `Song.rootNote` / `Song.scaleName`) are auto-appended to the
  prompt as `, 120 BPM, in C Minor; do not change tempo or key`
- Disk-full guard, output pruning (keeps last 50 clips), graceful degrade
  when SA3 dependencies aren't installed

## Architecture

```
Ableton Live 12 (host)
  └─ Extension (TypeScript)              src/extension.ts
        ├─ context-menu actions          AudioTrack / ArrangementSelection
        ├─ reads song.tempo / rootNote / scaleName / signature
        ├─ renders selected region       resources.renderPreFxAudio
        ├─ opens modal dialog            ui.showModalDialog (data: URL)
        └─ on result → song.createAudioTrack() per clip + createAudioClip
                                                ▲
                                                │  JSON.stringify result
   ┌────────────────────────────────────────────┘
   │
Modal dialog (HTML/JS)                   src/dialog.html
   ├─ 2-D pad: clicks → generate, dots = past results
   ├─ engine tabs (Magenta / Stable Audio 3)
   ├─ prompt mutation (X-axis)
   ├─ HTTP POST to backend
   └─ multi-select + AI x4 + insert / batch-insert
                                                ▲
                                                │  /generate, /sa3/generate
                                                │  /health, /sa3/health, /audio/{name}
   ┌────────────────────────────────────────────┘
FastAPI backend                          backend/server.py
   ├─ Magenta engine  (MLX, single-thread executor)
   ├─ SA3 engine      (Torch, lazy load, single-thread executor)
   ├─ disk-full guard + clip pruner
   └─ writes WAVs to /tmp/magenta_track_selection
```

## Repository layout

```
track-selection/
├── src/
│   ├── extension.ts        # Ableton extension entry point
│   ├── dialog.html         # 2-D exploration map UI
│   └── html.d.ts           # Loader shim so .html can be imported
├── backend/
│   ├── server.py           # FastAPI app + Magenta endpoint
│   ├── sa3_engine.py       # Lazy-loaded Stable Audio 3 wrapper
│   ├── start.sh            # Venv discovery + PYTHONPATH wiring + uvicorn
│   └── requirements.txt
├── build.ts                # esbuild bundler (.ts + inlined .html → dist/)
├── manifest.json           # Ableton extension manifest
├── package.json
├── tsconfig.json
└── .env.example            # Copy → .env
```

## Prerequisites

External pieces this repo expects on disk **next to** `track-selection/` (or
inside it where noted). They're not vendored because of size / upstream
licensing — clone them yourself.

| Dep | Where it goes | Why |
|---|---|---|
| **Ableton Live 12** (Beta or release with extensions support) | system app | host for the extension |
| **Ableton Extensions SDK 1.0.0-beta.0** | `../extensions-sdk-1.0.0-beta.0/` (sibling of `track-selection/`) | npm `file:` deps point here |
| **magenta-realtime** | `track-selection/magenta-realtime/` | Magenta engine source (MLX backend) |
| **stable-audio-3** | `track-selection/stable-audio-3/` | Stable Audio 3 engine source |
| **Python 3.12** | system | required by `magenta_rt` and `stable_audio_3` |
| **uv** (recommended) | `brew install uv` | fast venv + dep manager |
| **Node 20+** | system | extension build |

## Setup

Layout you should end up with:

```
hackathon/                                  ← any directory you like
├── extensions-sdk-1.0.0-beta.0/            ← unpack the Ableton SDK tgz here
└── track-selection/                        ← THIS REPO
    ├── magenta-realtime/                   ← clone github.com/magenta/magenta-realtime
    └── stable-audio-3/                     ← clone the Stable Audio 3 source you have access to
```

### 1. Clone & install the extension

```bash
git clone https://github.com/parth166/unfold.git track-selection
cd track-selection
cp .env.example .env       # then edit EXTENSION_HOST_PATH if your Live install is elsewhere
npm install
npm run build              # outputs dist/extension.js
```

### 2. Set up the backend venv

The recommended path is one shared venv at `track-selection/.venv` with
`magenta_rt` (MLX) installed; SA3 deps go on top of it.

```bash
# Magenta side
git clone https://github.com/magenta/magenta-realtime.git
cd magenta-realtime
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[mlx]"
cd ..

# SA3 side (uses the same venv)
cd stable-audio-3
uv pip install -e .         # or uv sync, depending on the upstream
cd ..

# FastAPI deps for the backend
uv pip install -r backend/requirements.txt
```

Don't have Stable Audio 3? That's fine — the backend will detect missing
`torch` / `stable_audio_3` and disable the SA3 tab in the UI (see
`/sa3/health`). Magenta still works.

### 3. Run the backend

```bash
cd backend
./start.sh                  # http://127.0.0.1:8765
./start.sh --dry-run        # sine-wave stub, useful for UI work without MLX
PORT=9000 ./start.sh        # override port
```

`start.sh` auto-discovers a venv with `magenta_rt + mlx` importable, prepends
in-tree `magenta-realtime/` and `stable-audio-3/` to `PYTHONPATH`, installs
FastAPI deps if missing, then runs `server.py`.

Smoke-test the API:

```bash
curl -s http://127.0.0.1:8765/health    | jq
curl -s http://127.0.0.1:8765/sa3/health | jq
```

### 4. Run the extension in Live

```bash
npm start
```

This rebuilds and launches Live (via `extensions-cli run`) with the
extension loaded. In Live, **right-click an audio track or an arrangement
selection** and choose **"Generate with Magenta or Stable Audio 3…"**.

## Using the dialog

1. Pick the engine tab (**Magenta** / **Stable Audio 3**). The SA3 tab
   disables itself if the backend reports it isn't available.
2. Type a base prompt (or pick a preset).
3. Click anywhere on the pad to generate — a busy dot appears and pulses
   while the backend works.
4. Click a finished dot to preview it inline; the readout shows the
   complexity / knob values that produced it.
5. **AI x4** drops 4 dots at the centres of the quadrants
   (`simple+chaotic`, `complex+chaotic`, `simple+strict`, `complex+strict`)
   and runs them in sequence — instant survey of the prompt's space.
6. **Shift-click** dots to add them to a batch.
7. **Insert into New Track** sends the previewed dot back to Live as a
   single new track. **Insert N selected** sends the whole batch — each
   one becomes its own new audio track.

For SA3, the project context auto-rides on the prompt. If your Live set is at
120 BPM in C minor, the actual prompt sent to the model is
`<your prompt>, 120 BPM, in C Minor; do not change tempo or key`. Hover any
finished dot to see the exact prompt that was sent.

## Backend reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Per-engine status (loaded, sample rate, model, error) |
| `/sa3/health` | GET | SA3 only |
| `/generate` | POST | Magenta — `{prompt, seconds, temperature, top_k, cfg_musiccoca, reset_state, audio_path}` |
| `/sa3/generate` | POST | Stable Audio 3 — `{prompt, duration, cfg_scale, steps, seed, negative_prompt}` |
| `/reset` | POST | Reset Magenta's autoregressive state |
| `/audio/{name}` | GET | Serve a generated WAV |

All clips land in `/tmp/magenta_track_selection`. The backend prunes oldest
files when the directory exceeds 50 clips, and refuses to write when free
disk space drops below 10 MB (returns HTTP 507).

## Development

```bash
npm run build       # tsc --noEmit + bundle
npm start           # build + extensions-cli run
```

`tsx build.ts` bundles `src/extension.ts`, inlining `dialog.html` as a
string import (see `src/html.d.ts`). The dialog is loaded as a `data:`
URL with the Ableton selection payload encoded in the URL hash.

## Limitations / known issues

- Apple Silicon only for now (Magenta's MLX backend; SA3 will run on CUDA
  too if you swap the device wiring).
- The X-axis prompt mutation is a small fixed tag pool — feel free to
  swap `SIMPLE_TAGS` / `COMPLEX_TAGS` in `dialog.html` for genre-specific
  pools.
- Magenta ignores the BPM/key constraint string (its conditioning is
  audio + MusicCoCa style embeddings, not free text). If you need
  tempo-locked Magenta output, condition on a reference clip at the
  desired tempo via the arrangement selection.

## License

MIT — see [`LICENSE`](LICENSE).

This project depends on, but does not redistribute,
[`magenta-realtime`](https://github.com/magenta/magenta-realtime) (Apache 2.0)
and Stable Audio 3 (per its own license terms — check what you have access
to before publishing derivatives).
