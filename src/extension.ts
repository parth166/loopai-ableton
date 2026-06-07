import {
  initialize,
  AudioTrack,
  type ActivationContext,
  type ArrangementSelection,
  type Handle,
} from "@ableton-extensions/sdk";

import dialogHtml from "./dialog.html";

const DIALOG_WIDTH = 820;
const DIALOG_HEIGHT = 760;

const COMMAND_GENERATE_FROM_SELECTION = "trackSelection.generate";
const COMMAND_GENERATE_FROM_TRACK = "trackSelection.generateOnTrack";

type Engine = "magenta" | "sa3";

interface InsertItem {
  wavPath: string;
  engine?: Engine | undefined;
  label?: string | undefined;
}

interface DialogResult {
  action: "insert" | "insert_multiple" | "cancel";
  engine?: Engine;
  wavPath?: string;          // single insert
  wavPaths?: string[];       // legacy multi
  items?: InsertItem[];      // preferred multi (carries per-clip engine + label)
  params?: Record<string, unknown>;
}

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

function readSongProps(api: ReturnType<typeof initialize<"1.0.0">>) {
  const song = api.application.song;
  // Most of these are exposed as `number`/`string` in the SDK but the runtime
  // sometimes hands back BigInts or undefined for unset scales. Be defensive.
  const tempo = Number(song.tempo);
  const sigNum = Number((song as unknown as { signatureNumerator?: number }).signatureNumerator ?? 4);
  const sigDen = Number((song as unknown as { signatureDenominator?: number }).signatureDenominator ?? 4);
  let rootNote: number | null = null;
  let scaleName: string | null = null;
  let scaleMode: boolean | null = null;
  try {
    rootNote = Number(song.rootNote);
    scaleName = String(song.scaleName ?? "");
    scaleMode = Boolean(song.scaleMode);
  } catch (_) {
    // older Live builds may not expose scale info — that's fine.
  }
  const rootName =
    typeof rootNote === "number" && rootNote >= 0 && rootNote < NOTE_NAMES.length
      ? NOTE_NAMES[rootNote]
      : null;
  return {
    bpm: Number.isFinite(tempo) ? tempo : null,
    timeSignature:
      Number.isFinite(sigNum) && Number.isFinite(sigDen) ? `${sigNum}/${sigDen}` : null,
    rootNote,
    rootName,
    scaleName,
    scaleMode,
  };
}

async function insertOnNewTrack(
  api: ReturnType<typeof initialize<"1.0.0">>,
  wavPath: string,
  startTime: number,
  duration: number,
  trackName: string,
): Promise<void> {
  const newTrack = await api.application.song.createAudioTrack();
  try {
    newTrack.name = trackName.slice(0, 64);
  } catch (err) {
    console.warn(`trackSelection: could not set new track name: ${err}`);
  }
  const imported = await api.resources.importIntoProject(wavPath);
  await newTrack.createAudioClip({
    filePath: imported,
    startTime,
    duration,
    isWarped: true,
  });
  console.log(`trackSelection: created new track "${newTrack.name}" with ${wavPath}`);
}

async function openDialogAndMaybeInsert(
  api: ReturnType<typeof initialize<"1.0.0">>,
  track: AudioTrack<"1.0.0">,
  rawStart: number | bigint,
  rawDuration: number | bigint,
): Promise<void> {
  // Live's SDK types these as `number`, but at runtime they are `bigint`
  // values (beat positions). Coerce to plain `number` everywhere we touch
  // them — JSON.stringify, native bridge calls, and arithmetic all need it.
  const startTime = Number(rawStart);
  const duration = Number(rawDuration);

  if (!(duration > 0)) {
    console.warn(
      "trackSelection: empty time selection — draw a range in the arrangement first.",
    );
    return;
  }

  console.log(
    `trackSelection: handler entered — track="${track.name}" ` +
      `start=${startTime} duration=${duration}`,
  );

  // Render the selected region to a WAV so Magenta can condition on it.
  let renderedAudioPath: string | null = null;
  try {
    console.log(
      `trackSelection: rendering "${track.name}" [${startTime}..${startTime + duration}] for audio conditioning…`,
    );
    renderedAudioPath = await api.resources.renderPreFxAudio(
      track,
      startTime,
      startTime + duration,
    );
    console.log(`trackSelection: rendered → ${renderedAudioPath}`);
  } catch (err) {
    console.warn(
      `trackSelection: renderPreFxAudio failed (${err}), falling back to text-only`,
    );
  }

  const songProps = readSongProps(api);
  console.log(
    `trackSelection: song props — bpm=${songProps.bpm} key=${songProps.rootName} ` +
      `scale=${songProps.scaleName} (mode=${songProps.scaleMode}) sig=${songProps.timeSignature}`,
  );

  const selectionPayload = {
    trackName: track.name,
    start: startTime,
    end: startTime + duration,
    renderedAudioPath,
    songProps,
  };

  const html = dialogHtml as unknown as string;
  const url =
    `data:text/html,${encodeURIComponent(html)}` +
    `#${encodeURIComponent(JSON.stringify(selectionPayload))}`;

  console.log(
    `trackSelection: opening dialog for "${track.name}" ` +
      `[${startTime}..${startTime + duration}] (Δ ${duration})`,
  );

  const raw = await api.ui.showModalDialog(url, DIALOG_WIDTH, DIALOG_HEIGHT);

  let result: DialogResult;
  try {
    result = JSON.parse(raw) as DialogResult;
  } catch {
    console.warn("trackSelection: dialog returned non-JSON:", raw);
    return;
  }

  // Normalize all "insert*" result shapes into a flat list of items.
  const items: InsertItem[] = [];
  if (result.action === "insert" && result.wavPath) {
    items.push({
      wavPath: result.wavPath,
      engine: result.engine,
    });
  } else if (result.action === "insert_multiple") {
    if (Array.isArray(result.items)) items.push(...result.items);
    else if (Array.isArray(result.wavPaths)) {
      for (const p of result.wavPaths) items.push({ wavPath: p, engine: result.engine });
    }
  }

  if (!items.length) {
    console.log("trackSelection: cancelled");
    return;
  }

  console.log(
    `trackSelection: dialog returned ${items.length} item(s) — ` +
      `creating new audio track(s) next to "${track.name}"`,
  );

  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    if (!it) continue;
    const engine = it.engine ?? result.engine ?? "magenta";
    const enginePrefix = engine === "sa3" ? "SA3" : "Magenta";
    const label =
      it.label && it.label.trim().length
        ? it.label.trim()
        : `${enginePrefix} · ${track.name}` + (items.length > 1 ? ` (${i + 1})` : "");
    try {
      await insertOnNewTrack(api, it.wavPath, startTime, duration, label);
    } catch (err) {
      console.error(`trackSelection: failed to insert ${it.wavPath}: ${err}`);
    }
  }

  console.log(`trackSelection: ${items.length} clip(s) inserted on new track(s)`);
}

export function activate(activation: ActivationContext) {
  const api = initialize(activation, "1.0.0");

  console.log("trackSelection: activating, registering context menu actions...");

  api.ui.registerContextMenuAction(
    "AudioTrack.ArrangementSelection",
    "Generate with Magenta or Stable Audio 3…",
    COMMAND_GENERATE_FROM_SELECTION,
  );

  api.ui.registerContextMenuAction(
    "AudioTrack",
    "Generate with Magenta or Stable Audio 3…",
    COMMAND_GENERATE_FROM_TRACK,
  );

  console.log(
    "trackSelection: registered on AudioTrack.ArrangementSelection and AudioTrack",
  );

  api.commands.registerCommand(
    COMMAND_GENERATE_FROM_SELECTION,
    async (arg: unknown) => {
      try {
        const selection = arg as ArrangementSelection;
        const firstHandle = selection.selected_lanes[0];
        if (!firstHandle) {
          console.warn(`${COMMAND_GENERATE_FROM_SELECTION}: no selected lane`);
          return;
        }
        const track = api.getObjectFromHandle(firstHandle, AudioTrack);
        if (!(track instanceof AudioTrack)) {
          console.warn(`${COMMAND_GENERATE_FROM_SELECTION}: not an AudioTrack`);
          return;
        }
        await openDialogAndMaybeInsert(
          api,
          track,
          selection.time_selection_start,
          selection.time_selection_end - selection.time_selection_start,
        );
      } catch (err) {
        console.error(`${COMMAND_GENERATE_FROM_SELECTION} error:`, err);
      }
    },
  );

  api.commands.registerCommand(
    COMMAND_GENERATE_FROM_TRACK,
    async (arg: unknown) => {
      try {
        const track = api.getObjectFromHandle(arg as Handle, AudioTrack);
        if (!(track instanceof AudioTrack)) {
          console.warn(`${COMMAND_GENERATE_FROM_TRACK}: not an AudioTrack`);
          return;
        }
        await openDialogAndMaybeInsert(api, track, 0, 4);
      } catch (err) {
        console.error(`${COMMAND_GENERATE_FROM_TRACK} error:`, err);
      }
    },
  );
}
