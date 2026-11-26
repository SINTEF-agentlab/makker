// How one tool call reads on its line in the thread: what it acted on (from its
// arguments) and a short note on how it went. Pure functions, unit-tested.

type Args = Record<string, unknown>;

const lineCount = (v: unknown): number => (v ? String(v).split("\n").length : 0);

/** read_file opens its reply with the range it returned ("@@ lines 101-200 of 222 @@").
 *  A file read as an image has no text to count. */
function readNote(content: string): string {
  const m = /^@@ lines (\d+)-(\d+) of (\d+)/.exec(content);
  if (!m) return content.trim() ? unitNote(content, "line") : "";
  const [from, to, total] = [Number(m[1]), Number(m[2]), Number(m[3])];
  return from === 1 && to === total ? `${total} ${total === 1 ? "line" : "lines"}` : `lines ${from}–${to}`;
}

/** grep lists matching lines under each file ("  474: text"), or just the files. */
function grepNote(content: string): string {
  if (/^No (matches|files) found/.test(content.trim())) return "no matches";
  const hits = content.split("\n").filter((l) => /^\s+\d+:/.test(l)).length;
  return hits ? `${hits} ${hits === 1 ? "match" : "matches"}` : unitNote(content, "file");
}

// A short note for the tools whose result (or arguments) says something worth a
// few words. A tool not listed has no note.
const NOTE: Record<string, (content: string, args: Args) => string> = {
  read_file: readNote,
  grep: grepNote,
  glob: (c) => listingNote(c, "file", "files"),
  ls: (c) => listingNote(c),
  write_file: (_c, a) => `${lineCount(a.content)} ${lineCount(a.content) === 1 ? "line" : "lines"}`,
  edit_file: (_c, a) => `+${lineCount(a.new_string)} −${lineCount(a.old_string)}`,
  // "attempt #3 (parent #2) · rmse=0.12 · id=<uuid>": the id is not for reading.
  record_attempt: (c) => c.split(" · id=")[0].split("\n")[0],
};

/** The note for a finished call, or "" when its tool has none. */
export function toolNote(name: string | null | undefined, args: Args | null | undefined, content: string): string {
  const note = name ? NOTE[name] : undefined;
  return note ? note(content, args ?? {}) : "";
}

/** Args worth showing on the line, in priority order. */
const PREVIEW_KEYS = [
  "title", "caption", "description", "code", "command", "pattern", "query", "file_path", "path", "slot",
];

/** A path as its last two segments: enough to tell files apart, short enough for a line. */
export function shortPath(path: string): string {
  const parts = path.replace(/\\/g, "/").split("/").filter(Boolean);
  return parts.length > 2 ? parts.slice(-2).join("/") : path;
}

/** What a tool call is acting on, in one line. */
export function argPreview(args: Args | null | undefined): string {
  if (!args) return "";
  for (const key of PREVIEW_KEYS) {
    const v = args[key];
    if (!v) continue;
    const line = String(v).split("\n").find((l) => l.trim()) ?? "";
    return key === "file_path" || key === "path" ? shortPath(line) : line.trim();
  }
  const first = Object.values(args)[0];
  if (first == null || typeof first === "object") return "";
  return String(first).split("\n")[0];
}

/** ls and glob return a single-line list repr (['a/', 'b/', …]); count the entries
 *  by their commas rather than quotes, so a filename containing an apostrophe
 *  (Python reprs it with double quotes) doesn't skew the count. */
export function listingNote(content: string, singular = "entry", plural = "entries"): string {
  const inner = String(content || "")
    .trim()
    .replace(/^\[/, "")
    .replace(/\]$/, "")
    .trim();
  const n = inner === "" || inner === "No files found" ? 0 : inner.split(",").length;
  return `${n} ${n === 1 ? singular : plural}`;
}

/** Count non-blank lines and label them ("3 matches", "1 file"). */
export function unitNote(content: string, singular: string, plural?: string): string {
  const n = String(content || "").split("\n").filter((l) => l.trim()).length;
  return `${n} ${n === 1 ? singular : plural || singular + "s"}`;
}

export function summarizeArgs(args: Args): string {
  return Object.entries(args)
    .map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`)
    .join("\n");
}

/** Parse a value that may be a dict or a JSON string into an object, else null. */
export function asObject(v: unknown): Record<string, unknown> | null {
  if (v && typeof v === "object" && !Array.isArray(v)) return v as Record<string, unknown>;
  if (typeof v === "string") {
    try {
      const o = JSON.parse(v);
      return o && typeof o === "object" && !Array.isArray(o) ? o : null;
    } catch {
      return null;
    }
  }
  return null;
}
