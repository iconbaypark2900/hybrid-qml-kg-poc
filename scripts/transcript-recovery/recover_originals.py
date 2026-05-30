"""Recover the user's ORIGINAL hetqml-next files.

Strategy:
  - For each file path, find every Read tool result across all Claude Code
    transcripts in the project.
  - Pick the EARLIEST Read by transcript timestamp — that's closest to the
    original on-disk content, before Claude modified anything.
  - Files that have only Writes (i.e. new files Claude created) are SKIPPED,
    because those weren't in the user's original folder.

Output: ./recovered_originals/ mirroring the relative path under hetqml-next/.
A `MANIFEST.txt` lists each file with its source transcript + line number.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "recovered_originals"
PROJECTS_DIR = Path(
    "/home/roc/.cursor/projects/"
    "home-roc-quantumGlobalGroup-hybrid-qml-kg-poc/agent-transcripts"
)


def extract_rel(fp: str) -> str | None:
    if not fp:
        return None
    norm = fp.replace("\\", "/")
    if "hetqml-next" not in norm.lower():
        return None
    idx = norm.lower().find("hetqml-next/")
    if idx == -1:
        return None
    rel = norm[idx + len("hetqml-next/"):].split("?")[0].split("#")[0].strip()
    return rel or None


_NUM_TAB = re.compile(r"^\s*\d+\t(.*)$")


def strip_n(s: str) -> str:
    out = []
    for line in s.splitlines():
        m = _NUM_TAB.match(line)
        out.append(m.group(1) if m else line)
    return "\n".join(out)


def main() -> int:
    # Per-path: list of (timestamp, content, transcript_name) from Reads
    reads: dict[str, list[tuple[str, str, str]]] = {}
    transcripts = sorted(PROJECTS_DIR.rglob("*.jsonl"))

    for path in transcripts:
        tu_paths: dict[str, str] = {}
        tu_ops: dict[str, str] = {}
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = rec.get("timestamp", "")
                msg = rec.get("message")
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for blk in content:
                    if not isinstance(blk, dict):
                        continue
                    bt = blk.get("type")
                    if bt == "tool_use":
                        if blk.get("name") in ("Read", "Edit"):
                            tid = blk.get("id")
                            rel = extract_rel((blk.get("input") or {}).get("file_path", ""))
                            if rel and tid:
                                tu_paths[tid] = rel
                                tu_ops[tid] = blk.get("name")
                    elif bt == "tool_result":
                        uid = blk.get("tool_use_id")
                        rel = tu_paths.get(uid)
                        if not rel or tu_ops.get(uid) != "Read":
                            continue
                        rc = blk.get("content")
                        text = ""
                        if isinstance(rc, list):
                            text = "".join(
                                s.get("text", "") for s in rc
                                if isinstance(s, dict) and s.get("type") == "text"
                            )
                        elif isinstance(rc, str):
                            text = rc
                        if not text or "<system-reminder>" in text[:200]:
                            # Skip empty or system-reminder leaks
                            pass
                        if not text:
                            continue
                        cleaned = strip_n(text)
                        reads.setdefault(rel, []).append((ts, cleaned, path.name))

    # For each file, pick the earliest Read (smallest timestamp).
    # Tie-break by content length (prefer the longer one — smaller is often
    # a partial offset/limit Read).
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_lines = []
    for rel in sorted(reads):
        items = reads[rel]
        # Group by content; pick the longest content among the earliest 3 Reads
        items.sort(key=lambda x: (x[0], -len(x[1])))
        ts, content, src = items[0]
        # If subsequent Reads have substantially longer content, use them
        # (the first Read might have been an offset+limit slice).
        for ts_b, c_b, src_b in items[1:5]:
            if len(c_b) > len(content) * 1.5:
                ts, content, src = ts_b, c_b, src_b
        target = OUT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        manifest_lines.append(f"{rel}\t{len(content)} bytes\tts={ts}\tsrc={src}")
        print(f"  {rel} <- {src[:8]}  ({len(content)} bytes, ts {ts})")

    (OUT / "MANIFEST.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"\n=== {len(reads)} files recovered to {OUT}/ ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
