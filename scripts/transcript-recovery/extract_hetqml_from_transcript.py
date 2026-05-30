"""Extract every hetqml-next file from a Claude Code transcript.

Walks Claude Code's JSONL transcript. Pulls files from:
  1. Write tool_use inputs (file_path + content)
  2. Read tool_result outputs paired with their tool_use file_path

Most recent occurrence per path wins. Output goes to ./recovered_hetqml_next/
mirroring the relative path under hetqml-next/.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRANSCRIPT = Path(sys.argv[1])
OUT_DIR = ROOT / "output" / "recovered_hetqml_next"


def extract_rel_path(file_path: str) -> str | None:
    if not file_path or "hetqml-next" not in file_path.lower():
        return None
    # Normalize separators
    norm = file_path.replace("\\", "/")
    idx = norm.lower().find("hetqml-next/")
    if idx == -1:
        return None
    rel = norm[idx + len("hetqml-next/"):]
    rel = rel.split("?")[0].split("#")[0].strip()
    return rel or None


def strip_cat_n(content: str) -> str:
    """Read tool returns content as 'NUMBER\tline' (cat -n format)."""
    out = []
    for line in content.splitlines():
        m = re.match(r"^\s*\d+\t(.*)$", line)
        out.append(m.group(1) if m else line)
    return "\n".join(out)


def main() -> int:
    files: dict[str, tuple[float, str]] = {}  # rel -> (timestamp_seq, content)
    tool_use_paths: dict[str, str] = {}       # tool_use_id -> rel_path (for Read pairing)
    tool_use_ops: dict[str, str] = {}         # tool_use_id -> 'Write'|'Read'
    seq = 0
    n_lines = 0
    n_writes = 0
    n_reads = 0

    with TRANSCRIPT.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_lines += 1
            seq += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")

                if btype == "tool_use":
                    name = block.get("name")
                    use_id = block.get("id")
                    inp = block.get("input") or {}

                    if name == "Write":
                        rel = extract_rel_path(inp.get("file_path", ""))
                        text = inp.get("content")
                        if rel and isinstance(text, str):
                            files[rel] = (seq, text)
                            n_writes += 1
                    elif name in ("Read", "Edit"):
                        rel = extract_rel_path(inp.get("file_path", ""))
                        if rel and use_id:
                            tool_use_paths[use_id] = rel
                            tool_use_ops[use_id] = name

                elif btype == "tool_result":
                    use_id = block.get("tool_use_id")
                    rel = tool_use_paths.get(use_id)
                    if not rel:
                        continue
                    op = tool_use_ops.get(use_id)
                    if op != "Read":
                        continue  # Edit results don't carry full content
                    rc = block.get("content")
                    text = ""
                    if isinstance(rc, list):
                        for sub in rc:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                text += sub.get("text", "")
                    elif isinstance(rc, str):
                        text = rc
                    if not text:
                        continue
                    cleaned = strip_cat_n(text)
                    # If file already has a more recent Write, keep it; else use Read
                    if rel not in files or files[rel][0] < seq:
                        files[rel] = (seq, cleaned)
                        n_reads += 1

    print(f"scanned {n_lines} lines; captured {n_writes} Writes, {n_reads} Reads; "
          f"{len(files)} unique paths under hetqml-next/")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for rel, (_seq, content) in sorted(files.items()):
        target = OUT_DIR / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        print(f"  wrote {target} ({len(content)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
