"""Aggregate hetqml-next file recovery across all Claude Code transcripts in
the hybrid-qml-kg-poc project. Most recent occurrence per relative path wins.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "recovered_hetqml_next"
TRANSCRIPTS = sorted(
    Path(
        "/home/roc/.cursor/projects/"
        "home-roc-quantumGlobalGroup-hybrid-qml-kg-poc/agent-transcripts"
    ).rglob("*.jsonl")
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
    files: dict[str, tuple[float, str]] = {}
    seq = 0
    grand_w = grand_r = 0
    for path in TRANSCRIPTS:
        print(f"=== {path.name} ===")
        tu_paths: dict[str, str] = {}
        tu_ops: dict[str, str] = {}
        n_w = n_r = 0
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
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
                for blk in content:
                    if not isinstance(blk, dict):
                        continue
                    bt = blk.get("type")
                    if bt == "tool_use":
                        name = blk.get("name")
                        inp = blk.get("input") or {}
                        if name == "Write":
                            rel = extract_rel(inp.get("file_path", ""))
                            text = inp.get("content")
                            if rel and isinstance(text, str):
                                files[rel] = (seq, text)
                                n_w += 1
                        elif name in ("Read", "Edit"):
                            rel = extract_rel(inp.get("file_path", ""))
                            tid = blk.get("id")
                            if rel and tid:
                                tu_paths[tid] = rel
                                tu_ops[tid] = name
                    elif bt == "tool_result":
                        uid = blk.get("tool_use_id")
                        rel = tu_paths.get(uid)
                        if not rel or tu_ops.get(uid) != "Read":
                            continue
                        rc = blk.get("content")
                        text = ""
                        if isinstance(rc, list):
                            text = "".join(
                                s.get("text", "")
                                for s in rc
                                if isinstance(s, dict) and s.get("type") == "text"
                            )
                        elif isinstance(rc, str):
                            text = rc
                        if not text:
                            continue
                        if rel not in files or files[rel][0] < seq:
                            files[rel] = (seq, strip_n(text))
                            n_r += 1
        print(f"  Writes={n_w} Reads={n_r}")
        grand_w += n_w
        grand_r += n_r

    OUT.mkdir(parents=True, exist_ok=True)
    for rel, (_seq, content) in sorted(files.items()):
        t = OUT / rel
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(content, encoding="utf-8")

    print(f"\n=== TOTAL: {len(files)} unique files "
          f"({grand_w} writes, {grand_r} reads aggregated) ===")
    for rel in sorted(files):
        print(f"  {rel} ({len(files[rel][1])} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
