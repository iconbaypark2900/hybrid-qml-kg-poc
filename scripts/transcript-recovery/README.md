# hetqml-next transcript recovery

One-off utilities to reconstruct `hetqml-next/` source files from Claude Code JSONL transcripts after an accidental overwrite session.

## Layout

- `extract_hetqml_from_transcript.py` — single transcript, latest version per file
- `recover_all.py` — all project transcripts, latest version per file
- `recover_originals.py` — all project transcripts, earliest Read per file (pre-edit originals)
- `assemble_original.py` — merge recovered originals with untouched on-disk files
- `output/recovered_hetqml_next/` — latest-version recovery output
- `output/recovered_originals/` — earliest-read recovery output

## Usage

```bash
cd scripts/transcript-recovery

# Recover originals from all transcripts
python recover_originals.py

# Or recover latest versions
python recover_all.py

# Single transcript
python extract_hetqml_from_transcript.py /path/to/transcript.jsonl

# Assemble final folder (adjust CURRENT/TARGET paths in script if needed)
python assemble_original.py
```
