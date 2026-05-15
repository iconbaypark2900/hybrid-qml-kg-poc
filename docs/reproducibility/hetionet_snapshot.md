# Hetionet v1.0 snapshot — content hashes

Recorded: 2026-05-04

Per `preregistration/osf_preregistration_v1.md` §3.1 and §9.2, the
Hetionet snapshot used by this project is identified by its SHA-256
content hash. The files below are tracked outside Git (see
`.gitignore`); these hashes are the canonical identifier for
reproducibility.

| File | Size (bytes) | Last modified (UTC) | SHA-256 |
|---|---:|---|---|
| `data/hetionet-v1.0-edges.sif` | 88,978,057 | 2025-11-04 16:56:29 UTC | `4b47bad290881ed468f2be9d11bf9adb7aaa2c5a1296de7ba09cdca058565a4e` |
| `data/hetionet-v1.0-nodes.tsv` | 2,462,952 | 2026-04-15 13:55:42 UTC | `f8a88e2bc21c576aa813c096fdef318e4502723ec4b7a97decb7c755c86cbc09` |

## Regenerating

```bash
python scripts/record_hetionet_hash.py
```

Idempotent — re-running produces the same hashes if the data files
have not changed. Run before OSF preregistration submission and
again before the pre-submission clean-room reproducibility check.
