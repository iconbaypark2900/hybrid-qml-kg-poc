"""Check the top candidates data."""
import json

with open("artifacts/predictions/top_candidates.json") as f:
    d = json.load(f)

print(f"Total candidates: {len(d)}")
for c in d:
    print(f"  {c['compound']:25s} -> {c['disease']:30s}  KG={c['kg_rotate_score']:.3f}  rev={c['signature_reversal_score']:.3f}")
