import json, sys

with open('BrainTumorAI/results/external_validation/cheng_figshare_v8/overlap_audit.json') as f:
    data = json.load(f)

print("Top-level keys:", list(data.keys()))
for k, v in data.items():
    if isinstance(v, dict):
        sub = list(v.keys())[:10]
        print(f"  {k}: dict with keys {sub}")
    elif isinstance(v, list):
        n = len(v)
        t = type(v[0]).__name__ if v else "empty"
        print(f"  {k}: list of len {n}, first item type: {t}")
        if v and isinstance(v[0], dict):
            print(f"    first item keys: {list(v[0].keys())}")
            print(f"    first item: {v[0]}")
    else:
        print(f"  {k}: {v}")

