import json, sys

with open('BrainTumorAI/results/external_validation/cheng_figshare_v8/overlap_audit.json') as f:
    data = json.load(f)

matches = data['matches']
print("exact_pixel_matches count:", len(matches.get('exact_pixel_matches', [])))
hsm = matches.get('high_similarity_matches', [])
print("high_similarity_matches count:", len(hsm))
if hsm:
    print("First match keys:", list(hsm[0].keys()))
    print("First match:", json.dumps(hsm[0], indent=2))
    # Check for split info
    all_keys = set()
    for m in hsm:
        all_keys.update(m.keys())
    print("All unique keys across matches:", sorted(all_keys))

