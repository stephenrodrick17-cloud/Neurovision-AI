"""Clarify Item C: 141 = edges or unique discovery test files?"""
import json, pandas as pd

with open('BrainTumorAI/results/external_validation/cheng_figshare_v8/overlap_audit.json') as f:
    oa = json.load(f)

hsm = oa['matches']['high_similarity_matches']
split_manifest = pd.read_csv('BrainTumorAI/results/split_manifest.csv')

# All test edges
test_edges = [m for m in hsm if m['local_split'] == 'test']
print(f"Test edges (match rows): {len(test_edges)}")

# Unique discovery test files
unique_disc_test_files = set(m['local_source_file'] for m in test_edges)
print(f"Unique discovery test files matched: {len(unique_disc_test_files)}")

# Unique Cheng slices matched to test
unique_cheng_test_slices = set(m['cheng_slice'] for m in test_edges)
print(f"Unique Cheng slices matched to test: {len(unique_cheng_test_slices)}")

# Cheng patient IDs from test matches
test_patient_ids = set(m['patient_id'] for m in test_edges)
print(f"Unique Cheng patients with test matches: {len(test_patient_ids)}")

# Cheng patient IDs from train matches
train_edges = [m for m in hsm if m['local_split'] == 'train']
train_patient_ids = set(m['patient_id'] for m in train_edges)
print(f"Unique Cheng patients with train matches: {len(train_patient_ids)}")

# Now: how many unique DISCOVERY TEST FILES have their matched Cheng patient also in train?
disc_test_also_train = set()
for m in test_edges:
    if m['patient_id'] in train_patient_ids:
        disc_test_also_train.add(m['local_source_file'])
print(f"\nDiscovery test files whose Cheng patient is also matched in train: {len(disc_test_also_train)}")
print(f"Discovery test files whose Cheng patient is NOT in train: {len(unique_disc_test_files - disc_test_also_train)}")

# Also check: is the mapping 1:1 or many:many?
# How many edges per discovery test file?
from collections import Counter
edges_per_disc = Counter(m['local_source_file'] for m in test_edges)
multi = {k: v for k, v in edges_per_disc.items() if v > 1}
print(f"\nDiscovery test files with >1 Cheng match: {len(multi)}")
if multi:
    for f, c in sorted(multi.items(), key=lambda x: -x[1])[:5]:
        print(f"  {f}: {c} matches")

# Edges per Cheng slice
edges_per_cheng = Counter(m['cheng_slice'] for m in test_edges)
multi_c = {k: v for k, v in edges_per_cheng.items() if v > 1}
print(f"Cheng slices matching >1 discovery test file: {len(multi_c)}")

# Same for ALL edges
print(f"\n--- All 846 edges ---")
all_disc = set(m['local_source_file'] for m in hsm)
print(f"Unique discovery files across all edges: {len(all_disc)}")
all_cheng = set(m['cheng_slice'] for m in hsm)
print(f"Unique Cheng slices across all edges: {len(all_cheng)}")

