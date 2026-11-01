import json
from collections import defaultdict

with open('BrainTumorAI/results/external_validation/cheng_figshare_v8/overlap_audit.json') as f:
    oa = json.load(f)

hsm = oa['matches']['high_similarity_matches']

# Build patient to splits map
patient_splits = defaultdict(set)
patient_files = defaultdict(list)
for m in hsm:
    patient_splits[m['patient_id']].add(m['local_split'])
    patient_files[m['patient_id']].append((m['local_split'], m['local_source_file'], m['cheng_slice']))

print(f"Total unique patients: {len(patient_splits)}")
train_patients = {pid for pid, splits in patient_splits.items() if 'train' in splits}
print(f"Patients with matches in train: {len(train_patients)}")

# Examine all test matches
test_matches = [m for m in hsm if m['local_split'] == 'test']
print(f"Total test matches: {len(test_matches)}")

test_in_train = [m for m in test_matches if m['patient_id'] in train_patients]
test_not_in_train = [m for m in test_matches if m['patient_id'] not in train_patients]

print(f"Test matches with patient in train: {len(test_in_train)}")
print(f"Test matches with patient NOT in train: {len(test_not_in_train)}")

print("\n--- Details of test matches where patient is NOT in train ---")
for i, m in enumerate(test_not_in_train, 1):
    pid = m['patient_id']
    splits = sorted(list(patient_splits[pid]))
    all_slices = [(s, f, c) for s, f, c in patient_files[pid]]
    print(f"{i}. Discovery File: {m['local_source_file']}")
    print(f"   Cheng Slice:    {m['cheng_slice']}")
    print(f"   Patient ID:     {pid}")
    print(f"   Patient Splits: {splits}")
    print(f"   All slices for this patient in Discovery ({len(all_slices)}):")
    for s, f, c in all_slices:
        print(f"      split={s:10s} file={f}")

print("\n--- Also check: where did 100812 and 101037 come from? ---")
for p in ['100812', '101037']:
    matching = [m for m in hsm if m['patient_id'] == p]
    print(f"Patient {p}: {len(matching)} matches in overlap_audit.json")
    for m in matching:
        print(f"   split={m['local_split']} file={m['local_source_file']} cheng={m['cheng_slice']}")

# Check if 100812 or 101037 exist in Cheng dataset at all:
import pandas as pd
cheng_manifest = pd.read_csv('BrainTumorAI/results/external_validation/cheng_figshare_v8/manifest.csv')
for p in ['100812', '101037']:
    sub = cheng_manifest[cheng_manifest['patient_id'].astype(str) == p]
    print(f"In Cheng manifest, patient {p}: {len(sub)} slices")

