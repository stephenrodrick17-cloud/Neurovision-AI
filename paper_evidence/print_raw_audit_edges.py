import json

audit_path = 'BrainTumorAI/results/external_validation/cheng_figshare_v8/overlap_audit.json'
with open(audit_path, 'r') as f:
    data = json.load(f)

matches = data['matches']['high_similarity_matches']

# Find all entries where local_split == 'test'
test_matches = [m for m in matches if m.get('local_split') == 'test']
print(f"Total test matches: {len(test_matches)}")

# Group by patient_id
from collections import defaultdict
p_to_matches = defaultdict(list)
for m in matches:
    p_to_matches[m['patient_id']].append(m)

train_patients = set(m['patient_id'] for m in matches if m.get('local_split') == 'train')
test_patients = set(m['patient_id'] for m in matches if m.get('local_split') == 'test')
non_train_test_patients = test_patients - train_patients

print(f"Patients in test but not train: {sorted(list(non_train_test_patients))}")

for p in sorted(list(non_train_test_patients)):
    print(f"\n=================== PATIENT: {p} ===================")
    for m in p_to_matches[p]:
        print(json.dumps(m, indent=2))

# Also search for patients MR054683, MR024780E, MR051461B, MR053110B
print("\n--- Checking MR* strings in audit ---")
for p_target in ['MR054683', 'MR024780E', 'MR051461B', 'MR053110B', '100812', '101037']:
    found = [m for m in matches if p_target.lower() in str(m).lower()]
    print(f"Target '{p_target}': found {len(found)} entries")
    if found:
        print(f"  First entry patient_id: {found[0]['patient_id']}, cheng_slice: {found[0]['cheng_slice']}, local_file: {found[0]['local_source_file']}")
