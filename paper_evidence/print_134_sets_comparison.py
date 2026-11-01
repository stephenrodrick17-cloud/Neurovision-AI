import json, pandas as pd
from collections import Counter

with open('BrainTumorAI/results/external_validation/cheng_figshare_v8/overlap_audit.json') as f:
    oa = json.load(f)

hsm = oa['matches']['high_similarity_matches']

train_edges = [m for m in hsm if m['local_split'] == 'train']
test_edges = [m for m in hsm if m['local_split'] == 'test']

train_patients = set(m['patient_id'] for m in train_edges)

# Set 1: Unique Cheng slices matched to test
cheng_slices_in_test = set(m['cheng_slice'] for m in test_edges)

# Set 2: Discovery test images whose matched Cheng patient is also matched in train
disc_test_images_patient_in_train = set(m['local_source_file'] for m in test_edges if m['patient_id'] in train_patients)

# Set 3: Discovery test images whose matched Cheng patient is NOT in train
disc_test_images_patient_NOT_in_train = set(m['local_source_file'] for m in test_edges if m['patient_id'] not in train_patients)

all_disc_test_matched = set(m['local_source_file'] for m in test_edges)

print(f"=== SET 1: Unique Cheng slices matched to test ===")
print(f"Count: {len(cheng_slices_in_test)}")
print(f"Type: Cheng .mat slice identifiers (archive://...)")
print(f"Sample (first 5): {sorted(list(cheng_slices_in_test))[:5]}")

print(f"\n=== SET 2: Discovery test images with train-matched Cheng patient ===")
print(f"Count: {len(disc_test_images_patient_in_train)}")
print(f"Type: Discovery dataset JPEG file paths")
print(f"Sample (first 5): {sorted(list(disc_test_images_patient_in_train))[:5]}")

print(f"\n=== SET 3: Discovery test images with Cheng patient NOT in train ===")
print(f"Count: {len(disc_test_images_patient_NOT_in_train)}")
print(f"Files: {sorted(list(disc_test_images_patient_NOT_in_train))}")
for f in sorted(list(disc_test_images_patient_NOT_in_train)):
    edge = [m for m in test_edges if m['local_source_file'] == f][0]
    print(f"  {f} -> Cheng patient {edge['patient_id']}, slice {edge['cheng_slice']}")

print(f"\n=== Breakdown of the 141 Discovery test match edges ===")
print(f"Total test match edges: {len(test_edges)}")
print(f"Unique Discovery test files: {len(all_disc_test_matched)}")
print(f"Unique Cheng slices: {len(cheng_slices_in_test)}")

counts = Counter(m['cheng_slice'] for m in test_edges)
multi_cheng = {k: v for k, v in counts.items() if v > 1}
print(f"\nCheng slices matching multiple Discovery test images ({len(multi_cheng)} slices):")
for cs, cnt in multi_cheng.items():
    matching_disc = [m['local_source_file'] for m in test_edges if m['cheng_slice'] == cs]
    print(f"  {cs} matches {cnt} Discovery test files: {matching_disc}")

