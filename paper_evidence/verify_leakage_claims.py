"""
Audit and verify all leakage claims from overlap_audit.json and split_manifest.csv.
Reprint every leakage number intended for the paper with absolute precision.
"""
import json
import pandas as pd
from collections import defaultdict, Counter

manifest_path = "BrainTumorAI/results/split_manifest.csv"
audit_path = "BrainTumorAI/results/external_validation/cheng_figshare_v8/overlap_audit.json"

df_manifest = pd.read_csv(manifest_path)
with open(audit_path, "r") as f:
    audit_data = json.load(f)

hsm = audit_data["matches"]["high_similarity_matches"]
exact = audit_data["matches"].get("exact_pixel_matches", [])

# 1. Total matches and edges
total_hsm_edges = len(hsm)
total_exact_edges = len(exact)

# Distinct Cheng slices across all edges
all_cheng_slices = set(m["cheng_slice"] for m in hsm)
all_discovery_files = set(m["local_source_file"] for m in hsm)

# Patient tracking
patient_to_splits = defaultdict(set)
patient_to_edges = defaultdict(list)
patient_to_slices = defaultdict(set)
patient_to_disc_files = defaultdict(set)

for m in hsm:
    pid = m["patient_id"]
    split = m["local_split"]
    c_slice = m["cheng_slice"]
    d_file = m["local_source_file"]
    
    patient_to_splits[pid].add(split)
    patient_to_edges[pid].append(m)
    patient_to_slices[pid].add(c_slice)
    patient_to_disc_files[pid].add(d_file)

total_unique_patients = len(patient_to_splits)

# Patients by split combination
train_patients = {p for p, s in patient_to_splits.items() if "train" in s}
val_patients = {p for p, s in patient_to_splits.items() if "validation" in s}
test_patients = {p for p, s in patient_to_splits.items() if "test" in s}

both_train_test = {p for p in train_patients if p in test_patients}
both_train_val = {p for p in train_patients if p in val_patients}
all_three = {p for p in train_patients if p in val_patients and p in test_patients}

train_only = {p for p, s in patient_to_splits.items() if s == {"train"}}
val_only = {p for p, s in patient_to_splits.items() if s == {"validation"}}
test_only = {p for p, s in patient_to_splits.items() if s == {"test"}}
val_and_test_only = {p for p, s in patient_to_splits.items() if s == {"validation", "test"}}

# Test matches breakdown
test_matches = [m for m in hsm if m["local_split"] == "test"]
test_disc_files = set(m["local_source_file"] for m in test_matches)
test_cheng_slices = set(m["cheng_slice"] for m in test_matches)
test_unique_patients = set(m["patient_id"] for m in test_matches)

# Of the 141 test discovery files:
test_files_with_train_patient = set()
test_files_without_train_patient = set()

for m in test_matches:
    if m["patient_id"] in train_patients:
        test_files_with_train_patient.add(m["local_source_file"])
    else:
        test_files_without_train_patient.add(m["local_source_file"])

report = {
    "source_files": {
        "manifest": manifest_path,
        "overlap_audit": audit_path
    },
    "matches_and_edges": {
        "exact_pixel_matches": total_exact_edges,
        "high_similarity_match_edges": total_hsm_edges,
        "distinct_cheng_slices_all": len(all_cheng_slices),
        "distinct_discovery_files_all": len(all_discovery_files)
    },
    "patient_counts": {
        "total_unique_patients": total_unique_patients,
        "patients_in_train": len(train_patients),
        "patients_in_validation": len(val_patients),
        "patients_in_test": len(test_patients),
        "patients_in_both_train_and_test": len(both_train_test),
        "patients_in_both_train_and_validation": len(both_train_val),
        "patients_in_all_three_splits": len(all_three),
        "patients_train_only": len(train_only),
        "patients_validation_only": len(val_only),
        "patients_test_only": len(test_only),
        "patients_validation_and_test_only": len(val_and_test_only)
    },
    "test_split_breakdown": {
        "total_test_match_edges": len(test_matches),
        "distinct_discovery_test_files": len(test_disc_files),
        "distinct_cheng_slices_in_test": len(test_cheng_slices),
        "distinct_patients_in_test": len(test_unique_patients),
        "test_images_with_patient_also_in_train": len(test_files_with_train_patient),
        "test_images_with_patient_NOT_in_train": len(test_files_without_train_patient)
    },
    "non_train_matched_test_details": []
}

for f in sorted(test_files_without_train_patient):
    m = [edge for edge in test_matches if edge["local_source_file"] == f][0]
    pid = m["patient_id"]
    report["non_train_matched_test_details"].append({
        "discovery_file": f,
        "cheng_slice": m["cheng_slice"],
        "patient_id": pid,
        "splits_for_patient": sorted(list(patient_to_splits[pid])),
        "all_discovery_files_for_patient": sorted(list(patient_to_disc_files[pid]))
    })

print(json.dumps(report, indent=2))

with open("paper_evidence/verified_leakage_claims.json", "w") as f_out:
    json.dump(report, f_out, indent=2)

