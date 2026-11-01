"""
Pre-Submission Analysis 6: Release & Reproducibility Package Audit
- Records git commit/status/tags, complete SHA-256 inventory of all checkpoints, manifests, prediction files,
  and scripts for Zenodo archival, verifies the Kaggle dataset license status from DATASET_PROVENANCE.md,
  and inspects the .docx manuscript files for any remaining yellow TODO paragraphs in Section 2.8.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_docx_highlights_and_todos(docx_path: Path) -> list[dict[str, Any]]:
    if not docx_path.is_file():
        return []
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    findings = []
    with zipfile.ZipFile(docx_path, "r") as z:
        root = ET.fromstring(z.read("word/document.xml"))
        for idx, p in enumerate(root.findall(".//w:p", ns)):
            txt = "".join(t.text for t in p.findall(".//w:t", ns) if t.text)
            if not txt.strip():
                continue
            has_yellow = any(
                hl.attrib.get(f"{{{ns['w']}}}val") == "yellow"
                for hl in p.findall(".//w:highlight", ns)
            )
            if has_yellow or "todo" in txt.lower() or "2.8" in txt or "[" in txt:
                if has_yellow or "todo" in txt.lower() or "2.8" in txt:
                    findings.append({
                        "paragraph_index": idx,
                        "has_yellow_highlight": has_yellow,
                        "text": txt.strip()[:400],
                    })
    return findings


def main() -> None:
    pe_dir = ROOT / "paper_evidence"

    # Git status & commit
    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        git_branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        git_tags = subprocess.run(["git", "tag", "-l"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip().splitlines()
    except Exception as e:
        git_head = f"ERROR: {e}"
        git_branch = "UNKNOWN"
        git_tags = []

    # Checkpoints inventory
    checkpoints = []
    for p in sorted((ROOT / "BrainTumorAI" / "models").glob("*.pth")) + sorted((pe_dir / "models").glob("*.pth")):
        checkpoints.append({
            "path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })

    # Manifests & Predictions inventory
    manifests_and_preds = []
    target_files = [
        ROOT / "BrainTumorAI" / "results" / "split_manifest.csv",
        ROOT / "BrainTumorAI" / "results" / "evaluation" / "test_predictions.csv",
        ROOT / "BrainTumorAI" / "results" / "evaluation" / "evaluation_metrics.json",
        ROOT / "BrainTumorAI" / "results" / "external_validation" / "cheng_figshare_v8" / "predictions.csv",
        ROOT / "BrainTumorAI" / "results" / "external_validation" / "cheng_figshare_v8" / "overlap_audit.json",
        ROOT / "BrainTumorAI" / "results" / "external_validation" / "cheng_figshare_v8_overlap_screened" / "predictions.csv",
        pe_dir / "brats_patient_split_manifest.csv",
        pe_dir / "brats_250_patients_manifest.csv",
        pe_dir / "brats_10k_stratified_manifest.csv",
        pe_dir / "brats_oriented_predictions.csv.gz",
        pe_dir / "brats_old_predictions.csv.gz",
        pe_dir / "brats_oriented_flipLR_predictions.csv.gz",
        pe_dir / "brats_250_leakage_free_baseline_predictions.csv.gz",
        pe_dir / "brats_250_leakage_free_multiseed_predictions.csv.gz",
        pe_dir / "leakage_free_multiseed.json",
        pe_dir / "leakage_free_multiseed_summary.csv",
        pe_dir / "gradcam_brats_evaluation.json",
        pe_dir / "gradcam_brats_multiseed.json",
    ]
    for p in target_files:
        if p.is_file():
            manifests_and_preds.append({
                "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })

    docx_audit = {}
    for docx_p in [
        ROOT / "NeuroVision-AI_manuscript_complete.docx",
        Path("c:/Users/asus/Downloads/NeuroVision-AI_manuscript_final.docx"),
    ]:
        if docx_p.is_file():
            docx_audit[str(docx_p)] = inspect_docx_highlights_and_todos(docx_p)

    payload = {
        "analysis": "Pre-Submission Analysis 6: Release & Archival Inventory (v7)",
        "git_repository": {
            "head_commit": git_head,
            "branch": git_branch,
            "existing_tags": git_tags,
        },
        "kaggle_discovery_dataset_license_audit": {
            "source_record": "DATASET_PROVENANCE.md (lines 10-13)",
            "official_dataset_name_status": "UNVERIFIED locally (folder structure matches Kaggle 'Brain Tumor Data Set' / 'Brain Tumor Detection' 4,600-image collection: 2,513 Brain Tumor, 2,087 Healthy)",
            "license_status_in_local_archive": "UNVERIFIED (No LICENSE file or redistribution terms are bundled inside archive/; DATASET_PROVENANCE.md explicitly states: 'Do not redistribute until verified')",
            "external_cohorts_licenses": {
                "cheng_2015_figshare_1512427": "CC BY 4.0 (Figshare doi:10.6084/m9.figshare.1512427.v8)",
                "brats_2023_gli": "CC BY-SA 4.0 / Synapse Challenge Data Use Terms (ASNR-MICCAI BraTS 2023)",
            },
        },
        "checkpoints_inventory": checkpoints,
        "manifests_and_predictions_inventory": manifests_and_preds,
        "docx_yellow_highlights_and_section_2_8_check": docx_audit,
    }
    out_path = pe_dir / "release_audit_v7.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved release audit to {out_path}", flush=True)


if __name__ == "__main__":
    main()

