from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

from .classifier_pipeline import (
    CLASS_TO_IDX,
    DEFAULT_CHECKPOINT,
    DEFAULT_RESULTS_DIR,
    PROJECT_ROOT,
    IDX_TO_CLASS,
    load_checkpoint,
    preprocess_image,
    sha256_file,
)
from .evaluate_classifier import compute_metrics
from .preprocess import _perceptual_hash, _structural_similarity

CHENG_SOURCE = "https://doi.org/10.6084/m9.figshare.1512427.v8"
CHENG_LICENSE = "CC BY 4.0"
CHENG_LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}
CHENG_PART_MD5 = {
    "brainTumorDataPublic_1-766.zip": "74b949ad33f042e6e103523091cd1428",
    "brainTumorDataPublic_767-1532.zip": "7e8a875500d2c8a346f270538e29890e",
    "brainTumorDataPublic_1533-2298.zip": "8227bf6080cb71f15a88be8d25c79ae7",
    "brainTumorDataPublic_2299-3064.zip": "b378a80d6174e5317d59eb28430c6652",
}
EXTERNAL_RESULTS_DIR = PROJECT_ROOT / "BrainTumorAI" / "results" / "external_validation"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _stable_stem(source: Path, root: Path) -> str:
    relative = source.relative_to(root).as_posix()
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
    return f"{source.stem}_{digest}"


def _mat_field(struct: Any, name: str) -> Any:
    if isinstance(struct, dict) and name in struct:
        return struct[name]
    if hasattr(struct, name):
        return getattr(struct, name)
    raise ValueError(f"MAT file has no cjdata.{name} field.")


def _mat_scalar(value: Any, field_name: str) -> Any:
    array = np.asarray(value).squeeze()
    if array.dtype.kind in {"S", "U"} and array.size > 1:
        scalar = "".join(str(item) for item in array.reshape(-1)).strip()
        if scalar:
            return scalar
    if array.size != 1:
        raise ValueError(f"MAT field cjdata.{field_name} must be a scalar.")
    scalar = array.item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    return scalar


def _decode_cheng_mat(mat_bytes: bytes, source_label: str) -> tuple[np.ndarray, str, int]:
    if b"\x89HDF\r\n\x1a\n" in mat_bytes[:4096]:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "Cheng MATLAB v7.3 files require h5py. Install it with "
                "`python -m pip install h5py` and retry."
            ) from exc
        try:
            with h5py.File(io.BytesIO(mat_bytes), "r") as contents:
                if "cjdata" not in contents:
                    raise ValueError(f"Cheng MAT file does not contain cjdata: {source_label}")
                record = contents["cjdata"]
                label_value = _mat_scalar(record["label"][()], "label")
                patient_data = np.asarray(record["PID"][()]).reshape(-1)
                patient_id = "".join(chr(int(code)) for code in patient_data if int(code)).strip()
                image = np.asarray(record["image"][()]).T
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Could not read Cheng MATLAB v7.3 file {source_label}: {exc}") from exc
    else:
        try:
            contents = loadmat(io.BytesIO(mat_bytes), squeeze_me=True, struct_as_record=False)
        except (OSError, ValueError, NotImplementedError) as exc:
            raise ValueError(f"Could not read Cheng MAT file {source_label}: {exc}") from exc
        if "cjdata" not in contents:
            raise ValueError(f"Cheng MAT file does not contain cjdata: {source_label}")
        record = contents["cjdata"]
        label_value = _mat_scalar(_mat_field(record, "label"), "label")
        patient_value = _mat_scalar(_mat_field(record, "PID"), "PID")
        if isinstance(patient_value, (int, float, np.integer, np.floating)):
            patient_number = float(patient_value)
            patient_id = str(int(patient_number)) if patient_number.is_integer() else str(patient_value)
        else:
            patient_id = str(patient_value).strip()
        image = np.squeeze(np.asarray(_mat_field(record, "image")))
    label_number = int(label_value)
    if float(label_value) != label_number or label_number not in CHENG_LABELS:
        raise ValueError(f"Unsupported Cheng tumor label {label_value!r} in {source_label}.")
    if not patient_id:
        raise ValueError(f"Empty Cheng patient ID in {source_label}.")
    if image.ndim != 2 or not np.issubdtype(image.dtype, np.number):
        raise ValueError(f"Cheng cjdata.image must be a numeric 2D array: {source_label}")
    if not np.isfinite(image).all():
        raise ValueError(f"Cheng image contains non-finite values: {source_label}")
    if float(image.max()) <= float(image.min()):
        raise ValueError(f"Cheng image has no intensity contrast: {source_label}")
    return image, patient_id, label_number


def _cheng_rows_from_archive(archive_path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    try:
        outer_archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Could not open Cheng archive {archive_path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    actual_md5: dict[str, str] = {}
    with outer_archive:
        names = set(outer_archive.namelist())
        missing_parts = set(CHENG_PART_MD5) - names
        if missing_parts:
            raise ValueError(f"Cheng archive is missing expected Figshare parts: {sorted(missing_parts)}")
        for part_name, expected_md5 in CHENG_PART_MD5.items():
            try:
                part_bytes = outer_archive.read(part_name)
            except (OSError, KeyError, RuntimeError) as exc:
                raise ValueError(f"Could not read nested Cheng archive {part_name}: {exc}") from exc
            actual_md5[part_name] = hashlib.md5(part_bytes).hexdigest()
            if actual_md5[part_name] != expected_md5:
                raise ValueError(
                    f"Cheng archive part {part_name} failed the official Figshare MD5 check: "
                    f"expected {expected_md5}, got {actual_md5[part_name]}."
                )
            try:
                with zipfile.ZipFile(io.BytesIO(part_bytes)) as part_archive:
                    mat_names = sorted(
                        name
                        for name in part_archive.namelist()
                        if not name.endswith("/") and name.lower().endswith(".mat")
                    )
                    for mat_name in mat_names:
                        image, patient_id, label_number = _decode_cheng_mat(
                            part_archive.read(mat_name),
                            f"{archive_path.name}!{part_name}!{mat_name}",
                        )
                        del image
                        rows.append(
                            {
                                "dataset": "cheng_figshare_v8",
                                "patient_id": patient_id,
                                "source_file": f"archive://{part_name}!{mat_name}",
                                "label_idx": CLASS_TO_IDX["tumor"],
                                "subtype": CHENG_LABELS[label_number],
                                "slice_id": Path(mat_name).stem,
                            }
                        )
            except (OSError, zipfile.BadZipFile, KeyError, RuntimeError) as exc:
                raise ValueError(f"Could not inspect Cheng archive part {part_name}: {exc}") from exc
    if not rows:
        raise ValueError(f"No MAT image files were found in Cheng archive: {archive_path}")
    return rows, actual_md5


def audit_cheng_overlap(
    archive_path: Path,
    local_manifest_path: Path = DEFAULT_RESULTS_DIR / "split_manifest.csv",
) -> dict[str, Any]:
    """Screen Cheng slices against all local splits for exact and high-similarity images."""
    if not local_manifest_path.is_file():
        raise FileNotFoundError(f"Local training manifest not found: {local_manifest_path}")
    local_rows = pd.read_csv(local_manifest_path)
    required = {"source_file", "split", "pixel_sha256", "phash", "status"}
    missing = required - set(local_rows.columns)
    if missing:
        raise ValueError(f"Local manifest is missing overlap-audit columns: {sorted(missing)}")
    local_rows = local_rows[local_rows["status"] == "readable"].reset_index(drop=True)
    local_hashes = [int(str(value), 16) for value in local_rows["phash"]]
    local_exact: dict[str, list[int]] = {}
    for index, digest in enumerate(local_rows["pixel_sha256"].astype(str)):
        local_exact.setdefault(digest, []).append(index)
    local_gray64: dict[int, np.ndarray] = {}

    def get_local_gray64(index: int) -> np.ndarray:
        if index not in local_gray64:
            path = PROJECT_ROOT / str(local_rows.iloc[index]["source_file"])
            try:
                with Image.open(path) as image:
                    _, _, local_gray64[index] = _perceptual_hash(image.convert("RGB"))
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Could not read local image for overlap audit: {path}") from exc
        return local_gray64[index]

    archive_path = archive_path.resolve()
    exact_edges: list[dict[str, str]] = []
    near_edges: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as outer_archive:
        for part_name, expected_md5 in CHENG_PART_MD5.items():
            part_bytes = outer_archive.read(part_name)
            if hashlib.md5(part_bytes).hexdigest() != expected_md5:
                raise ValueError(f"Cheng Figshare archive integrity check failed for {part_name}.")
            with zipfile.ZipFile(io.BytesIO(part_bytes)) as part_archive:
                for mat_name in part_archive.namelist():
                    if mat_name.endswith("/") or not mat_name.lower().endswith(".mat"):
                        continue
                    image_array, patient_id, _ = _decode_cheng_mat(
                        part_archive.read(mat_name),
                        f"{archive_path.name}!{part_name}!{mat_name}",
                    )
                    lower = float(image_array.min())
                    upper = float(image_array.max())
                    pixels = np.rint(
                        (image_array.astype(np.float64) - lower) * (255.0 / (upper - lower))
                    )
                    gray = np.clip(pixels, 0, 255).astype(np.uint8)
                    rgb = np.repeat(gray[:, :, None], 3, axis=2)
                    pixel_digest = hashlib.sha256(
                        f"{gray.shape[1]}x{gray.shape[0]}".encode("ascii") + rgb.tobytes()
                    ).hexdigest()
                    cheng_path = f"{part_name}!{mat_name}"
                    for local_index in local_exact.get(pixel_digest, []):
                        exact_edges.append(
                            {
                                "cheng_slice": cheng_path,
                                "patient_id": patient_id,
                                "local_source_file": str(local_rows.iloc[local_index]["source_file"]),
                                "local_split": str(local_rows.iloc[local_index]["split"]),
                            }
                        )

                    cheng_phash, _, cheng_gray64 = _perceptual_hash(Image.fromarray(gray, mode="L"))
                    cheng_hash_int = int(cheng_phash, 16)
                    for local_index, local_hash in enumerate(local_hashes):
                        if (cheng_hash_int ^ local_hash).bit_count() > 4:
                            continue
                        if _structural_similarity(cheng_gray64, get_local_gray64(local_index)) < 0.95:
                            continue
                        near_edges.append(
                            {
                                "cheng_slice": cheng_path,
                                "patient_id": patient_id,
                                "local_source_file": str(local_rows.iloc[local_index]["source_file"]),
                                "local_split": str(local_rows.iloc[local_index]["split"]),
                                "phash_hamming_distance": (cheng_hash_int ^ local_hash).bit_count(),
                            }
                        )
    exact_cheng = {edge["cheng_slice"] for edge in exact_edges}
    near_cheng = {edge["cheng_slice"] for edge in near_edges}
    all_edges = exact_edges + near_edges
    overlapping_patients = sorted({edge["patient_id"] for edge in all_edges})
    match_edges_by_split = {
        split: sum(edge["local_split"] == split for edge in all_edges)
        for split in ("train", "validation", "test")
    }
    return {
        "method": (
            "Exact decoded RGB pixel SHA-256 and the local conservative near-duplicate rule "
            "(pHash Hamming distance <=4 and 64x64 grayscale SSIM >=0.95)."
        ),
        "local_manifest": str(local_manifest_path.resolve()),
        "local_manifest_sha256": sha256_file(local_manifest_path),
        "local_sample_count": len(local_rows),
        "cheng_sample_count": 3064,
        "exact_match_slice_count": len(exact_cheng),
        "exact_match_edge_count": len(exact_edges),
        "near_duplicate_slice_count": len(near_cheng),
        "near_duplicate_edge_count": len(near_edges),
        "overlapping_patient_count": len(overlapping_patients),
        "overlapping_patient_ids": overlapping_patients,
        "match_edges_by_local_split": match_edges_by_split,
        "overlap_detected": bool(exact_cheng or near_cheng),
        "matches": {
            "exact_pixel_matches": exact_edges,
            "high_similarity_matches": near_edges,
        },
        "interpretation": (
            "A zero match count only excludes exact and high-similarity image matches under this "
            "screen; it does not prove patient independence or exclude other overlap."
        ),
    }


def prepare_cheng_dataset(data_root: Path, output_dir: Path) -> dict[str, Any]:
    """Convert the public Cheng .mat files into binary-labeled image rows."""
    data_root = data_root.resolve()
    if data_root.is_file() and data_root.suffix.lower() == ".zip":
        rows, verified_parts = _cheng_rows_from_archive(data_root)
        manifest = pd.DataFrame(rows).sort_values(["patient_id", "slice_id"], kind="stable")
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        metadata = {
            "dataset": "Cheng Figshare brain tumor dataset",
            "source": CHENG_SOURCE,
            "doi": "10.6084/m9.figshare.1512427.v8",
            "license": CHENG_LICENSE,
            "sample_count": len(manifest),
            "patient_count": int(manifest["patient_id"].nunique()),
            "class_mapping": {
                "1": "tumor (meningioma)",
                "2": "tumor (glioma)",
                "3": "tumor (pituitary)",
            },
            "evaluation_scope": (
                "Positive-class external sensitivity only. This dataset has no no-tumor class, so "
                "specificity, binary accuracy, and ROC-AUC are not estimable."
            ),
            "preprocessing": (
                "MAT cjdata.image is converted in memory to 8-bit grayscale by per-image min-max "
                "scaling; inference then uses the project's shared RGB, resize, and "
                "ImageNet-normalization transform."
            ),
            "patient_id_field": "cjdata.PID",
            "mask_used_for_classification": False,
            "source_archive": str(data_root),
            "source_archive_sha256": sha256_file(data_root),
            "verified_figshare_part_md5": verified_parts,
            "manifest": manifest_path.name,
            "manifest_sha256": sha256_file(manifest_path),
        }
        _write_json(output_dir / "dataset_metadata.json", metadata)
        return metadata
    if not data_root.is_dir():
        raise FileNotFoundError(f"Cheng data directory does not exist: {data_root}")
    source_files = sorted(
        path for path in data_root.rglob("*") if path.is_file() and path.suffix.lower() == ".mat"
    )
    if not source_files:
        raise FileNotFoundError(f"No .mat files found under Cheng data directory: {data_root}")

    output_dir = output_dir.resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for source in source_files:
        try:
            image, patient_id, label_number = _decode_cheng_mat(source.read_bytes(), str(source))
        except OSError as exc:
            raise ValueError(f"Could not read Cheng MAT file {source}: {exc}") from exc
        minimum = float(image.min())
        maximum = float(image.max())
        pixels = np.rint((image.astype(np.float64) - minimum) * (255.0 / (maximum - minimum)))
        destination = image_dir / f"{_stable_stem(source, data_root)}.png"
        Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="L").save(destination)
        rows.append(
            {
                "dataset": "cheng_figshare_v8",
                "patient_id": patient_id,
                "source_file": destination.relative_to(output_dir).as_posix(),
                "label_idx": CLASS_TO_IDX["tumor"],
                "subtype": CHENG_LABELS[label_number],
                "slice_id": source.stem,
            }
        )

    manifest = pd.DataFrame(rows).sort_values(["patient_id", "slice_id"], kind="stable")
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    metadata = {
        "dataset": "Cheng Figshare brain tumor dataset",
        "source": CHENG_SOURCE,
        "doi": "10.6084/m9.figshare.1512427.v8",
        "license": CHENG_LICENSE,
        "sample_count": len(manifest),
        "patient_count": int(manifest["patient_id"].nunique()),
        "class_mapping": {
            "1": "tumor (meningioma)",
            "2": "tumor (glioma)",
            "3": "tumor (pituitary)",
        },
        "evaluation_scope": (
            "Positive-class external sensitivity only. This dataset has no no-tumor class, so "
            "specificity, binary accuracy, and ROC-AUC are not estimable."
        ),
        "preprocessing": (
            "MAT cjdata.image converted to 8-bit grayscale by per-image min-max scaling; inference "
            "then uses the project's shared RGB, resize, and ImageNet-normalization transform."
        ),
        "patient_id_field": "cjdata.PID",
        "mask_used_for_classification": False,
        "manifest": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
    }
    _write_json(output_dir / "dataset_metadata.json", metadata)
    return metadata


def _nifti_suffix(path: Path) -> str:
    lower_name = path.name.lower()
    if lower_name.endswith(".nii.gz"):
        return ".nii.gz"
    if lower_name.endswith(".nii"):
        return ".nii"
    raise ValueError(f"Expected a .nii or .nii.gz file: {path}")


def _find_brats_segmentation(flair_path: Path, patient_id: str) -> Path:
    suffix = _nifti_suffix(flair_path)
    separator = flair_path.name[len(patient_id)]
    segmentation = flair_path.with_name(f"{patient_id}{separator}seg{suffix}")
    if segmentation.is_file():
        return segmentation
    candidates = [
        candidate
        for candidate in flair_path.parent.iterdir()
        if candidate.is_file()
        and re.fullmatch(
            rf"(?i){re.escape(patient_id)}[_-]seg\.nii(?:\.gz)?",
            candidate.name,
        )
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one segmentation NIfTI paired with {flair_path.name}; "
            f"found {len(candidates)} candidates."
        )
    return candidates[0]


def prepare_brats_dataset(
    data_root: Path,
    output_dir: Path,
    min_brain_pixels: int = 256,
) -> dict[str, Any]:
    """Create patient-linked FLAIR slice labels from BraTS NIfTI volumes and masks."""
    if min_brain_pixels < 1:
        raise ValueError("min_brain_pixels must be a positive integer.")
    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"BraTS data directory does not exist: {data_root}")
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "BraTS NIfTI preparation requires nibabel. Install it with "
            "`python -m pip install nibabel` and rerun this command."
        ) from exc

    flair_files = sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file()
        and re.search(r"(?i)(?:[_-](?:flair|t2f))\.nii(?:\.gz)?$", path.name)
    )
    if not flair_files:
        raise FileNotFoundError(
            f"No BraTS FLAIR NIfTI files found under {data_root}; supported names include *_flair.nii(.gz) and *_t2f.nii(.gz)."
        )

    output_dir = output_dir.resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen_patient_ids: set[str] = set()
    for flair_path in flair_files:
        suffix_match = re.search(r"(?i)(?:[_-])(?:flair|t2f)\.nii(?:\.gz)?$", flair_path.name)
        if suffix_match is None:
            raise ValueError(f"Could not extract a BraTS patient ID from FLAIR filename: {flair_path.name}")
        patient_id = flair_path.name[: suffix_match.start()]
        segmentation_path = _find_brats_segmentation(flair_path, patient_id)
        if patient_id in seen_patient_ids:
            raise ValueError(
                f"Multiple FLAIR volumes found for patient {patient_id}; select one acquisition per patient."
            )
        seen_patient_ids.add(patient_id)
        try:
            flair_image = nib.as_closest_canonical(nib.load(str(flair_path)))
            seg_image = nib.as_closest_canonical(nib.load(str(segmentation_path)))
            volume = np.asarray(flair_image.dataobj, dtype=np.float32)
            segmentation = np.asarray(seg_image.dataobj)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Could not read BraTS NIfTI pair for {patient_id}: {exc}") from exc
        if volume.ndim != 3 or segmentation.ndim != 3 or volume.shape != segmentation.shape:
            raise ValueError(
                f"BraTS FLAIR and segmentation must be matching 3D arrays for {patient_id}; "
                f"got {volume.shape} and {segmentation.shape}."
            )
        if not np.allclose(flair_image.affine, seg_image.affine, rtol=1e-5, atol=1e-5):
            raise ValueError(f"BraTS FLAIR and segmentation affine transforms do not match for {patient_id}.")
        if not np.isfinite(volume).all() or not np.isfinite(segmentation).all():
            raise ValueError(f"BraTS volume contains non-finite values for patient {patient_id}.")
        if np.any(segmentation < 0):
            raise ValueError(f"BraTS segmentation has negative labels for patient {patient_id}.")
        brain_pixels = volume != 0
        nonzero_values = volume[brain_pixels]
        if nonzero_values.size == 0:
            raise ValueError(f"BraTS FLAIR volume is entirely zero for patient {patient_id}.")
        low, high = np.percentile(nonzero_values, [0.5, 99.5])
        if high <= low:
            raise ValueError(f"BraTS FLAIR volume has no intensity contrast for patient {patient_id}.")

        saved_slices = 0
        for slice_index in range(volume.shape[2]):
            image_slice = volume[:, :, slice_index]
            brain_mask = image_slice != 0
            if int(np.count_nonzero(brain_mask)) < min_brain_pixels:
                continue
            scaled = np.rint(np.clip((image_slice - low) * (255.0 / (high - low)), 0, 255))
            image_name = (
                f"{_stable_stem(flair_path, data_root)}_z{slice_index:04d}.png"
            )
            destination = image_dir / image_name
            Image.fromarray(scaled.astype(np.uint8), mode="L").save(destination)
            tumor_present = bool(np.any(segmentation[:, :, slice_index] > 0))
            rows.append(
                {
                    "dataset": "brats_flair_2d",
                    "patient_id": patient_id,
                    "source_file": destination.relative_to(output_dir).as_posix(),
                    "label_idx": CLASS_TO_IDX["tumor"] if tumor_present else CLASS_TO_IDX["no_tumor"],
                    "subtype": "tumor-containing slice" if tumor_present else "no-tumor slice",
                    "slice_id": slice_index,
                }
            )
            saved_slices += 1
        if saved_slices == 0:
            raise ValueError(
                f"No BraTS slices met min_brain_pixels={min_brain_pixels} for patient {patient_id}."
            )

    manifest = pd.DataFrame(rows).sort_values(["patient_id", "slice_id"], kind="stable")
    if set(manifest["label_idx"].unique()) != set(CLASS_TO_IDX.values()):
        raise ValueError(
            "The prepared BraTS slices do not contain both tumor-positive and tumor-free slices; "
            "binary slice metrics cannot be calculated for this cohort."
        )
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    metadata = {
        "dataset": "BraTS FLAIR axial slices",
        "source": "Obtain from the official BraTS challenge dataset portal under its current access terms.",
        "sample_count": len(manifest),
        "patient_count": int(manifest["patient_id"].nunique()),
        "class_mapping": {
            "0": "no tumor voxels in this FLAIR axial slice",
            "1": "one or more tumor voxels in this FLAIR axial slice",
        },
        "evaluation_scope": (
            "External 2D slice classification. Specificity is against tumor-free slices from "
            "BraTS tumor-bearing patients, not healthy participants."
        ),
        "modality": "FLAIR",
        "orientation": "NIfTI volumes converted to closest canonical orientation; axial slices along axis 2.",
        "intensity_scaling": (
            "Per-volume nonzero voxel 0.5th and 99.5th percentiles scaled to 8-bit; "
            "project inference preprocessing is applied afterward."
        ),
        "slice_selection": f"Retain slices with at least {min_brain_pixels} nonzero FLAIR pixels.",
        "tumor_definition": "Any segmentation voxel greater than zero marks a tumor-containing slice.",
        "patient_id_field": "Prefix before the _flair or -flair filename suffix.",
        "manifest": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
    }
    _write_json(output_dir / "dataset_metadata.json", metadata)
    return metadata


def calculate_external_metrics(
    true_labels: list[int] | np.ndarray,
    tumor_scores: list[float] | np.ndarray,
    patient_ids: list[str] | np.ndarray,
    threshold: float = 0.5,
    seed: int = 42,
    bootstrap_replicates: int = 1000,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    raw_labels = np.asarray(true_labels)
    if raw_labels.ndim != 1 or not np.issubdtype(raw_labels.dtype, np.number):
        raise ValueError("Labels must be a one-dimensional numeric array.")
    if not np.isfinite(raw_labels).all() or not np.equal(raw_labels, np.floor(raw_labels)).all():
        raise ValueError("External labels must contain finite integer class indices.")
    labels = raw_labels.astype(np.int64)
    scores = np.asarray(tumor_scores, dtype=np.float64)
    patients = np.char.strip(np.asarray(patient_ids, dtype=str))
    if scores.ndim != 1 or patients.ndim != 1:
        raise ValueError("Labels, scores, and patient IDs must be one-dimensional.")
    if len(labels) == 0 or len(labels) != len(scores) or len(labels) != len(patients):
        raise ValueError("Labels, scores, and patient IDs must be non-empty and have equal lengths.")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Tumor scores must be finite values in the [0, 1] interval.")
    if np.any(np.char.str_len(patients) == 0):
        raise ValueError("Every external validation row must have a patient ID.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in the [0, 1] interval.")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer.")
    if not set(np.unique(labels)).issubset(set(CLASS_TO_IDX.values())):
        raise ValueError("External labels must use 0=no_tumor and 1=tumor.")

    patient_indices = {
        patient: np.flatnonzero(patients == patient)
        for patient in np.unique(patients)
    }
    y_pred = (scores >= threshold).astype(np.int64)
    result: dict[str, Any] = {
        "threshold": threshold,
        "sample_count": int(len(labels)),
        "patient_count": int(len(patient_indices)),
        "evaluation_unit": "2D image slice; confidence intervals resample whole patients.",
        "slice_counts_by_true_class": {
            IDX_TO_CLASS[label]: int(np.count_nonzero(labels == label))
            for label in IDX_TO_CLASS
        },
        "patient_counts_by_slice_class": {
            IDX_TO_CLASS[label]: int(
                sum(np.any(labels[index] == label) for index in patient_indices.values())
            )
            for label in IDX_TO_CLASS
        },
        "threshold_selection": "Fixed at 0.5; external labels are not used to tune the threshold.",
        "degenerate_predictions": bool(len(np.unique(y_pred)) == 1),
    }

    intervals: dict[str, dict[str, float]] = {}
    rng = np.random.default_rng(seed)
    if np.all(labels == CLASS_TO_IDX["tumor"]):
        sensitivities: list[float] = []
        for _ in range(bootstrap_replicates):
            sampled_patients = rng.choice(list(patient_indices), size=len(patient_indices), replace=True)
            sampled_indices = np.concatenate([patient_indices[patient] for patient in sampled_patients])
            sensitivities.append(float(np.mean(y_pred[sampled_indices] == 1)))
        result.update(
            {
                "external_validation_scope": "Tumor-positive cases only; no negative class is present.",
                "tumor_sensitivity_recall": float(np.mean(y_pred == 1)),
                "mean_tumor_score": float(np.mean(scores)),
                "metrics_not_estimable": ["specificity", "accuracy", "balanced_accuracy", "roc_auc", "average_precision"],
            }
        )
        intervals["tumor_sensitivity_recall"] = {
            "lower_95": float(np.quantile(sensitivities, 0.025)),
            "upper_95": float(np.quantile(sensitivities, 0.975)),
        }
    elif set(np.unique(labels)) == set(CLASS_TO_IDX.values()):
        result["external_validation_scope"] = (
            "Slice-level binary classification; negative BraTS slices are tumor-free slices, "
            "not healthy participants."
        )
        result.update(compute_metrics(labels, scores, threshold))

        strata: dict[tuple[bool, bool], list[str]] = {}
        for patient, indices in patient_indices.items():
            class_pattern = (
                bool(np.any(labels[indices] == CLASS_TO_IDX["no_tumor"])),
                bool(np.any(labels[indices] == CLASS_TO_IDX["tumor"])),
            )
            strata.setdefault(class_pattern, []).append(patient)
        for class_pattern in strata:
            if not any(class_pattern):
                raise ValueError(f"Invalid class pattern for patient stratum: {class_pattern}")

        metric_samples: dict[str, list[float]] = {
            name: []
            for name in (
                "accuracy",
                "balanced_accuracy",
                "tumor_sensitivity_recall",
                "no_tumor_specificity",
                "tumor_precision",
                "tumor_f1",
                "roc_auc",
                "average_precision",
            )
        }
        for _ in range(bootstrap_replicates):
            sampled_patients: list[str] = []
            for patients_in_stratum in strata.values():
                sampled_patients.extend(
                    rng.choice(patients_in_stratum, size=len(patients_in_stratum), replace=True).tolist()
                )
            sampled_indices = np.concatenate(
                [patient_indices[patient] for patient in sampled_patients]
            )
            sampled_metrics = compute_metrics(labels[sampled_indices], scores[sampled_indices], threshold)
            for name in metric_samples:
                metric_samples[name].append(float(sampled_metrics[name]))
        intervals = {
            name: {
                "lower_95": float(np.quantile(values, 0.025)),
                "upper_95": float(np.quantile(values, 0.975)),
            }
            for name, values in metric_samples.items()
        }
    else:
        raise ValueError("External validation labels must contain tumor cases.")

    result["confidence_intervals_patient_cluster_bootstrap_95"] = intervals
    return result, intervals


class _ExternalImageDataset(Dataset[torch.Tensor]):
    def __init__(self, rows: pd.DataFrame, image_root: Path) -> None:
        self.rows = rows.reset_index(drop=True)
        self.image_root = image_root.resolve()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        relative_path = Path(str(self.rows.iloc[index]["source_file"]))
        image_path = (self.image_root / relative_path).resolve()
        try:
            image_path.relative_to(self.image_root)
        except ValueError as exc:
            raise ValueError(f"Image path escapes external data root: {relative_path}") from exc
        try:
            with Image.open(image_path) as image:
                return preprocess_image(image)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not read external validation image: {image_path}") from exc


def _predict_cheng_archive(
    model: torch.nn.Module,
    rows: pd.DataFrame,
    archive_path: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    predictions = np.empty(len(rows), dtype=np.float64)
    with zipfile.ZipFile(archive_path) as outer_archive:
        for part_name, part_rows in rows.groupby(
            rows["source_file"].map(
                lambda value: str(value).removeprefix("archive://").split("!", 1)[0]
            ),
            sort=False,
        ):
            if part_name not in CHENG_PART_MD5:
                raise ValueError(f"Unexpected Cheng archive part in manifest: {part_name}")
            part_bytes = outer_archive.read(part_name)
            if hashlib.md5(part_bytes).hexdigest() != CHENG_PART_MD5[part_name]:
                raise ValueError(f"Cheng Figshare archive integrity check failed for {part_name}.")
            with zipfile.ZipFile(io.BytesIO(part_bytes)) as part_archive:
                batch_tensors: list[torch.Tensor] = []
                batch_rows: list[int] = []
                for row_index, row in part_rows.iterrows():
                    source_uri = str(row["source_file"])
                    if not source_uri.startswith("archive://"):
                        raise ValueError(f"Invalid Cheng archive URI in manifest: {source_uri}")
                    uri_part, separator, mat_name = source_uri[len("archive://") :].partition("!")
                    if not separator or uri_part != part_name or not mat_name.lower().endswith(".mat"):
                        raise ValueError(f"Invalid Cheng archive URI in manifest: {source_uri}")
                    image_array, _, _ = _decode_cheng_mat(
                        part_archive.read(mat_name),
                        f"{archive_path.name}!{part_name}!{mat_name}",
                    )
                    lower = float(image_array.min())
                    upper = float(image_array.max())
                    pixels = np.rint((image_array.astype(np.float64) - lower) * (255.0 / (upper - lower)))
                    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="L")
                    batch_tensors.append(preprocess_image(image))
                    batch_rows.append(int(row_index))
                    if len(batch_tensors) == batch_size:
                        _predict_external_batch(model, batch_tensors, batch_rows, predictions, device)
                        batch_tensors.clear()
                        batch_rows.clear()
                if batch_tensors:
                    _predict_external_batch(model, batch_tensors, batch_rows, predictions, device)
    if not np.isfinite(predictions).all():
        raise RuntimeError("External Cheng predictions are incomplete or non-finite.")
    return predictions


def _predict_external_batch(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    row_indices: list[int],
    predictions: np.ndarray,
    device: torch.device,
) -> None:
    with torch.inference_mode():
        logits = model(torch.stack(images).to(device))
        if logits.ndim != 2 or logits.shape[1] != len(CLASS_TO_IDX):
            raise RuntimeError("External checkpoint did not return two binary-class logits.")
        batch_scores = torch.softmax(logits, dim=1)[:, CLASS_TO_IDX["tumor"]].cpu().numpy()
    predictions[np.asarray(row_indices, dtype=np.int64)] = batch_scores


def evaluate_external_manifest(
    manifest_path: Path,
    checkpoint_path: Path,
    results_dir: Path,
    image_root: Path | None = None,
    dataset_name: str | None = None,
    threshold: float = 0.5,
    batch_size: int = 32,
    bootstrap_replicates: int = 1000,
    overlap_audit_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"External validation manifest not found: {manifest_path}")
    rows = pd.read_csv(manifest_path, dtype={"patient_id": str})
    required = {"source_file", "patient_id", "label_idx"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"External manifest is missing required columns: {sorted(missing)}")
    if rows.empty or rows["source_file"].isna().any() or rows["patient_id"].isna().any():
        raise ValueError("External manifest must contain rows with source_file and patient_id values.")
    rows["patient_id"] = rows["patient_id"].astype(str).str.strip()
    if rows["patient_id"].str.len().eq(0).any():
        raise ValueError("External manifest patient IDs must be non-empty.")
    if rows["source_file"].duplicated().any():
        raise ValueError("External manifest contains duplicate source_file rows.")
    numeric_labels = pd.to_numeric(rows["label_idx"], errors="raise")
    if not np.equal(numeric_labels, np.floor(numeric_labels)).all():
        raise ValueError("External label_idx values must be integers.")
    rows["label_idx"] = numeric_labels.astype(np.int64)
    if not set(rows["label_idx"].unique()).issubset(set(CLASS_TO_IDX.values())):
        raise ValueError("External label_idx must use 0=no_tumor and 1=tumor.")
    if CLASS_TO_IDX["tumor"] not in set(rows["label_idx"].unique()):
        raise ValueError("External validation manifest contains no tumor-positive examples.")
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be in the [0, 1] interval.")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer.")

    overlap_exclusion: dict[str, Any] | None = None
    if overlap_audit_path is not None:
        overlap_audit_path = overlap_audit_path.resolve()
        if not overlap_audit_path.is_file():
            raise FileNotFoundError(f"Overlap audit file not found: {overlap_audit_path}")
        overlap_audit = json.loads(overlap_audit_path.read_text(encoding="utf-8"))
        overlapping_patients = set(overlap_audit.get("overlapping_patient_ids", []))
        if not overlapping_patients:
            overlapping_patients = {
                str(edge["patient_id"])
                for edge in (
                    overlap_audit.get("matches", {}).get("exact_pixel_matches", [])
                    + overlap_audit.get("matches", {}).get("high_similarity_matches", [])
                )
            }
        matched_rows = rows["patient_id"].isin(overlapping_patients)
        overlap_exclusion = {
            "overlap_audit": str(overlap_audit_path),
            "overlap_audit_sha256": sha256_file(overlap_audit_path),
            "excluded_patient_count": int(rows.loc[matched_rows, "patient_id"].nunique()),
            "excluded_slice_count": int(matched_rows.sum()),
            "matched_patient_ids_excluded": sorted(
                set(rows.loc[matched_rows, "patient_id"].astype(str))
            ),
            "excluded_for_any_local_split": ["train", "validation", "test"],
            "remaining_overlap_screened_subset_only": True,
            "patient_independence_proven": False,
        }
        rows = rows.loc[~matched_rows].reset_index(drop=True)
        if rows.empty:
            raise ValueError("Overlap exclusion removed every external validation example.")
        if CLASS_TO_IDX["tumor"] not in set(rows["label_idx"].unique()):
            raise ValueError("Overlap exclusion removed every tumor-positive external example.")

    image_root = (image_root or manifest_path.parent).resolve()
    metadata_path = manifest_path.parent / "dataset_metadata.json"
    dataset_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else None
    )
    model, checkpoint_metadata = load_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    if dataset_metadata and dataset_metadata.get("source_archive"):
        archive_path = Path(dataset_metadata["source_archive"])
        if not archive_path.is_file():
            raise FileNotFoundError(f"Cheng source archive no longer exists: {archive_path}")
        expected_archive_hash = dataset_metadata.get("source_archive_sha256")
        if expected_archive_hash and sha256_file(archive_path) != expected_archive_hash:
            raise ValueError(f"Cheng source archive changed after manifest preparation: {archive_path}")
        tumor_scores = _predict_cheng_archive(model, rows, archive_path, device, batch_size)
    else:
        loader = DataLoader(
            _ExternalImageDataset(rows, image_root),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        scores: list[float] = []
        with torch.inference_mode():
            for images in loader:
                logits = model(images.to(device))
                if logits.ndim != 2 or logits.shape[1] != len(CLASS_TO_IDX):
                    raise RuntimeError("External checkpoint did not return two binary-class logits.")
                scores.extend(torch.softmax(logits, dim=1)[:, CLASS_TO_IDX["tumor"]].cpu().tolist())
        tumor_scores = np.asarray(scores, dtype=np.float64)
    true_labels = rows["label_idx"].to_numpy(dtype=np.int64)
    if len(tumor_scores) != len(rows):
        raise RuntimeError("External prediction count does not match manifest rows.")
    patient_ids = rows["patient_id"].to_numpy(dtype=str)
    metrics, _ = calculate_external_metrics(
        true_labels,
        tumor_scores,
        patient_ids,
        threshold=threshold,
        bootstrap_replicates=bootstrap_replicates,
    )
    if "subtype" in rows and np.all(true_labels == CLASS_TO_IDX["tumor"]):
        subtype_results: dict[str, Any] = {}
        for subtype, subtype_rows in rows.groupby("subtype", sort=True):
            subtype_indices = subtype_rows.index.to_numpy(dtype=np.int64)
            subtype_metrics, _ = calculate_external_metrics(
                true_labels[subtype_indices],
                tumor_scores[subtype_indices],
                patient_ids[subtype_indices],
                threshold=threshold,
                bootstrap_replicates=bootstrap_replicates,
            )
            subtype_results[str(subtype)] = {
                "sample_count": int(len(subtype_rows)),
                "patient_count": int(subtype_rows["patient_id"].nunique()),
                "tumor_sensitivity_recall": subtype_metrics["tumor_sensitivity_recall"],
                "confidence_interval_patient_cluster_bootstrap_95": (
                    subtype_metrics["confidence_intervals_patient_cluster_bootstrap_95"][
                        "tumor_sensitivity_recall"
                    ]
                ),
            }
        metrics["tumor_subtype_sensitivity_patient_cluster_bootstrap_95"] = subtype_results
    name = dataset_name or str(rows["dataset"].iloc[0] if "dataset" in rows else "external_dataset")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")
    if not name:
        raise ValueError("dataset_name must contain at least one letter or number.")
    output_dir = results_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    prediction_frame = rows.copy()
    prediction_frame["true_label"] = [
        IDX_TO_CLASS[index] for index in true_labels
    ]
    prediction_frame["tumor_score"] = tumor_scores
    prediction_frame["no_tumor_score"] = 1.0 - tumor_scores
    prediction_frame["predicted_label"] = [
        IDX_TO_CLASS[index] for index in (tumor_scores >= threshold).astype(np.int64)
    ]
    prediction_frame["threshold"] = threshold
    prediction_frame.to_csv(predictions_path, index=False)

    report = {
        "dataset": name,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_metadata": dataset_metadata,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_metadata": checkpoint_metadata,
        "threshold": threshold,
        "this_evaluation_fits_no_parameters_or_thresholds": True,
        "checkpoint_training_independence_from_external_cohort": (
            "Not verified by this evaluator; review checkpoint and training-data provenance."
        ),
        "software": {
            "torch": str(torch.__version__),
            "device": str(device),
        },
        "metrics": metrics,
        "patient_overlap_exclusion": overlap_exclusion,
        "predictions_file": str(predictions_path),
    }
    _write_json(output_dir / "metrics.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare public external MRI datasets and evaluate a frozen binary checkpoint."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    cheng_parser = commands.add_parser(
        "prepare-cheng",
        help="Convert extracted Cheng Figshare .mat files into a patient-linked manifest.",
    )
    cheng_parser.add_argument("--data-root", type=Path, required=True)
    cheng_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "BrainTumorAI" / "data" / "external" / "cheng",
    )

    brats_parser = commands.add_parser(
        "prepare-brats",
        help="Create FLAIR axial-slice labels from authorized BraTS NIfTI volumes and masks.",
    )
    brats_parser.add_argument("--data-root", type=Path, required=True)
    brats_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "BrainTumorAI" / "data" / "external" / "brats",
    )
    brats_parser.add_argument("--min-brain-pixels", type=int, default=256)

    overlap_parser = commands.add_parser(
        "audit-cheng-overlap",
        help="Screen Cheng images for exact/near duplicates against local train/validation/test.",
    )
    overlap_parser.add_argument("--archive", type=Path, required=True)
    overlap_parser.add_argument(
        "--local-manifest",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "split_manifest.csv",
    )
    overlap_parser.add_argument(
        "--output",
        type=Path,
        default=EXTERNAL_RESULTS_DIR / "cheng_figshare_v8" / "overlap_audit.json",
    )

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="Evaluate a frozen checkpoint on an external validation manifest.",
    )
    evaluate_parser.add_argument("--manifest", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    evaluate_parser.add_argument("--image-root", type=Path)
    evaluate_parser.add_argument("--dataset-name")
    evaluate_parser.add_argument("--results-dir", type=Path, default=EXTERNAL_RESULTS_DIR)
    evaluate_parser.add_argument("--threshold", type=float, default=0.5)
    evaluate_parser.add_argument("--batch-size", type=int, default=32)
    evaluate_parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    evaluate_parser.add_argument(
        "--overlap-audit",
        type=Path,
        help="Exclude every patient with any match from the Cheng/local overlap audit.",
    )

    args = parser.parse_args()
    if args.command == "prepare-cheng":
        print(json.dumps(prepare_cheng_dataset(args.data_root, args.output_dir), indent=2))
    elif args.command == "prepare-brats":
        print(
            json.dumps(
                prepare_brats_dataset(args.data_root, args.output_dir, args.min_brain_pixels),
                indent=2,
            )
        )
    elif args.command == "audit-cheng-overlap":
        report = audit_cheng_overlap(args.archive, args.local_manifest)
        _write_json(args.output, report)
        print(json.dumps(report, indent=2))
    else:
        report = evaluate_external_manifest(
            args.manifest,
            args.checkpoint,
            args.results_dir,
            image_root=args.image_root,
            dataset_name=args.dataset_name,
            threshold=args.threshold,
            batch_size=args.batch_size,
            bootstrap_replicates=args.bootstrap_replicates,
            overlap_audit_path=args.overlap_audit,
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
