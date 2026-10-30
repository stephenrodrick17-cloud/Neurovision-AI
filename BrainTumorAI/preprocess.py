from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold

from BrainTumorAI.classifier_pipeline import (
    CLASS_TO_IDX,
    DEFAULT_DATA_ROOT,
    DEFAULT_RESULTS_DIR,
    PROJECT_ROOT,
)

SOURCE_LABELS = {"Brain Tumor": "tumor", "Healthy": "no_tumor"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
PHASH_MAX_HAMMING_DISTANCE = 4


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _perceptual_hash(image: Image.Image) -> tuple[str, list[int], np.ndarray]:
    grayscale = np.asarray(image.convert("L"))
    grayscale_64 = cv2.resize(grayscale, (64, 64), interpolation=cv2.INTER_AREA)
    pixels = cv2.resize(grayscale_64.astype(np.float32), (32, 32))
    low_frequencies = cv2.dct(pixels)[:8, :8].flatten()[1:]
    median = float(np.median(low_frequencies))
    bits = [int(value > median) for value in low_frequencies]
    packed = sum(bit << position for position, bit in enumerate(bits))
    return f"{packed:016x}", bits, grayscale_64


def _structural_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_float = first.astype(np.float32)
    second_float = second.astype(np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mean_first = cv2.GaussianBlur(first_float, (11, 11), 1.5)
    mean_second = cv2.GaussianBlur(second_float, (11, 11), 1.5)
    variance_first = cv2.GaussianBlur(first_float * first_float, (11, 11), 1.5) - mean_first * mean_first
    variance_second = cv2.GaussianBlur(second_float * second_float, (11, 11), 1.5) - mean_second * mean_second
    covariance = cv2.GaussianBlur(first_float * second_float, (11, 11), 1.5) - mean_first * mean_second
    numerator = (2 * mean_first * mean_second + c1) * (2 * covariance + c2)
    denominator = (mean_first * mean_first + mean_second * mean_second + c1) * (
        variance_first + variance_second + c2
    )
    return float(np.mean(numerator / denominator))


def _group_near_duplicates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parent = list(range(len(rows)))
    rank = [0] * len(rows)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if rank[first_root] < rank[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        if rank[first_root] == rank[second_root]:
            rank[first_root] += 1

    exact_indices = _duplicate_groups(rows, "pixel_sha256")
    for indices in exact_indices:
        for index in indices[1:]:
            join(indices[0], index)

    bit_lookup = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)
    signatures = np.asarray([int(row["phash"], 16) for row in rows], dtype=np.uint64)
    grayscale_images = [row["_gray_64"] for row in rows]
    candidate_pair_count = 0
    high_similarity_pair_count = 0
    cross_label_candidate_count = 0
    cross_label_high_similarity_count = 0
    for first_index in range(len(rows) - 1):
        differences = np.bitwise_xor(signatures[first_index + 1 :], signatures[first_index])
        hamming_distances = bit_lookup[differences.view(np.uint8).reshape(-1, 8)].sum(axis=1)
        candidate_offsets = np.flatnonzero(hamming_distances <= PHASH_MAX_HAMMING_DISTANCE)
        for offset in candidate_offsets:
            second_index = first_index + 1 + int(offset)
            candidate_pair_count += 1
            if rows[first_index]["label"] != rows[second_index]["label"]:
                cross_label_candidate_count += 1
            similarity = _structural_similarity(
                grayscale_images[first_index],
                grayscale_images[second_index],
            )
            if similarity >= 0.95:
                high_similarity_pair_count += 1
                join(first_index, second_index)
                if rows[first_index]["label"] != rows[second_index]["label"]:
                    cross_label_high_similarity_count += 1

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)
    for indices in components.values():
        component_key = "\n".join(sorted(rows[index]["source_file"] for index in indices))
        group_id = hashlib.sha256(component_key.encode("utf-8")).hexdigest()
        for index in indices:
            rows[index]["duplicate_group"] = group_id
    multiple_sample_components = [indices for indices in components.values() if len(indices) > 1]
    return {
        "group_count": len(components),
        "multi_sample_group_count": len(multiple_sample_components),
        "samples_in_multi_sample_groups": sum(len(indices) for indices in multiple_sample_components),
        "coarse_phash_candidate_pairs_hamming_0_to_4": candidate_pair_count,
        "coarse_phash_candidate_pairs_cross_label_hamming_0_to_4": cross_label_candidate_count,
        "high_similarity_edges_phash_le_4_ssim_ge_0_95": high_similarity_pair_count,
        "high_similarity_cross_label_edges": cross_label_high_similarity_count,
        "near_duplicate_grouping": "Connected components of pHash Hamming distance <=4 and grayscale SSIM >=0.95; exact decoded-pixel matches are always grouped.",
    }


def _inspect_image(path: Path, source_label: str) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            original_mode = image.mode
            image_format = image.format
            width, height = image.size
            image.verify()
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            pixel_hash = _sha256_bytes(
                f"{rgb.width}x{rgb.height}".encode("ascii") + rgb.tobytes()
            )
            phash, phash_bits, gray_64 = _perceptual_hash(rgb)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        return {
            "source_file": path.relative_to(PROJECT_ROOT).as_posix(),
            "source_label": source_label,
            "label": SOURCE_LABELS[source_label],
            "status": "unreadable",
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "source_file": path.relative_to(PROJECT_ROOT).as_posix(),
        "source_label": source_label,
        "label": SOURCE_LABELS[source_label],
        "label_idx": CLASS_TO_IDX[SOURCE_LABELS[source_label]],
        "status": "readable",
        "format": image_format,
        "mode": original_mode,
        "width": width,
        "height": height,
        "file_sha256": _sha256_file(path),
        "pixel_sha256": pixel_hash,
        "phash": phash,
        "_phash_bits": phash_bits,
        "_gray_64": gray_64,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _duplicate_groups(rows: list[dict[str, Any]], hash_key: str) -> list[list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row[hash_key]].append(index)
    return [indices for indices in grouped.values() if len(indices) > 1]


def _legacy_split_overlap() -> dict[str, Any]:
    data_root = Path(__file__).resolve().parent / "dataset"
    split_hashes: dict[str, dict[str, list[str]]] = {}
    for split in ("train", "test"):
        entries: dict[str, list[str]] = defaultdict(list)
        for path in sorted((data_root / split).rglob("*")):
            if not path.is_file():
                continue
            try:
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    key = _sha256_bytes(
                        f"{rgb.width}x{rgb.height}".encode("ascii") + rgb.tobytes()
                    )
            except (OSError, ValueError, UnidentifiedImageError):
                continue
            entries[key].append(path.relative_to(data_root).as_posix())
        split_hashes[split] = entries

    shared = set(split_hashes["train"]) & set(split_hashes["test"])
    return {
        "cross_split_exact_pixel_hash_groups": len(shared),
        "train_samples_in_shared_groups": sum(len(split_hashes["train"][key]) for key in shared),
        "test_samples_in_shared_groups": sum(len(split_hashes["test"][key]) for key in shared),
    }


def _assign_splits(rows: list[dict[str, Any]], seed: int) -> dict[int, str]:
    if len(rows) < 20:
        raise ValueError("At least 20 readable images are required to create the three splits.")
    labels = np.asarray([row["label_idx"] for row in rows], dtype=np.int64)
    groups = np.asarray([row.get("duplicate_group", row["pixel_sha256"]) for row in rows], dtype=object)
    indices = np.arange(len(rows))

    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    train_validation, test = next(outer.split(indices, labels, groups))
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    train_local, validation_local = next(
        inner.split(
            train_validation,
            labels[train_validation],
            groups[train_validation],
        )
    )
    train = train_validation[train_local]
    validation = train_validation[validation_local]

    assignments = {
        **{int(index): "train" for index in train},
        **{int(index): "validation" for index in validation},
        **{int(index): "test" for index in test},
    }
    if len(assignments) != len(rows):
        raise RuntimeError("Some readable images were not assigned to a split.")
    split_groups: dict[str, set[str]] = {
        split: {
            rows[index].get("duplicate_group", rows[index]["pixel_sha256"])
            for index, assigned in assignments.items()
            if assigned == split
        }
        for split in ("train", "validation", "test")
    }
    if (
        split_groups["train"] & split_groups["validation"]
        or split_groups["train"] & split_groups["test"]
        or split_groups["validation"] & split_groups["test"]
    ):
        raise RuntimeError("Exact or high-similarity duplicate groups crossed split boundaries.")
    for split in ("train", "validation", "test"):
        split_labels = {rows[index]["label_idx"] for index, assigned in assignments.items() if assigned == split}
        if split_labels != set(CLASS_TO_IDX.values()):
            raise RuntimeError(f"The {split} split does not contain both classes.")
    return assignments


def audit_and_split(data_root: Path = DEFAULT_DATA_ROOT, results_dir: Path = DEFAULT_RESULTS_DIR, seed: int = 42) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing_directories = [
        str(data_root / source_label)
        for source_label in SOURCE_LABELS
        if not (data_root / source_label).is_dir()
    ]
    if missing_directories:
        raise FileNotFoundError(
            "Expected source class directories were not found: " + ", ".join(missing_directories)
        )

    for source_label in SOURCE_LABELS:
        source_dir = data_root / source_label
        for path in sorted(source_dir.iterdir(), key=lambda item: item.name.casefold()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                rows.append(_inspect_image(path, source_label))

    readable = [row for row in rows if row["status"] == "readable"]
    unreadable = [row for row in rows if row["status"] != "readable"]
    if not readable:
        raise ValueError(f"No readable images found under {data_root}.")

    duplicate_summary = _group_near_duplicates(readable)
    assignments = _assign_splits(readable, seed)
    for index, row in enumerate(readable):
        row["split"] = assignments[index]

    class_counts = Counter(row["label"] for row in readable)
    split_counts = {
        split: {
            label: sum(row["split"] == split and row["label"] == label for row in readable)
            for label in CLASS_TO_IDX
        }
        for split in ("train", "validation", "test")
    }
    exact_groups = _duplicate_groups(readable, "pixel_sha256")
    cross_label_exact_groups = [
        indices
        for indices in exact_groups
        if len({readable[index]["label"] for index in indices}) > 1
    ]
    for row in rows:
        row.pop("_phash_bits", None)
        row.pop("_gray_64", None)
    for row in unreadable:
        row.setdefault("split", "")
        row.setdefault("duplicate_group", "")
        row.setdefault("file_sha256", "")
        row.setdefault("pixel_sha256", "")
        row.setdefault("phash", "")

    dimensions = [(row["width"], row["height"]) for row in readable]
    report = {
        "dataset": {
            "directory": str(data_root),
            "source_class_mapping": SOURCE_LABELS,
            "model_class_to_idx": CLASS_TO_IDX,
            "source": "Not verifiable from files in the repository; obtain original citation and license before research redistribution.",
            "version": "Not present in local metadata.",
            "license": "Not present in local metadata.",
            "patient_or_volume_identifiers": "None found in the source directory or local metadata; patient-independent evaluation is not possible.",
            "raw_images_readable": len(readable),
            "raw_images_unreadable": len(unreadable),
            "class_counts": dict(class_counts),
            "format_counts": dict(Counter(row["format"] for row in readable)),
            "mode_counts": dict(Counter(row["mode"] for row in readable)),
            "unique_dimensions": len(set(dimensions)),
            "dimension_range": {
                "width_min": min(width for width, _ in dimensions),
                "width_max": max(width for width, _ in dimensions),
                "height_min": min(height for _, height in dimensions),
                "height_max": max(height for _, height in dimensions),
            },
            "exact_decoded_pixel_duplicate_groups": len(exact_groups),
            "samples_in_exact_duplicate_groups": sum(len(group) for group in exact_groups),
            "exact_duplicate_groups_with_conflicting_labels": len(cross_label_exact_groups),
            **duplicate_summary,
            "perceptual_hash_note": (
                "Coarse pHash matches are only candidates. Candidate pairs are grouped only when their "
                "64x64 grayscale SSIM is at least 0.95; review them manually before making clinical/data "
                "identity claims. No patient grouping is possible."
            ),
            "unreadable_files": unreadable,
            "split_method": (
                "Seeded StratifiedGroupKFold partitions with exact-pixel and high-SSIM near-duplicate "
                "components as groups; image-level groups only because patient identifiers are unavailable."
            ),
            "split_seed": seed,
            "split_counts": split_counts,
            "legacy_prepared_split_exact_pixel_overlap": _legacy_split_overlap(),
        },
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = results_dir / "split_manifest.csv"
    fields = [
        "source_file",
        "source_label",
        "label",
        "label_idx",
        "split",
        "file_sha256",
        "pixel_sha256",
        "duplicate_group",
        "phash",
        "format",
        "mode",
        "width",
        "height",
        "status",
        "error",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report["dataset"]["manifest"] = str(manifest_path)
    report_path = results_dir / "dataset_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the local brain-image dataset and create grouped splits.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = audit_and_split(args.data_root, args.results_dir, args.seed)
    print(json.dumps(report["dataset"], indent=2))


if __name__ == "__main__":
    main()
