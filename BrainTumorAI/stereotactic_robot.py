import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any


@dataclass
class CriticalStructure:
    name: str
    center_xyz: Tuple[float, float, float]
    radius_mm: float
    safety_penalty: float = 12.0


SKULL_RADIUS_MM = 92.0
BRAIN_HALF_DEPTH_MM = 70.0
VOXEL_SIZE_MM = 1.0


def _voxel_to_mm(px: int, py: int, pz: int, shape=(224, 224, 10)) -> Tuple[float, float, float]:
    h, w, d = shape
    x_mm = (px - w / 2.0) * VOXEL_SIZE_MM
    y_mm = -(py - h / 2.0) * VOXEL_SIZE_MM
    z_mm = (pz - d / 2.0) * VOXEL_SIZE_MM * 8.0
    return x_mm, y_mm, z_mm


def _mm_to_voxel(x_mm: float, y_mm: float, z_mm: float, shape=(224, 224, 10)) -> Tuple[int, int, int]:
    h, w, d = shape
    px = int(round(x_mm / VOXEL_SIZE_MM + w / 2.0))
    py = int(round(-y_mm / VOXEL_SIZE_MM + h / 2.0))
    pz = int(round(z_mm / (8.0 * VOXEL_SIZE_MM) + d / 2.0))
    return px, py, pz


def compute_tumor_centroid(heatmap: np.ndarray, mri_shape=(224, 224), threshold=0.4) -> Dict[str, Any]:
    h, w = mri_shape[:2]
    hm = np.clip(heatmap.astype(np.float32), 0, 1)
    if hm.shape[:2] != (h, w):
        hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
    hm = (hm - hm.min()) / max(hm.max() - hm.min(), 1e-6)
    hm_thr = np.where(hm > threshold, hm, 0.0)
    if hm_thr.sum() < 1.0:
        hm_thr = hm

    yy, xx = np.mgrid[0:h, 0:w]
    total = hm_thr.sum() + 1e-8
    cx_px = (xx * hm_thr).sum() / total
    cy_px = (yy * hm_thr).sum() / total
    cz_px = 5.0 + hm_thr[int(cy_px), int(cx_px)] * 4.0 - 2.0

    voxels_over_thr = int((hm > threshold).sum())
    volume_mm3 = voxels_over_thr * (VOXEL_SIZE_MM ** 2) * 6.0
    volume_mm3 = max(volume_mm3, 30.0)

    x_mm, y_mm, z_mm = _voxel_to_mm(int(cx_px), int(cy_px), int(cz_px), (h, w, 10))
    side = "Left" if x_mm < 0 else "Right"
    lobe = "Frontal" if y_mm > 15 else ("Parietal" if y_mm > -15 else "Occipital")

    return {
        "centroid_vox": (float(cx_px), float(cy_px), float(cz_px)),
        "centroid_mm": (x_mm, y_mm, z_mm),
        "volume_mm3": float(volume_mm3),
        "peak_activation": float(hm.max()),
        "side": side,
        "lobe": lobe,
        "heatmap_thresholded": hm_thr,
    }


def compute_burr_hole_entry(target_mm: Tuple[float, float, float]) -> Tuple[float, float, float]:
    tx, ty, tz = target_mm
    direction = np.array([tx, ty, tz], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([30.0, 40.0, 10.0])
        norm = np.linalg.norm(direction)
    direction /= norm
    offset = direction * (SKULL_RADIUS_MM + 4.0)
    entry = offset
    entry[2] = max(entry[2], 18.0)
    return float(entry[0]), float(entry[1]), float(entry[2])


def plan_trajectory(entry_mm: Tuple[float, float, float],
                    target_mm: Tuple[float, float, float]) -> Dict[str, Any]:
    e = np.array(entry_mm, dtype=np.float64)
    t = np.array(target_mm, dtype=np.float64)
    vec = t - e
    depth_mm = float(np.linalg.norm(vec))
    unit = vec / max(depth_mm, 1e-6)
    pitch_deg = float(np.degrees(np.arcsin(max(-1.0, min(1.0, unit[2])))))
    yaw_deg = float(np.degrees(np.arctan2(unit[1], unit[0])))
    roll_deg = 0.0
    approach = np.arctan2(np.linalg.norm(unit[:2]), abs(unit[2]))
    orthogonal_penalty = max(0.0, (approach - 1.2)) * 20.0
    return {
        "entry_mm": tuple(e.tolist()),
        "target_mm": tuple(t.tolist()),
        "unit_vector": tuple(unit.tolist()),
        "depth_mm": round(depth_mm, 2),
        "pitch_deg": round(pitch_deg, 2),
        "yaw_deg": round(yaw_deg, 2),
        "roll_deg": round(roll_deg, 2),
        "approach_angle_rad": round(approach, 3),
        "orthogonal_penalty": round(float(orthogonal_penalty), 2),
    }


def generate_critical_structures(shape=(224, 224, 10)) -> List[CriticalStructure]:
    h, w, d = shape
    def c(px, py, pz, r_mm, nm, p):
        x, y, z = _voxel_to_mm(px, py, pz, shape)
        return CriticalStructure(name=nm, center_xyz=(x, y, z), radius_mm=r_mm, safety_penalty=p)
    return [
        c(w // 2, h // 2 - 30, d // 2, 18, "Lateral Ventricles", 18.0),
        c(w // 2, h // 2 - 10, d // 2, 12, "Thalamus", 20.0),
        c(w // 2, h // 2 + 45, d // 2, 22, "Cerebellum", 15.0),
        c(w // 2 - 38, h // 2 + 8, d // 2, 10, "Internal Capsule (L)", 22.0),
        c(w // 2 + 38, h // 2 + 8, d // 2, 10, "Internal Capsule (R)", 22.0),
        c(w // 2, h // 2 + 55, d // 2 - 4, 10, "Brainstem", 30.0),
        c(w // 2, h // 2 - 55, d // 2, 16, "Motor Cortex", 16.0),
        c(w // 2, h // 2 + 70, d // 2, 16, "Visual Cortex (V1)", 16.0),
        c(w // 2, h // 2, d // 2 + 32, 14, "Sagittal Sinus Zone", 20.0),
    ]


def _segment_distance_mm(p1, p2, c) -> float:
    seg = np.array(p2, dtype=np.float64) - np.array(p1, dtype=np.float64)
    cp = np.array(c.center_xyz, dtype=np.float64) - np.array(p1, dtype=np.float64)
    t = np.dot(cp, seg) / max(np.dot(seg, seg), 1e-8)
    t = max(0.0, min(1.0, t))
    closest = np.array(p1, dtype=np.float64) + t * seg
    return float(np.linalg.norm(closest - np.array(c.center_xyz)))


def check_trajectory_safety(entry_mm: Tuple[float, float, float],
                            target_mm: Tuple[float, float, float],
                            structures: List[CriticalStructure]) -> Tuple[float, List[str]]:
    warnings = []
    score = 100.0
    for s in structures:
        d = _segment_distance_mm(entry_mm, target_mm, s)
        if d < s.radius_mm:
            score -= s.safety_penalty
            warnings.append(f"{s.name}: PENETRATION RISK ({d:.1f} / {s.radius_mm:.1f} mm)")
        elif d < s.radius_mm * 1.6:
            near = s.safety_penalty * 0.4 * (1.0 - (d - s.radius_mm) / max(0.6 * s.radius_mm, 1e-6))
            score -= near
            warnings.append(f"{s.name}: Near miss ({d:.1f} mm)")
    entry_arr = np.array(entry_mm, dtype=np.float64)
    target_arr = np.array(target_mm, dtype=np.float64)
    approach = np.arctan2(np.linalg.norm((target_arr - entry_arr)[:2]), abs((target_arr - entry_arr)[2]))
    if approach > 1.15:
        score -= 8
        warnings.append(f"Shallow approach angle ({np.degrees(approach):.0f} deg) — cortical surface hit risk")
    if np.linalg.norm(target_arr) > 78:
        score -= 5
        warnings.append("Target peripheral — eloquent cortical surface proximity")
    score = max(0.0, min(100.0, score))
    return round(score, 2), warnings


def robotic_arm_ik(entry_mm: Tuple[float, float, float],
                   target_mm: Tuple[float, float, float]) -> Dict[str, Any]:
    e = np.array(entry_mm, dtype=np.float64)
    t = np.array(target_mm, dtype=np.float64)
    vec = t - e
    depth = max(np.linalg.norm(vec), 1e-6)
    u = vec / depth
    base_offset = np.array([0.0, 0.0, SKULL_RADIUS_MM + 85.0])
    shoulder = base_offset + np.array([e[0] * 0.15, e[1] * 0.15, -10.0])
    to_entry = e - shoulder
    to_target = t - shoulder
    j1 = float(np.degrees(np.arctan2(to_entry[1], to_entry[0]))) % 360.0
    if j1 > 180: j1 -= 360.0
    j2 = float(np.degrees(np.arctan2(-to_entry[2], np.linalg.norm(to_entry[:2])))) - 20.0
    j3 = 42.0 - 0.28 * depth + float(np.degrees(np.arctan2(u[1], u[2]))) * 0.6
    j4 = float(np.degrees(np.arctan2(u[1], u[0])))
    j5 = float(np.degrees(np.arcsin(max(-1.0, min(1.0, u[2]))))) + 18.0
    j6 = (depth - 45.0) * 1.2
    j6 = float(j6) % 360.0
    if j6 > 180: j6 -= 360.0
    joints = {
        "J1_Base_Yaw": float(np.clip(j1, -175, 175)),
        "J2_Shoulder_Pitch": float(np.clip(j2, -85, 125)),
        "J3_Elbow_Pitch": float(np.clip(j3, -65, 170)),
        "J4_Wrist_Roll": float(np.clip(j4, -360, 360)),
        "J5_Wrist_Pitch": float(np.clip(j5, -120, 120)),
        "J6_EEF_Spin": float(np.clip(j6, -360, 360)),
    }
    target_eef = t
    actual_eef = t + np.array([np.random.RandomState(0).normal(0, 0.18),
                               np.random.RandomState(1).normal(0, 0.18),
                               np.random.RandomState(2).normal(0, 0.18)])
    traj_error_mm = float(np.clip(np.linalg.norm(actual_eef - target_eef), 0.02, 2.2))
    rpy_err = np.random.RandomState(3).randn(3) * 0.25
    eef_pose = {
        "x_mm": float(target_eef[0]),
        "y_mm": float(target_eef[1]),
        "z_mm": float(target_eef[2]),
        "roll_deg": float(np.degrees(np.arctan2(u[1], u[2]))) + rpy_err[0],
        "pitch_deg": float(np.degrees(np.arcsin(max(-1.0, min(1.0, u[0]))))) + rpy_err[1],
        "yaw_deg": float(np.degrees(np.arctan2(u[1], u[0]))) + rpy_err[2],
    }
    return {
        "joints_deg": joints,
        "eef_pose_mm_deg": eef_pose,
        "trajectory_error_mm": round(traj_error_mm, 3),
        "reach_feasible": (abs(joints["J2_Shoulder_Pitch"]) < 120) and (abs(joints["J3_Elbow_Pitch"]) < 165),
    }


def build_full_plan(heatmap: np.ndarray, mri_shape=(224, 224)) -> Dict[str, Any]:
    centroid = compute_tumor_centroid(heatmap, mri_shape)
    structures = generate_critical_structures()
    entry = compute_burr_hole_entry(centroid["centroid_mm"])
    traj = plan_trajectory(entry, centroid["centroid_mm"])
    safety, warnings = check_trajectory_safety(entry, centroid["centroid_mm"], structures)
    ik = robotic_arm_ik(entry, centroid["centroid_mm"])
    return {
        "centroid": centroid,
        "burr_hole_entry_mm": entry,
        "trajectory": traj,
        "critical_structures": structures,
        "safety_score": safety,
        "safety_warnings": warnings,
        "arm_ik": ik,
    }
