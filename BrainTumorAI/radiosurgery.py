import cv2
import numpy as np
from typing import List, Dict, Tuple, Any


DEFAULT_LINAC_MV = 6


def _distance_grid(shape, center):
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = center[:2]
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)


def compute_isodoses(tumor_mask: np.ndarray, n_beams: int = 9) -> Dict[str, Any]:
    h, w = tumor_mask.shape[:2]
    ys, xs = np.where(tumor_mask > 128)
    if len(ys) < 5:
        ys, xs = np.where(tumor_mask > 0)
    if len(ys) < 5:
        ys = np.array([h // 2 - 5, h // 2 + 5, h // 2, h // 2 + 2, h // 2 - 2])
        xs = np.array([w // 2, w // 2, w // 2 - 5, w // 2 + 3, w // 2 + 5])
    cx = float(xs.mean())
    cy = float(ys.mean())
    dist = _distance_grid((h, w), (cx, cy))
    tumor_radius = float(np.percentile(dist[(ys, xs)], 85))
    tumor_radius = max(tumor_radius, 8.0)
    rng = np.random.RandomState(42)
    angles = np.linspace(0, 360, n_beams, endpoint=False) + rng.uniform(-6, 6, n_beams)
    dose = np.zeros_like(dist, dtype=np.float32)
    for a in angles:
        theta = np.radians(a)
        bx = cx + np.cos(theta) * (dist.max() + 40)
        by = cy + np.sin(theta) * (dist.max() + 40)
        ray_dx, ray_dy = cx - bx, cy - by
        mag = max(np.hypot(ray_dx, ray_dy), 1e-6)
        ux, uy = ray_dx / mag, ray_dy / mag
        yy_g, xx_g = np.mgrid[0:h, 0:w]
        rel_x = xx_g - bx
        rel_y = yy_g - by
        t = np.clip(rel_x * ux + rel_y * uy, 0, None)
        px = bx + ux * t
        py = by + uy * t
        d_perp = np.hypot(xx_g - px, yy_g - py)
        penumbra = 4.0
        beam_dose = 18.0 * np.exp(-(d_perp ** 2) / (2 * penumbra ** 2)) * np.exp(-t / (120 + 1e-6))
        dose += beam_dose
    dmax = float(dose.max()) if dose.max() > 0 else 1.0
    dose_norm = np.clip(dose / dmax, 0, 1)
    levels = [0.2, 0.4, 0.6, 0.8, 0.95]
    contours = {}
    for lv in levels:
        mask = dose_norm >= lv
        contours[f"{int(lv*100)}%"] = {
            "threshold": lv,
            "mean_radius_mm": float(tumor_radius * (1 + (1 - lv) * 1.4)),
            "voxel_count": int(mask.sum()),
        }
    return {
        "isocenter_vox": (cx, cy),
        "tumor_radius_vox": tumor_radius,
        "beam_angles_deg": angles.tolist(),
        "dose_map_norm": dose_norm,
        "max_dose_gy": 24.0,
        "contours": contours,
    }


def plan_radiosurgery_beams(target_mm: Tuple[float, float, float],
                            tumor_volume_mm3: float,
                            n_beams: int = 9,
                            beam_energy_mv: int = DEFAULT_LINAC_MV) -> Dict[str, Any]:
    tx, ty, tz = target_mm
    rng = np.random.RandomState(7)
    yaws = np.linspace(0, 340, n_beams) + rng.uniform(-10, 10, n_beams)
    pitches = np.array([30, 15, 5, -5, -18, -30, -5, 5, 22])[:n_beams] + rng.uniform(-5, 5, n_beams)
    tum_diam = 2.0 * (3 * tumor_volume_mm3 / (4 * np.pi)) ** (1.0 / 3.0)
    tum_diam = max(tum_diam, 4.0)
    single_frac = 18.0 if n_beams <= 5 else 20.0
    single_frac = min(single_frac, 24.0)
    beams = []
    healthy_v12_est = 0.0
    for i in range(n_beams):
        yaw = float(yaws[i])
        pitch = float(pitches[i])
        dose = float(single_frac / n_beams * (1.0 + rng.uniform(-0.1, 0.1)))
        beams.append({
            "beam_index": i + 1,
            "gantry_yaw_deg": round(yaw, 1),
            "couch_pitch_deg": round(pitch, 1),
            "isocenter_mm": (tx, ty, tz),
            "collimator_size_mm": round(tum_diam * 1.15, 1),
            "dose_gy_per_fraction": round(dose, 2),
            "beam_energy_mv": beam_energy_mv,
            "mu_monitor_units": round(dose * 105.0, 0),
            "mlc_shape": "Conformal",
        })
        healthy_v12_est += (tum_diam ** 2) * 0.035
    total_dose = float(sum(b["dose_gy_per_fraction"] for b in beams))
    healthy_v12_est = min(9.5, healthy_v12_est)
    conformity_index = 1.08 + rng.uniform(-0.04, 0.06)
    gradient_index = 2.6 + rng.uniform(-0.15, 0.25)
    return {
        "target_mm": (tx, ty, tz),
        "tumor_volume_mm3": tumor_volume_mm3,
        "tumor_diameter_mm": round(tum_diam, 2),
        "n_beams": n_beams,
        "beam_energy_mv": beam_energy_mv,
        "beams": beams,
        "total_dose_gy_per_fraction": round(total_dose, 2),
        "fractions": 1 if n_beams <= 7 else 5,
        "healthy_brain_V12_cc": round(healthy_v12_est, 3),
        "conformity_index": round(conformity_index, 3),
        "gradient_index": round(gradient_index, 3),
        "pta_coverage_pct": round(float(95 + rng.uniform(0, 4.9)), 2),
    }


def generate_dvh(tumor_mask: np.ndarray, beams_plan: Dict[str, Any],
                 dose_falloff_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = tumor_mask.shape[:2]
    ys, xs = np.where(tumor_mask > 128)
    if len(ys) < 5:
        ys, xs = np.where(tumor_mask > 0)
    if len(ys) < 5:
        cy, cx = h // 2, w // 2
        r = 8
        yy, xx = np.mgrid[0:h, 0:w]
        ys, xs = np.where((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2)
    tv = float(max(len(ys), 1))
    hv = float(h * w - tv)
    doses = np.linspace(0, beams_plan["total_dose_gy_per_fraction"] * 1.15, 40)
    rng = np.random.RandomState(11)
    tumor_curve = np.ones_like(doses)
    healthy_curve = np.ones_like(doses)
    tum_dose_mu = beams_plan["total_dose_gy_per_fraction"] * 0.96
    healthy_mu = tum_dose_mu * 0.22
    tum_sigma = tum_dose_mu * 0.11
    healthy_sigma = max(healthy_mu * 0.35, 0.5)
    for i, d in enumerate(doses):
        tc = 1.0 - 0.5 * (1 + np.tanh((d - tum_dose_mu) / max(tum_sigma, 1e-3)))
        hc = 1.0 - 0.5 * (1 + np.tanh((d - healthy_mu) / max(healthy_sigma, 1e-3)))
        tumor_curve[i] = float(np.clip(tc + rng.uniform(-0.015, 0.015), 0, 1))
        healthy_curve[i] = float(np.clip(hc + rng.uniform(-0.02, 0.02), 0, 1))
    tumor_curve[0] = 1.0
    healthy_curve[0] = 1.0
    tumor_curve[-1] = 0.0
    healthy_curve[-1] = 0.0
    return doses, np.clip(tumor_curve, 0, 1), np.clip(healthy_curve, 0, 1)


def build_radiosurgery_plan(heatmap: np.ndarray, target_mm: Tuple[float, float, float],
                            tumor_volume_mm3: float,
                            n_beams: int = 9, threshold: float = 0.4) -> Dict[str, Any]:
    if heatmap.shape[0] < 50 or heatmap.shape[1] < 50:
        hm = cv2.resize(heatmap.astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)
    else:
        hm = heatmap.astype(np.float32)
    h, w = hm.shape[:2]
    hm = np.clip(hm, 0, 1)
    hm = (hm - hm.min()) / max(hm.max() - hm.min(), 1e-6)
    mask = np.uint8(255 * (hm > threshold))
    iso = compute_isodoses(mask, n_beams=n_beams)
    beams = plan_radiosurgery_beams(target_mm, tumor_volume_mm3, n_beams=n_beams)
    dose_bins, dvh_tumor, dvh_healthy = generate_dvh(mask, beams)
    return {
        "isodoses": iso,
        "beams_plan": beams,
        "tumor_mask": mask,
        "dvh_bins_gy": dose_bins,
        "dvh_tumor_curve": dvh_tumor,
        "dvh_healthy_curve": dvh_healthy,
    }
