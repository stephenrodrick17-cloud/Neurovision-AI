import plotly.graph_objects as go
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional


def _sphere_points(cx, cy, cz, r, n_pts=40):
    u = np.linspace(0, 2 * np.pi, n_pts)
    v = np.linspace(0, np.pi, n_pts)
    x = cx + r * np.outer(np.cos(u), np.sin(v))
    y = cy + r * np.outer(np.sin(u), np.sin(v))
    z = cz + r * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def _base_3d_layout(title: str) -> Dict[str, Any]:
    return {
        "scene": {
            "xaxis": {"visible": False, "showgrid": False, "zeroline": False},
            "yaxis": {"visible": False, "showgrid": False, "zeroline": False},
            "zaxis": {"visible": False, "showgrid": False, "zeroline": False},
            "aspectmode": "data",
            "bgcolor": "rgba(0,0,0,0)",
        },
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": dict(l=0, r=0, b=0, t=40),
        "height": 520,
        "showlegend": True,
        "legend": {
            "bgcolor": "rgba(0,0,0,0.4)",
            "bordercolor": "rgba(37,99,235,0.5)",
            "borderwidth": 1,
            "font": {"color": "white", "size": 11},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.02,
            "xanchor": "center",
            "x": 0.5,
        },
        "title": {
            "text": title,
            "font": {"color": "white", "size": 14, "family": "Space Grotesk"},
            "x": 0.02,
            "xanchor": "left",
        },
    }


def create_3d_brain_model(mri_slice, tumor_mask=None):
    depth = 10
    h, w = mri_slice.shape[:2]
    indices = np.where(mri_slice[..., 0] > 50) if mri_slice.ndim == 3 else np.where(mri_slice > 50)
    x = indices[1][::10]
    y = indices[0][::10]
    z_points = np.random.RandomState(7).randint(0, depth, size=len(x))
    traces = [go.Scatter3d(
        x=x, y=y, z=z_points, mode="markers",
        marker=dict(size=1.8, color="rgba(180,200,220,0.10)", line=dict(width=0)),
        name="Brain Parenchyma", showlegend=True, hoverinfo="skip",
    )]
    skull_x, skull_y, skull_z = _sphere_points(w / 2, h / 2, depth / 2, min(h, w) * 0.48, n_pts=24)
    traces.append(go.Surface(
        x=skull_x, y=skull_y, z=skull_z, opacity=0.06,
        colorscale=[[0, "rgba(100,150,220,0.05)"], [1, "rgba(160,200,255,0.08)"]],
        showscale=False, name="Skull Surface", showlegend=True,
    ))
    if tumor_mask is not None:
        tm = tumor_mask if tumor_mask.ndim == 2 else tumor_mask[..., 0]
        tumor_indices = np.where(tm > 128)
        tx = tumor_indices[1][::3]
        ty = tumor_indices[0][::3]
        tz = np.random.RandomState(3).randint(2, depth - 1, size=len(tx))
        traces.append(go.Scatter3d(
            x=tx, y=ty, z=tz, mode="markers",
            marker=dict(size=3, color="#ef4444", opacity=0.82,
                        line=dict(width=0.4, color="rgba(255,255,255,0.4)")),
            name="Tumor Volume", showlegend=True,
        ))
        if len(tx) > 0:
            cxs, cys, czs = tx.mean(), ty.mean(), tz.mean()
            r = max(3, float(np.percentile(np.sqrt((tx - cxs)**2 + (ty - cys)**2 + (tz - czs)**2), 80)))
            cx, cy, cz = _sphere_points(cxs, cys, czs, r, n_pts=20)
            traces.append(go.Surface(
                x=cx, y=cy, z=cz, opacity=0.25,
                colorscale=[[0, "rgba(239,68,68,0.05)"], [1, "rgba(239,68,68,0.35)"]],
                showscale=False, name="Tumor Isosurface", showlegend=True,
            ))
            entry_pt = [0.05 * w, 0.02 * h, 0.0]
            traces.append(go.Scatter3d(
                x=[entry_pt[0], cxs], y=[entry_pt[1], cys], z=[entry_pt[2], czs],
                mode="lines", line=dict(color="#10b981", width=4),
                name="Quick Surgical Path", showlegend=True,
            ))
    fig = go.Figure(data=traces)
    fig.update_layout(**_base_3d_layout("3D Surgical Planning Visualization"))
    fig.update_layout(scene_camera=dict(eye=dict(x=1.45, y=1.45, z=0.9)))
    return fig


def render_stereotactic_plan_3d(mri_slice,
                                tumor_mask: np.ndarray,
                                entry_mm: Tuple[float, float, float],
                                target_mm: Tuple[float, float, float],
                                critical_structures: List[Any],
                                heatmap: Optional[np.ndarray] = None) -> go.Figure:
    h, w = mri_slice.shape[:2]
    depth = 10
    scale_xy = 0.48
    scale_z = 0.52
    def to_plot_coords(mm):
        x_mm, y_mm, z_mm = mm
        px = w / 2 + (x_mm * scale_xy)
        py = h / 2 - (y_mm * scale_xy)
        pz = depth / 2 + (z_mm * scale_z)
        return px, py, pz
    traces = []
    indices = np.where(mri_slice[..., 0] > 50) if mri_slice.ndim == 3 else np.where(mri_slice > 50)
    bx = indices[1][::10]
    by = indices[0][::10]
    bz = np.random.RandomState(9).randint(0, depth, size=len(bx))
    traces.append(go.Scatter3d(
        x=bx, y=by, z=bz, mode="markers",
        marker=dict(size=1.6, color="rgba(170,195,230,0.09)"),
        name="Brain Parenchyma", hoverinfo="skip",
    ))
    skull_x, skull_y, skull_z = _sphere_points(w / 2, h / 2, depth / 2, min(h, w) * 0.48, n_pts=22)
    traces.append(go.Surface(
        x=skull_x, y=skull_y, z=skull_z, opacity=0.06,
        colorscale=[[0, "rgba(100,160,230,0.04)"], [1, "rgba(180,210,255,0.07)"]],
        showscale=False, name="Skull Surface",
    ))
    if tumor_mask is not None:
        tm = tumor_mask if tumor_mask.ndim == 2 else tumor_mask[..., 0]
        ty_vals, tx_vals = np.where(tm > 128)
        if len(tx_vals) > 0:
            tz = np.random.RandomState(3).randint(2, depth - 1, size=len(tx_vals))
            traces.append(go.Scatter3d(
                x=tx_vals[::3], y=ty_vals[::3], z=tz[::3], mode="markers",
                marker=dict(size=3.2, color="#ef4444", opacity=0.85),
                name="Tumor Volume",
            ))
            cxs, cys, czs = tx_vals.mean(), ty_vals.mean(), tz.mean()
            r = max(2.5, float(np.percentile(np.sqrt((tx_vals - cxs)**2 + (ty_vals - cys)**2 + (tz - czs)**2), 80)))
            cx, cy, cz = _sphere_points(cxs, cys, czs, r, n_pts=22)
            traces.append(go.Surface(
                x=cx, y=cy, z=cz, opacity=0.28,
                colorscale=[[0, "rgba(239,68,68,0.04)"], [1, "rgba(239,68,68,0.35)"]],
                showscale=False, name="Tumor Isosurface",
            ))
    for s in critical_structures:
        name = s.name
        cx0, cy0, cz0 = s.center_xyz
        r0 = s.radius_mm
        px0, py0, pz0 = to_plot_coords((cx0, cy0, cz0))
        r_plot = r0 * scale_xy
        xx, yy, zz = _sphere_points(px0, py0, pz0, r_plot, n_pts=16)
        traces.append(go.Surface(
            x=xx, y=yy, z=zz, opacity=0.10,
            colorscale=[[0, "rgba(251,191,36,0.03)"], [1, "rgba(251,191,36,0.25)"]],
            showscale=False, name=f"Zone: {name}", showlegend=True,
            hovertemplate=f"{name}<br>R={r0:.0f} mm<extra></extra>",
        ))
    e_px, e_py, e_pz = to_plot_coords(entry_mm)
    t_px, t_py, t_pz = to_plot_coords(target_mm)
    traces.append(go.Scatter3d(
        x=[e_px, t_px], y=[e_py, t_py], z=[e_pz, t_pz],
        mode="lines",
        line=dict(color="#10b981", width=6),
        name="Planned Trajectory",
    ))
    ring_u = np.linspace(0, 2 * np.pi, 28)
    ring_r = 6.0
    entry_n = np.array([t_px - e_px, t_py - e_py, t_pz - e_pz], dtype=np.float64)
    entry_n /= max(np.linalg.norm(entry_n), 1e-6)
    if abs(entry_n[2]) < 0.9:
        t1 = np.array([entry_n[1], -entry_n[0], 0.0], dtype=np.float64)
    else:
        t1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t1 /= max(np.linalg.norm(t1), 1e-6)
    t2 = np.cross(entry_n, t1)
    t2 /= max(np.linalg.norm(t2), 1e-6)
    ring_x = e_px + ring_r * np.cos(ring_u) * t1[0] + ring_r * np.sin(ring_u) * t2[0]
    ring_y = e_py + ring_r * np.cos(ring_u) * t1[1] + ring_r * np.sin(ring_u) * t2[1]
    ring_z = e_pz + ring_r * np.cos(ring_u) * t1[2] + ring_r * np.sin(ring_u) * t2[2]
    traces.append(go.Scatter3d(
        x=ring_x, y=ring_y, z=ring_z, mode="lines",
        line=dict(color="#fbbf24", width=4),
        name="Burr-Hole Entry (14 mm)",
    ))
    traces.append(go.Scatter3d(
        x=[e_px], y=[e_py], z=[e_pz], mode="markers",
        marker=dict(size=9, color="#fbbf24", line=dict(width=2, color="white")),
        name="Entry Point",
    ))
    traces.append(go.Scatter3d(
        x=[t_px], y=[t_py], z=[t_pz], mode="markers",
        marker=dict(size=9, color="#22d3ee", symbol="cross", line=dict(width=2, color="white")),
        name="Target Isocenter",
    ))
    fig = go.Figure(data=traces)
    fig.update_layout(**_base_3d_layout("6-DOF STEREOTACTIC ARM · TRAJECTORY PLAN"))
    fig.update_layout(scene_camera=dict(eye=dict(x=1.55, y=1.55, z=0.95)))
    return fig


def render_radiosurgery_plan_3d(mri_slice,
                                tumor_mask: np.ndarray,
                                target_mm: Tuple[float, float, float],
                                beams_plan: Dict[str, Any],
                                isodoses: Dict[str, Any]) -> go.Figure:
    h, w = mri_slice.shape[:2]
    depth = 10
    scale_xy = 0.48
    scale_z = 0.52
    def to_plot(mm):
        x, y, z = mm
        return (w / 2 + x * scale_xy, h / 2 - y * scale_xy, depth / 2 + z * scale_z)
    traces = []
    indices = np.where(mri_slice[..., 0] > 50) if mri_slice.ndim == 3 else np.where(mri_slice > 50)
    bx = indices[1][::10]
    by = indices[0][::10]
    bz = np.random.RandomState(13).randint(0, depth, size=len(bx))
    traces.append(go.Scatter3d(
        x=bx, y=by, z=bz, mode="markers",
        marker=dict(size=1.6, color="rgba(170,195,230,0.08)"),
        name="Brain Parenchyma", hoverinfo="skip",
    ))
    skull_x, skull_y, skull_z = _sphere_points(w / 2, h / 2, depth / 2, min(h, w) * 0.48, n_pts=22)
    traces.append(go.Surface(
        x=skull_x, y=skull_y, z=skull_z, opacity=0.05,
        colorscale=[[0, "rgba(100,160,230,0.04)"], [1, "rgba(180,210,255,0.07)"]],
        showscale=False, name="Skull",
    ))
    if tumor_mask is not None:
        tm = tumor_mask if tumor_mask.ndim == 2 else tumor_mask[..., 0]
        ty_vals, tx_vals = np.where(tm > 128)
        if len(tx_vals) > 0:
            tz = np.random.RandomState(3).randint(2, depth - 1, size=len(tx_vals))
            traces.append(go.Scatter3d(
                x=tx_vals[::3], y=ty_vals[::3], z=tz[::3], mode="markers",
                marker=dict(size=3, color="#ef4444", opacity=0.85),
                name="Tumor PTV",
            ))
    tx, ty, tz = to_plot(target_mm)
    traces.append(go.Scatter3d(
        x=[tx], y=[ty], z=[tz], mode="markers",
        marker=dict(size=10, color="#22d3ee", symbol="x", line=dict(width=2, color="white")),
        name="Isocenter",
    ))
    iso_levels = [(0.95, "#fde047"), (0.8, "#f97316"), (0.6, "#ef4444"), (0.4, "#a855f7"), (0.2, "#3b82f6")]
    for thresh, color in iso_levels:
        r_mm = 18 + (1.0 - thresh) * 70
        r_plot = r_mm * scale_xy
        x, y, z = _sphere_points(tx, ty, tz, r_plot, n_pts=18)
        traces.append(go.Surface(
            x=x, y=y, z=z, opacity=0.08,
            colorscale=[[0, f"{color}00"], [1, f"{color}44"]],
            showscale=False, name=f"Dose {int(thresh*100)}%", showlegend=True,
        ))
    linac_base_dist_mm = 420.0
    linac_base_dist_plot = linac_base_dist_mm * scale_xy
    for b in beams_plan.get("beams", []):
        yaw = np.radians(b["gantry_yaw_deg"])
        pitch = np.radians(b["couch_pitch_deg"])
        src_mm = (np.cos(pitch) * np.cos(yaw) * linac_base_dist_mm,
                  np.cos(pitch) * np.sin(yaw) * linac_base_dist_mm,
                  np.sin(pitch) * linac_base_dist_mm)
        sx, sy, sz = to_plot(src_mm)
        col = "#fb923c" if b["beam_index"] % 2 == 1 else "#f472b6"
        traces.append(go.Scatter3d(
            x=[sx, tx], y=[sy, ty], z=[sz, tz],
            mode="lines", line=dict(color=col, width=3),
            name=f"Beam {b['beam_index']:02d} ({b['gantry_yaw_deg']:.0f}°, {b['couch_pitch_deg']:.0f}°)",
            showlegend=b["beam_index"] <= 5,
        ))
        traces.append(go.Scatter3d(
            x=[sx], y=[sy], z=[sz], mode="markers",
            marker=dict(size=4, color=col), name=f"Linac Head {b['beam_index']:02d}", showlegend=False,
        ))
    fig = go.Figure(data=traces)
    fig.update_layout(**_base_3d_layout(
        f"CYBERKNIFE G4 · {beams_plan.get('n_beams', 9)} NON-COPLANAR BEAMS · {beams_plan.get('beam_energy_mv', 6)} MV"
    ))
    fig.update_layout(scene_camera=dict(eye=dict(x=1.55, y=1.55, z=0.95)))
    return fig


if __name__ == "__main__":
    pass
