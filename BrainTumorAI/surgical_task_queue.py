import uuid
import time
from copy import deepcopy
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


VALID_STATUSES = ["QUEUED", "PLANNING", "IN_PROGRESS", "COMPLETED", "CANCELLED"]
STATUS_FLOW = {
    "QUEUED": "PLANNING",
    "PLANNING": "IN_PROGRESS",
    "IN_PROGRESS": "COMPLETED",
    "COMPLETED": None,
    "CANCELLED": None,
}
PRIORITY_COLORS = {"HIGH": "#ef4444", "MED": "#f59e0b", "LOW": "#10b981"}
STATUS_COLORS = {
    "QUEUED": "#60a5fa",
    "PLANNING": "#8b5cf6",
    "IN_PROGRESS": "#f59e0b",
    "COMPLETED": "#10b981",
    "CANCELLED": "#ef4444",
}
ROBOTS = {
    "stereotactic_biopsy": "ROSA-One Brain",
    "resection": "Da Vinci Xi Neurosurgical",
    "srs": "CyberKnife G4 (SRS)",
    "laser_abl": "NeuroBlate System",
}
PROCEDURES = [
    "Stereotactic Biopsy",
    "Tumor Resection",
    "Radiosurgery (SRS)",
    "Laser Interstitial Ablation",
]


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(action: str, user: str = "NeuroVision-AI") -> Dict[str, str]:
    return {"timestamp": _now_iso(), "action": action, "user": user}


def default_queue() -> List[Dict[str, Any]]:
    base_date = datetime.now()
    seed = [
        {
            "patient_id": "P-001",
            "patient_name": "Aarav Sharma",
            "age": 58,
            "procedure_type": "Radiosurgery (SRS)",
            "assigned_robot": ROBOTS["srs"],
            "priority": "HIGH",
            "tumor_volume_mm3": 4320.0,
            "mni_coords_mm": (28, -14, 42),
            "lobe": "Right Parietal",
            "planned_date": (base_date + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"),
            "surgeon": "Dr. Kapoor",
            "endoscope_channel": "CH-03 (Karl Storz)",
            "overhead_camera_id": "OR-02-CAM-A",
            "n_beams": 9,
            "beam_energy_mv": 6,
        },
        {
            "patient_id": "P-002",
            "patient_name": "Priya Nair",
            "age": 44,
            "procedure_type": "Stereotactic Biopsy",
            "assigned_robot": ROBOTS["stereotactic_biopsy"],
            "priority": "MED",
            "tumor_volume_mm3": 1180.0,
            "mni_coords_mm": (-22, 36, 18),
            "lobe": "Left Frontal",
            "planned_date": (base_date + timedelta(hours=5)).strftime("%Y-%m-%d %H:%M"),
            "surgeon": "Dr. Iyer",
            "endoscope_channel": "CH-01 (Olympus)",
            "overhead_camera_id": "OR-01-CAM-B",
            "n_beams": 0,
            "beam_energy_mv": 0,
        },
        {
            "patient_id": "P-003",
            "patient_name": "Rohan Verma",
            "age": 61,
            "procedure_type": "Tumor Resection",
            "assigned_robot": ROBOTS["resection"],
            "priority": "HIGH",
            "tumor_volume_mm3": 18650.0,
            "mni_coords_mm": (8, -52, -8),
            "lobe": "Right Cerebellar",
            "planned_date": (base_date + timedelta(hours=26)).strftime("%Y-%m-%d %H:%M"),
            "surgeon": "Dr. Mishra",
            "endoscope_channel": "CH-05 (Zeiss Pentero)",
            "overhead_camera_id": "OR-03-CAM-A",
            "n_beams": 0,
            "beam_energy_mv": 0,
        },
        {
            "patient_id": "P-004",
            "patient_name": "Ananya Desai",
            "age": 35,
            "procedure_type": "Laser Interstitial Ablation",
            "assigned_robot": ROBOTS["laser_abl"],
            "priority": "LOW",
            "tumor_volume_mm3": 2750.0,
            "mni_coords_mm": (-16, -8, -18),
            "lobe": "Left Temporal (Mesial)",
            "planned_date": (base_date + timedelta(days=2)).strftime("%Y-%m-%d %H:%M"),
            "surgeon": "Dr. Bhatt",
            "endoscope_channel": "CH-02 (Medtronic)",
            "overhead_camera_id": "OR-02-CAM-C",
            "n_beams": 0,
            "beam_energy_mv": 0,
        },
    ]
    out = []
    for i, s in enumerate(seed):
        uid = f"T-{int(time.time())}-{i+1:03d}"
        out.append({
            "id": uid,
            **s,
            "status": ["QUEUED", "PLANNING", "IN_PROGRESS", "QUEUED"][i],
            "progress_pct": [0, 32, 68, 0][i],
            "created_at": _now_iso(),
            "audit_log": [_log(f"Task created with status {['QUEUED','PLANNING','IN_PROGRESS','QUEUED'][i]}")],
            "error_log": [],
            "notes": "",
        })
    return out


def add_task(queue: List[Dict[str, Any]], patient_id: str, patient_name: str,
             procedure_type: str, priority: str, tumor_volume_mm3: float,
             mni_coords_mm, lobe: str, planned_date: str, age: int = 50,
             surgeon: str = "Unassigned", endoscope_channel: str = "CH-01",
             overhead_camera_id: str = "OR-01-CAM-A",
             n_beams: int = 0, beam_energy_mv: int = 6,
             notes: str = "") -> Dict[str, Any]:
    robot_lookup = {
        "Stereotactic Biopsy": ROBOTS["stereotactic_biopsy"],
        "Tumor Resection": ROBOTS["resection"],
        "Radiosurgery (SRS)": ROBOTS["srs"],
        "Laser Interstitial Ablation": ROBOTS["laser_abl"],
    }
    robot = robot_lookup.get(procedure_type, ROBOTS["stereotactic_biopsy"])
    uid = f"T-{int(time.time())}-{uuid.uuid4().hex[:3].upper()}"
    task = {
        "id": uid,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "age": int(age),
        "procedure_type": procedure_type,
        "assigned_robot": robot,
        "priority": priority,
        "tumor_volume_mm3": float(tumor_volume_mm3),
        "mni_coords_mm": tuple(mni_coords_mm),
        "lobe": lobe,
        "planned_date": planned_date,
        "surgeon": surgeon,
        "endoscope_channel": endoscope_channel,
        "overhead_camera_id": overhead_camera_id,
        "n_beams": int(n_beams),
        "beam_energy_mv": int(beam_energy_mv),
        "status": "QUEUED",
        "progress_pct": 0,
        "created_at": _now_iso(),
        "audit_log": [_log("Task created and queued", "NeuroVision-AI")],
        "error_log": [],
        "notes": notes,
    }
    queue.insert(0, task)
    return task


def find_task(queue: List[Dict[str, Any]], task_id: str) -> Optional[Dict[str, Any]]:
    for t in queue:
        if t["id"] == task_id:
            return t
    return None


def advance_task_status(queue: List[Dict[str, Any]], task_id: str,
                        user: str = "Operator", emergency_stop: bool = False) -> bool:
    t = find_task(queue, task_id)
    if t is None:
        return False
    if emergency_stop:
        if t["status"] in ("PLANNING", "IN_PROGRESS"):
            t["status"] = "CANCELLED"
            t["audit_log"].append(_log("EMERGENCY STOP — task cancelled", user))
            t["error_log"].append({"time": _now_iso(), "msg": "User-initiated E-STOP"})
            return True
        return False
    next_s = STATUS_FLOW.get(t["status"])
    if next_s is None:
        return False
    t["audit_log"].append(_log(f"Status: {t['status']} -> {next_s}", user))
    t["status"] = next_s
    if t["status"] == "PLANNING":
        t["progress_pct"] = max(t["progress_pct"], 20)
    elif t["status"] == "IN_PROGRESS":
        t["progress_pct"] = max(t["progress_pct"], 55)
    elif t["status"] == "COMPLETED":
        t["progress_pct"] = 100
    return True


def cancel_task(queue: List[Dict[str, Any]], task_id: str,
                reason: str = "User cancelled", user: str = "Operator") -> bool:
    t = find_task(queue, task_id)
    if t is None or t["status"] in ("COMPLETED", "CANCELLED"):
        return False
    t["status"] = "CANCELLED"
    t["audit_log"].append(_log(f"Task cancelled: {reason}", user))
    return True


def procedure_uses_radiation(procedure_type: str) -> bool:
    return "Radiosurgery" in procedure_type or "SRS" in procedure_type
