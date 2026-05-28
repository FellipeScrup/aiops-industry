"""BPM context enrichment for Smart Factory log events.

Maps (station, current_task) pairs to high-level BPM activity context derived
from the Smart Factory process models (García-Bañuelos et al., 2025 / Seiger et al., 2023).

Two processes are modelled in the factory:
  Storage:    DPS → VGR → EC_1 (color detection) → HBW (store)
  Production: HBW → VGR → OV (heat) → VGR → MM (mill) → VGR → SM (sort) → VGR → DPS

Reference: García-Bañuelos et al. (2025). Procedia Computer Science, 257, 856–863.
           DOI: 10.1016/j.procs.2025.03.110
"""

from __future__ import annotations

# ── BPM activity table ────────────────────────────────────────────────────────
# Structure: list of (station, task_keywords, activity_name, process, next_station)
# Matching: task_keywords are checked as substrings (case-insensitive) in current_task.
# First match wins.

_BPM_TABLE: list[tuple[str, list[str], str, str, str]] = [
    # ── VGR_1 (Vacuum Gripper Robot — central transporter) ────────────────────
    ("VGR_1", ["transport", "oven"],
        "Pickup and transport to Oven",       "Production", "OV_1"),
    ("VGR_1", ["transport", "milling"],
        "Pickup and transport to Milling Machine", "Production", "MM_1"),
    ("VGR_1", ["transport", "sink"],
        "Pickup and transport to Sink (SM)",  "Production", "SM_1"),
    ("VGR_1", ["transport", "dps"],
        "Deliver workpiece to DPS",           "Production", "DPS"),
    ("VGR_1", ["transport", "hbw"],
        "Transport workpiece to HBW",         "Storage",    "HBW_1"),
    ("VGR_1", ["transport", "warehouse"],
        "Transport workpiece to HBW",         "Storage",    "HBW_1"),
    ("VGR_1", ["transport"],
        "Transport workpiece (destination TBD)", "Production/Storage", "-"),
    ("VGR_1", ["pick", "dps"],
        "Get Workpiece from DPS",             "Storage",    "EC_1"),
    ("VGR_1", ["pick"],
        "Pick up workpiece",                  "Production/Storage", "-"),
    ("VGR_1", ["deliver"],
        "Deliver workpiece to DPS",           "Production", "DPS"),
    ("VGR_1", ["move", "dps"],
        "Move to DPS",                        "Production", "DPS"),
    ("VGR_1", ["move"],
        "Move to target station",             "Production/Storage", "-"),

    # ── HBW_1 (High-Bay Warehouse) ────────────────────────────────────────────
    ("HBW_1", ["unload"],
        "Unload workpiece from warehouse slot", "Production", "VGR_1"),
    ("HBW_1", ["store"],
        "Store workpiece in warehouse slot",    "Storage",    "idle"),
    ("HBW_1", ["calibrat"],
        "Calibrate warehouse components",       "Maintenance", "-"),
    ("HBW_1", ["fetch"],
        "Fetch workpiece from slot",            "Production", "VGR_1"),

    # ── OV_1 (Oven) ───────────────────────────────────────────────────────────
    ("OV_1", ["heat"],
        "Heat workpiece in oven",               "Production", "MM_1 via VGR_1"),
    ("OV_1", ["load"],
        "Load workpiece into oven",             "Production", "OV_1"),
    ("OV_1", ["unload"],
        "Unload workpiece from oven",           "Production", "MM_1 via VGR_1"),

    # ── MM_1 (Milling Machine) ────────────────────────────────────────────────
    ("MM_1", ["mill"],
        "Mill workpiece",                       "Production", "SM_1 via VGR_1"),
    ("MM_1", ["load"],
        "Load workpiece into milling machine",  "Production", "MM_1"),
    ("MM_1", ["unload"],
        "Unload workpiece from milling machine","Production", "SM_1 via VGR_1"),

    # ── SM_1 (Sorting Machine) ────────────────────────────────────────────────
    ("SM_1", ["sort"],
        "Sort workpiece by color/type",         "Production", "DPS via VGR_1"),
    ("SM_1", ["convey"],
        "Convey workpiece on sorting belt",     "Production", "SM_1"),
    ("SM_1", ["separate"],
        "Separate workpiece",                   "Production", "DPS via VGR_1"),

    # ── EC_1 (Environment & Camera) ───────────────────────────────────────────
    ("EC_1", ["detect", "color"],
        "Detect workpiece color/type",          "Storage",    "HBW_1 via VGR_1"),
    ("EC_1", ["detect"],
        "Sensor detection",                     "Ambient",    "-"),

    # ── WT_1 (Workstation Transport belt) ────────────────────────────────────
    ("WT_1", ["convey"],
        "Convey workpiece on transport belt",   "Production", "-"),
    ("WT_1", ["transport"],
        "Transport workpiece on belt",          "Production", "-"),
]


def get_bpm_context(station: str, current_task: str) -> dict[str, str] | None:
    """Return BPM activity context for a given station and task.

    Args:
        station: Station identifier (e.g. 'VGR_1').
        current_task: Raw task string from the event log.

    Returns:
        Dict with keys activity, process, next_station, or None if no match.
    """
    if not current_task:
        return None

    task_lower = current_task.lower()

    for row_station, keywords, activity, process, next_station in _BPM_TABLE:
        if row_station != station:
            continue
        if all(kw in task_lower for kw in keywords):
            return {
                "activity":     activity,
                "process":      process,
                "next_station": next_station,
            }

    return None


def format_bpm_context(station: str, current_task: str) -> str:
    """Return a formatted BPM context string for embedding, or empty string."""
    ctx = get_bpm_context(station, current_task)
    if ctx is None:
        return ""
    return (
        f"BPM: {ctx['activity']} | "
        f"Process: {ctx['process']} | "
        f"Next: {ctx['next_station']}"
    )
