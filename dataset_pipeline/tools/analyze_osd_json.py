#!/usr/bin/env python
import json
import math
import sys
from pathlib import Path
from itertools import combinations

import numpy as np


def bbox_center_and_extent_from_aabb(aabb: dict):
    """Compute center and extent from axis-aligned bbox."""
    minb = np.array(aabb["min_bound"], dtype=float)
    maxb = np.array(aabb["max_bound"], dtype=float)
    center = (minb + maxb) / 2.0
    extent = maxb - minb
    return center, extent


def main(json_path: str):
    json_path = Path(json_path)
    with open(json_path, "r") as f:
        detections = json.load(f)

    print(f"\nLoaded {len(detections)} detections from {json_path}\n")

    # --- 1) Global stats ---
    classes = sorted({d.get("class_name", "UNKNOWN") for d in detections})
    print("Classes found:", ", ".join(classes))
    print()

    msp_objects = []  # this will be our lightweight MSP-style representation

    # --- 2) Per-object summary ---
    for idx, det in enumerate(detections):
        name = det.get("class_name", "UNKNOWN")
        conf = det.get("confidence", None)

        pcd = det.get("pcd", None)
        num_points = len(pcd.get("points", [])) if isinstance(pcd, dict) else 0

        aabb = det.get("axis_aligned_bbox", None)
        obb = det.get("oriented_bbox", None)

        center = None
        extent = None

        if aabb is not None:
            center, extent = bbox_center_and_extent_from_aabb(aabb)
        elif obb is not None:
            center = np.array(obb["center"], dtype=float)
            extent = np.array(obb["extent"], dtype=float)

        print(f"Object {idx:02d}: {name}")
        if conf is not None:
            print(f"  confidence: {conf:.3f}")
        print(f"  #points in pcd: {num_points}")

        if center is not None and extent is not None:
            print(f"  center (x,y,z): {center.tolist()}")
            print(f"  extent (dx,dy,dz): {extent.tolist()}")
        else:
            print("  [no 3D bbox info]")

        print()

        # add to MSP-style list
        msp_objects.append(
            {
                "id": idx,
                "label": name,
                "confidence": float(conf) if conf is not None else None,
                "num_points": int(num_points),
                "center": center.tolist() if center is not None else None,
                "extent": extent.tolist() if extent is not None else None,
            }
        )

    # --- 3) Pairwise distances between centers (optional, just for inspection) ---
    print("\nPairwise distances between object centers (meters):")
    for (i, a), (j, b) in combinations(enumerate(msp_objects), 2):
        ca, cb = a["center"], b["center"]
        if ca is None or cb is None:
            continue
        da = np.array(ca)
        db = np.array(cb)
        dist = float(np.linalg.norm(da - db))
        print(f"  {i:02d} ({a['label']}) <-> {j:02d} ({b['label']}): {dist:.3f} m")

    # --- 4) Dump a small summary JSON next to the original file ---
    out_path = json_path.with_name(json_path.stem + "_msp_summary.json")
    with open(out_path, "w") as f:
        json.dump(msp_objects, f, indent=2)
    print(f"\nWrote MSP-style summary to: {out_path}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/analyze_osd_json.py /path/to/indoor.json")
        sys.exit(1)
    main(sys.argv[1])
