#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe landing inference script with optional 2-model ensemble and sky masking.
- Model #1 (required): detects main obstacles (person, vehicle, building, tree).
- Model #2 (optional): used to add WATER (class id = 4) masks.
- Sky mask (optional): simple HSV heuristic to remove sky from safe area.
Outputs a visualized image with the largest safe circle and prints per-image summary.

USAGE (examples):
  python -m src.infer_safe \
    --weights "modified_uavid_dataset/safe_landing_runs_new2/weights/best.pt" \
    --source "modified_uavid_dataset/test_data/Images" \
    --outdir "pred_safe" \
    --device mps \
    --sky \
    --conf 0.35

  # Ensemble: add water from second model
  python -m src.infer_safe \
    --weights  "modified_uavid_dataset/safe_landing_runs_new2/weights/best.pt" \
    --weights2 "modified_uavid_dataset/safe_landing_ft_person_water/weights/best.pt" \
    --source "modified_uavid_dataset/test_data/Images" \
    --outdir "pred_safe" \
    --device mps \
    --sky \
    --conf 0.35

Notes:
- Class mapping is assumed as: 0:person, 1:vehicle, 2:building, 3:tree, 4:water, 5:animal
- By default, obstacles = {person, vehicle, building, tree}. WATER is added only if --weights2 is provided
  (or if model #1 also contains water class predictions).
- Use --exclude to ignore any of the above classes as obstacles (e.g. --exclude tree,water).
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# -------------------------
# Class mapping / constants
# -------------------------
CLASS_NAME_TO_ID = {
    "person": 0,
    "vehicle": 1,
    "building": 2,
    "tree": 3,
    "water": 4,
    "animal": 5,
}

DEFAULT_OBSTACLE_NAMES = ["person", "vehicle", "building", "tree"]
SENTINEL_FULLSAFE_RADIUS = 65533  # to mimic earlier logs


# -------------------------
# Utilities
# -------------------------
def imread_any(path: str) -> Optional[np.ndarray]:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    # Fallback for environments without fromfile Unicode support
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def imwrite_any(path: str, img: np.ndarray) -> bool:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(path, img)
    return ok


def parse_exclude(s: str) -> Set[int]:
    if not s:
        return set()
    toks = [t.strip().lower() for t in s.split(",") if t.strip()]
    cids = set()
    for t in toks:
        if t.isdigit():
            cids.add(int(t))
        elif t in CLASS_NAME_TO_ID:
            cids.add(CLASS_NAME_TO_ID[t])
        else:
            # silently ignore unknown tokens
            pass
    return cids


def select_device(dev: str):
    # ultralytics accepts "cpu", "mps", integer GPU index, or "0", "0,1" etc.
    # We just pass through the value. Validation will happen at runtime.
    if dev is None:
        return "cpu"
    return dev


def masks_for_classes(result, class_ids: Set[int], out_hw: Tuple[int, int]) -> np.ndarray:
    """Return a binary 0/255 mask for instances in 'class_ids' from a YOLOv8 segmentation result."""
    h, w = out_hw
    if result is None or result.masks is None or result.boxes is None:
        return np.zeros((h, w), np.uint8)

    masks = result.masks.data  # [N, h', w'] torch tensor
    clses = result.boxes.cls   # [N] torch tensor of class ids

    if masks is None or clses is None or len(masks) == 0:
        return np.zeros((h, w), np.uint8)

    masks_np = masks.cpu().numpy()
    clses_np = clses.cpu().numpy().astype(int)

    out = np.zeros((h, w), np.uint8)
    for mi, ci in zip(masks_np, clses_np):
        if ci in class_ids:
            m = (cv2.resize(mi, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8) * 255
            out = cv2.bitwise_or(out, m)
    return out


def fast_sky_mask(bgr: np.ndarray, top_frac: float = 0.6) -> np.ndarray:
    """Heuristic sky mask: blue + bright-gray in upper region."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue  = cv2.inRange(hsv, (90, 20, 70), (135, 255, 255))   # blue sky
    cloud = cv2.inRange(hsv, (0,  0, 180), (180, 60, 255))    # bright clouds
    mask = cv2.bitwise_or(blue, cloud)
    # keep only top part
    m = np.zeros_like(mask)
    m[: int(h * top_frac), :] = 255
    mask = cv2.bitwise_and(mask, m)
    # close gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask


def largest_safe_circle(obstacles_mask: np.ndarray) -> Tuple[bool, Tuple[int, int], int]:
    """
    Given a binary mask (255 = obstacle, 0 = free), compute largest inscribed circle in free area.
    Returns (ok, center(x,y), radius_px).
    """
    h, w = obstacles_mask.shape[:2]
    # Allowed = invert
    allowed = cv2.bitwise_not(obstacles_mask)

    if allowed.max() == 0:
        # No free pixel
        return False, (w // 2, h // 2), 0

    if obstacles_mask.max() == 0:
        # No obstacles at all: fully safe
        return True, (w // 2, h // 2), SENTINEL_FULLSAFE_RADIUS

    dist = cv2.distanceTransform((allowed > 0).astype(np.uint8), cv2.DIST_L2, 3)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist)
    radius = int(maxVal)
    center = (int(maxLoc[0]), int(maxLoc[1]))
    ok = radius > 0
    return ok, center, radius


def overlay_visual(bgr: np.ndarray,
                   obstacles_mask: np.ndarray,
                   water_mask: Optional[np.ndarray],
                   sky_mask: Optional[np.ndarray],
                   center: Tuple[int, int],
                   radius: int) -> np.ndarray:
    """Create a visualization image with masks and the safe circle."""
    vis = bgr.copy()

    # Color overlays
    overlay = vis.copy()
    # Obstacles: red
    red = np.zeros_like(vis)
    red[:, :, 2] = 255
    overlay = np.where(obstacles_mask[..., None] > 0, cv2.addWeighted(overlay, 0.5, red, 0.5, 0), overlay)

    # Water: blue (if provided separately)
    if water_mask is not None:
        blue = np.zeros_like(vis)
        blue[:, :, 0] = 255
        overlay = np.where(water_mask[..., None] > 0, cv2.addWeighted(overlay, 0.5, blue, 0.5, 0), overlay)

    # Sky: light-cyan
    if sky_mask is not None:
        cyan = np.zeros_like(vis)
        cyan[:, :, 0] = 255
        cyan[:, :, 1] = 255
        overlay = np.where(sky_mask[..., None] > 0, cv2.addWeighted(overlay, 0.5, cyan, 0.3, 0), overlay)

    vis = overlay

    # Draw safe circle if not sentinel
    if radius > 0 and radius != SENTINEL_FULLSAFE_RADIUS:
        cv2.circle(vis, center, radius, (0, 255, 0), 2)
        cv2.circle(vis, center, 4, (0, 255, 0), -1)

    # Put label
    label = f"r={radius}px" if radius != SENTINEL_FULLSAFE_RADIUS else "FULL SAFE"
    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def is_image_file(p: str) -> bool:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    return p.lower().endswith(exts)


def collect_sources(src: str) -> List[str]:
    if os.path.isdir(src):
        files = sorted([p for p in glob.glob(os.path.join(src, "*")) if is_image_file(p)])
        return files
    elif is_image_file(src):
        return [src]
    else:
        # try glob
        files = sorted([p for p in glob.glob(src) if is_image_file(p)])
        return files


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=str, help="Primary model weights (YOLOv8-seg).")
    ap.add_argument("--weights2", type=str, default=None, help="Optional second model for WATER class.")
    ap.add_argument("--source", required=True, type=str, help="Image file, glob, or directory.")
    ap.add_argument("--outdir", type=str, default="pred_safe", help="Output directory for visualizations.")
    ap.add_argument("--device", type=str, default="cpu", help='cpu | mps | 0 | "0,1" etc.')
    ap.add_argument("--imgsz", type=int, default=768, help="Inference size.")
    ap.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    ap.add_argument("--exclude", type=str, default="", help="Comma-separated class names/ids to ignore as obstacles.")
    ap.add_argument("--sky", action="store_true", help="Enable heuristic sky mask.")
    ap.add_argument("--obstacle-dilation-kernel-size", type=int, default=50,
                    help="Kernel size for dilating obstacle masks to expand unsafe area.")
    args = ap.parse_args()

    device = select_device(args.device)

    # Load models
    print(f"Loading model: {args.weights}")
    model1 = YOLO(args.weights)

    model2 = None
    if args.weights2:
        print(f"Loading secondary model (water): {args.weights2}")
        model2 = YOLO(args.weights2)

    # Determine obstacle classes from model1 (default), minus exclude
    default_obstacles = set(CLASS_NAME_TO_ID[n] for n in DEFAULT_OBSTACLE_NAMES)
    excludes = parse_exclude(args.exclude)
    obstacle_cids = sorted(list(default_obstacles - excludes))

    # Add water to obstacle set if we either have model2 or user didn't exclude it and model1 can detect it
    add_water = (CLASS_NAME_TO_ID["water"] not in excludes) and (model2 is not None)
    if add_water:
        pass  # Water will be added from model2 predictions explicitly

    src_paths = collect_sources(args.source)
    if not src_paths:
        print(f"No images found under: {args.source}")
        sys.exit(1)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    for ip in src_paths:
        img = imread_any(ip)
        if img is None:
            print(f"Skipping (cannot read): {ip}")
            continue

        h, w = img.shape[:2]

        # Model #1 inference
        res1 = model1.predict(img, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)
        res1 = res1[0] if isinstance(res1, list) else res1
        base_mask = masks_for_classes(res1, set(obstacle_cids), (h, w))

        # Optional: WATER from model #2
        water_mask = None
        if model2 is not None:
            res2 = model2.predict(img, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False, classes=[CLASS_NAME_TO_ID["water"]])
            res2 = res2[0] if isinstance(res2, list) else res2
            water_mask = masks_for_classes(res2, {CLASS_NAME_TO_ID["water"]}, (h, w))
            base_mask = cv2.bitwise_or(base_mask, water_mask)

        # Optional: SKY
        sky_mask = fast_sky_mask(img) if args.sky else None
        if sky_mask is not None:
            base_mask = cv2.bitwise_or(base_mask, sky_mask)

        # Post-process mask (denoise & thicken a bit)
        if base_mask.max() > 0:
            k = max(3, (args.imgsz // 128) | 1)  # odd kernel size
            base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
            # Dilate obstacles to expand unsafe area (new addition)
            if args.obstacle_dilation_kernel_size > 0:
                dk = (args.obstacle_dilation_kernel_size // 2) * 2 + 1  # Ensure odd kernel size
                base_mask = cv2.dilate(base_mask, np.ones((dk, dk), np.uint8), iterations=1)

        # Largest safe circle
        ok, center, radius = largest_safe_circle(base_mask)

        # Visualization
        vis = overlay_visual(img, base_mask, water_mask, sky_mask, center, radius)

        # Save
        name = Path(ip).stem + "_safe.jpg"
        op = str(Path(args.outdir) / name)
        imwrite_any(op, vis)
        print(f"-> {op}  (ok={ok}, r={radius})")

    # Done


if __name__ == "__main__":
    main()
