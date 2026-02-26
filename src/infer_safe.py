#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe landing inference with:
- Model #1 (segment): person, vehicle, building, tree  (+optional animal)
- Model #2 (optional, segment): water
- Optional sky masking (HSV)
- Optional lane-mark based road ban (white/yellow lane -> Hough -> thick road ribbon)

Usage (examples):
  python -m src.infer_safe \
    --weights "modified_uavid_dataset/safe_landing_runs_new2/weights/best.pt" \
    --weights2 "modified_uavid_dataset/safe_landing_ft_person_water/weights/best.pt" \
    --source "modified_uavid_dataset/test_data/Images" \
    --outdir "pred_safe" \
    --device mps \
    --imgsz 832 \
    --conf 0.35 \
    --sky \
    --ban-road-lanes \
    --lane-thickness 12 \
    --lane-dilate 21 \
    --save-lane-debug

Notes:
- Classes assumed: 0:person, 1:vehicle, 2:building, 3:tree, 4:water, 5:animal
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

CLASS_NAME_TO_ID = {
    "person": 0,
    "vehicle": 1,
    "building": 2,
    "tree": 3,
    "water": 4,
    "animal": 5,
}

DEFAULT_OBSTACLE_NAMES = ["person", "vehicle", "building", "tree"]
SENTINEL_FULLSAFE_RADIUS = 65533

# optional GeoTIFF reader
try:
    import rasterio
except Exception:
    rasterio = None



def imread_any(path: str) -> Optional[np.ndarray]:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img

def imwrite_any(path: str, img: np.ndarray) -> bool:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(path, img)

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
    return cids

def select_device(dev: str):
    return "cpu" if dev is None else dev

def masks_for_classes(result, class_ids: Set[int], out_hw: Tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    if result is None or result.masks is None or result.boxes is None:
        return np.zeros((h, w), np.uint8)
    masks = result.masks.data
    clses = result.boxes.cls
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
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue  = cv2.inRange(hsv, (90, 20, 70), (135, 255, 255))
    cloud = cv2.inRange(hsv, (0,  0, 180), (180, 60, 255))
    mask = cv2.bitwise_or(blue, cloud)
    gate = np.zeros_like(mask); gate[: int(h * top_frac), :] = 255
    mask = cv2.bitwise_and(mask, gate)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask

def largest_safe_circle(obstacles_mask: np.ndarray) -> Tuple[bool, Tuple[int, int], int]:
    h, w = obstacles_mask.shape[:2]
    allowed = cv2.bitwise_not(obstacles_mask)
    if allowed.max() == 0:
        return False, (w // 2, h // 2), 0
    if obstacles_mask.max() == 0:
        return True, (w // 2, h // 2), SENTINEL_FULLSAFE_RADIUS
    dist = cv2.distanceTransform((allowed > 0).astype(np.uint8), cv2.DIST_L2, 3)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(dist)
    return (int(maxVal) > 0), (int(maxLoc[0]), int(maxLoc[1])), int(maxVal)

def overlay_visual(bgr: np.ndarray,
                   obstacles_mask: np.ndarray,
                   water_mask: Optional[np.ndarray],
                   sky_mask: Optional[np.ndarray],
                   center: Tuple[int, int],
                   radius: int) -> np.ndarray:
    vis = bgr.copy()
    overlay = vis.copy()
    # Obstacles -> red
    red = np.zeros_like(vis); red[:, :, 2] = 255
    overlay = np.where(obstacles_mask[..., None] > 0, cv2.addWeighted(overlay, 0.5, red, 0.5, 0), overlay)
    # Water -> blue
    if water_mask is not None:
        blue = np.zeros_like(vis); blue[:, :, 0] = 255
        overlay = np.where(water_mask[..., None] > 0, cv2.addWeighted(overlay, 0.5, blue, 0.5, 0), overlay)
    # Sky -> cyan
    if sky_mask is not None:
        cyan = np.zeros_like(vis); cyan[:, :, 0:2] = 255
        overlay = np.where(sky_mask[..., None] > 0, cv2.addWeighted(overlay, 0.5, cyan, 0.3, 0), overlay)
    vis = overlay
    if radius > 0 and radius != SENTINEL_FULLSAFE_RADIUS:
        cv2.circle(vis, center, radius, (0, 255, 0), 2)
        cv2.circle(vis, center, 4, (0, 255, 0), -1)
    label = f"r={radius}px" if radius != SENTINEL_FULLSAFE_RADIUS else "FULL SAFE"
    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return vis

def is_image_file(p: str) -> bool:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    return p.lower().endswith(exts)

def collect_sources(src: str) -> List[str]:
    if os.path.isdir(src):
        return sorted([p for p in glob.glob(os.path.join(src, "*")) if is_image_file(p)])
    elif is_image_file(src):
        return [src]
    else:
        return sorted([p for p in glob.glob(src) if is_image_file(p)])

# --------------------------
# Lane-based Road Ban (CV) |
# --------------------------
def _lane_candidates_hsv(bgr: np.ndarray) -> np.ndarray:
    """White + Yellow candidates via HSV + tophat to highlight thin bright marks."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # white marks (bright low-sat)
    white = cv2.inRange(hsv, (0, 0, 200), (180, 60, 255))
    # yellow marks
    yellow = cv2.inRange(hsv, (15, 80, 120), (35, 255, 255))
    cand = cv2.bitwise_or(white, yellow)

    # Top-hat (enhance thin bright lines)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    tophat = cv2.morphologyEx(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.MORPH_TOPHAT, kernel)
    tophat = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Combine & clean
    comb = cv2.bitwise_or(cand, (tophat > 160).astype(np.uint8) * 255)
    comb = cv2.morphologyEx(comb, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return comb

def lane_road_mask_from_hough(bgr: np.ndarray,
                              hough_thresh: int = 60,
                              min_len: int = 40,
                              max_gap: int = 10,
                              thickness: int = 12,
                              dilate_ksize: int = 21,
                              dbg: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns (road_mask_from_lanes, debug_vis). road_mask is 0/255.
    """
    h, w = bgr.shape[:2]
    cand = _lane_candidates_hsv(bgr)

    edges = cv2.Canny(cand, 50, 150, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_thresh,
                            minLineLength=min_len, maxLineGap=max_gap)
    road_mask = np.zeros((h, w), np.uint8)
    dbg_vis = bgr.copy() if dbg else None

    if lines is not None:
        for l in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, l)
            # draw a thick ribbon along each detected lane line
            cv2.line(road_mask, (x1, y1), (x2, y2), 255, thickness)
            if dbg_vis is not None:
                cv2.line(dbg_vis, (x1, y1), (x2, y2), (0, 215, 255), 2)  # orange

    if dilate_ksize > 0:
        k = (dilate_ksize // 2) * 2 + 1
        road_mask = cv2.dilate(road_mask, np.ones((k, k), np.uint8), iterations=1)

    return road_mask, dbg_vis

def load_static_ban(path: str, target_hw: tuple[int, int]) -> np.ndarray:
    """Load a static ban raster (TIF/PNG). Returns 0/255 uint8 resized to (h,w)."""
    h, w = target_hw
    m = None
    # Prefer rasterio for GeoTIFF
    if rasterio is not None and path.lower().endswith((".tif", ".tiff")):
        try:
            with rasterio.open(path) as ds:
                arr = ds.read(1)  # first band
                m = (arr > 0).astype(np.uint8) * 255
        except Exception:
            m = None
    if m is None:
        # Fallback: OpenCV
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"static-ban cannot be read: {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m = (img > 0).astype(np.uint8) * 255

    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m




# --------------
# Main
# --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=str)
    ap.add_argument("--weights2", type=str, default=None, help="Optional second model for WATER class.")
    ap.add_argument("--source", required=True, type=str)
    ap.add_argument("--outdir", type=str, default="pred_safe")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--exclude", type=str, default="")
    ap.add_argument("--sky", action="store_true")
    ap.add_argument("--obstacle-dilation-kernel-size", type=int, default=50)

    # Lane-based road ban
    ap.add_argument("--ban-road-lanes", action="store_true", help="Use lane markings to ban roads.")
    ap.add_argument("--lane-hough-thresh", type=int, default=60)
    ap.add_argument("--lane-min-length", type=int, default=40)
    ap.add_argument("--lane-max-gap", type=int, default=10)
    ap.add_argument("--lane-thickness", type=int, default=12, help="pixels")
    ap.add_argument("--lane-dilate", type=int, default=21, help="additional dilation (px)")
    ap.add_argument("--save-lane-debug", action="store_true")

    ap.add_argument("--static-ban", dest="static_ban", type=str, default=None,
                    help="Path to static ban mask (TIF/PNG). Resized to frame size.")
    ap.add_argument("--margin", type=int, default=0,
                    help="Ban a border band of N pixels around the frame.")

    args = ap.parse_args()
    device = select_device(args.device)

    print(f"Loading model: {args.weights}")
    model1 = YOLO(args.weights)
    model2 = YOLO(args.weights2) if args.weights2 else None

    default_obstacles = set(CLASS_NAME_TO_ID[n] for n in DEFAULT_OBSTACLE_NAMES)
    excludes = parse_exclude(args.exclude)
    obstacle_cids = sorted(list(default_obstacles - excludes))

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
        # Model #1
        res1 = model1.predict(img, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)
        res1 = res1[0] if isinstance(res1, list) else res1
        base_mask = masks_for_classes(res1, set(obstacle_cids), (h, w))

        # ---- Static ban mask (OSM-derived) ----

        if args.static_ban:
            try:
                sb = load_static_ban(args.static_ban, (h, w))
                base_mask = cv2.bitwise_or(base_mask, sb)
            except Exception as e:
                print(f"[WARN] static-ban load failed: {e}")

        # ---- Frame margin ban ----
        if args.margin and args.margin > 0:
            border = np.ones((h, w), np.uint8) * 255
            # keep interior 0, make a ring of 'margin' pixels as 255
            cv2.rectangle(border,
                          (args.margin, args.margin),
                          (w - args.margin - 1, h - args.margin - 1),
                          0, thickness=-1)
            base_mask = cv2.bitwise_or(base_mask, border)

        # Optional WATER
        water_mask = None
        if model2 is not None:
            res2 = model2.predict(img, imgsz=args.imgsz, conf=args.conf, device=device,
                                  verbose=False, classes=[CLASS_NAME_TO_ID["water"]])
            res2 = res2[0] if isinstance(res2, list) else res2
            water_mask = masks_for_classes(res2, {CLASS_NAME_TO_ID["water"]}, (h, w))
            base_mask = cv2.bitwise_or(base_mask, water_mask)

        # Optional SKY
        sky_mask = fast_sky_mask(img) if args.sky else None
        if sky_mask is not None:
            base_mask = cv2.bitwise_or(base_mask, sky_mask)

        # Lane-based road ban
        if args.ban_road_lanes:
            road_mask, dbg_vis = lane_road_mask_from_hough(
                img,
                hough_thresh=args.lane_hough_thresh,
                min_len=args.lane_min_length,
                max_gap=args.lane_max_gap,
                thickness=args.lane_thickness,
                dilate_ksize=args.lane_dilate,
                dbg=args.save_lane_debug
            )
            base_mask = cv2.bitwise_or(base_mask, road_mask)
            if args.save_lane_debug and dbg_vis is not None:
                op_dbg = str(Path(args.outdir) / (Path(ip).stem + "_lanes.jpg"))
                imwrite_any(op_dbg, dbg_vis)

        # Post-process obstacles (close + (optional) dilate)
        if base_mask.max() > 0:
            k = max(3, (args.imgsz // 128) | 1)
            base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
            if args.obstacle_dilation_kernel_size > 0:
                dk = (args.obstacle_dilation_kernel_size // 2) * 2 + 1
                base_mask = cv2.dilate(base_mask, np.ones((dk, dk), np.uint8), iterations=1)

        ok, center, radius = largest_safe_circle(base_mask)
        vis = overlay_visual(img, base_mask, water_mask, sky_mask, center, radius)

        out_path = str(Path(args.outdir) / (Path(ip).stem + "_safe.jpg"))
        imwrite_any(out_path, vis)
        print(f"-> {out_path}  (ok={ok}, r={radius})")

if __name__ == "__main__":
    main()
