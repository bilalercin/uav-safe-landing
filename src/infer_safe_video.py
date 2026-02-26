# src/infer_safe_video.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe landing for videos:
- Model #1 (seg): person, vehicle, building, tree (+optional animal)
- Model #2 (optional, seg): water
- Optional sky masking (HSV)
- Optional lane-mark based road ban (HSV+TopHat + Canny+Hough -> thick ribbon)
- Optional static ban raster (GeoTIFF/PNG) resized to frame
- Outputs annotated MP4 + per-frame summary JSON

Example:
  python -m src.infer_safe_video \
    --weights "modified_uavid_dataset/safe_landing_runs_new2/weights/best.pt" \
    --weights2 "modified_uavid_dataset/safe_landing_ft_person_water/weights/best.pt" \
    --source "/path/to/video.mp4" \
    --out "pred_safe/video_safe.mp4" \
    --device mps \
    --imgsz 832 \
    --conf 0.35 \
    --sky \
    --ban-road-lanes \
    --lane-thickness 12 \
    --lane-dilate 21 \
    --static-ban "static_ban_izmit.tif" \
    --margin 20
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Set, Tuple, List

import cv2
import numpy as np
from ultralytics import YOLO

# ---- copy minimal utilities from image pipeline ----
CLASS_NAME_TO_ID = {
    "person": 0, "vehicle": 1, "building": 2, "tree": 3, "water": 4, "animal": 5,
}
DEFAULT_OBSTACLE_NAMES = ["person", "vehicle", "building", "tree"]
SENTINEL_FULLSAFE_RADIUS = 65533

try:
    import rasterio
except Exception:
    rasterio = None


def select_device(dev: str):
    return "cpu" if dev is None else dev


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
    whiteish = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))
    blueish  = cv2.inRange(hsv, (90, 20, 70), (135, 255, 255))
    mask = cv2.bitwise_or(whiteish, blueish)
    gate = np.zeros_like(mask); gate[: int(h * top_frac), :] = 255
    mask = cv2.bitwise_and(mask, gate)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask


def largest_safe_circle(obstacles_mask: np.ndarray):
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

    red = np.zeros_like(vis); red[:, :, 2] = 255               # obstacles
    overlay = np.where(obstacles_mask[..., None] > 0, cv2.addWeighted(overlay, 0.5, red, 0.5, 0), overlay)

    if water_mask is not None:
        blue = np.zeros_like(vis); blue[:, :, 0] = 255         # water
        overlay = np.where(water_mask[..., None] > 0, cv2.addWeighted(overlay, 0.5, blue, 0.5, 0), overlay)

    if sky_mask is not None:
        cyan = np.zeros_like(vis); cyan[:, :, 0:2] = 255       # sky
        overlay = np.where(sky_mask[..., None] > 0, cv2.addWeighted(overlay, 0.5, cyan, 0.3, 0), overlay)

    vis = overlay

    if radius > 0 and radius != SENTINEL_FULLSAFE_RADIUS:
        cv2.circle(vis, center, radius, (0, 255, 0), 2)
        cv2.circle(vis, center, 4, (0, 255, 0), -1)

    label = f"r={radius}px" if radius != SENTINEL_FULLSAFE_RADIUS else "FULL SAFE"
    cv2.putText(vis, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


# ---- lane-mark based road ban ----
def _lane_candidates_hsv(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 200), (180, 60, 255))
    yellow = cv2.inRange(hsv, (15, 80, 120), (35, 255, 255))
    cand = cv2.bitwise_or(white, yellow)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    tophat = cv2.morphologyEx(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.MORPH_TOPHAT, kernel)
    tophat = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    comb = cv2.bitwise_or(cand, (tophat > 160).astype(np.uint8) * 255)
    comb = cv2.morphologyEx(comb, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return comb


def lane_road_mask_from_hough(bgr: np.ndarray,
                              hough_thresh: int = 60,
                              min_len: int = 40,
                              max_gap: int = 10,
                              thickness: int = 12,
                              dilate_ksize: int = 21) -> np.ndarray:
    h, w = bgr.shape[:2]
    cand = _lane_candidates_hsv(bgr)
    edges = cv2.Canny(cand, 50, 150, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_thresh,
                            minLineLength=min_len, maxLineGap=max_gap)
    road_mask = np.zeros((h, w), np.uint8)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            cv2.line(road_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness)
    if dilate_ksize > 0:
        k = (dilate_ksize // 2) * 2 + 1
        road_mask = cv2.dilate(road_mask, np.ones((k, k), np.uint8), iterations=1)
    return road_mask


def load_static_ban(path: str, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    m = None
    if rasterio is not None and path.lower().endswith((".tif", ".tiff")):
        try:
            with rasterio.open(path) as ds:
                arr = ds.read(1)
                m = (arr > 0).astype(np.uint8) * 255
        except Exception:
            m = None
    if m is None:
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


def main():
    ap = argparse.ArgumentParser()
    # core
    ap.add_argument("--weights", required=True, type=str)
    ap.add_argument("--weights2", type=str, default=None, help="Optional second model for WATER class.")
    ap.add_argument("--source", required=True, type=str, help="Video path or camera index (e.g., 0).")
    ap.add_argument("--out", type=str, default="pred_safe/video_safe.mp4", help="Output MP4 path.")
    ap.add_argument("--outdir", type=str, default="pred_safe", help="Folder for side outputs.")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--exclude", type=str, default="")
    ap.add_argument("--sky", action="store_true")
    ap.add_argument("--obstacle-dilation-kernel-size", type=int, default=50)
    ap.add_argument("--margin", type=int, default=0)
    # lanes
    ap.add_argument("--ban-road-lanes", action="store_true")
    ap.add_argument("--lane-hough-thresh", type=int, default=60)
    ap.add_argument("--lane-min-length", type=int, default=40)
    ap.add_argument("--lane-max-gap", type=int, default=10)
    ap.add_argument("--lane-thickness", type=int, default=12)
    ap.add_argument("--lane-dilate", type=int, default=21)
    # static ban
    ap.add_argument("--static-ban", dest="static_ban", type=str, default=None,
                    help="Path to static ban raster (TIF/PNG). Resized per-frame.")
    # perf
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame (>=1).")
    ap.add_argument("--save-frames", action="store_true", help="Also save annotated frames as JPG.")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    # models
    print(f"[video] Loading seg model: {args.weights}")
    model1 = YOLO(args.weights)
    model2 = YOLO(args.weights2) if args.weights2 else None

    # obstacles
    default_obstacles = set(CLASS_NAME_TO_ID[n] for n in DEFAULT_OBSTACLE_NAMES)
    excludes = parse_exclude(args.exclude)
    obstacle_cids = sorted(list(default_obstacles - excludes))

    # video source (file or cam index)
    src = args.source
    cap: cv2.VideoCapture
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        # read one frame to initialize
        ret, fr0 = cap.read()
        if not ret:
            raise RuntimeError("Video has no frames.")
        height, width = fr0.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = args.out
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # optional static ban (loaded per frame size)
    static_ban_cache = None
    if args.static_ban:
        try:
            static_ban_cache = load_static_ban(args.static_ban, (height, width))
            print(f"[video] Static ban loaded: {args.static_ban}")
        except Exception as e:
            print(f"[WARN] static-ban load failed: {e}")

    # summary
    summary: List[dict] = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fidx = 0
    wrote = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.stride > 1 and (fidx % args.stride != 0):
            # fast path: just write raw frame to keep timing (or copy last result). Here we pass-through.
            writer.write(frame)
            fidx += 1
            continue

        h, w = frame.shape[:2]
        # model #1
        res1 = model1.predict(frame, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)
        res1 = res1[0] if isinstance(res1, list) else res1
        base_mask = masks_for_classes(res1, set(obstacle_cids), (h, w))

        # static ban
        if static_ban_cache is not None:
            base_mask = cv2.bitwise_or(base_mask, static_ban_cache)

        # frame margin ban
        if args.margin and args.margin > 0:
            border = np.ones((h, w), np.uint8) * 255
            cv2.rectangle(border,
                          (args.margin, args.margin),
                          (w - args.margin - 1, h - args.margin - 1),
                          0, thickness=-1)
            base_mask = cv2.bitwise_or(base_mask, border)

        # water (model #2)
        water_mask = None
        if model2 is not None:
            res2 = model2.predict(frame, imgsz=args.imgsz, conf=args.conf, device=device,
                                  verbose=False, classes=[CLASS_NAME_TO_ID["water"]])
            res2 = res2[0] if isinstance(res2, list) else res2
            water_mask = masks_for_classes(res2, {CLASS_NAME_TO_ID["water"]}, (h, w))
            base_mask = cv2.bitwise_or(base_mask, water_mask)

        # sky
        sky_mask = fast_sky_mask(frame) if args.sky else None
        if sky_mask is not None:
            base_mask = cv2.bitwise_or(base_mask, sky_mask)

        # lanes -> road ban
        if args.ban_road_lanes:
            road_mask = lane_road_mask_from_hough(
                frame,
                hough_thresh=args.lane_hough_thresh,
                min_len=args.lane_min_length,
                max_gap=args.lane_max_gap,
                thickness=args.lane_thickness,
                dilate_ksize=args.lane_dilate
            )
            base_mask = cv2.bitwise_or(base_mask, road_mask)

        # post-process
        if base_mask.max() > 0:
            k = max(3, (args.imgsz // 128) | 1)
            base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
            if args.obstacle_dilation_kernel_size > 0:
                dk = (args.obstacle_dilation_kernel_size // 2) * 2 + 1
                base_mask = cv2.dilate(base_mask, np.ones((dk, dk), np.uint8), iterations=1)

        ok, center, radius = largest_safe_circle(base_mask)
        vis = overlay_visual(frame, base_mask, water_mask, sky_mask, center, radius)

        # write
        writer.write(vis)
        wrote += 1

        if args.save_frames:
            fp = str(Path(args.outdir) / f"frame_{fidx:06d}_safe.jpg")
            cv2.imwrite(fp, vis)

        t_sec = fidx / fps if fps > 0 else 0.0
        summary.append({"frame": fidx, "t_sec": round(t_sec, 3), "ok": bool(ok), "radius_px": int(radius)})

        if fidx % 50 == 0:
            if total > 0:
                print(f"[video] {fidx}/{total} frames processed...")
            else:
                print(f"[video] {fidx} frames processed...")

        fidx += 1

    cap.release()
    writer.release()

    # summary JSON
    sum_path = str(Path(args.outdir) / "summary_video.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump({
            "video": args.source,
            "out": args.out,
            "frames_written": wrote,
            "fps_in": fps,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "sky": bool(args.sky),
            "ban_road_lanes": bool(args.ban_road_lanes),
            "static_ban": args.static_ban if args.static_ban else None,
            "margin": args.margin,
            "per_frame": summary
        }, f, ensure_ascii=False, indent=2)

    print(f"[video] Done. Wrote: {args.out}")
    print(f"[video] Summary: {sum_path}")


if __name__ == "__main__":
    main()
