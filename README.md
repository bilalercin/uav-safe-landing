# Safe Landing Zone Detection for UAVs

## Description
This project focuses on **UAV safe landing zone (LZ) selection**, leveraging **YOLOv8-Seg based obstacle segmentation**, **water masking (ensemble)**, **sky mask**, and **single-click safe circle** calculation.

It provides a machine learning solution for identifying safe landing zones for Unmanned Aerial Vehicles (UAVs) by utilizing object detection and/or image segmentation techniques to detect potential obstacles (e.g., people, water) within the landing area and generates safe landing maps.

## 1) Objective and Scope
- To segment obstacles such as **humans, vehicles, buildings, and trees** in the image and determine a **safe area for landing**.
- To prohibit **water** (lake/sea) surfaces (using a second model in an ensemble).
- To remove the sky from the image using an **HSV-based** approach (sky mask).
- As a result, draw the **largest safe circle**, save it to the visual, and print summaries like `ok` and `radius`.

> Class Map: `0: person`, `1: vehicle`, `2: building`, `3: tree`, `4: water`, `5: animal`

---

## 2) Data Preparation (Dataset Merge)
### 2.1 UAVID → YOLOv8-Seg Conversion
- We converted UAVID colored label masks (RGB) to **YOLOv8-Seg polygon** format using a **color → class** mapping.
- The (BGR) color-mapping we used (summary):
  - **building** `(128,0,0 RGB) → BGR (0,0,128) → cid=2`
  - **tree** `(0,128,0 RGB) → BGR (0,128,0) → cid=3`
  - **moving/static car** `(64,0,128 / 192,0,192 RGB) → BGR (128,0,64 / 192,0,192) → cid=1`
  - **human** `(64,64,0 RGB) → BGR (0,64,64) → cid=0`
- We used `AREA_MIN` to discard small fragments and `approxPolyDP` for contour simplification.

### 2.2 Roboflow “Water” → class id 4
- We downloaded **water** data in YOLOv8-Seg format from Roboflow, rewrote the labels so that the **first token was 4** (water), and copied them to the `images/{train,val}`, `labels/{train,val}` structure.

### 2.3 COCO 2017 (val) → Adding Person Polygons
- We read COCO `person` segmentations with `pycocotools` and converted them to YOLOv8-Seg polygon lines with **normalization**.
- We enriched the dataset with `train/val` split ratio and naming (`coco_person_train_*`).

### 2.4 YAML Generation
- `safe-landing.yaml` was created for the merged dataset:
  ```yaml
  path: datasets/safe-landing
  train: images/train
  val: images/val
  names:
    0: person
    1: vehicle
    2: building
    3: tree
    4: water
  ```

## 3) Training (Ultralytics YOLOv8-Seg)
### 3.1 Basic Training
```bash
yolo task=segment mode=train \
  model=yolov8n-seg.pt \
  data=/PATH/TO/safe-landing.yaml \
  imgsz=768 batch=4 epochs=80 amp=True workers=2 device=0 \
  project=/PATH/TO/runs name=safe_landing_runs_new2
```

### 3.2 Fine-tuning (with person+water)
```bash
yolo task=segment mode=train \
  model=/PATH/TO/runs/safe_landing_runs_new2/weights/best.pt \
  data=/PATH/TO/safe-landing.yaml \
  imgsz=768 batch=4 epochs=10 amp=True workers=2 device=0 \
  project=/PATH/TO/runs name=safe_landing_ft_person_water
```
Selected model: `safe_landing_ft_person_water/weights/best.pt` (ensemble water performance is better).

## 4) Inference – src/infer_safe.py
Key features:

Model #1 (mandatory): obstacles (person, vehicle, building, tree)

Model #2 (optional): to add water mask (water=cid4)

Sky mask (optional): HSV-based sky removal

Post-masking: morphological closing + dilation for safety margin

Circle calculation: largest inscribed circle in the free area (distance transform)

### Example Usage (Local)
```bash
python -m src.infer_safe \
  --weights   "modified_uavid_dataset/safe_landing_runs_new2/weights/best.pt" \
  --weights2  "modified_uavid_dataset/safe_landing_ft_person_water/weights/best.pt" \
  --source    "modified_uavid_dataset/test_data/Images" \
  --outdir    "pred_safe" \
  --device    mps \
  --imgsz     768 \
  --conf      0.35 \
  --sky \
  --obstacle-dilation-kernel-size 50
```
Output image: `_safe.jpg`

Console: `-> pred_safe/xxx_safe.jpg (ok=True, r=738)`

`r=65533` → FULL SAFE (no obstacles)

Colors: `red=obstacle`, `blue=water`, `cyan=sky`, `green=circle`

Note: You can exclude specific classes with `--exclude` (e.g., `--exclude tree`).

## 5) Environment and Setup
### Requirements
```bash
python>=3.10
ultralytics
opencv-python
numpy
tqdm
pycocotools   # (for adding COCO person)
```
### Colab (summary):
```bash
pip install ultralytics opencv-python-headless numpy tqdm pycocotools
```
### Local (Mac/Windows/Linux):
```bash
pip install ultralytics opencv-python numpy tqdm pycocotools
```
We worked with `--device mps` for Apple Silicon; use `--device 0` for NVIDIA GPUs.

## 6) Roadmap
Water: Reinforced with ensemble model, covering what a single model missed.

Sky: Simple HSV mask added (upper band + morphological closing to reduce false positives).

Road prohibition (experimental): Road detection / lane cues from image will be added later; static OSM mask was to be used at that stage.

VisDrone (optional): A separate YOLOv8-Detect model was prepared for vehicle/pedestrian detection (integration trials were performed).

## Troubleshooting
`r=65533` appears too often → Do not decrease `--conf`; check water/sky masking, increase `--obstacle-dilation-kernel-size`.

Water/sky mix-up → Turn off `--sky` and compare; use second model (`--weights2`) for water.

Safe areas within roads → Static (OSM) or lane-based road masking will be added in the next phase.

## License and Attributions
Datasets: UAVID, COCO, Roboflow (Water). Please adhere to relevant licenses and attributions.

Model and code skeleton: Ultralytics YOLOv8.
