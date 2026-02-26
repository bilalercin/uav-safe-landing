import os, zipfile, glob, shutil, json, random, pathlib, argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2

# === Parametreler ===
NAMES = ["person","vehicle","building","tree","water"]
# person = 0, water = 4 -> diğerleri UAVID'den gelecek
COLOR2CLS = {           # UAVID renkleri (BGR)
    (0,  0,128): 2,     # building (RGB 128,0,0)
    (0,128,  0): 3,     # tree     (RGB 0,128,0)
    (128,0, 64): 1,     # moving car  (RGB 64,0,128)
    (192,0,192): 1,     # static car  (RGB 192,0,192)
    (0, 64, 64): 0,     # human       (RGB 64,64,0)
}
TOL = 1
AREA_MIN = 120
EPS_FRAC = 0.002

def ensure_dirs(root):
    for d in ["images/train","images/val","labels/train","labels/val"]:
        Path(root, d).mkdir(parents=True, exist_ok=True)

def inrange_color(mask_bgr, bgr, tol=TOL):
    lo = np.array([max(0, bgr[0]-tol), max(0, bgr[1]-tol), max(0, bgr[2]-tol)], np.uint8)
    hi = np.array([min(255, bgr[0]+tol), min(255, bgr[1]+tol), min(255, bgr[2]+tol)], np.uint8)
    return cv2.inRange(mask_bgr, lo, hi)

def yolo_poly(cnt, w, h):
    cnt = cnt.reshape(-1,2).astype(np.float32)
    cnt[:,0] = np.clip(cnt[:,0], 0, w-1)/w
    cnt[:,1] = np.clip(cnt[:,1], 0, h-1)/h
    return " ".join([f"{x:.6f} {y:.6f}" for x,y in cnt])

def convert_uavid(uavid_root, out_root):
    TR_IMG = Path(uavid_root, "train_data/Images")
    TR_LAB = Path(uavid_root, "train_data/Labels")
    VA_IMG = Path(uavid_root, "val_data/Images")
    VA_LAB = Path(uavid_root, "val_data/Labels")
    if not TR_IMG.exists():
        print("[UAVID] klasör bulunamadı, atlanıyor.")
        return
    def _convert(img_dir, lab_dir, split):
        out_img = Path(out_root, f"images/{split}")
        out_lab = Path(out_root, f"labels/{split}")
        for ip in sorted(glob.glob(str(img_dir)+"/*")):
            name = Path(ip).stem
            lp = None
            for ext in [".png",".jpg",".jpeg",".tif"]:
                cand = Path(lab_dir, name+ext)
                if cand.is_file():
                    lp = str(cand); break
            if lp is None:
                continue
            img = cv2.imread(ip)
            lab = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
            if lab is None or img is None or lab.ndim != 3:
                continue
            h,w = img.shape[:2]
            lines=[]
            for bgr, cls in COLOR2CLS.items():
                mask = inrange_color(lab, bgr, TOL)
                if mask.sum()==0:
                    continue
                cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    if cv2.contourArea(c) < AREA_MIN:
                        continue
                    peri = cv2.arcLength(c, True)
                    c = cv2.approxPolyDP(c, EPS_FRAC*peri, True)
                    lines.append(f"{cls} {yolo_poly(c,w,h)}")
            if not lines:
                continue
            cv2.imwrite(str(out_img/ f"{name}.jpg"), img)
            (out_lab/ f"{name}.txt").write_text("\n".join(lines))
    _convert(TR_IMG, TR_LAB, "train")
    _convert(VA_IMG, VA_LAB, "val")
    print("[UAVID] dönüştürme bitti.")

def unzip(zpath, outdir):
    with zipfile.ZipFile(zpath, 'r') as zf:
        zf.extractall(outdir)

def roboflow_merge(rf_zip, dataset_root, prefix, target_cid):
    tmp = Path(dataset_root, "_tmp_rf"); tmp.mkdir(exist_ok=True)
    out = Path(tmp, prefix); out.mkdir(exist_ok=True)
    unzip(rf_zip, out)
    # Roboflow export genelde out/* tek klasör açıyor
    entries = [p for p in out.glob("*") if p.is_dir()]
    data_root = entries[0] if len(entries)==1 else out
    split_map = {"train":"train","valid":"val","val":"val"}
    copied = 0
    for sp_src, sp_dst in split_map.items():
        img_dir = Path(data_root, sp_src, "images")
        lbl_dir = Path(data_root, sp_src, "labels")
        if not img_dir.is_dir():
            continue
        for img in img_dir.glob("*"):
            base = img.stem
            ext  = img.suffix
            new_name = f"{prefix}_{sp_dst}_{base}{ext}"
            dst_img  = Path(dataset_root, "images", sp_dst, new_name)
            shutil.copy2(img, dst_img)
            src_txt = Path(lbl_dir, base+".txt")
            dst_txt = Path(dataset_root, "labels", sp_dst, f"{prefix}_{sp_dst}_{base}.txt")
            if src_txt.exists():
                lines_out=[]
                for ln in src_txt.read_text().splitlines():
                    xs = ln.split()
                    if not xs:
                        continue
                    xs[0] = str(target_cid)
                    lines_out.append(" ".join(xs))
                dst_txt.parent.mkdir(parents=True, exist_ok=True)
                dst_txt.write_text("\n".join(lines_out)+("\n" if lines_out else ""))
            copied += 1
    print(f"[Roboflow-{prefix}] kopyalanan görsel:", copied)

def add_coco_person(coco_root, dataset_root, max_images=600, val_ratio=0.15, prefix="coco_person"):
    ann_file = Path(coco_root, "annotations/instances_val2017.json")
    if not ann_file.exists():
        print("[COCO] Bulunamadı, atlanıyor. (İstersen --coco-root ver.)"); return
    from pycocotools.coco import COCO
    coco = COCO(str(ann_file))
    cat_id = coco.getCatIds(catNms=['person'])[0]
    img_ids = coco.getImgIds(catIds=[cat_id])
    random.seed(0); random.shuffle(img_ids); img_ids = img_ids[:max_images]
    added_tr = added_val = 0
    for img_id in tqdm(img_ids, desc="COCO person"):
        img_info = coco.loadImgs([img_id])[0]
        w, h = img_info["width"], img_info["height"]
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cat_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        polys_all=[]
        for a in anns:
            seg = a.get("segmentation", None)
            if not seg or isinstance(seg, dict):
                continue
            for poly in seg:
                if len(poly) < 6: continue
                xs = poly[0::2]; ys = poly[1::2]
                poly_norm=[]
                for x,y in zip(xs,ys):
                    poly_norm += [max(0,min(1,x/w)), max(0,min(1,y/h))]
                polys_all.append(poly_norm)
        if not polys_all:
            continue
        split = "val" if random.random() < val_ratio else "train"
        name  = f"{prefix}_{split}_{Path(img_info['file_name']).stem}"
        src_img = Path(coco_root, "val2017", img_info["file_name"])
        if not src_img.exists():
            continue
        dst_img = Path(dataset_root, "images", split, name+".jpg")
        shutil.copy2(src_img, dst_img)
        dst_lbl = Path(dataset_root, "labels", split, name+".txt")
        with open(dst_lbl, "w") as f:
            for poly in polys_all:
                if len(poly) >= 6:
                    f.write("0 " + " ".join(f"{v:.6f}" for v in poly) + "\n")
        if split=="val": added_val += 1
        else: added_tr += 1
    print(f"[COCO] eklendi -> train:{added_tr}  val:{added_val}")

def write_yaml(dataset_root, yaml_path):
    text = "path: {}\ntrain: images/train\nval: images/val\nnames:\n".format(dataset_root)
    text += "\n".join([f"  {i}: {n}" for i,n in enumerate(NAMES)]) + "\n"
    Path(yaml_path).write_text(text)
    print("YAML yazıldı ->", yaml_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="data/safe-landing", type=str)
    ap.add_argument("--uavid-root",   default="", type=str, help="UAVID local kök (opsiyonel)")
    ap.add_argument("--rf-water-zip", default="", type=str, help="Roboflow WATER zip yolu (YOLOv8-Seg export)")
    ap.add_argument("--coco-root",    default="", type=str, help="COCO2017 kök (opsiyonel)")
    ap.add_argument("--yaml-out",     default="safe-landing.yaml", type=str)
    args = ap.parse_args()

    ds_root = Path(args.dataset_root); ensure_dirs(ds_root)

    if args.uavid_root:
        convert_uavid(args.uavid_root, ds_root)

    if args.rf_water_zip and Path(args.rf_water_zip).exists():
        roboflow_merge(args.rf_water_zip, ds_root, prefix="water", target_cid=4)

    if args.coco_root:
        add_coco_person(args.coco_root, ds_root)

    # YOLO cache temizlik (varsa)
    for p in list(ds_root.glob("labels/*.cache")) + list(ds_root.glob("labels/*/*.cache")):
        try: p.unlink()
        except: pass

    write_yaml(ds_root, args.yaml_out)
