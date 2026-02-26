from ultralytics import YOLO
from pathlib import Path
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source",  required=True, help="görsel/klasör/video")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=0)   # CPU: 'cpu'
    ap.add_argument("--name", default="pred_local")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=".",
        name=args.name,
        save=True
    )
    print("Bitti. Çıkış klasörü: ./", args.name)

