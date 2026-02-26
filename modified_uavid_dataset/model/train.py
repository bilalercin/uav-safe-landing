from ultralytics import YOLO
from pathlib import Path
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="safe-landing.yaml")
    ap.add_argument("--weights", default="yolov8n-seg.pt")   # veya kendi ft ağırlığın
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=0)                   # CPU istiyorsan 'cpu'
    ap.add_argument("--name", default="safe_landing_runs_local")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=2,
        device=args.device,
        project=".",
        name=args.name,
        amp=True
    )
