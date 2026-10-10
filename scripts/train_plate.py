"""Plaka tespiti modeli eğitimi.

Veri seti : keremberke/license-plate-object-detection (HuggingFace, Roboflow kaynaklı, COCO formatı)
Başlangıç : yolo11m.pt (COCO ön eğitimli)
Çıktı     : runs/plate/yolo11m_plate/weights/best.pt  ->  models/plate_yolo11m.pt

Kullanım (proje kökünden):
    pip install huggingface_hub
    python scripts/train_plate.py                 # varsayılan ayarlar
    python scripts/train_plate.py --epochs 50 --batch 8
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS_DIR = ROOT / "datasets" / "plates"
REPO = "keremberke/license-plate-object-detection"
SPLITS = {"train": "train", "valid": "val", "test": "test"}


def download_and_extract():
    from huggingface_hub import hf_hub_download

    raw = DS_DIR / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        out = raw / split
        if out.exists() and any(out.rglob("*")):
            print(f"[veri] {split} zaten var, atlanıyor")
            continue
        print(f"[veri] {split}.zip indiriliyor...")
        zp = hf_hub_download(REPO, f"data/{split}.zip", repo_type="dataset")
        with zipfile.ZipFile(zp) as z:
            z.extractall(out)
    return raw


def coco_to_yolo(raw: Path):
    """Her split'teki COCO json'u YOLO formatına çevirir. Tek sınıf: plate."""
    for split, ysplit in SPLITS.items():
        img_out = DS_DIR / "images" / ysplit
        lbl_out = DS_DIR / "labels" / ysplit
        if img_out.exists() and any(img_out.iterdir()):
            print(f"[dönüşüm] {ysplit} zaten hazır")
            continue
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        jsons = list((raw / split).rglob("*.json"))
        if not jsons:
            raise FileNotFoundError(f"{raw / split} içinde COCO json bulunamadı")
        ann_path = jsons[0]
        coco = json.loads(ann_path.read_text(encoding="utf-8"))
        img_root = ann_path.parent

        images = {im["id"]: im for im in coco["images"]}
        boxes = {}
        for a in coco["annotations"]:
            boxes.setdefault(a["image_id"], []).append(a["bbox"])

        n = 0
        for iid, im in images.items():
            src = img_root / im["file_name"]
            if not src.exists():
                hits = list(img_root.rglob(Path(im["file_name"]).name))
                if not hits:
                    continue
                src = hits[0]
            W, H = im["width"], im["height"]
            lines = []
            for x, y, w, h in boxes.get(iid, []):
                if w <= 1 or h <= 1:
                    continue
                cx, cy = (x + w / 2) / W, (y + h / 2) / H
                lines.append(f"0 {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
            dst = img_out / src.name
            shutil.copy2(src, dst)
            (lbl_out / (src.stem + ".txt")).write_text("\n".join(lines))
            n += 1
        print(f"[dönüşüm] {ysplit}: {n} görsel")

    yaml = DS_DIR / "plates.yaml"
    yaml.write_text(
        f"path: {DS_DIR.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: plate\n",
        encoding="utf-8",
    )
    return yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11m.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    raw = download_and_extract()
    data_yaml = coco_to_yolo(raw)

    from ultralytics import YOLO

    model = YOLO(args.model)
    name = f"{Path(args.model).stem}_plate"
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=10,
        cos_lr=True,
        project=str(ROOT / "runs" / "plate"),
        name=name,
        exist_ok=True,
        device=0,
        amp=True,
        plots=True,
    )
    best = ROOT / "runs" / "plate" / name / "weights" / "best.pt"
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy2(best, ROOT / "models" / f"plate_{Path(args.model).stem}.pt")
    print(f"[bitti] En iyi model: models/plate_{Path(args.model).stem}.pt")


if __name__ == "__main__":  # Windows'ta DataLoader için gerekli
    main()
