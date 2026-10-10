"""Adım 1 — Kişi tespiti (YOLO11, COCO ön eğitimli, yalnızca 'person' sınıfı)."""
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

PERSON_CLASS = 0  # COCO'da 'person'


@dataclass
class Detections:
    xyxy: np.ndarray  # (N, 4) piksel koordinatı
    conf: np.ndarray  # (N,)

    def __len__(self):
        return len(self.conf)


class PersonDetector:
    def __init__(self, model="yolo11l.pt", imgsz=1280, conf=0.35, iou=0.5, half=True, device=0):
        self.model = YOLO(model)
        self.kw = dict(imgsz=imgsz, conf=conf, iou=iou, half=half, device=device,
                       classes=[PERSON_CLASS], verbose=False)

    def warmup(self, frame, n=3):
        """İlk çağrılar (CUDA başlatma) FPS ölçümünü bozmasın diye."""
        for _ in range(n):
            self.model.predict(frame, **self.kw)

    def __call__(self, frame) -> Detections:
        r = self.model.predict(frame, **self.kw)[0]
        b = r.boxes
        return Detections(xyxy=b.xyxy.cpu().numpy(), conf=b.conf.cpu().numpy())
