"""Araç tespiti + takibi (YOLO11, COCO: car / motorcycle / bus / truck) + global kimlik."""
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "Otomobil", 3: "Motosiklet", 5: "Otobus", 7: "Kamyon"}


@dataclass
class VTracks:
    xyxy: np.ndarray
    conf: np.ndarray
    ids: np.ndarray
    cls: np.ndarray

    def __len__(self):
        return len(self.ids)


class VehicleTracker:
    def __init__(self, model="yolo11l.pt", tracker_cfg="configs/botsort_traffic.yaml", imgsz=960,
                 det_conf=0.1, half=True, device=0):
        self.model = YOLO(model)
        self.kw = dict(tracker=tracker_cfg, persist=True, imgsz=imgsz, conf=det_conf, iou=0.5, half=half,
                       device=device, classes=list(VEHICLE_CLASSES), verbose=False)
        self.segment = 0
        self._map, self._next = {}, 1
        self._cls_votes = defaultdict(Counter)  # gid -> sınıf oyları (tür kareden kareye titremesin)

    def reset(self):
        for t in getattr(getattr(self.model, "predictor", None), "trackers", None) or []:
            t.reset()
        self.segment += 1

    @property
    def total_ids(self):
        return self._next - 1

    def vtype(self, gid):
        c = self._cls_votes.get(gid)
        return VEHICLE_CLASSES[c.most_common(1)[0][0]] if c else "?"

    def __call__(self, frame):
        r = self.model.track(frame, **self.kw)[0]
        b = r.boxes
        if b is None or b.id is None or len(b) == 0:
            z = np.zeros(0)
            return VTracks(np.zeros((0, 4)), z, z.astype(int), z.astype(int))
        ids = []
        for i in b.id.int().cpu().tolist():
            key = (self.segment, i)
            if key not in self._map:
                self._map[key] = self._next
                self._next += 1
            ids.append(self._map[key])
        ids = np.array(ids, dtype=int)
        cls = b.cls.int().cpu().numpy()
        conf = b.conf.cpu().numpy()
        for g, c, s in zip(ids, cls, conf):
            self._cls_votes[int(g)][int(c)] += float(s)
        return VTracks(b.xyxy.cpu().numpy(), conf, ids, cls)
