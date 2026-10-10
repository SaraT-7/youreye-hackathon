"""Adım 2 — Kişi takibi (ByteTrack / BoT-SORT) + sahne kesmesinde takibi sıfırlama.

Neden global kimlik: takip algoritması sıfırlanınca kimlik sayacı 1'den yeniden başlar.
(klip, yerel_id) çiftini artan bir global kimliğe eşleyerek farklı kliplerdeki kişilerin
aynı numarayı almasını engelliyoruz.
"""
from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

from src.task1_cafe.filters import suppress_nested

PERSON_CLASS = 0


@dataclass
class Tracks:
    xyxy: np.ndarray  # (N, 4)
    conf: np.ndarray  # (N,)
    ids: np.ndarray   # (N,) global kimlikler

    def __len__(self):
        return len(self.ids)


class PersonTracker:
    def __init__(self, model="yolo11l.pt", tracker_cfg="configs/botsort_cafe.yaml", imgsz=1280,
                 det_conf=0.1, iou=0.5, half=True, device=0, nested_filter=True):
        self.model = YOLO(model)
        self.suppressed = 0  # bastırılan iç içe kutu sayısı (özet için)
        if nested_filter:
            # Takip algoritmasının callback'inden ÖNCE kaydedilir -> tespitler takibe girmeden süzülür.
            self.model.add_callback("on_predict_postprocess_end", self._filter_nested)
        self.kw = dict(tracker=tracker_cfg, persist=True, imgsz=imgsz, conf=det_conf, iou=iou,
                       half=half, device=device, classes=[PERSON_CLASS], verbose=False)
        self.segment = 0
        self._map = {}       # (segment, local_id) -> global_id
        self._next_gid = 1

    def _filter_nested(self, predictor):
        for i, r in enumerate(predictor.results):
            b = r.boxes
            if b is None or len(b) < 2:
                continue
            keep, new_conf = suppress_nested(b.xyxy.cpu().numpy(), b.conf.cpu().numpy())
            if len(keep) == len(b):
                continue
            self.suppressed += len(b) - len(keep)
            feats = getattr(r, "feats", None)  # BoT-SORT ReID özellikleri kutularla aynı sırada tutulmalı
            idx = torch.as_tensor(keep, dtype=torch.long, device=b.data.device)
            nr = r[idx]
            data = nr.boxes.data.clone()
            data[:, 4] = torch.as_tensor(new_conf, dtype=data.dtype, device=data.device)
            nr.update(boxes=data)
            if feats is not None:
                nr.feats = feats[idx.to(feats.device)]
            predictor.results[i] = nr

    def reset(self):
        """Sahne kesmesinde çağrılır: izler silinir, yeni klip yeni kimlikler alır."""
        pred = getattr(self.model, "predictor", None)
        for t in getattr(pred, "trackers", None) or []:
            t.reset()
        self.segment += 1

    def _gid(self, local_id):
        key = (self.segment, int(local_id))
        if key not in self._map:
            self._map[key] = self._next_gid
            self._next_gid += 1
        return self._map[key]

    @property
    def total_ids(self):
        return self._next_gid - 1

    def __call__(self, frame) -> Tracks:
        r = self.model.track(frame, **self.kw)[0]
        b = r.boxes
        if b is None or b.id is None or len(b) == 0:
            return Tracks(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))
        ids = np.array([self._gid(i) for i in b.id.int().cpu().tolist()], dtype=int)
        return Tracks(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), ids)
