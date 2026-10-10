"""Sahne kesmesi tespiti.

Ardışık karelerin küçültülmüş (160x90) gri görüntüleri arasındaki ortalama mutlak fark (MAD).
Kafe videosunda ölçülen değerler: normal karelerde medyan 0.4, en yüksek ~3.2;
6 gerçek kesmede 16–28. Varsayılan eşik 10 bu ikisinin arasında geniş pay bırakır.
"""
import cv2
import numpy as np


class SceneCutDetector:
    def __init__(self, threshold=10.0, cooldown=5, size=(160, 90)):
        self.threshold = threshold
        self.cooldown = cooldown  # bir kesmeden sonra kaç kare yeni kesme aranmaz (geçiş kareleri için)
        self.size = size
        self.prev = None
        self._wait = 0
        self.last_score = 0.0

    def __call__(self, frame) -> bool:
        g = cv2.cvtColor(cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        cut = False
        if self.prev is not None:
            self.last_score = float(np.mean(cv2.absdiff(g, self.prev)))
            if self._wait > 0:
                self._wait -= 1
            elif self.last_score > self.threshold:
                cut = True
                self._wait = self.cooldown
        self.prev = g
        return cut
