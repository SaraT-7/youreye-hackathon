"""Sanal çizgiden geçiş sayımı (Aşama 1 ve Aşama 2 ortak).

Yöntem
- Her iz için bir referans noktası kullanılır (kişi: ayak = kutunun alt ortası).
- Noktanın çizgiye işaretli uzaklığı d = ((B-A) x (P-A)) / |B-A| hesaplanır.
  İşaret, noktanın çizginin hangi tarafında olduğunu verir.
- Histerezis: |d| < band ise iz "belirsiz bölgede" sayılır ve tarafı güncellenmez.
  Böylece çizgi üzerinde titreyen bir kutu tekrar tekrar sayılmaz.
- Bir izin onaylı tarafı (-1 / +1) değiştiğinde geçiş kaydedilir:
  -1 -> +1 = "in", +1 -> -1 = "out".
- Her kimlik her yönde EN FAZLA BİR KEZ sayılır.
- Nokta, çizgi parçasının izdüşüm aralığının dışındaysa (margin payıyla) yok sayılır;
  yani çizginin uzantısından geçenler sayılmaz.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CrossEvent:
    frame: int
    track_id: int
    direction: str  # "in" | "out"


@dataclass
class LineCounter:
    a: tuple                 # çizgi başlangıcı (x, y)
    b: tuple                 # çizgi sonu (x, y)
    band: float = 15.0       # histerezis yarı genişliği (piksel)
    margin: float = 0.05     # çizgi parçası uzunluğunun bu oranı kadar uç payı
    in_sign: int = 1         # hangi taraf "içeri" (+1: A->B yönünün sağı, görüntü koordinatlarında)
    _side: dict = field(default_factory=dict)
    _counted: set = field(default_factory=set)
    events: list = field(default_factory=list)

    def __post_init__(self):
        self.A = np.array(self.a, dtype=float)
        self.B = np.array(self.b, dtype=float)
        self.AB = self.B - self.A
        self.L = float(np.linalg.norm(self.AB))

    @property
    def n_in(self):
        return sum(e.direction == "in" for e in self.events)

    @property
    def n_out(self):
        return sum(e.direction == "out" for e in self.events)

    def signed_distance(self, p):
        ap = np.asarray(p, dtype=float) - self.A
        return float(self.AB[0] * ap[1] - self.AB[1] * ap[0]) / self.L

    def _within_segment(self, p):
        t = float(np.dot(np.asarray(p, dtype=float) - self.A, self.AB)) / (self.L ** 2)
        return -self.margin <= t <= 1 + self.margin

    def reset_tracks(self):
        """Sahne kesmesinde: taraf hafızası silinir, sayaçlar korunur."""
        self._side.clear()

    def update(self, frame_idx, track_ids, points):
        """Bu karedeki izleri işler; yeni geçiş olaylarını döndürür."""
        new = []
        for tid, p in zip(track_ids, points):
            tid = int(tid)
            if not self._within_segment(p):
                continue
            d = self.signed_distance(p) * self.in_sign
            if abs(d) < self.band:
                continue                      # belirsiz bölge: taraf değişmez
            side = 1 if d > 0 else -1
            prev = self._side.get(tid)
            self._side[tid] = side
            if prev is None or prev == side:
                continue
            direction = "in" if side == 1 else "out"
            key = (tid, direction)
            if key in self._counted:
                continue
            self._counted.add(key)
            ev = CrossEvent(frame_idx, tid, direction)
            self.events.append(ev)
            new.append(ev)
        return new
