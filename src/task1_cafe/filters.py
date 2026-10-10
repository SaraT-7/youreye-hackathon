"""İç içe (aynı kişiye ait) kutuları bastırma.

Sorun: masada oturan bir kişi hem "masanın üstü" (baş + gövde) hem "baştan ayağa" iki ayrı kutu
olarak bulunabiliyor. NMS bunu yakalamıyor, çünkü küçük kutu büyüğün içinde olduğu için IoU
genelde ~0.4–0.5 kalıyor (NMS eşiği 0.5). Sonuç: aynı kişiye iki kimlik, ve kutu şekli kareden
kareye değiştikçe kimlik kopması.

Kural: küçük kutu S, büyük kutu L için
  1) S'nin alanının en az %80'i L'nin içinde (içerme oranı, IoU değil),
  2) üst kenarları yakın (|S.y1 - L.y1| < 0.25 * L.yükseklik)  -> ikisi de aynı başı içeriyor,
  3) S'nin genişliği L'ninkinin en az %60'ı                  -> iki kişiyi kapsayan geniş kutu değil.
Üçü de sağlanırsa S silinir; L kalır (ayak noktası sayım için gerekli) ve güveni max(S, L) olur.
Masa arkasında duran ikinci bir kişi genelde koşul 2'yi sağlamaz (başı farklı yükseklikte).
"""
import numpy as np


def suppress_nested(xyxy, conf, contain_thr=0.8, top_tol=0.25, min_width_ratio=0.6):
    """(keep_indices, new_conf) döndürür. new_conf, keep sırasındaki güvenlerdir."""
    n = len(conf)
    if n < 2:
        return np.arange(n), np.asarray(conf, dtype=float).copy()
    xyxy = np.asarray(xyxy, dtype=float)
    conf = np.asarray(conf, dtype=float).copy()
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    area = w * h
    alive = np.ones(n, dtype=bool)
    for li in np.argsort(-area):              # büyükten küçüğe
        if not alive[li]:
            continue
        for si in range(n):
            if si == li or not alive[si] or area[si] > area[li]:
                continue
            ix1 = max(xyxy[li, 0], xyxy[si, 0]); iy1 = max(xyxy[li, 1], xyxy[si, 1])
            ix2 = min(xyxy[li, 2], xyxy[si, 2]); iy2 = min(xyxy[li, 3], xyxy[si, 3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if (inter / max(area[si], 1e-6) >= contain_thr
                    and abs(xyxy[si, 1] - xyxy[li, 1]) < top_tol * h[li]
                    and w[si] >= min_width_ratio * w[li]):
                alive[si] = False
                conf[li] = max(conf[li], conf[si])
    keep = np.flatnonzero(alive)
    return keep, conf[keep]
