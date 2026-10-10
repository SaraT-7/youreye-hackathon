"""Ekrana çizim yardımcıları (OpenCV)."""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_box(img, xyxy, label, color=(0, 200, 255), thickness=2, font_scale=0.55):
    """Kutu + üstünde dolgulu etiket."""
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), base = cv2.getTextSize(label, FONT, font_scale, 1)
        ty = max(y1, th + base + 2)
        cv2.rectangle(img, (x1, ty - th - base - 2), (x1 + tw + 4, ty), color, -1)
        cv2.putText(img, label, (x1 + 2, ty - base), FONT, font_scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_panel(img, lines, origin=None, font_scale=0.7, pad=10, bg=(30, 30, 30), alpha=0.65):
    """Yarı saydam bilgi paneli. origin verilmezse sağ üst köşe."""
    sizes = [cv2.getTextSize(t, FONT, font_scale, 2)[0] for t in lines]
    w = max(s[0] for s in sizes) + 2 * pad
    line_h = max(s[1] for s in sizes) + 12
    h = line_h * len(lines) + pad
    H, W = img.shape[:2]
    x, y = origin if origin else (W - w - 15, 15)
    roi = img[y:y + h, x:x + w]  # yalnızca panel bölgesini karıştır (tam kare kopyası pahalı)
    cv2.addWeighted(np.full_like(roi, bg), alpha, roi, 1 - alpha, 0, roi)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x + pad, y + pad + line_h * (i + 1) - 10), FONT, font_scale,
                    (255, 255, 255), 2, cv2.LINE_AA)


# Kimlik başına sabit renk (aynı kişi hep aynı renkte)
_ID_COLORS = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237), (164, 123, 232),
              (0, 131, 0), (214, 80, 98), (72, 73, 227), (180, 180, 40), (40, 180, 180)]


def id_color(track_id):
    return _ID_COLORS[int(track_id) % len(_ID_COLORS)]


def draw_trail(img, points, color, thickness=2):
    if len(points) >= 2:
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)


def draw_count_line(img, a, b, in_sign=1, flash=False, label_in="ICERI"):
    """Sayım çizgisi + 'içeri' yönünü gösteren ok. flash=True iken (geçiş anında) kalın ve yeşil."""
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    color = (60, 220, 60) if flash else (0, 215, 255)
    cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, 6 if flash else 3, cv2.LINE_AA)
    ab = b - a
    n = np.array([-ab[1], ab[0]]) / (np.linalg.norm(ab) + 1e-9) * in_sign  # "in" tarafına bakan normal
    m = (a + b) / 2
    tip = m + n * 45
    cv2.arrowedLine(img, tuple(m.astype(int)), tuple(tip.astype(int)), color, 3, cv2.LINE_AA, tipLength=0.35)
    cv2.putText(img, label_in, tuple((tip + n * 8 + np.array([8, 8])).astype(int)), FONT, 0.6, color, 2, cv2.LINE_AA)
