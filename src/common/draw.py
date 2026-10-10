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
