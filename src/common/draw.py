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


def draw_trail(img, points, color, thickness=2, max_jump=120):
    """İz çizgisi. Kimlik başka bir nesneye atladığında oluşan uzun düz çizgileri çizmez."""
    pts = list(points)
    for p, q in zip(pts[:-1], pts[1:]):
        if abs(p[0] - q[0]) + abs(p[1] - q[1]) <= max_jump:
            cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, thickness, cv2.LINE_AA)


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


# ---------------------------------------------------------------- kart tarzı panel
ACCENT = (255, 170, 40)      # BGR: turuncu-mavi yerine sıcak sarı vurgu
CARD_BG = (28, 28, 28)
MUTED = (170, 170, 170)


def _blend_rect(img, x1, y1, x2, y2, color, alpha):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    cv2.addWeighted(np.full_like(roi, color), alpha, roi, 1 - alpha, 0, roi)


def draw_card(img, x, y, w, title, big=None, rows=(), footer=None, accent=ACCENT):
    """Başlık şeridi + (isteğe bağlı) büyük sayı + satırlar + alt bilgi. rows: [(sol, sağ), ...]"""
    line_h, pad = 26, 12
    h = 34 + (52 if big is not None else 0) + line_h * len(rows) + (24 if footer else 0) + pad
    _blend_rect(img, x, y, x + w, y + h, CARD_BG, 0.72)
    cv2.rectangle(img, (x, y), (x + w, y + 30), accent, -1)
    cv2.putText(img, title, (x + pad, y + 21), FONT, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    cy = y + 34
    if big is not None:
        cv2.putText(img, str(big), (x + pad, cy + 42), FONT, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cy += 52
    for left, right in rows:
        cy += line_h
        cv2.putText(img, str(left), (x + pad, cy - 6), FONT, 0.56, MUTED, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(str(right), FONT, 0.6, 2)
        cv2.putText(img, str(right), (x + w - pad - tw, cy - 6), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if footer:
        cv2.putText(img, footer, (x + pad, cy + 18), FONT, 0.45, MUTED, 1, cv2.LINE_AA)
    return h


def draw_plate_badge(img, x_center, y_top, text, conf=None, valid=False):
    """Plaka görünümlü rozet: beyaz zemin, siyah yazı; geçerli formatta yeşil, değilse turuncu kenar."""
    if not text:
        return
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.7, 2)
    w, h = tw + 18, th + 14
    x1 = int(x_center - w / 2); y1 = int(y_top)
    H, W = img.shape[:2]
    x1 = min(max(x1, 0), W - w); y1 = min(max(y1, 0), H - h - 16)
    border = (60, 200, 60) if valid else (0, 160, 255)
    cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (250, 250, 250), -1)
    cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), border, 2)
    cv2.putText(img, text, (x1 + 9, y1 + h - 8), FONT, 0.7, (15, 15, 15), 2, cv2.LINE_AA)
    if conf is not None:
        cv2.putText(img, f"{conf:.2f}", (x1, y1 + h + 14), FONT, 0.45, border, 1, cv2.LINE_AA)


def draw_table(img, x, y, title, header, rows, col_x, accent=ACCENT, width=None):
    """Basit tablo kartı. col_x: her sütunun karta göre x konumu."""
    line_h, pad = 26, 12
    width = width or (col_x[-1] + 160)
    h = 34 + line_h * (len(rows) + 1) + pad
    _blend_rect(img, x, y, x + width, y + h, CARD_BG, 0.72)
    cv2.rectangle(img, (x, y), (x + width, y + 30), accent, -1)
    cv2.putText(img, title, (x + pad, y + 21), FONT, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    cy = y + 34 + line_h
    for cx, hd in zip(col_x, header):
        cv2.putText(img, hd, (x + cx, cy - 8), FONT, 0.48, MUTED, 1, cv2.LINE_AA)
    for r in rows:
        cy += line_h
        for i, (cx, val) in enumerate(zip(col_x, r)):
            col = (255, 255, 255)
            if isinstance(val, tuple):  # (metin, renk)
                val, col = val
            cv2.putText(img, str(val), (x + cx, cy - 8), FONT, 0.55, col, 2 if i == len(r) - 1 else 1, cv2.LINE_AA)
    return h
