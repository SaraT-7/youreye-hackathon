"""Plaka bölgesi tespiti (araç kırpımı içinde, kendi eğittiğimiz YOLO11m) + OCR (EasyOCR)."""
import cv2
import numpy as np
from ultralytics import YOLO

from src.task2_vehicle.plate_text import ALNUM


class PlateDetector:
    """Araç kırpımlarında plaka arar. Tüm kırpımlar tek çağrıda (batch) işlenir."""

    def __init__(self, model="models/plate_yolo11m.pt", imgsz=640, conf=0.35, half=True, device=0):
        self.model = YOLO(model)
        self.kw = dict(imgsz=imgsz, conf=conf, half=half, device=device, verbose=False, max_det=1)

    def __call__(self, crops):
        """crops: BGR liste -> her biri için (xyxy kırpım koordinatında, conf) veya None."""
        if not crops:
            return []
        out = []
        for r in self.model.predict(crops, **self.kw):
            if r.boxes is None or len(r.boxes) == 0:
                out.append(None)
            else:
                out.append((r.boxes.xyxy[0].cpu().numpy(), float(r.boxes.conf[0])))
        return out


class PlateOCR:
    """EasyOCR tanıma modeli; plaka kırpımı zaten tek satır olduğu için metin bulma (CRAFT) adımı atlanır."""

    def __init__(self, gpu=True, target_h=64):
        import easyocr
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        self.target_h = target_h

    def variants(self, plate_bgr, trim_left=0.07):
        """Aynı plaka için 3 ön işleme: gri, CLAHE, keskinleştirilmiş. En güvenli okuma seçilir.
        trim_left: soldaki mavi AB şeridini (PL/TR yazısı) kırpar; OCR'ı en çok şaşırtan kısım."""
        h, w = plate_bgr.shape[:2]
        x0 = int(w * trim_left)
        img = plate_bgr[:, x0:] if w - x0 > 10 else plate_bgr
        h, w = img.shape[:2]
        s = self.target_h / max(h, 1)
        big = cv2.resize(img, (max(int(w * s), 1), self.target_h), interpolation=cv2.INTER_CUBIC)
        g = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 8)).apply(g)
        blur = cv2.GaussianBlur(g, (0, 0), 2.0)
        sharp = cv2.addWeighted(g, 1.8, blur, -0.8, 0)  # unsharp mask
        return [g, clahe, sharp]

    def _read(self, g):
        res = self.reader.recognize(g, allowlist=ALNUM, detail=1, paragraph=False, decoder="beamsearch")
        if not res:
            return "", 0.0
        res = sorted(res, key=lambda r: r[0][0][0])  # soldan sağa
        return "".join(r[1] for r in res), float(np.mean([r[2] for r in res]))

    def __call__(self, plate_bgr):
        """-> (ham_metin, güven): 3 ön işlemeden en yüksek güvenli okuma"""
        best = ("", 0.0)
        for g in self.variants(plate_bgr):
            t, c = self._read(g)
            if c > best[1]:
                best = (t, c)
        return best


class FastPlateOCR:
    """fast-plate-ocr: düşük çözünürlüklü plakalar için eğitilmiş, çok ülkeli (Avrupa dahil) plaka OCR modeli.
    Sürümler arası API farkı için iki arayüz denenir; görüntü geçici dosya üzerinden verilir (en uyumlu yol)."""

    def __init__(self, model_name=None):
        import os
        import tempfile
        self._tmp = os.path.join(tempfile.gettempdir(), "plate_ocr_tmp.png")
        try:
            from fast_plate_ocr import LicensePlateRecognizer  # v1.x
            self.model = LicensePlateRecognizer(model_name or "cct-xs-v1-global-model")
        except ImportError:
            from fast_plate_ocr import ONNXPlateRecognizer  # v0.x
            self.model = ONNXPlateRecognizer(model_name or "global-plates-mobile-vit-v2-model")

    def __call__(self, plate_bgr):
        cv2.imwrite(self._tmp, plate_bgr)
        try:
            out = self.model.run(self._tmp, return_confidence=True)
        except TypeError:
            out = self.model.run(self._tmp)
        # v1.x: [PlatePrediction(plate=..., char_probs=array)]  |  v0.x: (["PLATE__"], probs) veya ["PLATE__"]
        probs = None
        if isinstance(out, tuple) and len(out) == 2 and not hasattr(out[0], "plate"):
            out, probs = out
        item = out[0] if isinstance(out, (list, tuple)) and len(out) else out
        if hasattr(item, "plate"):
            text = item.plate
            probs = getattr(item, "char_probs", None)
        else:
            text = str(item)
        text = str(text).replace("_", "").upper()
        conf = 0.5
        if probs is not None:
            c = np.asarray(probs, dtype=float).reshape(-1)
            if c.size:
                conf = float(c[:max(len(text), 1)].mean())
        return text, conf
