"""Video okuma / yazma yardımcıları."""
from pathlib import Path

import cv2


class VideoReader:
    """cv2.VideoCapture sarmalayıcısı; kareleri (index, frame) olarak döndürür."""

    def __init__(self, path, max_frames=None, start_frame=0):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Video açılamadı: {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.start_frame = start_frame
        remaining = max(total - start_frame, 0)
        self.total = min(remaining, max_frames) if max_frames else remaining

    def __iter__(self):
        idx = self.start_frame
        n = 0
        while n < self.total:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
            n += 1
        self.cap.release()

    def __len__(self):
        return self.total


class VideoWriter:
    """MP4 yazıcı. Klasörü kendisi oluşturur."""

    def __init__(self, path, fps, width, height):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Çıktı videosu oluşturulamadı: {path}")

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
