"""FPS ölçümü: üstel hareketli ortalama (ekranda titremesin diye) + genel ortalama."""
import time


class FPSMeter:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.ema = None
        self.total_time = 0.0
        self.count = 0
        self._t0 = None

    def start(self):
        self._t0 = time.perf_counter()

    def stop(self):
        dt = time.perf_counter() - self._t0
        self.total_time += dt
        self.count += 1
        inst = 1.0 / dt if dt > 0 else 0.0
        self.ema = inst if self.ema is None else (1 - self.alpha) * self.ema + self.alpha * inst
        return dt

    @property
    def fps(self):
        return self.ema or 0.0

    @property
    def avg_fps(self):
        return self.count / self.total_time if self.total_time > 0 else 0.0

    @property
    def avg_ms(self):
        return 1000 * self.total_time / self.count if self.count else 0.0
