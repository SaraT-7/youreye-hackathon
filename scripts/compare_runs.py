"""outputs/task1 (veya verilen klasör) altındaki step*.json özetlerini tablo halinde karşılaştırır.

    python scripts/compare_runs.py
    python scripts/compare_runs.py outputs/task2
"""
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/task1")
rows = [json.loads(p.read_text(encoding="utf-8")) | {"_file": p.stem} for p in sorted(folder.glob("step*.json"))]
if not rows:
    sys.exit(f"{folder} içinde özet bulunamadı")
cols = [("_file", "çalıştırma"), ("frames", "kare"), ("avg_fps_pipeline", "FPS"), ("avg_model_ms", "model ms"),
        ("persons_per_frame_avg", "kişi/kare"), ("persons_per_frame_max", "maks"),
        ("conf_mean", "ort. güven"), ("conf_low_share(<0.5)", "güven<0.5")]
w = [max(len(h), *(len(str(r.get(k, ""))) for r in rows)) for k, h in cols]
print("  ".join(h.ljust(x) for (_, h), x in zip(cols, w)))
print("  ".join("-" * x for x in w))
for r in rows:
    print("  ".join(str(r.get(k, "")).ljust(x) for (k, _), x in zip(cols, w)))
