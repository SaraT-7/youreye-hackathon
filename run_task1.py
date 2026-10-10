"""Aşama 1 — Kafe: kişi sayımı ve cinsiyet sınıflandırma.

Şu anki durum: ADIM 1 (yalnızca kişi tespiti + görselleştirme + FPS)

Örnekler (proje kökünden):
    python run_task1.py                                   # tüm video, yolo11l, imgsz 1280, conf 0.35
    python run_task1.py --max-frames 300 --show           # ilk 300 kare, canlı pencere
    python run_task1.py --model yolo11m.pt --imgsz 960    # karşılaştırma için
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.common.draw import draw_box, draw_panel
from src.common.fps import FPSMeter
from src.common.video import VideoReader, VideoWriter
from src.task1_cafe.detector import PersonDetector


def parse_args():
    p = argparse.ArgumentParser(description="Aşama 1 — kişi tespiti")
    p.add_argument("--video", default="data/task1_cafe.mp4")
    p.add_argument("--out", default=None, help="Çıktı videosu (varsayılan: outputs/task1/step1_<model>_<imgsz>_c<conf>.mp4)")
    p.add_argument("--model", default="yolo11l.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--no-half", action="store_true", help="FP16 kapat (FP32)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--show", action="store_true", help="İşlerken pencerede göster (q ile çık)")
    return p.parse_args()


def main():
    a = parse_args()
    tag = f"{Path(a.model).stem}_{a.imgsz}_c{round(a.conf*100):02d}"
    out_path = Path(a.out or f"outputs/task1/step1_{tag}.mp4")

    reader = VideoReader(a.video, max_frames=a.max_frames, start_frame=a.start_frame)
    writer = VideoWriter(out_path, reader.fps, reader.width, reader.height)
    det = PersonDetector(a.model, imgsz=a.imgsz, conf=a.conf, half=not a.no_half)

    fps_total = FPSMeter()   # okuma hariç tüm işlem (tespit + çizim)
    fps_model = FPSMeter()   # yalnızca model
    counts, confs = [], []
    warmed = False

    print(f"[bilgi] {a.video}: {reader.width}x{reader.height} @ {reader.fps:.0f} FPS, {len(reader)} kare")
    print(f"[bilgi] model={a.model} imgsz={a.imgsz} conf={a.conf} half={not a.no_half}")

    try:
        for idx, frame in reader:
            if not warmed:
                det.warmup(frame)
                warmed = True

            fps_total.start()
            fps_model.start()
            d = det(frame)
            fps_model.stop()

            for box, c in zip(d.xyxy, d.conf):
                draw_box(frame, box, f"person {c:.2f}")
            draw_panel(frame, [
                f"Kisi (karede): {len(d)}",
                f"FPS: {fps_total.fps:5.1f}  (model {fps_model.avg_ms:4.1f} ms)",
                f"{Path(a.model).stem} @ {a.imgsz}",
            ])
            fps_total.stop()

            writer.write(frame)
            counts.append(len(d))
            confs.extend(d.conf.tolist())

            if a.show:
                try:
                    cv2.imshow("Asama 1 - Adim 1", cv2.resize(frame, (1280, 720)))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    print("[uyarı] OpenCV pencere desteği yok (headless sürüm). --show kapatıldı, video kaydı sürüyor.")
                    a.show = False
            if idx % 100 == 0:
                print(f"  kare {idx:5d}/{reader.start_frame + len(reader)}  kişi={len(d):2d}  FPS={fps_total.fps:5.1f}")

    except KeyboardInterrupt:
        print("\n[uyarı] Ctrl+C ile durduruldu; o ana kadarki kısım kaydediliyor.")
    finally:
        # Video dosyasının sonunu (moov) yazmak için şart; yoksa MP4 açılmaz.
        writer.release()
        if a.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    if not counts:
        print("[uyarı] Hiç kare işlenmedi.")
        return
    confs = np.array(confs) if confs else np.zeros(1)
    summary = {
        "model": a.model, "imgsz": a.imgsz, "conf_thr": a.conf, "half": not a.no_half,
        "frames": len(counts),
        "avg_fps_pipeline": round(fps_total.avg_fps, 1),
        "avg_model_ms": round(fps_model.avg_ms, 1),
        "persons_per_frame_avg": round(float(np.mean(counts)), 2),
        "persons_per_frame_max": int(np.max(counts)),
        "conf_mean": round(float(confs.mean()), 3),
        "conf_low_share(<0.5)": round(float((confs < 0.5).mean()), 3),
        "output": str(out_path),
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[özet]")
    for k, v in summary.items():
        print(f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
