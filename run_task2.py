"""Aşama 2 — Şehre giren araç sayımı ve plaka tespiti.

Akış: araç tespiti + takip (YOLO11l + BoT-SORT) -> şehir giriş çizgisinde sayım
      -> araç kırpımında plaka tespiti (kendi eğitilen YOLO11m) -> OCR (EasyOCR)
      -> format doğrulama/düzeltme (PL/TR) -> araç başına oylama -> CSV/JSON + kırpım görselleri

Örnekler (proje kökünden):
    python run_task2.py                              # tüm video
    python run_task2.py --max-frames 900 --show      # ilk 30 sn, canlı pencere
    python run_task2.py --preview-line               # sayım çizgisini kontrol et
    python run_task2.py --plate-format tr            # Türkiye plaka formatı ile doğrulama
"""
import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

import re

from src.common.draw import FONT, draw_box, draw_card, draw_count_line, draw_plate_badge, draw_table, draw_trail, id_color
from src.common.fps import FPSMeter
from src.common.line_counter import LineCounter
from src.common.scene_cut import SceneCutDetector
from src.common.video import VideoReader, VideoWriter
from src.task2_vehicle.plate_text import PlateVoter, normalize


def parse_args():
    p = argparse.ArgumentParser(description="Aşama 2 — araç sayımı + plaka")
    p.add_argument("--video", default="data/task2_traffic.mp4")
    p.add_argument("--out", default=None)
    p.add_argument("--model", default="yolo11l.pt")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--tracker", default="configs/botsort_traffic.yaml")
    p.add_argument("--plate-model", default="models/plate_yolo11m.pt")
    p.add_argument("--plate-conf", type=float, default=0.35)
    p.add_argument("--plate-format", default="pl", choices=["pl", "tr", "any"],
                   help="Doğrulama formatı. Video Polonya plakaları içerdiği için varsayılan 'pl'")
    p.add_argument("--min-vehicle-h", type=int, default=90, help="Plaka aranacak en küçük araç yüksekliği (piksel)")
    p.add_argument("--plate-zone-y", type=float, default=0.55,
                   help="Plaka yalnızca alt kenarı karenin bu oranından aşağıdaki araçlarda aranır (yakın = büyük plaka)")
    p.add_argument("--min-plate-w", type=int, default=50,
                   help="OCR yapılacak en küçük plaka genişliği (piksel). Daha küçük plakalar okunamayacak kadar bulanık")
    p.add_argument("--plate-every", type=int, default=2, help="Araç başına kaç karede bir plaka okuması")
    p.add_argument("--line", default="60,840,1110,840", help="Şehir giriş çizgisi x1,y1,x2,y2")
    p.add_argument("--band", type=float, default=15.0)
    p.add_argument("--in-sign", type=int, default=1, choices=[1, -1], help="+1: kameraya doğru gelen = şehre giriş")
    p.add_argument("--cut-thr", type=float, default=10.0)
    p.add_argument("--no-ocr", action="store_true", help="OCR kapalı (hız karşılaştırması)")
    p.add_argument("--ocr", default="auto", choices=["auto", "fastplate", "easyocr"],
                   help="auto: fast-plate-ocr kuruluysa onu, değilse EasyOCR kullanır")
    p.add_argument("--no-half", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--show", action="store_true")
    p.add_argument("--preview-line", action="store_true")
    return p.parse_args()


def short_plate(text, n=9):
    """Ekran için güvenli plaka metni: yalnızca A-Z/0-9, en fazla n karakter."""
    m = re.findall(r"[A-Z0-9]+", str(text).upper())
    return "".join(m)[:n]


def fmt_time(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def draw_recent(img, rows, voter, origin=(15, 15), n=6):
    """Sol üstte son geçişler tablosu (en yeni üstte)."""
    table = []
    for t, i, vt, _ in rows[-n:][::-1]:
        pl, pc, valid, _ = voter.result(i)
        pl = short_plate(pl)
        table.append([fmt_time(t), f"#{i}", vt[:10],
                      (pl or "-", (90, 220, 90) if valid else ((0, 170, 255) if pl else (150, 150, 150)))])
    draw_table(img, origin[0], origin[1], "SON GECISLER", ["SURE", "ID", "TUR", "PLAKA"], table,
               col_x=[12, 110, 175, 310], width=470)


def main():
    a = parse_args()
    line = [float(v) for v in a.line.split(",")]
    if a.preview_line:
        r = VideoReader(a.video, max_frames=1)
        for _, f in r:
            draw_count_line(f, line[:2], line[2:], a.in_sign, label_in="SEHRE GIRIS")
            Path("outputs/task2").mkdir(parents=True, exist_ok=True)
            cv2.imwrite("outputs/task2/line_preview.jpg", f)
        print("[bilgi] outputs/task2/line_preview.jpg kaydedildi")
        return

    from src.task2_vehicle.plate import FastPlateOCR, PlateDetector, PlateOCR
    from src.task2_vehicle.tracker import VehicleTracker

    tag = f"{Path(a.model).stem}_{a.imgsz}_{a.plate_format}" + ("_noocr" if a.no_ocr else f"_{a.ocr}")
    out_path = Path(a.out or f"outputs/task2/task2_{tag}.mp4")
    crops_dir = out_path.parent / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    reader = VideoReader(a.video, max_frames=a.max_frames, start_frame=a.start_frame)
    writer = VideoWriter(out_path, reader.fps, reader.width, reader.height)
    tracker = VehicleTracker(a.model, a.tracker, imgsz=a.imgsz, half=not a.no_half)
    plates = PlateDetector(a.plate_model, conf=a.plate_conf, half=not a.no_half)
    ocr, ocr_name = None, "kapalı"
    if not a.no_ocr:
        if a.ocr in ("auto", "fastplate"):
            try:
                ocr, ocr_name = FastPlateOCR(), "fast-plate-ocr"
            except Exception as e:  # kurulu değilse / model inmediyse
                if a.ocr == "fastplate":
                    raise
                print(f"[uyarı] fast-plate-ocr yüklenemedi ({e}); EasyOCR kullanılıyor")
        if ocr is None:
            ocr, ocr_name = PlateOCR(), "EasyOCR"
    print(f"[bilgi] OCR: {ocr_name}")
    voter = PlateVoter(every=a.plate_every, fmt=a.plate_format)
    counter = LineCounter(tuple(line[:2]), tuple(line[2:]), band=a.band, in_sign=a.in_sign)
    cuts = SceneCutDetector(threshold=a.cut_thr)

    fps_total, fps_det, fps_plate, fps_ocr = FPSMeter(), FPSMeter(), FPSMeter(), FPSMeter()
    trails = defaultdict(lambda: deque(maxlen=30))
    last_plate_box = {}                 # gid -> (xyxy tam kare, conf)  çizim için
    best = {}                           # gid -> (skor, araç kırpımı, plaka kırpımı, det_conf, ocr_conf)
    vconf = defaultdict(list)           # gid -> araç tespit güvenleri
    events, recent = [], []
    flash_until = -1
    print(f"[bilgi] {a.video}: {reader.width}x{reader.height} @ {reader.fps:.0f} FPS, {len(reader)} kare")

    try:
        for idx, frame in reader:
            fps_total.start()
            H, W = frame.shape[:2]
            if cuts(frame):
                tracker.reset(); trails.clear(); counter.reset_tracks()
                print(f"  [kesme] kare {idx} -> takip sıfırlandı")

            fps_det.start()
            t = tracker(frame)
            fps_det.stop()

            # --- Plaka: büyük araçlarda ve okuma sırası gelmiş olanlarda, tek batch
            need = [k for k in range(len(t))
                    if (t.xyxy[k][3] - t.xyxy[k][1]) >= a.min_vehicle_h and t.xyxy[k][3] >= a.plate_zone_y * H
                    and voter.wants(int(t.ids[k]), idx)]
            if need:
                fps_plate.start()
                vcrops, offs = [], []
                for k in need:
                    x1, y1, x2, y2 = t.xyxy[k].astype(int)
                    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
                    vcrops.append(frame[y1:y2, x1:x2].copy()); offs.append((x1, y1))
                dets = plates(vcrops)
                fps_plate.stop()
                for k, vc, (ox, oy), d in zip(need, vcrops, offs, dets):
                    gid = int(t.ids[k])
                    if d is None:
                        voter._last[gid] = idx
                        continue
                    (px1, py1, px2, py2), pconf = d
                    last_plate_box[gid] = ((px1 + ox, py1 + oy, px2 + ox, py2 + oy), pconf, idx)
                    pc = vc[max(int(py1) - 2, 0):int(py2) + 2, max(int(px1) - 3, 0):int(px2) + 3]
                    if ocr is None or pc.shape[1] < a.min_plate_w or pc.shape[0] < 8:
                        voter._last[gid] = idx
                        continue
                    fps_ocr.start()
                    raw, oconf = ocr(pc)
                    fps_ocr.stop()
                    text, valid = normalize(raw, a.plate_format)
                    w = voter.add(gid, idx, text, valid, oconf, pconf)
                    score = w * pc.shape[1]  # büyük ve güvenli okuma = daha iyi kayıt görseli
                    if w > 0 and score > best.get(gid, (0,))[0]:
                        best[gid] = (score, vc, pc.copy(), pconf, oconf)

            # --- Sayım
            refs = [((b[0] + b[2]) / 2, b[3]) for b in t.xyxy]  # aracın ön-alt noktası
            for ev in counter.update(idx, t.ids, refs):
                flash_until = idx + int(reader.fps // 2)
                events.append(ev)
                recent.append([ev.frame / reader.fps, ev.track_id, tracker.vtype(ev.track_id), ""])
                print(f"  [geçiş] kare {ev.frame}  #{ev.track_id} {tracker.vtype(ev.track_id)}  "
                      f"{'GİRİŞ' if ev.direction == 'in' else 'ÇIKIŞ'}")

            # --- Çizim
            draw_count_line(frame, line[:2], line[2:], a.in_sign, flash=idx <= flash_until, label_in="SEHRE GIRIS")
            for box, c, gid, ref in zip(t.xyxy, t.conf, t.ids, refs):
                gid = int(gid); col = id_color(gid)
                vconf[gid].append(float(c))
                trails[gid].append(ref); draw_trail(frame, trails[gid], col)
                pl, pconf_v, valid, _ = voter.result(gid)
                txt = f"#{gid} {tracker.vtype(gid)} {c:.2f}" + (" G" if (gid, "in") in counter._counted else "")
                draw_box(frame, box, txt, color=col)
                if pl:
                    draw_plate_badge(frame, (box[0] + box[2]) / 2, box[3] + 6, short_plate(pl), pconf_v, valid)
                pb = last_plate_box.get(gid)
                if pb and idx - pb[2] <= 3:
                    (qx1, qy1, qx2, qy2), qc, _ = pb
                    cv2.rectangle(frame, (int(qx1), int(qy1)), (int(qx2), int(qy2)), (0, 255, 0), 2)
            types = defaultdict(int)
            for e in events:
                if e.direction == "in":
                    types[tracker.vtype(e.track_id)] += 1
            n = max(fps_total.count, 1)
            rows_card = [(k, v) for k, v in sorted(types.items())] or [("-", 0)]
            rows_card += [("Karede arac", len(t)), ("Cikan", counter.n_out), ("FPS", f"{fps_total.fps:.1f}")]
            draw_card(frame, W - 330, 15, 315, "SEHRE GIREN ARAC", big=counter.n_in, rows=rows_card,
                      footer=f"tespit {fps_det.avg_ms:.0f} | plaka {fps_plate.total_time * 1e3 / n:.0f} | "
                             f"OCR {fps_ocr.total_time * 1e3 / n:.0f} ms/kare")
            draw_recent(frame, recent, voter)
            fps_total.stop()
            writer.write(frame)

            if a.show:
                try:
                    cv2.imshow("Asama 2", cv2.resize(frame, (1280, 720)))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    a.show = False
            if idx % 150 == 0:
                print(f"  kare {idx:5d}/{reader.start_frame + len(reader)}  araç={len(t):2d}  "
                      f"giren={counter.n_in:3d}  FPS={fps_total.fps:5.1f}")
    except KeyboardInterrupt:
        print("\n[uyarı] Ctrl+C ile durduruldu; o ana kadarki kısım kaydediliyor.")
    finally:
        writer.release()
        if a.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    # --- Kayıtlar
    first_seen = {}  # plaka -> ilk geçen araç kimliği (tekrar geçiş kontrolü)
    rows = []
    for e in sorted(events, key=lambda e: e.frame):
        pl, pc, valid, nreads = voter.result(e.track_id)
        b = best.get(e.track_id)
        vimg = pimg = ""
        if b is not None:
            vimg = str(crops_dir / f"vehicle_{e.track_id}.jpg"); cv2.imwrite(vimg, b[1])
            pimg = str(crops_dir / f"plate_{e.track_id}.jpg"); cv2.imwrite(pimg, b[2])
        repeat = ""
        if pl and valid:
            if pl in first_seen and first_seen[pl] != e.track_id:
                repeat = f"tekrar (ilk: #{first_seen[pl]})"
            first_seen.setdefault(pl, e.track_id)
        rows.append({
            "frame": e.frame, "time_s": round(e.frame / reader.fps, 2), "direction": e.direction,
            "vehicle_id": e.track_id, "vehicle_type": tracker.vtype(e.track_id),
            "vehicle_conf": round(float(np.mean(vconf[e.track_id])) if vconf[e.track_id] else 0.0, 3),
            "plate": pl, "plate_vote_share": round(pc, 3), "plate_format_valid": valid, "ocr_reads": nreads,
            "plate_det_conf": round(b[3], 3) if b else "", "ocr_conf": round(b[4], 3) if b else "",
            "repeat": repeat, "vehicle_img": vimg, "plate_img": pimg,
        })
    csv_path = out_path.with_name(out_path.stem + "_events.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    out_path.with_name(out_path.stem + "_events.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    ins = [r for r in rows if r["direction"] == "in"]
    nfr = max(fps_total.count, 1)
    summary = {
        "frames": fps_total.count, "avg_fps_pipeline": round(fps_total.avg_fps, 1),
        "det_ms": round(fps_det.avg_ms, 1),
        "plate_ms_per_frame": round(fps_plate.total_time * 1e3 / nfr, 1),
        "ocr_ms_per_frame": round(fps_ocr.total_time * 1e3 / nfr, 1),
        "entered": counter.n_in, "exited": counter.n_out, "total_vehicle_ids": tracker.total_ids,
        "entered_by_type": {k: sum(r["vehicle_type"] == k for r in ins) for k in {r["vehicle_type"] for r in ins}},
        "entered_with_plate": sum(bool(r["plate"]) for r in ins),
        "entered_with_valid_plate": sum(bool(r["plate_format_valid"]) for r in ins),
        "repeat_passes": sum(bool(r["repeat"]) for r in rows),
        "plate_format": a.plate_format, "ocr": ocr_name, "line": line,
        "events_csv": str(csv_path), "output": str(out_path),
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[özet]")
    for k, v in summary.items():
        print(f"  {k:26s} {v}")


if __name__ == "__main__":
    main()
