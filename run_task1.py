"""Aşama 1 — Kafe: kişi sayımı ve cinsiyet sınıflandırma.

Şu anki durum: ADIM 4 (tespit + takip + sahne kesmesi + giriş/çıkış sayımı + cinsiyet tahmini)

Not: Cinsiyet çıktısı yalnızca görsel model tahminidir; kesin değildir ve hata payı içerir.

Örnekler (proje kökünden):
    python run_task1.py                                       # tüm video, BoT-SORT + ReID
    python run_task1.py --max-frames 600 --show               # ilk 600 kare (2 kesme içerir), canlı pencere
    python run_task1.py --tracker configs/bytetrack_cafe.yaml # ByteTrack ile karşılaştırma
    python run_task1.py --preview-line                        # çizgiyi birkaç karede çizip kaydeder, çıkar
    python run_task1.py --line 800,300,1100,300               # çizgiyi elle ayarlama (x1,y1,x2,y2; 1920x1080)
    python run_task1.py --no-gender                           # cinsiyet modeli kapalı (hız karşılaştırması)
"""
import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from src.common.draw import draw_box, draw_count_line, draw_panel, draw_trail, id_color
from src.common.line_counter import LineCounter
from src.common.fps import FPSMeter
from src.common.scene_cut import SceneCutDetector
from src.common.video import VideoReader, VideoWriter
from src.task1_cafe.gender import GenderVoter
from src.task1_cafe.tracker import PersonTracker


def parse_args():
    p = argparse.ArgumentParser(description="Aşama 1 — kişi tespiti + takip")
    p.add_argument("--video", default="data/task1_cafe.mp4")
    p.add_argument("--out", default=None)
    p.add_argument("--model", default="yolo11l.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--tracker", default="configs/botsort_cafe.yaml")
    p.add_argument("--det-conf", type=float, default=0.1,
                   help="Takip algoritmasına verilen en düşük tespit güveni (ikinci eşleştirme turu için)")
    p.add_argument("--cut-thr", type=float, default=10.0, help="Sahne kesmesi eşiği (kareler arası fark)")
    p.add_argument("--no-nested-filter", action="store_true", help="İç içe kutu bastırmayı kapat (karşılaştırma için)")
    p.add_argument("--line", default="790,230,1070,230",
                   help="Sayım çizgisi x1,y1,x2,y2 (orijinal piksel). Varsayılan: kapı önü")
    p.add_argument("--band", type=float, default=15.0, help="Histerezis bandı (piksel)")
    p.add_argument("--in-sign", type=int, default=1, choices=[1, -1],
                   help="+1: çizginin altı (kameraya yakın taraf) içeri; -1: tersi")
    p.add_argument("--preview-line", action="store_true",
                   help="Çizgiyi her klibin ilk karesine çizip outputs/task1/line_preview_*.jpg kaydeder ve çıkar")
    p.add_argument("--no-gender", action="store_true", help="Cinsiyet tahminini kapat")
    p.add_argument("--gender-every", type=int, default=5, help="Kimlik başına kaç karede bir cinsiyet örneği")
    p.add_argument("--gender-max", type=int, default=15, help="Kimlik başına en fazla örnek")
    p.add_argument("--gender-thr", type=float, default=0.65, help="Bu güvenin altı 'belirsiz' (?)")
    p.add_argument("--clip-arch", default="ViT-B-16")
    p.add_argument("--clip-pretrained", default="laion2b_s34b_b88k")
    p.add_argument("--trail", type=int, default=30, help="Çizilecek iz uzunluğu (kare)")
    p.add_argument("--no-half", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def preview_line(a, line):
    reader = VideoReader(a.video)
    cuts = SceneCutDetector(threshold=a.cut_thr)
    saved = 0
    for idx, frame in reader:
        if idx == 0 or cuts(frame):
            draw_count_line(frame, line[:2], line[2:], a.in_sign)
            out = Path(f"outputs/task1/line_preview_{saved + 1}_f{idx}.jpg")
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), frame)
            saved += 1
    print(f"[bilgi] {saved} önizleme kaydedildi: outputs/task1/line_preview_*.jpg")


def main():
    a = parse_args()
    line = [float(v) for v in a.line.split(",")]
    if a.preview_line:
        preview_line(a, line)
        return
    trk_name = Path(a.tracker).stem.replace("_cafe", "")
    tag = f"{trk_name}_{Path(a.model).stem}_{a.imgsz}" + ("_nofilter" if a.no_nested_filter else "")
    tag += "_nogender" if a.no_gender else ""
    out_path = Path(a.out or f"outputs/task1/step4_{tag}.mp4")

    reader = VideoReader(a.video, max_frames=a.max_frames, start_frame=a.start_frame)
    writer = VideoWriter(out_path, reader.fps, reader.width, reader.height)
    tracker = PersonTracker(a.model, a.tracker, imgsz=a.imgsz, det_conf=a.det_conf, half=not a.no_half,
                            nested_filter=not a.no_nested_filter)
    cuts = SceneCutDetector(threshold=a.cut_thr)
    counter = LineCounter(tuple(line[:2]), tuple(line[2:]), band=a.band, in_sign=a.in_sign)
    events_rows = []
    flash_until = -1
    gender = None
    if not a.no_gender:
        from src.task1_cafe.gender import ClipGender
        print(f"[bilgi] CLIP yükleniyor: {a.clip_arch} / {a.clip_pretrained} (ilk seferde indirilir)")
        gender = ClipGender(a.clip_arch, a.clip_pretrained)
    voter = GenderVoter(every=a.gender_every, max_samples=a.gender_max, decide_thr=a.gender_thr)
    fps_gender = FPSMeter()
    gid_clip = {}

    fps_total, fps_model = FPSMeter(), FPSMeter()
    trails = defaultdict(lambda: deque(maxlen=a.trail))
    track_len = defaultdict(int)     # global id -> görüldüğü kare sayısı
    cut_frames, per_frame = [], []

    print(f"[bilgi] {a.video}: {reader.width}x{reader.height} @ {reader.fps:.0f} FPS, {len(reader)} kare")
    print(f"[bilgi] model={a.model} imgsz={a.imgsz} tracker={a.tracker} det_conf={a.det_conf}")

    try:
        for idx, frame in reader:
            fps_total.start()
            if cuts(frame):
                tracker.reset()
                trails.clear()
                counter.reset_tracks()
                cut_frames.append(idx)
                print(f"  [kesme] kare {idx} (fark={cuts.last_score:.1f}) -> takip sıfırlandı, klip #{tracker.segment}")

            fps_model.start()
            t = tracker(frame)
            fps_model.stop()

            feet = [((b[0] + b[2]) / 2, b[3]) for b in t.xyxy]  # ayak noktası = kutunun alt ortası
            for ev in counter.update(idx, t.ids, feet):
                flash_until = idx + int(reader.fps // 2)
                events_rows.append([ev.frame, round(ev.frame / reader.fps, 2), tracker.segment + 1,
                                    ev.track_id, ev.direction])
                print(f"  [geçiş] kare {ev.frame}  #{ev.track_id}  {'GİRİŞ' if ev.direction == 'in' else 'ÇIKIŞ'}")

            # --- Cinsiyet: yalnızca örneğe ihtiyacı olan kimlikler, tek seferde (batch)
            if gender is not None:
                need = [k for k, (b, gid) in enumerate(zip(t.xyxy, t.ids))
                        if voter.wants(int(gid), idx, b[3] - b[1])]
                if need:
                    fps_gender.start()
                    H, W = frame.shape[:2]
                    crops, ok_k = [], []
                    for k in need:
                        x1, y1, x2, y2 = t.xyxy[k].astype(int)
                        c = frame[max(y1, 0):min(y2, H), max(x1, 0):min(x2, W)]
                        if c.shape[0] > 8 and c.shape[1] > 8:  # kare dışına taşan/boş kırpımları atla
                            crops.append(c.copy()); ok_k.append(k)
                    need = ok_k
                    for k, pm in zip(need, gender.prob_male(crops)):
                        b = t.xyxy[k]
                        voter.add(int(t.ids[k]), idx, pm, b[3] - b[1], t.conf[k])
                    fps_gender.stop()

            draw_count_line(frame, line[:2], line[2:], a.in_sign, flash=idx <= flash_until)
            for box, c, gid, foot in zip(t.xyxy, t.conf, t.ids, feet):
                col = id_color(gid)
                trails[gid].append(foot)
                track_len[gid] += 1
                draw_trail(frame, trails[gid], col)
                gid_clip.setdefault(int(gid), tracker.segment + 1)
                tag_txt = "".join(m for k, m in (("in", " G"), ("out", " C")) if (int(gid), k) in counter._counted)
                g_txt = ""
                if gender is not None:
                    lab, gconf, _ = voter.result(int(gid))
                    g_txt = f" | {lab} {gconf:.2f}" if lab != "?" else " | ?"
                draw_box(frame, box, f"#{gid} {c:.2f}{g_txt}{tag_txt}", color=col)
                cv2.circle(frame, tuple(map(int, foot)), 4, col, -1)
            lines = [f"Giren: {counter.n_in}   Cikan: {counter.n_out}", f"Karede kisi: {len(t)}"]
            if gender is not None:
                labs = [voter.result(int(g))[0] for g in t.ids]
                lines.append(f"  Erkek {labs.count('E')} / Kadin {labs.count('K')} / ? {labs.count('?')}")
                ent = [voter.result(e.track_id)[0] for e in counter.events if e.direction == "in"]
                lines.append(f"Girenler: E {ent.count('E')} / K {ent.count('K')} / ? {ent.count('?')}")
            lines += [f"Klip: {tracker.segment + 1}",
                      f"FPS: {fps_total.fps:5.1f}  (tespit {fps_model.avg_ms:4.1f} ms"
                      + (f", cinsiyet {fps_gender.total_time * 1000 / max(fps_total.count, 1):4.1f} ms)" if gender else ")")]
            if gender is not None:
                lines.append("Cinsiyet: model tahmini, kesin degildir")
            draw_panel(frame, lines)
            fps_total.stop()

            writer.write(frame)
            per_frame.append(len(t))

            if a.show:
                try:
                    cv2.imshow("Asama 1 - Adim 2", cv2.resize(frame, (1280, 720)))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    print("[uyarı] OpenCV pencere desteği yok; --show kapatıldı.")
                    a.show = False
            if idx % 100 == 0:
                print(f"  kare {idx:5d}/{reader.start_frame + len(reader)}  iz={len(t):2d}  "
                      f"kimlik={tracker.total_ids:3d}  FPS={fps_total.fps:5.1f}")
    except KeyboardInterrupt:
        print("\n[uyarı] Ctrl+C ile durduruldu; o ana kadarki kısım kaydediliyor.")
    finally:
        writer.release()
        if a.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    if not per_frame:
        print("[uyarı] Hiç kare işlenmedi.")
        return
    lens = np.array(list(track_len.values())) if track_len else np.zeros(1)
    fps_src = reader.fps
    summary = {
        "tracker": a.tracker, "model": a.model, "imgsz": a.imgsz, "det_conf": a.det_conf,
        "frames": len(per_frame),
        "avg_fps_pipeline": round(fps_total.avg_fps, 1),
        "avg_model_ms": round(fps_model.avg_ms, 1),
        "scene_cuts": cut_frames,
        "clips": tracker.segment + 1,
        "tracks_per_frame_avg": round(float(np.mean(per_frame)), 2),
        "total_ids": tracker.total_ids,
        "track_len_median_frames": int(np.median(lens)),
        "short_tracks(<1s)": int((lens < fps_src).sum()),
        "short_track_share": round(float((lens < fps_src).mean()), 3),
        "nested_boxes_suppressed": tracker.suppressed,
        "line": line, "band_px": a.band,
        "entered": counter.n_in, "exited": counter.n_out,
        "events_csv": str(out_path.with_name(out_path.stem + "_events.csv")),
        "output": str(out_path),
    }
    with open(out_path.with_name(out_path.stem + "_events.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "clip", "track_id", "direction", "gender", "gender_conf"])
        for r in events_rows:
            lab, gc, _ = voter.result(r[3])
            w.writerow(r + ([lab, round(gc, 3)] if gender is not None else ["", ""]))
    if gender is not None:
        track_rows = []
        for gid, n in sorted(track_len.items()):
            lab, gc, ns = voter.result(gid)
            pm = voter.p_male(gid)
            track_rows.append([gid, gid_clip.get(gid, ""), n, ns, "" if pm is None else round(pm, 3), lab, round(gc, 3)])
        with open(out_path.with_name(out_path.stem + "_tracks.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["track_id", "clip", "frames", "gender_samples", "p_male", "gender", "gender_conf"])
            w.writerows(track_rows)
        labs_all = [r[5] for r in track_rows]
        ent = [voter.result(e.track_id)[0] for e in counter.events if e.direction == "in"]
        summary.update({
            "gender_ms_per_frame": round(fps_gender.total_time * 1000 / max(len(per_frame), 1), 1),
            "gender_tracks": {"E": labs_all.count("E"), "K": labs_all.count("K"), "?": labs_all.count("?")},
            "gender_entered": {"E": ent.count("E"), "K": ent.count("K"), "?": ent.count("?")},
            "gender_note": "Görsel model tahminidir; kesin değildir, hata payı içerir.",
        })
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[özet]")
    for k, v in summary.items():
        print(f"  {k:26s} {v}")


if __name__ == "__main__":
    main()
