"""Plaka doğrulama sayfası + doğruluk hesabı.

1) python scripts/plate_review.py
   -> outputs/task2/plate_review.html : her geçiş için araç görseli, plaka kırpımı, sistemin okuduğu plaka
   -> outputs/task2/plate_ground_truth.csv : 'true_plate' sütununu siz doldurun (okunamıyorsa '?' yazın)
2) Doldurduktan sonra tekrar: python scripts/plate_review.py --score
   -> tam eşleşme oranı + karakter doğruluğu (1 - CER)
"""
import argparse
import csv
import html
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--events", default=None)
ap.add_argument("--score", action="store_true")
a = ap.parse_args()
d = Path("outputs/task2")
ev = Path(a.events) if a.events else sorted(d.glob("task2_*_events.csv"), key=lambda p: p.stat().st_mtime)[-1]
rows = list(csv.DictReader(open(ev, encoding="utf-8")))
gt_path = d / "plate_ground_truth.csv"


def lev(x, y):
    prev = list(range(len(y) + 1))
    for i, cx in enumerate(x, 1):
        cur = [i]
        for j, cy in enumerate(y, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cx != cy)))
        prev = cur
    return prev[-1]


if a.score:
    gt = list(csv.DictReader(open(gt_path, encoding="utf-8")))
    lab = [g for g in gt if g["true_plate"].strip() and g["true_plate"].strip() != "?"]
    if not lab:
        raise SystemExit("plate_ground_truth.csv içinde doldurulmuş true_plate yok")
    exact = sum(g["predicted"].replace(" ", "") == g["true_plate"].replace(" ", "").upper() for g in lab)
    chars = sum(len(g["true_plate"].replace(" ", "")) for g in lab)
    errs = sum(lev(g["predicted"], g["true_plate"].replace(" ", "").upper()) for g in lab)
    unread = sum(g["true_plate"].strip() == "?" for g in gt)
    print(f"Etiketlenen: {len(lab)}  (insan gözüyle de okunamayan: {unread})")
    print(f"Tam eşleşme: {exact}/{len(lab)} = %{100 * exact / len(lab):.1f}")
    print(f"Karakter doğruluğu (1-CER): %{100 * (1 - errs / max(chars, 1)):.1f}")
    raise SystemExit

with open(gt_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["vehicle_id", "time_s", "vehicle_type", "predicted", "true_plate"])
    for r in rows:
        w.writerow([r["vehicle_id"], r["time_s"], r["vehicle_type"], r["plate"], ""])

cells = []
for r in rows:
    def img(p):
        return f'<img src="crops/{html.escape(p.replace(chr(92), "/").split("/")[-1])}">' if p else "<i>görsel yok</i>"
    valid = "✔ format" if r["plate_format_valid"] == "True" else "✖ format"
    cells.append(f"""<tr><td>#{r['vehicle_id']}<br>{r['time_s']} s<br>{r['vehicle_type']}</td>
<td class=v>{img(r['vehicle_img'])}</td><td class=p>{img(r['plate_img'])}</td>
<td><b>{html.escape(r['plate'] or '-')}</b><br>oy payı {r['plate_vote_share']}<br>OCR {r['ocr_conf']}<br>{valid}<br>{r['repeat']}</td></tr>""")
(d / "plate_review.html").write_text(f"""<!doctype html><meta charset=utf-8><title>Plaka doğrulama</title>
<style>body{{font-family:sans-serif;margin:16px}}td{{border-bottom:1px solid #ddd;padding:6px;vertical-align:middle}}
.v img{{height:140px}}.p img{{height:60px;image-rendering:pixelated}}</style>
<h2>Plaka doğrulama — {html.escape(ev.name)}</h2><p>Gerçek plakaları plate_ground_truth.csv dosyasına yazın,
sonra <code>python scripts/plate_review.py --score</code></p>
<table><tr><th>Araç</th><th>Araç görseli</th><th>Plaka kırpımı</th><th>Sistem</th></tr>{''.join(cells)}</table>""", encoding="utf-8")
print(f"[bilgi] {d / 'plate_review.html'} ve {gt_path} oluşturuldu ({len(rows)} geçiş)")
