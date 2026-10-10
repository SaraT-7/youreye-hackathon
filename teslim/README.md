# Arabica × YourEye Hackathon — Görüntü İşleme Çözümü

Kayseri, 10.10.2026 · Bireysel katılım

İki saha senaryosu için uçtan uca çalışan, tek komutla çalıştırılabilen bir görüntü işleme hattı:

1. **Kafe içi kişi sayımı ve cinsiyet sınıflandırma** — `run_task1.py`
2. **Şehre giren araç sayımı ve plaka tespiti / OCR** — `run_task2.py`

---

## Hızlı başlangıç

```powershell
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Görev videoları data/ altına (git dışı):
#   data/task1_cafe.mp4, data/task2_traffic.mp4
# Kafe videosundaki bozuk MPEG-4 bloklarından kaçınmak için H.264 kopyası:
ffmpeg -i data/task1_cafe.mp4 -c:v libx264 -crf 18 -an data/task1_cafe_h264.mp4

# Plaka modeli: models/plate_yolo11m.pt (notebooks/train_plate_colab.ipynb ile eğitildi)

python run_task1.py            # -> outputs/task1/step4_*.mp4, *_events.csv, *_tracks.csv, *.json
python run_task2.py            # -> outputs/task2/task2_*.mp4, *_events.csv/json, crops/
```

`pip install "fast-plate-ocr[onnx]"` plaka OCR'ı için gereklidir (yoksa EasyOCR'a düşer).

Her betik `--help` ile tüm ayarları gösterir (`--show` canlı pencere, `--max-frames N` kısa deneme).

Test ortamı: Windows, Python 3.10, RTX 4050 Laptop (6 GB), CUDA 12.1, Ultralytics 8.4.78.

---

## Mimari

```
src/
  common/        video okuma/yazma, FPS ölçer, çizim, sahne kesmesi, sanal çizgi sayacı (iki görev ortak)
  task1_cafe/    kişi takibi, iç içe kutu filtresi, CLIP cinsiyet + kişi bazında oylama
  task2_vehicle/ araç takibi, plaka tespiti + OCR, plaka formatı doğrulama + araç bazında oylama
configs/         takip algoritması ayarları (BoT-SORT / ByteTrack)
scripts/         plaka modeli eğitimi, çalıştırma karşılaştırması, plaka doğrulama aracı
notebooks/       Colab plaka eğitim defteri
docs/            karşılaştırma grafikleri, ekran görüntüleri
```

---

## Aşama 1 — Kafe: kişi sayımı ve cinsiyet

**Akış:** kare → sahne kesmesi kontrolü → kişi tespiti → iç içe kutu filtresi → takip → kapı çizgisinde giriş/çıkış → cinsiyet (kişi başına seyrek örnekleme + oylama) → çizim + kayıt

| Bileşen | Seçim | Neden |
|---|---|---|
| Kişi tespiti | YOLO11l, COCO ön eğitimli, `imgsz 1280`, güven ≥ 0.35 | 5 yapılandırma karşılaştırıldı (`docs/asama1_model_karsilastirma.pdf`); uzaktaki/karanlıktaki kişiler için çözünürlük model boyutundan etkili |
| Takip | BoT-SORT + ReID (`configs/botsort_cafe.yaml`) | ByteTrack'e göre daha az kimlik (28/30), kısa iz payı %10 → %3.6, hız aynı |
| İç içe kutu filtresi | İçerme ≥ 0.8, üst kenar farkı < 0.25·h, genişlik oranı ≥ 0.6 | Masa arkasında oturan kişiye çift kutu → çift kimlik sorununu çözer (tüm videoda 2.102 kutu bastırıldı) |
| Sahne kesmesi | Kareler arası ortalama mutlak fark > 10 | Video 7 klipten oluşuyor; 6 kesmenin 6'sı yakalandı, yanlış alarm yok. Kesmede takip sıfırlanır, global kimlik korunur |
| Sayım | Ayak noktası + işaretli uzaklık + ±15 px histerezis, kimlik başına yön başına 1 sayım | Çizgide titreyen kutu tekrar sayılmaz; 7 senaryolu birim testi |
| Cinsiyet | CLIP ViT-B/16 (LAION-2B), tüm vücut kırpımı, 5'er metin kalıbı | Kamera yukarıdan bakıyor, yüzler çoğunlukla görünmüyor; eğitim gerektirmez |
| Cinsiyet oylaması | Kimlik başına 5 karede bir, en fazla 15 örnek, ağırlık = güven × boy; < 3 örnek veya güven < 0.65 → "?" | Etiket titremez; GPU yükü düşük (3.5 ms/kare) |

### Tasarım kararları ve gerekçeleri

- **YOLO11m → YOLO11l:** İlk testte YOLO11m karanlıkta (özellikle siyah giyen) ve uzakta oturan kişileri kaçırdı. YOLO11l + 1280 px'e geçildi; daha ağır olmasına rağmen tespit aşamasında ~30 FPS sağladığı için sorun olmadı (karşılaştırma: `docs/asama1_model_karsilastirma.pdf`).
- **İnce ayar yapılmadı:** CrowdHuman ile ince ayar için ne zaman ne de ölçüm yapacak etiketli kafe verisi vardı.
- **FP16** ile ~2 kat hızlanma; TensorRT daha hızlı ama ilk derleme uzun ve Windows kurulumu riskli olduğu için ertelendi.
- **Takip:** ByteTrack ile başlandı; kimlik kopmaları görülünce BoT-SORT + ReID ile karşılaştırıldı. ByteTrack yalnızca kutu konumuna baktığı için oturan birinin önünden biri geçince kimlik kopuyordu. Aynı 600 karelik testte toplam kimlik 30 → 28, medyan iz süresi 8.0 → 8.6 sn, 1 sn'den kısa kopuk iz payı %10 → %3.6; hız neredeyse aynı (25.7 / 25.3 FPS) → BoT-SORT seçildi.
- **Sayım için çizgi** (bölge yerine): dokümandaki "giriş / çıkış veya sayım çizgisi" isterine doğrudan uyuyor.
- **Sahne kesmesi:** Video farklı saatlerden 7 klibin birleşimi; kesmeler takibi bozacağı için ele alındı. Önce renk histogramı düşünüldü; ölçümde klipler aynı sahneyi gösterdiğinden kare farkı (MAD) çok daha net ayırdı (normal ≤ 3.2, kesme 16–28). Ekrandaki saati OCR ile okumak alternatifti, ama saat parlak pencerenin üstünde olduğundan yanlış okuma → sahte kesme riski vardı.
- **Tüm vücuttan cinsiyet:** Kişiler çoğunlukla oturuyor; sırt, gövde ve yüz standart biçimde görünmüyor. Yüz tabanlı yöntem sırtı dönük kişilerde sonuç üretmeyeceği için tüm vücut kırpımı + CLIP seçildi.

### Karşılaşılan sorunlar ve çözümler

| Sorun | Neden | Çözüm | Etki |
|---|---|---|---|
| Aynı kişiye 2 kutu / 2 kimlik | Üst gövde kutusu tam vücut kutusunun içinde; IoU ~0.4–0.5 < NMS eşiği 0.5 | İç içe kutu filtresi (içerme ≥ 0.8, üst kenar farkı < 0.25·h, genişlik oranı ≥ 0.6); 4 sentetik senaryo ile test | 600 karede 721, tüm videoda 2.102 kutu bastırıldı |
| Kimlik kopması (#19 → #29) | Kutu şekli zıplıyor, kısa kapanmalar, görünüm bilgisi yok | `track_buffer` 2 → 5 sn, `match_thresh` 0.80 → 0.85, `new_track_thresh` 0.45 → 0.50, BoT-SORT + ReID | kimlik 30 → 28, kısa iz %10 → %3.6 |
| Video ~30. saniyede çöküyor | Kafe videosunda bozuk MPEG-4 blokları (`marker does not match f_code` ×241.943) | H.264'e yeniden kodlama, kare sayısı korunarak (2.159) | Tüm video sorunsuz |

**Sonuçlar (tüm video, 2.159 kare):**

| Metrik | Değer |
|---|---|
| Ortalama işlem hızı | **22.6 FPS** tek başına (tespit+takip 36.3 ms, cinsiyet 3.5 ms / kare); teslim videosu Aşama 2 ile aynı anda üretildiği için o çalıştırmada 14.1 FPS |
| Kapıdan giren / çıkan | **6 / 4** |
| Girenlerin cinsiyet tahmini | Erkek 4, Kadın 1, Belirsiz 1 |
| Tüm kimliklerin cinsiyet tahmini | Erkek 37, Kadın 22, Belirsiz 4 |
| Toplam kimlik / medyan iz süresi | 63 / 247 kare (~10 sn) |

> **Not:** Cinsiyet sınıflandırması yalnızca görsel model çıktısıdır. Kişinin kimliğini veya kendini nasıl tanımladığını belirlemez; saç, kıyafet, siluet gibi ipuçlarına dayanır ve hata payı içerir.

**Teslim videoları:** `teslim/videolar/gorev1_kafe_kisi_cinsiyet.mp4`, `teslim/videolar/gorev2_arac_plaka.mp4` (H.264'e sıkıştırılmış).

**Çıktılar:** sonuç videosu, `*_events.csv` (her geçiş: kare, saniye, klip, kimlik, yön, cinsiyet), `*_tracks.csv` (her kimlik: klip, süre, örnek sayısı, P(erkek), etiket), özet `*.json`.

---

## Aşama 2 — Şehir girişi: araç sayımı, plaka tespiti, OCR

**Akış:** kare → araç tespiti + takip → şehir giriş çizgisi → (yakın araçlarda) araç kırpımında plaka tespiti → OCR → format düzeltme/doğrulama → araç bazında oylama → kayıt + kırpım görselleri

| Bileşen | Seçim | Neden |
|---|---|---|
| Araç tespiti + tür | YOLO11l COCO (otomobil/motosiklet/otobüs/kamyon), tür = güven ağırlıklı oylama | Bonus: araç türü sınıflandırma; tür kareden kareye titremez |
| Takip | BoT-SORT + ReID (`configs/botsort_traffic.yaml`) | Aşama 1 ile aynı gerekçe |
| Sayım | Aşama 1'deki ortak çizgi sayacı; kameraya doğru gelen = şehre giriş | Kod tekrarı yok |
| Plaka tespiti | **Kendi eğittiğimiz YOLO11m** (aşağıda), araç kırpımı içinde, toplu (batch) | Plaka tüm karede ~80×20 px; araç kırpımında modele ~3 kat büyük görünür |
| OCR | **fast-plate-ocr** (`cct-xs-v1-global-model`); yedek: EasyOCR (3 ön işleme, AB şeridi kırpma) | Plakaya özel model düşük çözünürlükte çok daha iyi (aşağıdaki tablo) |
| Format | Polonya (varsayılan) ve Türkiye kuralları (`--plate-format pl/tr/any`), karakter karışıklığı düzeltmesi (0↔O, 1↔I, 8↔B…) | **Video Polonya plakaları içeriyor**; TR doğrulaması da uygulandı (dokümandaki bonus) |
| Plaka oylaması | Araç başına ≤ 25 okuma; okunabilir mesafede (plaka ≥ 50 px, araç karenin alt %45'inde); en sık uzunluktaki okumalar arasında **karakter bazında** ağırlıklı oylama | Tek okuma çoğu zaman kısmen yanlış; doğru karakterler birleşir |
| Tekrar geçiş | Aynı plaka farklı kimlikle tekrar geçerse işaretlenir | Bonus |

### Plaka tespit modeli eğitimi

- **Veri:** `keremberke/license-plate-object-detection` (HuggingFace, Roboflow kaynaklı) — 6.176 eğitim / 1.765 doğrulama / 882 test, tek sınıf
- **Model:** YOLO11m (COCO ön eğitimli başlangıç), 640 px, 40 epoch, AdamW, cos LR, otomatik batch (13), Colab T4, 2.1 saat
- **Doğrulama seti:** Precision **0.987**, Recall **0.960**, mAP50 **0.982**, mAP50-95 **0.703**; T4'te 8.4 ms/görsel

**Eğitim eğrisi** (`docs/plaka_egitim_egrisi.png`):

| Epoch | 1 | 3 | 8 | 12 | 19 | 26 | 29 | 34 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| mAP50 | 0.890 | 0.948 | 0.967 | 0.970 | 0.980 | 0.982 | 0.982 | 0.987 | **0.983** |
| mAP50-95 | 0.546 | 0.605 | 0.648 | 0.669 | 0.687 | 0.692 | 0.697 | 0.701 | **0.704** |

- İlk epoch'ta bile mAP50 0.89 — COCO ön eğitimli başlangıç sayesinde.
- mAP50 ~18. epoch'ta 0.98'de doydu; sonraki kazanç kutunun plakaya oturmasında (mAP50-95).
- Son 10 epoch'ta mozaik veri artırma kapanınca mAP50-95 0.697 → 0.704.
- Eğitim kayıpları düşerken doğrulama skoru düşmedi → aşırı öğrenme yok; erken durdurma (sabır 10) tetiklenmedi.

### OCR'ın gelişimi

| Deneme | Sonuç | Ders |
|---|---|---|
| EasyOCR, tüm mesafelerde | Okunan 13 plakanın **0'ı** formata uyuyor | Okuma bütçesi araç uzaktayken (plaka ~40×12 px) tükeniyordu |
| + yalnızca yakın araçlar (plaka ≥ 50 px, karenin alt %45'i) | Yine 0 geçerli; güven 0.04–0.21 | Genel amaçlı OCR 8–10 px karakterde yetersiz |
| + 3 ön işleme, AB şeridi kırpma, karakter bazında oylama | Sınırlı iyileşme | — |
| **fast-plate-ocr** (plakaya özel, düşük çözünürlükle eğitilmiş, çok ülkeli) | Formata uyan plaka **%80** | Plakaya özel model belirleyici oldu |

**Sonuçlar (tüm video, 9.184 kare, 5 dk 6 sn):**

| Metrik | Değer |
|---|---|
| Ortalama işlem hızı | **17.7 FPS** (tespit+takip 30.4 ms, plaka 16.5 ms, OCR 4.0 ms / kare) |
| Şehre giren araç | **313** (çıkan 0; videoda tek yönlü trafik) |
| Türlere göre | Otomobil 278, Kamyon 29, Motosiklet 4, Otobus 2 |
| Plaka okunan araç | **307 / 313** (%98) |
| Polonya formatına uyan plaka | **251 / 313** (%80) |
| Tekrar geçiş | 0 |

> Formata uymak **doğru okunmuş olmak demek değildir** (ör. `WA4862A` formata uyar ama gerçek plaka `WA4869A`). Gerçek doğruluk için `scripts/plate_review.py` ile örneklem etiketlenip `--score` çalıştırılmalıdır (zaman yetmediği için tam ölçüm yapılmadı).

**Çıktılar:** sonuç videosu (sağ üst sayaç paneli, sol üst son geçişler listesi), `*_events.csv/json` (zaman, araç ID, tür, araç güveni, plaka, oy payı, format geçerliliği, plaka tespit güveni, OCR güveni, tekrar, görsel yolları), `crops/vehicle_<id>.jpg` ve `crops/plate_<id>.jpg`.

**Plaka doğrulama aracı:** `python scripts/plate_review.py` → `outputs/task2/plate_review.html` (araç + plaka kırpımı + okunan metin) ve doldurulacak `plate_ground_truth.csv`; `--score` ile tam eşleşme ve karakter doğruluğu.

---

## Kullanılan model ve kütüphaneler

| Ad | Kullanım | Lisans |
|---|---|---|
| Ultralytics YOLO11l / YOLO11m | Kişi/araç tespiti; plaka modeli tabanı | AGPL-3.0 |
| BoT-SORT, ByteTrack (Ultralytics içi) | Çoklu nesne takibi | AGPL-3.0 |
| OpenCLIP ViT-B/16 (`laion2b_s34b_b88k`) | Sıfır-atış (zero-shot) cinsiyet tahmini | MIT |
| fast-plate-ocr | Plaka metni tanıma (ana) | MIT |
| EasyOCR | Plaka metni tanıma (yedek) | Apache-2.0 |
| keremberke/license-plate-object-detection | Plaka modeli eğitim verisi | Roboflow kaynaklı |
| OpenCV, NumPy, PyTorch, ffmpeg | Video, görüntü işleme, çıkarım | — |

---

## Bilinen hatalar ve kısıtlar

- **Etiketli doğruluk verisi yok.** Kişi/araç tespiti ve takip için MOTA/IDF1 hesaplanamadı; karşılaştırmalar görsel kontrol ve dolaylı göstergelerle (kimlik sayısı, iz süresi, güven dağılımı) yapıldı.
- **Kafe videosunda bozuk MPEG-4 blokları** var (`marker does not match f_code`, ~240 bin kez); OpenCV ile işlerken çökme yaşandı. H.264'e yeniden kodlanarak çözüldü (kare sayısı korunuyor: 2.159).
- **Personel / müşteri ayrımı yok:** kapıdan sık geçen garsonlar da sayılabilir (kimlik başına yön başına 1 sayım bunu kısmen sınırlar).
- **Kimlik kopmaları tamamen bitmedi:** uzun kapanmalarda (> 5 sn) kişi yeni kimlik alabilir. Sayım çizgiye dayandığı için çizgiden uzaktaki kopmalar sayımı etkilemez.
- **Cinsiyet tahmini** saç/kıyafet gibi klişelere açık ipuçlarına dayanır; küçük ve sırtı dönük kırpımlarda zayıflar.
- **Plaka OCR'ı:** video 1080p ve araçlar hareketli; plakalar en yakın noktada ~70×20 px, karakter yüksekliği ~8–10 px ve hareket bulanıklığı var. Formata uyan okumalarda bile tek karakter hataları olabiliyor (`9`↔`2` gibi); etiketli tam doğruluk ölçümü yapılmadı.
- **Sayım çizgisi elle ayarlı:** başka bir kamera açısında `--line` ile yeniden belirlenmeli.
- **Plakalar Polonya formatında;** dokümandaki Türkiye formatı doğrulaması uygulandı ama bu videoda doğrudan sınanamadı.
- **FPS:** tüm modeller açıkken Aşama 1 ~22 FPS (video 25 FPS); Aşama 2 OCR nedeniyle daha yavaş.

## Geliştirme önerileri

- Plaka OCR'ı için 4K kaynak video, birkaç kareyi hizalayıp birleştiren süper çözünürlük veya fast-plate-ocr'ın bu kameradan birkaç yüz plaka ile ince ayarı.
- TensorRT / FP16 dışa aktarımı ile 1.5–2 kat hız.
- Personeli ayırmak için kıyafet renginden basit bir sınıflandırıcı veya kısa süreli yeniden tanıma (ReID) galerisi.
- Cinsiyet için yaya özellik modeli (PA-100K) ile ince ayar ve etiketli küçük bir doğrulama seti.
- Sonuçlar için basit web paneli (Streamlit) ve canlı kamera akışı desteği.
