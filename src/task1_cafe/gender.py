"""Adım 4 — Cinsiyet sınıflandırma (CLIP zero-shot, tüm vücut kırpımı) + kişi bazında oylama.

ÖNEMLİ: Çıktı yalnızca görsel bir model tahminidir. Kişinin kimliğini veya kendini nasıl
tanımladığını belirlemez; saç, kıyafet, siluet gibi ipuçlarına dayanır ve hata payı içerir.

Yapı
- ClipGender : kırpımları toplu (batch) olarak CLIP'e verir, P(erkek) döndürür.
- GenderVoter: her kimlik için örnekleri biriktirir; ne zaman yeni örnek alınacağına ve
               son etiketin ne olacağına karar verir (torch gerektirmez, test edilebilir).
"""
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

# Birden fazla metin kalıbının ortalaması tek kalıba göre daha kararlı sonuç verir (prompt ensembling).
PROMPTS = {
    "male": ["a photo of a man", "a photo of a male person", "a man sitting in a cafe",
             "a man walking", "a CCTV image of a man"],
    "female": ["a photo of a woman", "a photo of a female person", "a woman sitting in a cafe",
               "a woman walking", "a CCTV image of a woman"],
}
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def letterbox_square(crop, size=224):
    """Kırpımı kesmeden kareye tamamlar (baş ve ayaklar kaybolmasın), sonra size x size yapar."""
    h, w = crop.shape[:2]
    s = max(h, w)
    canvas = np.full((s, s, 3), 124, dtype=np.uint8)  # CLIP ortalamasına yakın gri
    y0, x0 = (s - h) // 2, (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = crop
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


class ClipGender:
    def __init__(self, arch="ViT-B-16", pretrained="laion2b_s34b_b88k", device="cuda"):
        import open_clip
        import torch

        self.torch = torch
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=device)
        self.model.eval()
        tok = open_clip.get_tokenizer(arch)
        with torch.no_grad(), torch.autocast("cuda", enabled=device == "cuda"):
            feats = []
            for cls in ("male", "female"):
                t = self.model.encode_text(tok(PROMPTS[cls]).to(device)).float()
                t = t / t.norm(dim=-1, keepdim=True)
                m = t.mean(0)
                feats.append(m / m.norm())
        self.text = torch.stack(feats)            # (2, D): [erkek, kadın]
        self.scale = float(self.model.logit_scale.exp())

    def prob_male(self, crops):
        """crops: BGR numpy listesi -> her biri için P(erkek)."""
        if not crops:
            return np.zeros(0)
        torch = self.torch
        x = np.stack([letterbox_square(c)[:, :, ::-1] for c in crops]).astype(np.float32) / 255.0
        x = (x - CLIP_MEAN) / CLIP_STD
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad(), torch.autocast("cuda", enabled=self.device == "cuda"):
            img = self.model.encode_image(x).float()
        img = img / img.norm(dim=-1, keepdim=True)
        probs = (self.scale * img @ self.text.T).softmax(dim=-1)
        return probs[:, 0].cpu().numpy()


@dataclass
class GenderVoter:
    every: int = 5            # bir kimlik için en sık kaç karede bir örnek alınır
    max_samples: int = 15     # kimlik başına en fazla örnek (sonrası gereksiz; FPS korunur)
    min_samples: int = 3      # etiket göstermek için gereken asgari örnek
    min_height: int = 80      # bundan kısa kırpımlar (uzak/çok kapanmış) kullanılmaz (piksel)
    decide_thr: float = 0.65  # ortalama olasılık bu eşiğin altındaysa "belirsiz"
    _samples: dict = field(default_factory=lambda: defaultdict(list))  # gid -> [(p_male, ağırlık)]
    _last: dict = field(default_factory=dict)                          # gid -> son örnek karesi

    def wants(self, gid, frame_idx, box_h):
        """Bu karede bu kimlikten yeni örnek alınmalı mı?"""
        if box_h < self.min_height or len(self._samples[gid]) >= self.max_samples:
            return False
        last = self._last.get(gid)
        return last is None or frame_idx - last >= self.every

    def add(self, gid, frame_idx, p_male, box_h, det_conf):
        # Büyük ve güvenli tespitler daha bilgilidir -> ağırlık = güven * min(1, yükseklik/200)
        w = float(det_conf) * min(1.0, box_h / 200.0)
        self._samples[gid].append((float(p_male), w))
        self._last[gid] = frame_idx

    def result(self, gid):
        """(etiket, güven, örnek_sayısı). etiket: 'E' (erkek), 'K' (kadın), '?' (belirsiz/yetersiz)."""
        s = self._samples.get(gid, [])
        if not s:
            return "?", 0.0, 0
        p = np.array([a for a, _ in s]); w = np.array([b for _, b in s]) + 1e-6
        pm = float(np.average(p, weights=w))
        conf = max(pm, 1 - pm)
        if len(s) < self.min_samples or conf < self.decide_thr:
            return "?", conf, len(s)
        return ("E" if pm >= 0.5 else "K"), conf, len(s)

    def p_male(self, gid):
        s = self._samples.get(gid, [])
        if not s:
            return None
        p = np.array([a for a, _ in s]); w = np.array([b for _, b in s]) + 1e-6
        return float(np.average(p, weights=w))
