"""Plaka metni: temizleme, format doğrulama/düzeltme ve araç bazında oylama (torch gerektirmez).

Video Polonya plakaları içeriyor (ör. WA 4869A, WU 5213U); görev dokümanı Türkiye formatı istiyor.
Bu yüzden format bölgeye göre seçilebilir: --plate-format pl | tr | any
"""
import re

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8"}

# Polonya: 2–3 harf bölge kodu + 4–5 karakter (harf/rakam). Toplam 7–8.
PL_RE = re.compile(r"^[A-Z]{2,3}[A-Z0-9]{4,5}$")
# Türkiye: il kodu 01–81 + 1–3 harf + 2–4 rakam (ör. 38 ABC 123, 06 A 1234). Toplam 7–8 (bazı 5–6).
TR_RE = re.compile(r"^(0[1-9]|[1-7][0-9]|8[01])[A-Z]{1,3}[0-9]{2,4}$")


def clean(text):
    return "".join(ch for ch in text.upper() if ch in ALNUM)


def fix_pl(t):
    """PL: AB şeridindeki 'PL' önekini at, bölge kodundaki rakamları harfe çevir."""
    if t.startswith("PL") and len(t) >= 9:
        t = t[2:]
    if len(t) < 7 or len(t) > 8:
        return t, False
    # bölge kodu: ilk 2 karakter kesin harf; 3. karakter harfse 3 harfli kod olabilir
    head = "".join(TO_LETTER.get(c, c) for c in t[:2])
    t = head + t[2:]
    return t, bool(PL_RE.match(t))


def fix_tr(t):
    """TR: ilk iki karakter rakam, ardından harf bloğu, sonra rakam bloğu."""
    if t.startswith("TR") and len(t) >= 7:
        t = t[2:]
    if len(t) < 5 or len(t) > 9:
        return t, False
    head = "".join(TO_DIGIT.get(c, c) for c in t[:2])
    rest = t[2:]
    # harf bloğu uzunluğunu (1–3) en az karakter değişikliği gerektirecek şekilde seç
    best = None
    for k in (1, 2, 3):
        L, D = rest[:k], rest[k:]
        if not 2 <= len(D) <= 4:
            continue
        cost = sum(c.isdigit() for c in L) + sum(c.isalpha() for c in D)
        cand = head + "".join(TO_LETTER.get(c, c) for c in L) + "".join(TO_DIGIT.get(c, c) for c in D)
        if TR_RE.match(cand) and (best is None or cost < best[0]):
            best = (cost, cand)
    if best:
        return best[1], True
    cand = head + rest
    return cand, bool(TR_RE.match(cand))


def normalize(text, fmt="pl"):
    """(düzeltilmiş_metin, format_geçerli_mi)"""
    t = clean(text)
    if fmt == "pl":
        return fix_pl(t)
    if fmt == "tr":
        return fix_tr(t)
    a, ok_a = fix_pl(t)
    if ok_a:
        return a, True
    b, ok_b = fix_tr(t)
    return (b, True) if ok_b else (t, False)


@dataclass
class PlateVoter:
    """Araç başına plaka okumalarını biriktirir. Geçerli formattaki okumalar 2 kat ağırlık alır."""
    every: int = 2
    max_reads: int = 25
    fmt: str = "pl"
    valid_bonus: float = 2.0
    _votes: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    _n: dict = field(default_factory=lambda: defaultdict(int))
    _last: dict = field(default_factory=dict)
    _valid: dict = field(default_factory=dict)
    _best_score: dict = field(default_factory=dict)  # kayıt için en iyi kırpımın skoru

    def wants(self, tid, frame_idx):
        if self._n[tid] >= self.max_reads:
            return False
        last = self._last.get(tid)
        return last is None or frame_idx - last >= self.every

    def add(self, tid, frame_idx, text, valid, ocr_conf, det_conf):
        self._last[tid] = frame_idx
        self._n[tid] += 1
        if len(text) < 4:
            return 0.0
        w = float(ocr_conf) * float(det_conf) * (self.valid_bonus if valid else 1.0)
        self._votes[tid][text] += w
        self._valid[text] = valid
        return w

    def result(self, tid, fmt=None):
        """(plaka, güven, format_geçerli, okuma_sayısı).

        1) En sık görülen uzunluktaki okumalar alınır,
        2) her karakter pozisyonu için ağırlıklı oylama yapılır (okumalar kısmen doğru olsa bile
           doğru karakterler birleşir), 3) sonuç format kuralından geçirilir.
        Karakter oylaması geçersiz formata düşerse, tam metin oylamasının kazananı kullanılır.
        güven = karakter başına kazanan oy payının ortalaması."""
        v = self._votes.get(tid)
        if not v:
            return "", 0.0, False, self._n.get(tid, 0)
        whole, ww = max(v.items(), key=lambda kv: kv[1])
        by_len = defaultdict(float)
        for t, w in v.items():
            by_len[len(t)] += w
        L = max(by_len, key=by_len.get)
        pos = [defaultdict(float) for _ in range(L)]
        for t, w in v.items():
            if len(t) == L:
                for i, ch in enumerate(t):
                    pos[i][ch] += w
        chars = [max(p, key=p.get) for p in pos]
        conf = float(np.mean([p[c] / sum(p.values()) for p, c in zip(pos, chars)]))
        cand = "".join(chars)
        fmt = fmt or self.fmt
        cand_n, ok = normalize(cand, fmt)
        if ok or not self._valid.get(whole, False):
            return cand_n, conf, ok, self._n[tid]
        return whole, ww / (sum(v.values()) + 1e-9), True, self._n[tid]
