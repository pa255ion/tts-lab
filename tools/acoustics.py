#!/usr/bin/env python3
"""Objective measures of synthetic speech, for tuning naturalness.

    tools/acoustics.py [-s] out.wav labels [more.wav more-labels ...]

labels is either `ttslab frames -e <engine> "<text>"` output (CSV, for the
same text and options) or a Festival/CMU ARCTIC .lab file, for comparing
against natural recordings. Prints, per file:

  * HNR: harmonics-to-noise ratio in the middle of vowels (Boersma 1993
    autocorrelation method). Natural modal voice: roughly 15-25 dB; below
    ~12 dB sounds hoarse or breathy.
  * Obstruent peak level relative to the median vowel level, per class
    (stop bursts, fricatives). In natural speech /s/ peaks about 5-15 dB
    below the vowels and stop bursts lower still.
  * Long-term average spectrum (LTAS) in octave bands, relative to the
    500 Hz band.
  * F0 (estimated from the audio, so synthetic and natural are measured
    the same way) over vowels: spread in semitones, and how much it moves
    in 10 ms.
  * H1-H2 in vowels: the level of the first harmonic over the second, a
    standard voice-quality measure (higher = breathier/laxer, lower =
    tenser/creakier; typical modal male speech ~0-6 dB).
  * Vowel level spread: standard deviation of vowel levels (stress and
    phrasing make natural vowels differ by several dB).

-s pools all files into one summary (per-file values are normalized first:
levels re that file's median vowel, F0 re its median).

An objective number is not a listener. These are for spotting gross
departures from natural speech, not for ranking voices.
"""
import csv
import sys
import wave

import numpy as np

VOWELS = set("iy ih eh ae aa ah ao uh uw er ax ey ay oy aw ow".split())
STOPS = set("p t k b d g".split())
FRICS = set("f v th dh s z sh zh ch jh".split())


def read_wav(path):
    with wave.open(path) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)
    return x / 32768.0, sr


def read_segments(path):
    """-> [(phone, t_start, t_end)] from a ttslab frames CSV or a .lab file."""
    if path.endswith(".lab"):
        out, t0 = [], 0.0
        for line in open(path):
            f = line.split()
            if len(f) < 3 or f[0] == "#":
                continue
            t = float(f[0])
            out.append([f[2], t0, t])
            t0 = t
        return out
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            t, p = float(r["t_ms"]) / 1000.0, r["phone"]
            if out and out[-1][0] == p:
                out[-1][2] = t + 0.005
            else:
                out.append([p, t, t + 0.005])
    return out


def f0_track(x, sr, a, b, step=0.01, f0_lo=60, f0_hi=400):
    """Autocorrelation F0 every `step` s in [a, b); None where unvoiced."""
    out = []
    h = int(0.02 * sr)
    t = a
    while t < b:
        c = int(t * sr)
        seg = x[max(0, c - h) : c + h]
        f = None
        if len(seg) == 2 * h:
            seg = (seg - seg.mean()) * np.hanning(len(seg))
            r = np.correlate(seg, seg, "full")[len(seg) - 1 :]
            lo, hi = int(sr / f0_hi), int(sr / f0_lo)
            if r[0] > 0:
                k = lo + int(np.argmax(r[lo:hi]))
                if r[k] / r[0] > 0.45:
                    f = sr / k
        out.append(f)
        t += step
    return out


def h1h2(x, sr, f0):
    n = len(x)
    nfft = 1 << (8 * n).bit_length()
    sp = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(n), nfft))
    hz = sr / nfft

    def peak(f):
        lo, hi = int((f * 0.85) / hz), int((f * 1.15) / hz) + 1
        return 20 * np.log10(sp[lo:hi].max() + 1e-12)

    return peak(f0) - peak(2 * f0)


def hnr(x, sr, f0_lo=60, f0_hi=400):
    """Boersma's autocorrelation HNR for one windowed stretch."""
    w = np.hanning(len(x))
    xw = (x - x.mean()) * w
    n = 1 << (2 * len(x) - 1).bit_length()
    r = np.fft.irfft(np.abs(np.fft.rfft(xw, n)) ** 2)[: len(x)]
    rw = np.fft.irfft(np.abs(np.fft.rfft(w, n)) ** 2)[: len(x)]
    if r[0] <= 0:
        return None
    rn = (r / r[0]) / (rw / rw[0])
    lo, hi = int(sr / f0_hi), min(int(sr / f0_lo), len(x) // 2)
    k = lo + int(np.argmax(rn[lo:hi]))
    peak = min(rn[k], 0.9999)
    return 10 * np.log10(peak / (1 - peak))


def level_db(x):
    return 10 * np.log10(np.mean(x * x) + 1e-12)


def measure(wav_path, lab_path):
    """Per-file measurements, normalized so that files can be pooled."""
    x, sr = read_wav(wav_path)
    m = {"hnr": [], "vlev": [], "stop": {}, "fric": {}, "st": [], "dst": [], "h12": [], "f0": [], "ltas": {}}
    vlev = []
    obstr = []
    f0s = []
    for p, a, b in read_segments(lab_path):
        i, j = int(a * sr), min(int(b * sr), len(x))
        if j - i < int(0.02 * sr):
            continue
        if p in VOWELS:
            mid = (i + j) // 2
            h = int(0.02 * sr)
            v = hnr(x[mid - h : mid + h], sr)
            if v is not None:
                m["hnr"].append(v)
            vlev.append(level_db(x[i:j]))
            tr = f0_track(x, sr, a + 0.01, b - 0.01)
            v = [f for f in tr if f]
            f0s += v
            for u, w in zip(tr, tr[1:]):
                if u and w:
                    m["dst"].append(abs(12 * np.log2(w / u)))
            if v:
                m["h12"].append(h1h2(x[mid - 2 * h : mid + 2 * h], sr, float(np.median(v))))
        elif p in STOPS or p in FRICS:
            win = int(0.01 * sr)
            pk = max(level_db(x[k : k + win]) for k in range(i, max(i + 1, j - win), win // 2))
            obstr.append((p, pk))
    vref = float(np.median(vlev)) if vlev else 0.0
    m["vlev"] = [v - vref for v in vlev]
    for p, pk in obstr:
        m["stop" if p in STOPS else "fric"].setdefault(p, []).append(pk - vref)
    if f0s:
        m["f0"] = f0s
        m["st"] = list(12 * np.log2(np.array(f0s) / np.median(f0s)))

    # LTAS over the whole file (silence contributes nothing).
    n = 1024
    win = np.hanning(n)
    spec = np.zeros(n // 2 + 1)
    for k in range(0, len(x) - n, n // 2):
        spec += np.abs(np.fft.rfft(x[k : k + n] * win)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    for c in BANDS:
        sel = (freqs >= c / np.sqrt(2)) & (freqs < min(c * np.sqrt(2), sr / 2))
        if sel.any():
            m["ltas"][c] = spec[sel].sum()
    ref = m["ltas"][500]
    m["ltas"] = {c: [10 * np.log10(v / ref + 1e-12)] for c, v in m["ltas"].items()}
    return m


BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]


def pool(ms):
    out = {"hnr": [], "vlev": [], "stop": {}, "fric": {}, "st": [], "dst": [], "h12": [], "f0": [], "ltas": {}}
    for m in ms:
        for k in ("hnr", "vlev", "st", "dst", "h12", "f0"):
            out[k] += m[k]
        for k in ("stop", "fric", "ltas"):
            for p, v in m[k].items():
                out[k].setdefault(p, []).extend(v)
    return out


def show(title, m):
    print(f"== {title}")
    if m["hnr"]:
        print(f"  vowel HNR        median {np.median(m['hnr']):5.1f} dB  (10th pct {np.percentile(m['hnr'], 10):5.1f}, n={len(m['hnr'])})")
    for name, k in (("stop peak", "stop"), ("fricative peak", "fric")):
        if m[k]:
            cells = "  ".join(f"{p} {np.median(v):+5.1f}" for p, v in sorted(m[k].items()))
            print(f"  {name:15s}  re vowels (dB): {cells}")
    print("  LTAS re 500 Hz   " + "  ".join(f"{c}:{np.mean(m['ltas'][c]):+5.1f}" for c in BANDS if c in m["ltas"]))
    if m["vlev"]:
        print(f"  vowel level      spread (sd) {np.std(m['vlev']):4.1f} dB")
    if m["st"]:
        print(f"  F0 (vowels)      median {np.median(m['f0']):5.1f} Hz, sd {np.std(m['st']):4.2f} st, |dF0| per 10 ms mean {np.mean(m["dst"]):4.2f} median {np.median(m["dst"]):4.2f} p90 {np.percentile(m["dst"], 90):4.2f} st")
    if m["h12"]:
        print(f"  H1-H2 (vowels)   median {np.median(m['h12']):+5.1f} dB")


def main():
    args = sys.argv[1:]
    summary = bool(args) and args[0] == "-s"
    if summary:
        args = args[1:]
    if not args or len(args) % 2:
        sys.exit(__doc__)
    ms = []
    for k in range(0, len(args), 2):
        ms.append(measure(args[k], args[k + 1]))
        if not summary:
            show(args[k], ms[-1])
    if summary:
        show(f"{len(ms)} files pooled", pool(ms))


if __name__ == "__main__":
    main()
