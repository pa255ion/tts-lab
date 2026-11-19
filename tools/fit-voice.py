#!/usr/bin/env python3
"""Fit a voice profile for the refined formant engine to a speaker's recordings.

    tools/align.py voice NAME                       # first: phone alignment
    tools/fit-voice.py fit NAME [--iters 4] [--no-bandwidths] [--loci] [--full]
    tools/fit-voice.py score NAME [--voice PROFILE] [--style S] [--ref PROFILE]
    ttslab say --voice data/voice/NAME/profile.txt "Hello."

Analysis by synthesis. The engine is run on the same sentences the speaker
recorded, with the current profile; both the recordings and the synthetic
speech are measured the same way at the same (aligned) points; and each
profile value is moved by the difference. Measuring the synthesizer's own
output, rather than copying measurements into it, cancels the biases of the
measurements (LPC formants of a nasal, bandwidths, F0 estimates) and of the
engine's coarticulation model (a target is never quite reached in fast
speech, so copying sentence-medial formants into the targets would reduce
the vowels twice). Measured, per phone or per style:

  * F1-F3 of vowels (middle; 20% and 80% of diphthongs), nasals, liquids and
    glides, by LPC with formant picking guided by the expected values;
    stressed vowels only, and the sustained-vowel and h_d prompts when there
    are enough of them;
  * with --loci only, the formant targets (loci) of non-velar obstruents,
    from F2/F3 at the edges of the neighbouring vowels (they cost
    intelligibility: NOTES §8); the classic locus equation per place is
    always computed, for the report;
  * B1-B3 (LPC), only as a ratio to the synthetic measurement, clamped;
  * frication and burst peak levels relative to the vowels (written half
    way to the measured value, in log, unless --full; likewise breath);
  * durations per phone (shrunk toward 1 when there are few tokens);
  * per style: median F0 (-> base F0), F0 spread in semitones (-> pitch
    range), speech rate, H1-H2 (-> Rd offset) and vowel HNR (-> breath noise);
  * the token-to-token spread of F1-F3 that the engine does not reproduce
    (between takes of the same prompt when there are several takes;
    otherwise from the residuals): --variation in ttslab.

`fit` writes data/voice/NAME/profile.txt. `score` synthesizes the aligned
recordings' sentences (with --voice, or the stock voice) and reports how far
the synthetic speech is from the recordings on the same measures.
"""
import argparse
import csv
import itertools
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acoustics  # noqa: E402
import align  # noqa: E402

ROOT = align.ROOT
BIN = align.BIN
SR = 16000

VOWELS = set("iy ih eh ae aa ah ao uh uw er".split())
DIPH = set("ey ay ow aw oy".split())
SONOR_C = set("m n ng l r w y".split())
STOPS = set("p t k b d g".split())
FRIC = set("f v th dh s z sh zh ch jh".split())
VELAR = set("k g ng".split())
SENTENCES = set("QPSR")  # script sections read as ordinary sentences
PLACE = {"p": "labial", "b": "labial", "m": "labial", "f": "labial", "v": "labial",
         "t": "alveolar", "d": "alveolar", "n": "alveolar", "s": "alveolar", "z": "alveolar",
         "th": "dental", "dh": "dental", "sh": "postalveolar", "zh": "postalveolar",
         "ch": "postalveolar", "jh": "postalveolar", "k": "velar", "g": "velar"}


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

class Profile:
    def __init__(self):
        self.name = ""
        self.ph = {}  # phone -> dict(f, fe, b, sd, dur, amp, n)
        self.styles = {}  # name -> dict(f0, range, rate, rd, breath)
        self.comments = []

    @staticmethod
    def parse(text):
        p = Profile()
        for line in text.splitlines():
            line = line.split("#")[0].split()
            if not line:
                continue
            if line[0] == "name":
                p.name = " ".join(line[1:])
            elif line[0] == "phone":
                e = {"dur": 1.0, "amp": 1.0, "sd": [0.0, 0.0, 0.0]}
                i = 2
                while i < len(line):
                    k = line[i]
                    if k in ("f", "fe", "b", "sd"):
                        e[k] = [float(v) for v in line[i + 1 : i + 4]]
                        i += 4
                    else:
                        e[k] = float(line[i + 1])
                        i += 2
                if "fe" not in e:
                    e["fe"] = list(e["f"])
                p.ph[line[1]] = e
            elif line[0] == "style":
                s = {"f0": 110.0, "range": 1.0, "rate": 1.0, "rd": 0.0, "breath": 1.0}
                for k, v in zip(line[2::2], line[3::2]):
                    s[k] = float(v)
                p.styles[line[1]] = s
        return p

    def copy(self):
        q = Profile.parse(self.text())
        q.comments = list(self.comments)
        return q

    def text(self):
        out = ["# tts-lab voice profile, fitted by tools/fit-voice.py (see src/voice_profile.zig)"]
        out += ["# " + c for c in self.comments]
        out.append(f"name {self.name}")
        for ph, e in self.ph.items():
            s = f"phone {ph:2s} f {e['f'][0]:.0f} {e['f'][1]:.0f} {e['f'][2]:.0f}"
            if ph in DIPH:
                s += f" fe {e['fe'][0]:.0f} {e['fe'][1]:.0f} {e['fe'][2]:.0f}"
            s += f" b {e['b'][0]:.0f} {e['b'][1]:.0f} {e['b'][2]:.0f}"
            if any(e["sd"]):
                s += f" sd {e['sd'][0]:.0f} {e['sd'][1]:.0f} {e['sd'][2]:.0f}"
            if abs(e["dur"] - 1) > 0.005:
                s += f" dur {e['dur']:.3f}"
            if abs(e["amp"] - 1) > 0.005:
                s += f" amp {e['amp']:.3f}"
            if e.get("n"):
                s += f" n {int(e['n'])}"
            out.append(s)
        for name, s in self.styles.items():
            out.append(f"style {name} f0 {s['f0']:.1f} range {s['range']:.3f} rate {s['rate']:.3f} rd {s['rd']:.3f} breath {s['breath']:.3f}")
        return "\n".join(out) + "\n"


def stock():
    out = subprocess.run([BIN, "profile"], capture_output=True, text=True, check=True).stdout
    return Profile.parse(out)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def lpc_roots(seg, order=18):
    seg = np.append(seg[0], seg[1:] - 0.7 * seg[:-1]) * np.hamming(len(seg))
    r = np.correlate(seg, seg, "full")[len(seg) - 1 : len(seg) + order]
    if r[0] <= 0:
        return np.zeros(0), np.zeros(0)
    r[0] *= 1 + 1e-6
    # Levinson-Durbin.
    a = np.zeros(order + 1)
    a[0] = 1
    e = r[0]
    for i in range(1, order + 1):
        k = -(r[i] + a[1:i] @ r[i - 1 : 0 : -1]) / e
        a[1 : i + 1] = a[1 : i + 1] + k * a[i - 1 :: -1][: i]
        e *= 1 - k * k
        if e <= 0:
            break
    z = np.roots(a)
    z = z[np.imag(z) > 0]
    f = np.angle(z) * SR / (2 * np.pi)
    b = -np.log(np.abs(z)) * SR / np.pi
    keep = (f > 120) & (f < 5200) & (b < 800)
    o = np.argsort(f[keep])
    return f[keep][o], b[keep][o]


def pick(f, b, ref):
    """Choose F1<F2<F3 among LPC candidates closest (log) to ref."""
    best, arg = 1e9, None
    n = len(f)
    if n < 3:
        return None
    for c in itertools.combinations(range(n), 3):
        cost = sum(abs(math.log(f[c[k]] / ref[k])) for k in range(3)) + 0.0004 * sum(b[c[k]] for k in range(3))
        if cost < best:
            best, arg = cost, c
    if arg is None or best > 1.6:
        return None
    return [f[i] for i in arg], [b[i] for i in arg]


def formants_at(x, t, ref, half=0.0125):
    i, j = int((t - half) * SR), int((t + half) * SR)
    if i < 0 or j > len(x):
        return None
    f, b = lpc_roots(x[i:j])
    return pick(f, b, ref)


def formants_over(x, ts, ref):
    got = [formants_at(x, t, ref) for t in ts]
    got = [g for g in got if g]
    if not got:
        return None
    F = np.median([g[0] for g in got], axis=0)
    B = np.median([g[1] for g in got], axis=0)
    return list(F), list(B)


def rms_db(x):
    return 10 * np.log10(np.mean(x * x) + 1e-12)


# ---------------------------------------------------------------------------
# Utterances
# ---------------------------------------------------------------------------

class Utt:
    """One recording (or its synthetic counterpart): audio and timed phones
    with the front end's features."""

    def __init__(self, key, x, segs):
        self.key, self.x, self.segs = key, x, segs  # segs: dicts with phone, t0, t1, stress, accent, ...


def feature_segs(text):
    out = subprocess.run([BIN, "segs", text], capture_output=True, text=True, check=True).stdout
    return [r for r in csv.DictReader(out.splitlines())]


def natural_utts(name):
    sp = align.script_prompts()
    base = os.path.join(ROOT, "data", "voice", name)
    utts = []
    for style, take, pid, wav in align.voice_files(name):
        lab = os.path.join(base, "align", style, take, pid + ".lab")
        if pid not in sp or not os.path.exists(lab):
            continue
        sec, text = sp[pid]
        spoken = align.spoken_text(pid, text)
        feats = [r for r in feature_segs(spoken) if r["phone"] != "pau"]
        L = [s for s in align.read_lab(lab) if s[0] != "pau"]
        if [r["phone"] for r in feats] != [s[0] for s in L]:
            continue
        segs = [dict(r, t0=a, t1=b, i=k) for k, (r, (_, a, b)) in enumerate(zip(feats, L))]
        utts.append(Utt((style, take, pid, sec, spoken), align.read_wav(wav), segs))
    return utts


def synthesize(prof, utts, style, workdir):
    """Synthesize each distinct sentence of `utts` with `prof` -> {spoken: Utt}."""
    texts = sorted({u.key[4] for u in utts})
    pfile = os.path.join(workdir, "profile.txt")
    with open(pfile, "w") as f:
        f.write(prof.text())
    lst = os.path.join(workdir, "list.txt")
    with open(lst, "w") as f:
        f.write("\n".join(texts) + "\n")
    tag = os.path.join(workdir, "s")
    subprocess.run([BIN, "batch", lst, tag, "-e", "refined", "--voice", pfile, "--style", style],
                   capture_output=True, text=True, check=True)
    out = {}
    for k, t in enumerate(texts):
        x = align.read_wav(f"{tag}_{k:03d}.wav")
        rows = [r for r in csv.DictReader(open(f"{tag}_{k:03d}.csv")) if r["phone"] != "pau"]
        segs = []
        for i, r in enumerate(rows):
            a = float(r["start_ms"]) / 1000
            segs.append(dict(r, t0=a, t1=a + float(r["dur_ms"]) / 1000, i=i))
        out[t] = Utt(None, x, segs)
    return out


def measure(u, prof):
    """Per-token measures of one utterance -> {token index: dict}."""
    x, segs = u.x, u.segs
    m = {}
    vlev = []
    for s in segs:
        if s["phone"] in VOWELS | DIPH and s["t1"] - s["t0"] > 0.03:
            vlev.append(rms_db(x[int(s["t0"] * SR) : int(s["t1"] * SR)]))
    vref = float(np.median(vlev)) if vlev else 0.0
    for k, s in enumerate(segs):
        p, a, b = s["phone"], s["t0"], s["t1"]
        d = b - a
        r = {"dur": d}
        e = prof.ph.get(p)
        if p in VOWELS or p in SONOR_C:
            if d >= 0.03 and e:
                mid = (a + b) / 2
                ts = [mid - 0.01, mid, mid + 0.01] if d > 0.07 else [mid]
                if u.key and u.key[3] == "V":
                    ts = list(np.linspace(a + 0.25 * d, b - 0.25 * d, 7))
                g = formants_over(x, ts, e["f"])
                if g:
                    r["F"], r["B"] = g
        elif p in DIPH and d >= 0.05 and e:
            g1 = formants_over(x, [a + 0.2 * d], e["f"])
            g2 = formants_over(x, [a + 0.8 * d], e["fe"])
            if g1 and g2:
                r["F"], r["B"], r["Fe"] = g1[0], g1[1], g2[0]
        if p in STOPS or p in FRIC:
            # Formants at the edges of the neighbouring vowels.
            if k + 1 < len(segs) and segs[k + 1]["phone"] in VOWELS | DIPH:
                n = segs[k + 1]
                en = prof.ph[n["phone"]]
                if n["t1"] - n["t0"] > 0.05:
                    g = formants_at(x, n["t0"] + 0.015, en["f"])
                    gm = formants_at(x, (n["t0"] + n["t1"]) / 2, en["f"])
                    if g:
                        r["on"] = g[0]
                    if g and gm:
                        r["on_mid"] = gm[0]
            if k > 0 and segs[k - 1]["phone"] in VOWELS | DIPH:
                pv = segs[k - 1]
                ep = prof.ph[pv["phone"]]
                if pv["t1"] - pv["t0"] > 0.05:
                    g = formants_at(x, pv["t1"] - 0.015, ep["fe"])
                    if g:
                        r["off"] = g[0]
            # Peak level re the utterance's median vowel.
            i, j = int(a * SR), int(b * SR)
            w = int(0.01 * SR)
            if j - i > w:
                r["lev"] = max(rms_db(x[q : q + w]) for q in range(i, j - w, w // 2)) - vref
        if p in VOWELS | DIPH and d > 0.06:
            mid = (a + b) / 2
            h = int(0.02 * SR)
            c = int(mid * SR)
            if h < c < len(x) - h:
                v = acoustics.hnr(x[c - h : c + h], SR)
                if v is not None:
                    r["hnr"] = v
                tr = [f for f in acoustics.f0_track(x, SR, a + 0.01, b - 0.01) if f]
                if tr:
                    r["f0"] = tr
                    if c - 2 * h > 0 and c + 2 * h < len(x):
                        r["h12"] = acoustics.h1h2(x[c - 2 * h : c + 2 * h], SR, float(np.median(tr)))
        m[k] = r
    return m


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def stressed(s):
    return s["phone"] in VOWELS | DIPH and s["stress"] in ("1", "2")


def pairs(nat, syn_by_text, nat_m, syn_m):
    """-> list of (utt, seg, nat measure, syn measure) for every token."""
    out = []
    for u in nat:
        su = syn_by_text[u.key[4]]
        if len(su.segs) != len(u.segs):
            continue
        for k, s in enumerate(u.segs):
            out.append((u, s, nat_m[id(u)][k], syn_m[u.key[4]][k]))
    return out


def med(v):
    return float(np.median(v)) if len(v) else None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def update(prof, base, P, style_utts, it, use_bw, loci):
    """One analysis-by-synthesis step on all phone targets."""
    by = defaultdict(list)
    for u, s, n, y in P:
        by[s["phone"]].append((u, s, n, y))
    lines = []
    for ph, toks in sorted(by.items()):
        e, e0 = prof.ph[ph], base.ph[ph]
        sel = toks
        if ph in VOWELS | DIPH:
            sel = [t for t in toks if stressed(t[1])]
            clean = [t for t in sel if t[0].key[3] in ("V", "H")]
            if len(clean) >= 3:
                sel = clean
        # Formant targets of sonorants.
        if ph in VOWELS | DIPH | (SONOR_C - VELAR):
            d = [np.subtract(n["F"], y["F"]) for _, _, n, y in sel if "F" in n and "F" in y]
            if len(d) >= 3:
                dd = np.median(d, axis=0)
                for j in range(3):
                    e["f"][j] = clamp(e["f"][j] + dd[j], 0.6 * e0["f"][j], 1.6 * e0["f"][j])
                if use_bw:
                    rb = [np.divide(n["B"], np.maximum(y["B"], 20)) for _, _, n, y in sel if "B" in n and "B" in y]
                    rb = np.median(rb, axis=0) ** 0.5
                    for j in range(3):
                        e["b"][j] = clamp(e["b"][j] * rb[j], 0.6 * e0["b"][j], 3.0 * e0["b"][j])
                e["n"] = len(d)
            if ph in DIPH:
                d = [np.subtract(n["Fe"], y["Fe"]) for _, _, n, y in sel if "Fe" in n and "Fe" in y]
                if len(d) >= 3:
                    dd = np.median(d, axis=0)
                    for j in range(3):
                        e["fe"][j] = clamp(e["fe"][j] + dd[j], 0.6 * e0["fe"][j], 1.6 * e0["fe"][j])
            else:
                e["fe"] = list(e["f"])
        if ph in SONOR_C:
            # Consonant measurements are less reliable (LPC in nasals, short
            # glides): keep them nearer the stock values.
            for key in ("f", "fe"):
                e[key] = [clamp(v, 0.75 * v0, 1.35 * v0) for v, v0 in zip(e[key], e0[key])]
        for key in ("f", "fe"):
            # Keep the formants in order and apart.
            f = e[key]
            f[1] = max(f[1], f[0] + 150)
            f[2] = max(f[2], f[1] + 200)
        # Loci of non-velar obstruents, from the vowel edges next to them.
        if loci and (ph in STOPS or ph in FRIC) and ph not in VELAR:
            d = []
            for _, _, n, y in sel:
                for key in ("on", "off"):
                    if key in n and key in y:
                        d.append(np.subtract(n[key], y[key])[1:3])
            if len(d) >= 5:
                dd = np.median(d, axis=0)
                for j in (1, 2):
                    e["f"][j] = clamp(e["f"][j] + 1.5 * dd[j - 1], 0.6 * e0["f"][j], 1.5 * e0["f"][j])
                e["fe"] = list(e["f"])
                e["n"] = len(d)
        # Frication / burst level.
        if ph in STOPS or ph in FRIC:
            d = [n["lev"] - y["lev"] for _, _, n, y in sel if "lev" in n and "lev" in y]
            if len(d) >= 3:
                e["amp"] = clamp(e["amp"] * 10 ** (0.8 * med(d) / 20), 0.2, 5.0)
        # Duration (neutral sentences only: not held vowels or carrier
        # phrases), shrunk toward no change.
        dd = [math.log(n["dur"] / y["dur"]) for u, _, n, y in toks
              if u.key[0] == "neutral" and u.key[3] in SENTENCES and y["dur"] > 0 and n["dur"] > 0]
        if dd:
            w = len(dd) / (len(dd) + 20)
            e["dur"] = clamp(e["dur"] * math.exp(w * med(dd)), 0.4, 2.5)
    # Styles.
    for style, P_all_s in style_utts.items():
        st = prof.styles[style]
        # Pitch and tempo from sentences, when there are any.
        P_s = [t for t in P_all_s if t[0].key[3] in SENTENCES] or P_all_s
        nf = [f for _, s, n, _ in P_s for f in n.get("f0", [])]
        yf = [f for _, s, _, y in P_s for f in y.get("f0", [])]
        if len(nf) > 20 and len(yf) > 20:
            st["f0"] = clamp(st["f0"] * math.exp(math.log(np.median(nf) / np.median(yf))), 50, 400)
            sn = np.std(12 * np.log2(np.array(nf) / np.median(nf)))
            sy = np.std(12 * np.log2(np.array(yf) / np.median(yf)))
            st["range"] = clamp(st["range"] * (sn / max(sy, 0.1)) ** 0.8, 0.2, 4)
        nd = sum(n["dur"] for _, _, n, _ in P_s)
        yd = sum(y["dur"] for _, _, _, y in P_s)
        if style != "neutral" and nd > 0 and yd > 0:
            st["rate"] = clamp(st["rate"] * yd / nd, 0.5, 2)
        h = [n["h12"] - y["h12"] for _, s, n, y in P_all_s if "h12" in n and "h12" in y and stressed(s)]
        if len(h) > 10:
            # About +5 dB of H1-H2 per +1 of Rd in this engine (measured by
            # the sweep in `fit`); damped.
            st["rd"] = clamp(st["rd"] + 0.6 * med(h) / SLOPE["rd"], -0.4, 1.2)
        q = [n["hnr"] - y["hnr"] for _, s, n, y in P_all_s if "hnr" in n and "hnr" in y and stressed(s)]
        if len(q) > 10:
            st["breath"] = clamp(st["breath"] * 2 ** (0.6 * med(q) / SLOPE["breath"]), 0.2, 6)
    return lines


SLOPE = {"rd": 5.0, "breath": -3.0}  # dB of H1-H2 per Rd; dB of HNR per doubling of breath


def spreads(prof, P):
    """Token-to-token F1-F3 spread the engine does not model."""
    # Between takes of the same prompt and token.
    takes = defaultdict(dict)
    for u, s, n, y in P:
        if "F" in n:
            takes[(u.key[0], u.key[2], s["i"])][u.key[1]] = (s["phone"], n["F"])
    diffs = defaultdict(list)
    for v in takes.values():
        vals = list(v.values())
        if len(vals) >= 2:
            ph = vals[0][0]
            F = np.array([x[1] for x in vals])
            diffs[ph] += list(F - F.mean(0))
    res = defaultdict(list)
    for u, s, n, y in P:
        if "F" in n and "F" in y:
            res[s["phone"]].append(np.subtract(n["F"], y["F"]))
    for ph, e in prof.ph.items():
        if len(diffs[ph]) >= 6:
            d = np.array(diffs[ph])
            # Deviations from a mean of k takes shrink by sqrt((k-1)/k); ~2 takes.
            e["sd"] = list(1.4826 * np.median(np.abs(d), 0) * math.sqrt(2))
        elif len(res[ph]) >= 6:
            d = np.array(res[ph])
            mad = 1.4826 * np.median(np.abs(d - np.median(d, 0)), 0)
            # Residuals also hold context effects the engine gets wrong and
            # measurement error: count half of them.
            e["sd"] = list(0.5 * mad)
        e["sd"] = [min(v, 0.15 * f) for v, f in zip(e["sd"], e["f"])]


def calibrate(prof, nat, style, workdir):
    """Measure the engine's H1-H2 per unit Rd and HNR per doubling of breath."""
    sub = nat[: min(25, len(nat))]
    out = {}
    for key, lo, hi in (("rd", 0.0, 0.4), ("breath", 1.0, 2.0)):
        vals = []
        for v in (lo, hi):
            q = prof.copy()
            q.styles[style][key] = v
            syn = synthesize(q, sub, style, workdir)
            meas = [measure(syn[u.key[4]], q) for u in sub]
            k = "h12" if key == "rd" else "hnr"
            vals.append(med([r[k] for m in meas for r in m.values() if k in r]))
        out[key] = (vals[1] - vals[0]) / (hi - lo if key == "rd" else 1.0)
    return out


def score(nat, syn, nat_m, syn_m, label):
    P = pairs(nat, syn, nat_m, {t: syn_m[t] for t in syn_m})
    rows = []
    fe = [[], [], []]
    for u, s, n, y in P:
        if stressed(s) and "F" in n and "F" in y:
            for j in range(3):
                fe[j].append(abs(n["F"][j] - y["F"][j]) / n["F"][j])
    # Per-vowel-type medians compared (speaker's typical vowel vs ours).
    byv = defaultdict(lambda: ([], []))
    for u, s, n, y in P:
        if stressed(s) and "F" in n and "F" in y:
            byv[s["phone"]][0].append(n["F"])
            byv[s["phone"]][1].append(y["F"])
    ve = [[], [], []]
    for ph, (a, b) in byv.items():
        if len(a) >= 5:
            ma, mb = np.median(a, 0), np.median(b, 0)
            for j in range(3):
                ve[j].append(abs(ma[j] - mb[j]) / ma[j])
    nf = [f for _, _, n, _ in P for f in n.get("f0", [])]
    yf = [f for _, _, _, y in P for f in y.get("f0", [])]
    sd = lambda v: np.std(12 * np.log2(np.array(v) / np.median(v)))
    dl = [math.log(n["dur"] / y["dur"]) for _, s, n, y in P if n["dur"] > 0 and y["dur"] > 0]
    h12 = [n["h12"] - y["h12"] for _, s, n, y in P if "h12" in n and "h12" in y and stressed(s)]
    hnr = [n["hnr"] - y["hnr"] for _, s, n, y in P if "hnr" in n and "hnr" in y and stressed(s)]
    lev = [n["lev"] - y["lev"] for _, s, n, y in P if "lev" in n and "lev" in y]

    def ltas(x):
        n = 1024
        win = np.hanning(n)
        spec = np.zeros(n // 2 + 1)
        for k in range(0, len(x) - n, n // 2):
            spec += np.abs(np.fft.rfft(x[k : k + n] * win)) ** 2
        fr = np.fft.rfftfreq(n, 1 / SR)
        b = {}
        for c in acoustics.BANDS:
            sel = (fr >= c / np.sqrt(2)) & (fr < min(c * np.sqrt(2), SR / 2))
            b[c] = spec[sel].sum()
        return np.array([10 * np.log10(b[c] / b[500] + 1e-12) for c in acoustics.BANDS])

    ln = np.mean([ltas(u.x) for u in nat], 0)
    ly = np.mean([ltas(syn[u.key[4]].x) for u in nat], 0)
    print(f"== {label}  ({len(nat)} utterances, {len(P)} tokens)")
    print(f"  per-token formant error (stressed vowels): F1 {np.median(fe[0]) * 100:4.1f}%  F2 {np.median(fe[1]) * 100:4.1f}%  F3 {np.median(fe[2]) * 100:4.1f}%  (median |nat-syn|/nat)")
    print(f"  per-vowel median formant error:            F1 {np.mean(ve[0]) * 100:4.1f}%  F2 {np.mean(ve[1]) * 100:4.1f}%  F3 {np.mean(ve[2]) * 100:4.1f}%  (mean over {len(ve[0])} vowels)")
    print(f"  F0 median {np.median(nf):5.1f} vs {np.median(yf):5.1f} Hz;  spread {sd(nf):4.2f} vs {sd(yf):4.2f} st")
    print(f"  duration error (log, per phone) rms {np.sqrt(np.mean(np.square(dl))):4.2f}, median nat/syn {math.exp(np.median(dl)):4.2f}")
    print(f"  H1-H2 nat-syn {med(h12):+5.1f} dB;  HNR nat-syn {med(hnr):+5.1f} dB;  obstruent level nat-syn {med(lev):+5.1f} dB")
    print("  LTAS re 500 Hz (nat / syn): " + "  ".join(f"{c}:{a:+.0f}/{b:+.0f}" for c, a, b in zip(acoustics.BANDS, ln, ly)) + f"   mean |diff| {np.mean(np.abs(ln - ly)):4.1f} dB")


def locus_report(P):
    """Classic locus equations, F2onset = k F2mid + c, per place."""
    pts = defaultdict(list)
    for u, s, n, y in P:
        if s["phone"] in STOPS and "on" in n and "on_mid" in n:
            pts[PLACE[s["phone"]]].append((n["on_mid"][1], n["on"][1]))
    out = []
    for place, v in sorted(pts.items()):
        if len(v) < 8:
            continue
        a = np.array(v)
        k, c = np.polyfit(a[:, 0], a[:, 1], 1)
        locus = c / (1 - k) if k < 0.95 else float("nan")
        out.append(f"locus equation {place}: F2on = {k:.2f} F2mid + {c:.0f}  (locus {locus:.0f} Hz, n={len(v)})")
    return out


def cmd_fit(a):
    nat = natural_utts(a.name)
    if not nat:
        sys.exit(f"no aligned recordings for {a.name}: run tools/align.py voice {a.name} first")
    styles = sorted({u.key[0] for u in nat}, key=lambda s: (s != "neutral", s))
    print(f"{len(nat)} aligned recordings, styles: {', '.join(styles)}", file=sys.stderr)
    base = stock()
    prof = base.copy()
    prof.name = a.name
    st0 = prof.styles.pop("neutral")
    for s in styles:
        prof.styles[s] = dict(st0)
    work = tempfile.mkdtemp(prefix="fitvoice")
    try:
        slope = calibrate(prof, [u for u in nat if u.key[0] == styles[0]], styles[0], work)
        SLOPE["rd"] = slope["rd"] if abs(slope["rd"]) > 0.5 else 5.0
        SLOPE["breath"] = slope["breath"] if abs(slope["breath"]) > 0.3 else -3.0
        print(f"  engine: {SLOPE['rd']:+.1f} dB H1-H2 per Rd, {SLOPE['breath']:+.1f} dB HNR per doubling of breath", file=sys.stderr)
        nat_m = {}
        for it in range(a.iters + 1):
            P_all, style_P = [], {}
            for s in styles:
                us = [u for u in nat if u.key[0] == s]
                syn = synthesize(prof, us, s, work)
                syn_m = {t: measure(syn[t], prof) for t in syn}
                for u in us:
                    if id(u) not in nat_m or it == 1:
                        # Natural formants are re-measured once the
                        # references have moved toward the speaker.
                        nat_m[id(u)] = measure(u, prof)
                P = pairs(us, syn, nat_m, syn_m)
                P_all += P
                style_P[s] = P
            fe = [abs(n["F"][1] - y["F"][1]) / n["F"][1] for _, s, n, y in P_all if stressed(s) and "F" in n and "F" in y]
            print(f"  iteration {it}: stressed-vowel F2 error {np.median(fe) * 100:.1f}%", file=sys.stderr)
            if it == a.iters:
                break
            update(prof, base, P_all, style_P, it, a.bandwidths, a.loci)
        spreads(prof, P_all)
        if not a.full:
            # Intelligibility guard. Matched all the way, the speaker's
            # (quieter) consonant levels and breathiness cost the engine
            # 4-9 points of ASR word error rate on held-out sentences (NOTES,
            # 2026-11-19): natural speech has redundant cues this engine
            # lacks, and its louder consonants and cleaner voice make up for
            # them. Go half way (in log), and cap the breath noise.
            for e in prof.ph.values():
                e["amp"] = e["amp"] ** 0.5
            for st in prof.styles.values():
                st["breath"] = min(st["breath"] ** 0.5, 1.5)
        prof.comments = [f"{len(nat)} recordings, styles {', '.join(styles)}; {a.iters} iterations"
                         + ("" if a.full else "; consonant levels and breath half way to the measured values")
                         + ("; obstruent loci fitted" if a.loci else "")] + locus_report(P_all)
        out = a.out or os.path.join(ROOT, "data", "voice", a.name, "profile.txt")
        with open(out, "w") as f:
            f.write(prof.text())
        print(f"wrote {os.path.relpath(out, ROOT)}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def cmd_score(a):
    nat = natural_utts(a.name)
    if a.style:
        nat = [u for u in nat if u.key[0] == a.style]
    style = a.style or "neutral"
    if a.voice:
        prof = Profile.parse(open(a.voice).read())
        if style not in prof.styles:
            style = next(iter(prof.styles))
    else:
        prof = stock()
    # Both sides are measured with the same formant references, so scores of
    # different voices on the same recordings are comparable.
    ref_path = a.ref or os.path.join(ROOT, "data", "voice", a.name, "profile.txt")
    ref = Profile.parse(open(ref_path).read()) if os.path.exists(ref_path) else stock()
    work = tempfile.mkdtemp(prefix="fitvoice")
    try:
        nat = [u for u in nat if u.key[0] == (a.style or u.key[0])]
        syn = synthesize(prof, nat, style, work)
        nat_m = {id(u): measure(u, ref) for u in nat}
        syn_m = {t: measure(syn[t], ref) for t in syn}
        score(nat, syn, nat_m, syn_m, f"{a.name} vs {'stock voice' if not a.voice else a.voice}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("name")
    f.add_argument("--iters", type=int, default=4)
    f.add_argument("--no-bandwidths", dest="bandwidths", action="store_false")
    f.add_argument("--loci", action="store_true", help="also fit the loci of p b t d f v th dh s z sh zh ch jh (costs intelligibility)")
    f.add_argument("--full", action="store_true", help="match consonant levels and breath fully (costs intelligibility)")
    f.add_argument("--out")
    s = sub.add_parser("score")
    s.add_argument("name")
    s.add_argument("--voice")
    s.add_argument("--style")
    s.add_argument("--ref", help="profile whose targets guide formant picking (default: NAME's own)")
    a = ap.parse_args()
    {"fit": cmd_fit, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
