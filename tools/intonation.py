#!/usr/bin/env python3
"""Intonation measures: synthetic vs natural (CMU ARCTIC) F0 contours.

    zig build -Doptimize=ReleaseFast
    AUDIO=60 tools/fetch-arctic.sh
    nix develop .#eval -c tools/intonation.py [--set a|b|ab] [--n N]
          [--speakers bdl,rms,jmk] [--engine "-e refined" ...]

For each ARCTIC prompt with audio, our front end's plan (`ttslab segs`) gives
the words, accents and phrases. For a natural recording the planned phones
are aligned to the speaker's phone labels (as in fit-durations.py), so every
measure is taken at the same planned accents and phrases in both. F0 is
estimated from the audio the same way for both (autocorrelation, 10 ms),
in semitones re the speaker's median.

Per phrase (stretch between planned pauses):
  * range: 90th - 10th percentile of vowel F0
  * accent height: F0 peak in [vowel onset, vowel end + 80 ms] re the
    phrase's 10th percentile; peak delay from vowel onset
  * unaccented vowels re the phrase's 10th percentile
  * declination: slope of accent peaks over time, phrases with >= 2 accents
  * first - last accent peak
  * the end: F0 over the second half of the last vowel re the 10th pct,
    after a period and before a comma separately
  * |dF0| per 10 ms within vowels; F0 sd over the whole utterance
  * accent peak variability: sd of accent heights (after removing the
    linear declination), i.e. how unequal the accents are
"""
import argparse
import csv
import os
import subprocess
import sys
import wave

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data", "arctic")
BIN = os.environ.get("TTSLAB", os.path.join(ROOT, "zig-out", "bin", "ttslab"))
OUT = os.path.join(ROOT, "out", "intonation")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module  # noqa: E402

fd = import_module("fit-durations")
VOWELS = fd.VOWELS


def read_wav(path):
    with wave.open(path) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)
    return x / 32768.0, sr


def f0_track(x, sr, step=0.01, lo=55, hi=400):
    """Normalized autocorrelation pitch every 10 ms; nan where unvoiced."""
    h = int(0.02 * sr)
    n = len(x) // int(step * sr)
    out = np.full(n, np.nan)
    kl, kh = int(sr / hi), int(sr / lo)
    win = np.hanning(2 * h)
    rw = np.correlate(win, win, "full")[2 * h - 1 :]
    for i in range(n):
        c = int(i * step * sr)
        seg = x[c - h : c + h]
        if c - h < 0 or len(seg) < 2 * h:
            continue
        seg = (seg - seg.mean()) * win
        e = np.dot(seg, seg)
        if e < 1e-7:
            continue
        r = np.correlate(seg, seg, "full")[2 * h - 1 :] / e / (rw / rw[0])
        k = kl + int(np.argmax(r[kl:kh]))
        if r[k] < 0.5:
            continue
        # Prefer the shortest lag whose peak is nearly as good (avoid octave-down errors).
        for d in (2, 3):
            kk = int(round(k / d))
            if kk >= kl:
                j = max(kl, kk - 2) + int(np.argmax(r[max(kl, kk - 2) : kk + 3]))
                if r[j] > 0.85 * r[k]:
                    k = j
                    break
        # Parabolic interpolation.
        if kl < k < kh - 1:
            a, b, cc = r[k - 1], r[k], r[k + 1]
            den = a - 2 * b + cc
            k = k + (0.5 * (a - cc) / den if den != 0 else 0)
        out[i] = sr / k
    return out


def clean(f0):
    """Drop octave jumps against a running median, then a 3-point median."""
    v = f0.copy()
    med = np.nanmedian(v)
    v[(v < 0.6 * med) | (v > 2.0 * med)] = np.nan
    out = v.copy()
    for i in range(1, len(v) - 1):
        w = v[i - 1 : i + 2]
        if not np.isnan(v[i]) and np.sum(~np.isnan(w)) == 3:
            out[i] = np.median(w)
    return out


def ours(text, opts, tag, id_):
    os.makedirs(OUT, exist_ok=True)
    wav = os.path.join(OUT, f"{tag}_{id_}.wav")
    segs = subprocess.run([BIN, "segs", *opts, text], capture_output=True, text=True, check=True).stdout
    subprocess.run([BIN, "say", *opts, "--no-play", "-o", wav, text], capture_output=True, check=True)
    rows = list(csv.DictReader(segs.splitlines()))
    t = 0.0
    for r in rows:
        r["start"] = t
        t += float(r["dur_ms"]) / 1000
        r["end"] = t
    return rows, wav


def natural_times(segs, lab_path):
    """Planned segs with start/end from the speaker's labels (None if unmatched)."""
    nat, t0 = [], 0.0
    for line in open(lab_path):
        f = line.split()
        if len(f) < 3 or f[0] == "#":
            continue
        t = float(f[0])
        nat.append((fd.MAP.get(f[2], f[2]), t0, t))
        t0 = t
    pairs = fd.align([s["phone"] for s in segs], [p for p, _, _ in nat])
    out = [dict(s, start=None, end=None) for s in segs]
    for i, j in pairs:
        if fd.CLASSES.get(segs[i]["phone"]) == fd.CLASSES.get(nat[j][0]):
            out[i]["start"], out[i]["end"] = nat[j][1], nat[j][2]
    return out


def phrases(segs):
    out, cur = [], []
    for i, s in enumerate(segs):
        if s["phone"] == "pau":
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(i)
    if cur:
        out.append(cur)
    return out


def measure(segs, f0, ref_hz, final_kinds):
    """final_kinds: list of 'period'/'comma' per phrase."""
    st = 12 * np.log2(f0 / ref_hz)
    T = lambda t: int(round(t / 0.01))
    m = {k: [] for k in "range acc_h acc_delay acc_rel unacc decl first_last end_period end_comma dst acc_var vowel_st dst_mid dst_edge".split()}
    for pi, ph in enumerate(phrases(segs)):
        vs = [segs[i] for i in ph if segs[i]["phone"] in VOWELS and segs[i]["start"] is not None]
        if len(vs) < 2:
            continue
        vf = []
        for v in vs:
            seg = st[T(v["start"]) : T(v["end"]) + 1]
            vf += [x for x in seg if not np.isnan(x)]
            for q, (a, b) in enumerate(zip(seg, seg[1:])):
                if not (np.isnan(a) or np.isnan(b)):
                    m["dst"].append(abs(b - a))
                    rel = (q + 0.5) / max(1, len(seg) - 1)
                    m["dst_mid" if 0.25 <= rel <= 0.75 else "dst_edge"].append(abs(b - a))
        if len(vf) < 10:
            continue
        m["vowel_st"] += vf
        base = np.percentile(vf, 10)
        m["range"].append(np.percentile(vf, 90) - base)
        t_ph0 = vs[0]["start"]
        peaks = []
        for v in vs:
            if v["accent"] == "1":
                a, b = T(v["start"]), T(v["end"] + 0.08)
                seg = st[a : b + 1]
                if np.all(np.isnan(seg)):
                    continue
                k = int(np.nanargmax(seg))
                m["acc_h"].append(seg[k] - base)
                m["acc_delay"].append(1000 * ((a + k) * 0.01 - v["start"]))
                m["acc_rel"].append(((a + k) * 0.01 - v["start"]) / max(0.02, v["end"] - v["start"]))
                peaks.append(((a + k) * 0.01 - t_ph0, seg[k]))
            else:
                seg = st[T(v["start"]) : T(v["end"]) + 1]
                if np.any(~np.isnan(seg)):
                    m["unacc"].append(np.nanmean(seg) - base)
        if len(peaks) >= 2:
            t, p = np.array(peaks).T
            if t[-1] - t[0] > 0.3:
                slope, icpt = np.polyfit(t, p, 1)
                m["decl"].append(slope)
                m["acc_var"] += list(p - (slope * t + icpt))
            m["first_last"].append(p[0] - p[-1])
        lv = vs[-1]
        seg = st[T((lv["start"] + lv["end"]) / 2) : T(lv["end"]) + 1]
        if np.any(~np.isnan(seg)) and pi < len(final_kinds):
            m["end_" + final_kinds[pi]].append(np.nanmean(seg) - base)
    return m


def kinds_of(text, n):
    """Punctuation that ends each of the n phrases."""
    ks = []
    for c in text:
        if c in ",;:":
            ks.append("comma")
        elif c in ".!?":
            ks.append("period")
    ks = [k for i, k in enumerate(ks)]
    while len(ks) < n:
        ks.append("period")
    return ks[:n]


def summarize(name, ms):
    pool = {k: sum((m[k] for m in ms), []) for k in ms[0]}
    q = lambda k, f=np.median: f(pool[k]) if pool[k] else float("nan")
    row = {
        "phrase range p90-p10 (st)": q("range"),
        "accent height re p10 (st)": q("acc_h"),
        "accent height sd (st)": q("acc_h", np.std),
        "accent peak delay (ms)": q("acc_delay"),
        "accent peak pos (0=onset,1=end)": q("acc_rel"),
        "unaccented re p10 (st)": q("unacc"),
        "declination (st/s)": q("decl"),
        "accent residual sd (st)": q("acc_var", np.std),
        "first - last accent (st)": q("first_last"),
        "end of sentence re p10 (st)": q("end_period"),
        "before comma re p10 (st)": q("end_comma"),
        "|dF0| median (st/10ms)": q("dst"),
        "|dF0| p90 (st/10ms)": np.percentile(pool["dst"], 90) if pool["dst"] else float("nan"),
        "|dF0| p90, mid-vowel": np.percentile(pool["dst_mid"], 90) if pool["dst_mid"] else float("nan"),
        "|dF0| p90, vowel edges": np.percentile(pool["dst_edge"], 90) if pool["dst_edge"] else float("nan"),
        "F0 sd, vowels (st)": q("vowel_st", np.std),
    }
    return name, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="ab")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--speakers", default="bdl,rms,jmk")
    ap.add_argument("--engine", action="append", default=None, help='ttslab options, e.g. "-e refined"; repeatable')
    ap.add_argument("--no-natural", action="store_true")
    a = ap.parse_args()
    engines = a.engine or ["-e refined"]
    prompts = []
    for line in open(os.path.join(DATA, "prompts.tsv")):
        id_, text = line.rstrip("\n").split("\t", 1)
        if id_[7] in a.set and int(id_[8:]) <= a.n:
            prompts.append((id_, text))
    cols = []
    plans = {}
    for ei, e in enumerate(engines):
        opts = e.split()
        ms = []
        for id_, text in prompts:
            segs, wav = ours(text, opts, f"e{ei}", id_)
            plans.setdefault(id_, segs)
            x, sr = read_wav(wav)
            f0 = clean(f0_track(x, sr))
            ms.append((segs, f0, kinds_of(text, len(phrases(segs)))))
        ref = np.nanmedian(np.concatenate([f for _, f, _ in ms]))
        cols.append(summarize(e, [measure(s, f, ref, k) for s, f, k in ms]))
    if not a.no_natural:
        for spk in a.speakers.split(","):
            ms = []
            for id_, text in prompts:
                wp = os.path.join(DATA, spk, id_ + ".wav")
                if not os.path.exists(wp):
                    continue
                segs = natural_times(plans[id_], os.path.join(DATA, spk, id_ + ".lab"))
                x, sr = read_wav(wp)
                f0 = clean(f0_track(x, sr))
                ms.append((segs, f0, kinds_of(text, len(phrases(segs)))))
            if ms:
                ref = np.nanmedian(np.concatenate([f for _, f, _ in ms]))
                cols.append(summarize(f"{spk} (n={len(ms)})", [measure(s, f, ref, k) for s, f, k in ms]))
    names = list(cols[0][1])
    w = max(len(n) for n in names)
    print(" " * w + "".join(f" | {c[0][:16]:>16s}" for c in cols))
    for n in names:
        print(f"{n:{w}s}" + "".join(f" | {c[1][n]:16.2f}" for c in cols))


if __name__ == "__main__":
    main()
