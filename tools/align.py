#!/usr/bin/env python3
"""Forced alignment: find where each phone of a known text is in a recording.

    tools/align.py train  [--speakers bdl,rms,jmk] [--n 200]
    tools/align.py accuracy   [--speakers bdl,rms,jmk,slt,clb] [--from 201 --to 300]
    tools/align.py voice  NAME [--iters 3]

A small monophone HMM aligner, in the HTK tradition: 39 MFCC features
(13 cepstra + deltas + delta-deltas, mean/variance normalized per
utterance) every 5 ms, three left-to-right states per phone with diagonal
Gaussian mixtures, and an optional short pause between words. Viterbi only.

`train` builds the speaker-independent models from CMU ARCTIC recordings and
their phone labels (tools/fetch-arctic.sh, AUDIO=300) into
data/models/aligner.npz. `accuracy` measures boundary accuracy against the ARCTIC
labels on sentences (and speakers) not used for training. `voice` aligns a
speaker's recordings of voice/script.tsv (data/voice/NAME/STYLE/takeN/ID.wav),
first with the speaker-independent models, then adapts the means to the
speaker (MAP) and realigns, `--iters` times. The labels are written as
data/voice/NAME/align/STYLE/takeN/ID.lab, in the ARCTIC/Festival format that
tools/acoustics.py and tools/fit-voice.py read.

Phone sequences come from `ttslab segs` (the same lexicon and letter-to-sound
rules the synthesizer uses), so the alignment is of what we would have said;
the optional pauses absorb the speaker's own phrasing.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import wave

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ARCTIC = os.path.join(ROOT, "data", "arctic")
MODEL = os.path.join(ROOT, "data", "models", "aligner.npz")
BIN = os.path.join(ROOT, "zig-out", "bin", "ttslab")
SCRIPT = os.path.join(ROOT, "voice", "script.tsv")

SR = 16000
HOP = 80  # 5 ms
WIN = 400  # 25 ms
NSTATE = 3
NMIX = 4
MAP = {"ax": "ah", "axr": "er", "ssil": "pau", "dx": "t", "H#": "pau", "h#": "pau"}
PHONES = ("pau aa ae ah ao aw ay b ch d dh eh er ey f g hh ih iy jh k l m n ng ow oy p r s sh t th uh uw v w y z zh").split()
PIDX = {p: i for i, p in enumerate(PHONES)}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def read_wav(path):
    with wave.open(path) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
    if sw != 2:
        raise SystemExit(f"{path}: need 16-bit PCM")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        # Linear-interpolation resampling is enough for MFCCs (and for LPC
        # below 5 kHz); proper filtering would need scipy.
        n = int(len(x) * SR / sr)
        if sr > SR:
            k = int(round(sr / SR))
            x = np.convolve(x, np.ones(k) / k, "same")
        x = np.interp(np.arange(n) * sr / SR, np.arange(len(x)), x)
    return x


def _mel_bank(nfft=512, nmel=26, lo=60, hi=7600):
    def mel(f):
        return 2595 * np.log10(1 + f / 700)

    def imel(m):
        return 700 * (10 ** (m / 2595) - 1)

    pts = imel(np.linspace(mel(lo), mel(hi), nmel + 2))
    bins = np.floor((nfft + 1) * pts / SR).astype(int)
    fb = np.zeros((nmel, nfft // 2 + 1))
    for m in range(1, nmel + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, a:b] = (np.arange(a, b) - a) / max(1, b - a)
        fb[m - 1, b:c] = (c - np.arange(b, c)) / max(1, c - b)
    return fb


FB = _mel_bank()
DCT = np.cos(np.pi / 26 * (np.arange(26)[None, :] + 0.5) * np.arange(13)[:, None])


def frames_of(x):
    n = 1 + max(0, (len(x) - WIN) // HOP)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    xp = np.append(x[0], x[1:] - 0.97 * x[:-1])
    return xp[idx] * np.hamming(WIN)


def deltas(c, k=2):
    pad = np.pad(c, ((k, k), (0, 0)), mode="edge")
    num = sum(i * (pad[k + i : len(c) + k + i] - pad[k - i : len(c) + k - i]) for i in range(1, k + 1))
    return num / (2 * sum(i * i for i in range(1, k + 1)))


def mfcc(x):
    fr = frames_of(x)
    p = np.abs(np.fft.rfft(fr, 512)) ** 2
    c = np.log(p @ FB.T + 1e-10) @ DCT.T
    d = deltas(c)
    f = np.hstack([c, d, deltas(d)])
    return (f - f.mean(0)) / (f.std(0) + 1e-8)


def t2f(t):
    """Time (s) -> frame index (frame i covers [i*HOP, i*HOP+WIN))."""
    return int(round((t * SR - WIN / 2) / HOP))


def f2t(i):
    return (i * HOP + WIN / 2) / SR


# ---------------------------------------------------------------------------
# Labels and phone sequences
# ---------------------------------------------------------------------------

def read_lab(path):
    out, t0 = [], 0.0
    for line in open(path):
        f = line.split()
        if len(f) < 3 or f[0] == "#":
            continue
        t = float(f[0])
        p = MAP.get(f[2], f[2])
        if p in PIDX:
            if out and out[-1][0] == "pau" and p == "pau":
                out[-1][2] = t
            else:
                out.append([p, t0, t])
        t0 = t
    return out


def write_lab(path, segs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("#\n")
        for p, a, b in segs:
            f.write(f"{b:.4f} 125 {p}\n")


_seg_cache = {}


def our_segs(text):
    """-> [(phone, word index or -1)] from ttslab's front end."""
    if text not in _seg_cache:
        out = subprocess.run([BIN, "segs", text], capture_output=True, text=True, check=True).stdout
        _seg_cache[text] = [(r["phone"], int(r["word"])) for r in csv.DictReader(out.splitlines())]
    return _seg_cache[text]


def spoken_text(pid, text):
    """The part of a script line that is actually said."""
    text = re.sub(r"\([^)]*\)", "", text).replace("*", "")
    if pid.startswith("V"):
        m = re.search(r"\{[^}]*\}", text)
        return m.group(0) if m else text
    return text


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Models:
    """NSTATE states per phone, NMIX-component diagonal GMMs."""

    def __init__(self, mu, var, w):
        self.mu, self.var, self.w = mu, var, w  # [P*S, M, D], [P*S, M, D], [P*S, M]
        self._prep()

    def _prep(self):
        self.ivar = 1 / self.var
        self.const = np.log(self.w + 1e-12) - 0.5 * np.sum(np.log(2 * np.pi * self.var), axis=2)

    def loglik(self, x, states):
        """x [T, D] -> [T, len(states)] log-likelihoods."""
        mu, iv, cst = self.mu[states], self.ivar[states], self.const[states]
        # (x - mu)^2 / var summed over D, for all T, S, M.
        a = np.einsum("td,smd->tsm", x * x, iv)
        b = np.einsum("td,smd->tsm", x, mu * iv)
        c = np.sum(mu * mu * iv, axis=2)
        ll = cst[None] - 0.5 * (a - 2 * b + c[None])
        m = ll.max(axis=2)
        return m + np.log(np.exp(ll - m[..., None]).sum(axis=2))

    def save(self, path):
        np.savez_compressed(path, mu=self.mu, var=self.var, w=self.w)

    @staticmethod
    def load(path):
        z = np.load(path)
        return Models(z["mu"], z["var"], z["w"])


def fit_gmm(x, nmix, iters=8, floor=None):
    """EM for a diagonal GMM, initialized by splitting."""
    d = x.shape[1]
    if len(x) > 20000:
        x = x[np.random.default_rng(0).choice(len(x), 20000, replace=False)]
    if floor is None:
        floor = 0.01
    mu = x.mean(0, keepdims=True)
    var = np.maximum(x.var(0, keepdims=True), floor)
    w = np.ones(1)
    while len(w) < nmix and len(x) >= 20 * (2 * len(w)):
        e = 0.2 * np.sqrt(var)
        mu = np.vstack([mu - e, mu + e])
        var = np.vstack([var, var])
        w = np.concatenate([w, w]) / 2
        for _ in range(iters):
            ll = (np.log(w)[None] - 0.5 * np.sum(np.log(2 * np.pi * var), 1)[None]
                  - 0.5 * (((x[:, None, :] - mu[None]) ** 2) / var[None]).sum(2))
            ll -= ll.max(1, keepdims=True)
            g = np.exp(ll)
            g /= g.sum(1, keepdims=True)
            n = g.sum(0) + 1e-6
            mu = (g.T @ x) / n[:, None]
            var = np.maximum((g.T @ (x * x)) / n[:, None] - mu * mu, floor)
            w = n / n.sum()
    out_mu = np.zeros((nmix, d))
    out_var = np.ones((nmix, d))
    out_w = np.full(nmix, 1e-8)
    out_mu[: len(w)], out_var[: len(w)], out_w[: len(w)] = mu, var, w
    return out_mu, out_var, out_w


def estimate(state_frames, nmix=NMIX, prior=None):
    """state_frames: list over all P*S states of [n, D] arrays."""
    S = len(PHONES) * NSTATE
    D = 39
    mu = np.zeros((S, nmix, D))
    var = np.ones((S, nmix, D))
    w = np.full((S, nmix), 1.0 / nmix)
    for s in range(S):
        x = state_frames[s]
        if x is None or len(x) < 10:
            if prior is not None:
                mu[s], var[s], w[s] = prior.mu[s], prior.var[s], prior.w[s]
            continue
        mu[s], var[s], w[s] = fit_gmm(x, nmix)
    return Models(mu, var, w)


# ---------------------------------------------------------------------------
# Viterbi over a left-to-right state graph with optional (skippable) models
# ---------------------------------------------------------------------------

def build_graph(phones, optional):
    """phones: list of phone names; optional[i]: may be skipped.
    -> (state ids, phone index of each graph state, skip-entry list)."""
    states, owner = [], []
    for i, p in enumerate(phones):
        base = PIDX[p] * NSTATE
        n = 1 if (optional[i] and p == "pau") else NSTATE
        # A short optional pause is a single state: the middle pause state.
        ids = [base + 1] if n == 1 else [base + k for k in range(NSTATE)]
        states += ids
        owner += [i] * len(ids)
    return np.array(states), np.array(owner)


def viterbi(ll, owner, optional):
    """ll [T, N] log-likelihoods for the graph's N states. Each state may
    stay, advance to the next state, or (from the last state of a phone)
    jump over following optional phones. Must start in the first
    non-optional-skippable prefix and end in the last phones."""
    T, N = ll.shape
    nph = owner[-1] + 1
    first = np.zeros(nph, int)
    last = np.zeros(nph, int)
    for j in range(N - 1, -1, -1):
        first[owner[j]] = j
    for j in range(N):
        last[owner[j]] = j
    # Entry states of phone i reachable from the end of phone i-1 or, skipping
    # optional phones, earlier ones. pred2[j] = extra predecessor (or -1).
    extra = [[] for _ in range(N)]
    for i in range(1, nph):
        k = i - 1
        while k >= 1 and optional[k]:
            extra[first[i]].append(last[k - 1])
            k -= 1
    maxe = max(1, max(len(e) for e in extra))
    pred = np.full((N, maxe), -1)
    for j, e in enumerate(extra):
        pred[j, : len(e)] = e
    starts = [first[0]]
    k = 0
    while k < nph - 1 and optional[k]:
        starts.append(first[k + 1])
        k += 1
    ends = [last[nph - 1]]
    k = nph - 1
    while k > 0 and optional[k]:
        ends.append(last[k - 1])
        k -= 1
    NEG = -1e30
    delta = np.full(N, NEG)
    delta[starts] = ll[0, starts]
    bp = np.zeros((T, N), np.int32)
    ar = np.arange(N)
    has_extra = pred[:, 0] >= 0
    for t in range(1, T):
        stay = delta
        adv = np.concatenate([[NEG], delta[:-1]])
        best = np.where(adv > stay, ar - 1, ar)
        val = np.maximum(adv, stay)
        if has_extra.any():
            pe = np.where(pred >= 0, delta[np.maximum(pred, 0)], NEG)
            k = pe.argmax(1)
            v2 = pe[ar, k]
            better = v2 > val
            best = np.where(better, pred[ar, k], best)
            val = np.where(better, v2, val)
        delta = val + ll[t]
        bp[t] = best
    j = max(ends, key=lambda e: delta[e])
    path = np.zeros(T, int)
    for t in range(T - 1, -1, -1):
        path[t] = j
        j = bp[t, j]
    return path


def align_utt(models, feats, phones, optional):
    states, owner = build_graph(phones, optional)
    T = len(feats)
    if T < len(states):
        return None
    ll = models.loglik(feats, states)
    path = viterbi(ll, owner, optional)
    segs = []
    for t, j in enumerate(path):
        i = owner[j]
        if segs and segs[-1][0] == i:
            segs[-1][2] = t
        else:
            segs.append([i, t, t])
    out = []
    for i, a, b in segs:
        out.append([phones[i], f2t(a) - HOP / SR / 2 if a > 0 else 0.0, f2t(b) + HOP / SR / 2])
    out[-1][2] = max(out[-1][2], f2t(T - 1))
    return out, path, states


def sequence_from_segs(segs):
    """ttslab segs -> phones with an optional pause between words."""
    phones, optional = ["pau"], [True]
    last_w = None
    for p, w in segs:
        if p == "pau":
            if phones[-1] != "pau":
                phones.append("pau")
                optional.append(True)
            last_w = None
            continue
        if last_w is not None and w != last_w and phones[-1] != "pau":
            phones.append("pau")
            optional.append(True)
        phones.append(p)
        optional.append(False)
        last_w = w
    if phones[-1] != "pau":
        phones.append("pau")
        optional.append(True)
    return phones, optional


def sequence_from_lab(lab):
    return [p for p, _, _ in lab], [False] * len(lab)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def arctic_ids(lo, hi):
    return [f"arctic_a{i:04d}" for i in range(lo, hi + 1)]


def prompts():
    return dict(l.rstrip("\n").split("\t") for l in open(os.path.join(ARCTIC, "prompts.tsv")))


def collect(models, utts, seqfn, out_frames=None):
    """Align utts [(feats, phones, optional)] and gather frames per state."""
    S = len(PHONES) * NSTATE
    acc = [[] for _ in range(S)]
    results = []
    for feats, phones, optional in utts:
        r = align_utt(models, feats, phones, optional)
        results.append(r)
        if r is None:
            continue
        _, path, states = r
        st = states[path]
        for s in np.unique(st):
            acc[s].append(feats[st == s])
    return [np.vstack(a) if a else None for a in acc], results


def cmd_train(a):
    spk = a.speakers.split(",")
    utts, init = [], [[] for _ in range(len(PHONES) * NSTATE)]
    for s in spk:
        for pid in arctic_ids(1, a.n):
            wav = os.path.join(ARCTIC, s, pid + ".wav")
            lab = os.path.join(ARCTIC, s, pid + ".lab")
            if not (os.path.exists(wav) and os.path.exists(lab)):
                continue
            f = mfcc(read_wav(wav))
            L = read_lab(lab)
            # Flat start from the labels: each phone split in three.
            for p, t0, t1 in L:
                i0, i1 = max(0, t2f(t0)), min(len(f), t2f(t1))
                if i1 - i0 < NSTATE:
                    continue
                cuts = np.linspace(i0, i1, NSTATE + 1).astype(int)
                for k in range(NSTATE):
                    init[PIDX[p] * NSTATE + k].append(f[cuts[k] : cuts[k + 1]])
            utts.append((f, *sequence_from_lab(L)))
    print(f"{len(utts)} utterances from {','.join(spk)}", file=sys.stderr)
    m = estimate([np.vstack(x) if x else None for x in init])
    for it in range(a.iters):
        frames, res = collect(m, utts, None)
        m = estimate(frames, prior=m)
        print(f"  re-estimation {it + 1}: {sum(r is not None for r in res)} aligned", file=sys.stderr)
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    m.save(MODEL)
    print(f"wrote {os.path.relpath(MODEL, ROOT)}")


def boundary_errors(hyp, ref):
    """Match phones by a string alignment, compare start times of matched
    phones whose predecessor also matched (i.e. true shared boundaries)."""
    a = [p for p, _, _ in hyp]
    b = [p for p, _, _ in ref]
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else 1.5))
    i, j, pairs = n, m, []
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1] and D[i, j] == D[i - 1, j - 1]:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        elif D[i, j] == D[i, j - 1] + 1:
            j -= 1
        else:
            i, j = i - 1, j - 1
    pairs = set(pairs)
    err = []
    for (i, j) in pairs:
        if i > 0 and (i - 1, j - 1) in pairs and a[i] != "pau" and a[i - 1] != "pau":
            err.append(abs(hyp[i][1] - ref[j][1]))
    return err, len(pairs) / max(1, len(b))


def cmd_eval(a):
    m = Models.load(MODEL)
    pr = prompts()
    for s in a.speakers.split(","):
        utts, refs, utts_lex = [], [], []
        for pid in arctic_ids(a.lo, a.hi):
            wav = os.path.join(ARCTIC, s, pid + ".wav")
            lab = os.path.join(ARCTIC, s, pid + ".lab")
            if not (os.path.exists(wav) and os.path.exists(lab)):
                continue
            f = mfcc(read_wav(wav))
            L = read_lab(lab)
            refs.append(L)
            utts.append((f, *sequence_from_lab(L)))
            utts_lex.append((f, *sequence_from_segs(our_segs(pr[pid]))))
        if not utts:
            continue
        rows = []
        for name, U in (("label phones", utts), ("lexicon phones", utts_lex)):
            mm = m
            for it in range(a.iters + 1):
                frames, res = collect(mm, U, None)
                errs, cover = [], []
                for r, L in zip(res, refs):
                    if r is None:
                        continue
                    e, c = boundary_errors(r[0], L)
                    errs += e
                    cover.append(c)
                e = np.array(errs) * 1000
                rows.append(f"  {name:15s} adapt {it}: median {np.median(e):4.1f} ms, within 20 ms {np.mean(e <= 20) * 100:4.1f}%, within 40 ms {np.mean(e <= 40) * 100:4.1f}%  (n={len(e)}, phones matched {np.mean(cover) * 100:.0f}%)")
                if it < a.iters:
                    mm = adapt(m, frames)
        print(f"== {s} {pid[:8]} {a.lo}-{a.hi} ({len(utts)} utterances)")
        print("\n".join(rows))


def adapt(prior, frames, tau=10.0):
    """MAP adaptation of the means toward the speaker's own frames."""
    mu = prior.mu.copy()
    for s, x in enumerate(frames):
        if x is None or len(x) == 0:
            continue
        # Responsibilities of the prior's mixture components.
        ll = (np.log(prior.w[s] + 1e-12)[None] - 0.5 * np.sum(np.log(2 * np.pi * prior.var[s]), 1)[None]
              - 0.5 * (((x[:, None, :] - prior.mu[s][None]) ** 2) / prior.var[s][None]).sum(2))
        ll -= ll.max(1, keepdims=True)
        g = np.exp(ll)
        g /= g.sum(1, keepdims=True)
        n = g.sum(0)
        mu[s] = (tau * prior.mu[s] + g.T @ x) / (tau + n)[:, None]
    return Models(mu, prior.var, prior.w)


def script_prompts():
    out = {}
    for line in open(SCRIPT, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        pid, sec, text = line.rstrip("\n").split("\t")
        out[pid] = (sec, text)
    if os.path.exists(os.path.join(ARCTIC, "prompts.tsv")):
        for pid, text in prompts().items():
            out[pid] = ("R", text)
    return out


def voice_files(name):
    """-> [(style, take, id, wav path)] for a recorded voice."""
    base = os.path.join(ROOT, "data", "voice", name)
    out = []
    for style in sorted(os.listdir(base)):
        sd = os.path.join(base, style)
        if style == "align" or not os.path.isdir(sd):
            continue
        for take in sorted(os.listdir(sd)):
            td = os.path.join(sd, take)
            if not os.path.isdir(td):
                continue
            for fn in sorted(os.listdir(td)):
                if fn.endswith(".wav"):
                    out.append((style, take, fn[:-4], os.path.join(td, fn)))
    return out


def cmd_voice(a):
    m0 = Models.load(MODEL)
    sp = script_prompts()
    utts, keys = [], []
    for style, take, pid, wav in voice_files(a.name):
        if pid not in sp or pid == "Q01":
            continue
        f = mfcc(read_wav(wav))
        utts.append((f, *sequence_from_segs(our_segs(spoken_text(pid, sp[pid][1])))))
        keys.append((style, take, pid))
    print(f"{len(utts)} recordings of {a.name}", file=sys.stderr)
    m = m0
    for it in range(a.iters + 1):
        frames, res = collect(m, utts, None)
        fails = sum(r is None for r in res)
        print(f"  pass {it}: {len(res) - fails} aligned{f', {fails} too short' if fails else ''}", file=sys.stderr)
        if it < a.iters:
            m = adapt(m0, frames)
    for (style, take, pid), r in zip(keys, res):
        if r is not None:
            write_lab(os.path.join(ROOT, "data", "voice", a.name, "align", style, take, pid + ".lab"), r[0])
    print(f"wrote data/voice/{a.name}/align/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--speakers", default="bdl,rms,jmk")
    t.add_argument("--n", type=int, default=200)
    t.add_argument("--iters", type=int, default=2)
    e = sub.add_parser("accuracy")
    e.add_argument("--speakers", default="bdl,rms,jmk,slt,clb")
    e.add_argument("--from", dest="lo", type=int, default=201)
    e.add_argument("--to", dest="hi", type=int, default=300)
    e.add_argument("--iters", type=int, default=2)
    v = sub.add_parser("voice")
    v.add_argument("name")
    v.add_argument("--iters", type=int, default=3)
    a = ap.parse_args()
    {"train": cmd_train, "accuracy": cmd_eval, "voice": cmd_voice}[a.cmd](a)


if __name__ == "__main__":
    main()
