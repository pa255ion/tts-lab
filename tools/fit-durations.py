#!/usr/bin/env python3
"""Fit the refined duration model to CMU ARCTIC phone timings.

    tools/fetch-arctic.sh                  # labels + prompts -> data/arctic/
    zig build -Doptimize=ReleaseFast
    nix develop .#eval -c tools/fit-durations.py [--speakers bdl,rms,jmk] [--write]

For every ARCTIC prompt, `ttslab segs` gives our front end's segments and
features (phone, stress, accent, function word, phrase-final). These are
aligned to the hand-checked automatic labels of each speaker, and a
multiplicative ("sums of products", van Santen 1994) model is fitted in the
log domain:

    log dur = log base[phone] + sum of context factors + speaker offset

The context factors are the ones Klatt's (1979) rules use, so the result is
still a rule system — only the numbers come from data. Trained on the
arctic_a set, tested on arctic_b. Prints the fit statistics; --write
regenerates src/duration_model.zig.
"""
import csv
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data", "arctic")
BIN = os.path.join(ROOT, "zig-out", "bin", "ttslab")

VOWELS = set("iy ih eh ae aa ah ao uh uw er ey ay oy aw ow".split())
VLESS_STOPS = set("p t k ch".split())
VOICED_OBS = set("b d g jh v dh z zh".split())
SONORANT_C = set("m n ng l r w y".split())
CLASSES = {
    **{p: "V" for p in VOWELS},
    **{p: "S" for p in "p t k b d g ch jh".split()},
    **{p: "F" for p in "f v th dh s z sh zh hh".split()},
    **{p: "N" for p in "m n ng l r w y".split()},
    "pau": "P",
}
# ARCTIC (festvox "radio" phone set) -> ours.
MAP = {"ax": "ah", "axr": "er", "ssil": "pau", "dx": "t"}

V_FEATS = [
    "unstressed", "secondary", "function_word", "accent", "phrase_final",
    "polysyllabic", "not_word_final", "next_vless_stop", "next_voiced_obs",
    "next_sonorant", "final_next_voiced",
]
C_FEATS = [
    "cluster", "phrase_final", "function_word", "word_initial", "word_final",
    "stressed_onset",
]


def ours(id_, text):
    path = os.path.join(DATA, "ours", id_ + ".csv")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = subprocess.run([BIN, "segs", "--timing", "klatt", text], capture_output=True, text=True, check=True).stdout
        with open(path, "w") as f:
            f.write(out)
    with open(path) as f:
        return list(csv.DictReader(f))


def lab(path):
    out, t0 = [], 0.0
    for line in open(path):
        f = line.split()
        if len(f) < 3 or f[0] == "#":
            continue
        t = float(f[0])
        out.append((MAP.get(f[2], f[2]), t - t0))
        t0 = t
    return out


def align(a, b):
    """Needleman-Wunsch over phone labels; returns index pairs (i, j)."""
    n, m = len(a), len(b)
    cost = np.zeros((n + 1, m + 1))
    cost[:, 0] = np.arange(n + 1) * 1.0
    cost[0, :] = np.arange(m + 1) * 1.0
    back = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = 0.0 if a[i - 1] == b[j - 1] else (0.6 if CLASSES.get(a[i - 1]) == CLASSES.get(b[j - 1]) else 2.0)
            opts = (cost[i - 1, j - 1] + s, cost[i - 1, j] + 1.0, cost[i, j - 1] + 1.0)
            k = int(np.argmin(opts))
            cost[i, j] = opts[k]
            back[i, j] = k
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        k = back[i, j]
        if k == 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def features(segs, i):
    s = segs[i]
    p = s["phone"]
    word = s["word"]
    same_word = [k for k in range(len(segs)) if segs[k]["word"] == word]
    nxt = segs[i + 1]["phone"] if i + 1 < len(segs) else "pau"
    prv = segs[i - 1]["phone"] if i > 0 else "pau"
    final = s["phrase_final"] == "1"
    if p in VOWELS:
        later_v = any(segs[k]["phone"] in VOWELS for k in same_word if k > i)
        nv = sum(1 for k in same_word if segs[k]["phone"] in VOWELS)
        return "V", {
            "unstressed": s["stress"] == "0",
            "secondary": s["stress"] == "2",
            "function_word": s["function_word"] == "1",
            "accent": s["accent"] == "1",
            "phrase_final": final,
            "polysyllabic": nv > 1,
            "not_word_final": later_v,
            "next_vless_stop": nxt in VLESS_STOPS,
            "next_voiced_obs": nxt in VOICED_OBS,
            "next_sonorant": nxt in SONORANT_C,
            "final_next_voiced": final and (nxt in VOICED_OBS or nxt in SONORANT_C),
        }
    is_c = lambda q: q != "pau" and q not in VOWELS
    return "C", {
        "cluster": is_c(prv) or is_c(nxt),
        "phrase_final": final,
        "function_word": s["function_word"] == "1",
        "word_initial": i == same_word[0],
        "word_final": i == same_word[-1],
        "stressed_onset": nxt in VOWELS and segs[i + 1]["stress"] == "1",
    }


def main():
    speakers = ["bdl", "rms", "jmk"]
    if "--speakers" in sys.argv:
        speakers = sys.argv[sys.argv.index("--speakers") + 1].split(",")
    prompts = []
    for line in open(os.path.join(DATA, "prompts.tsv")):
        id_, text = line.rstrip("\n").split("\t", 1)
        prompts.append((id_, text))

    rows = []  # (set, speaker, phone, kind, feats, klatt_ms, nat_ms, is_pause_after_comma)
    pauses = {"ours": [], "nat": []}
    for id_, text in prompts:
        segs = ours(id_, text)
        a = [s["phone"] for s in segs]
        for spk in speakers:
            lp = os.path.join(DATA, spk, id_ + ".lab")
            if not os.path.exists(lp):
                continue
            nat = lab(lp)
            b = [p for p, _ in nat]
            pairs = align(a, b)
            matched = dict(pairs)
            for i, j in pairs:
                p = a[i]
                if p == "pau" or b[j] != p:
                    continue
                kind, f = features(segs, i)
                rows.append((id_[7], spk, p, kind, f, float(segs[i]["dur_ms"]), 1000 * nat[j][1]))
            # Sentence-internal pauses we plan: what did the speaker do there?
            for i in range(1, len(a) - 1):
                if a[i] != "pau":
                    continue
                pauses["ours"].append(float(segs[i]["dur_ms"]))
                j = matched.get(i)
                pauses["nat"].append(1000 * nat[j][1] if j is not None and b[j] == "pau" else 0.0)

    phones = sorted({r[2] for r in rows})
    out = {}
    for kind, names in (("V", V_FEATS), ("C", C_FEATS)):
        sel = [r for r in rows if r[3] == kind]
        ph = sorted({r[2] for r in sel})
        cols = ph + names + speakers[1:]

        def design(rs):
            X = np.zeros((len(rs), len(cols)))
            for n, r in enumerate(rs):
                X[n, cols.index(r[2])] = 1
                for k in names:
                    X[n, cols.index(k)] = float(r[4][k])
                if r[1] in speakers[1:]:
                    X[n, cols.index(r[1])] = 1
            return X

        train = [r for r in sel if r[0] == "a"]
        test = [r for r in sel if r[0] == "b"]
        Xtr, ytr = design(train), np.log([r[6] for r in train])
        lam = 1.0
        w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(len(cols)), Xtr.T @ ytr)
        spk_off = np.mean([0.0] + [w[cols.index(s)] for s in speakers[1:]])
        Xte, yte = design(test), np.log([r[6] for r in test])
        pred = Xte @ w
        klatt = np.log([r[5] for r in test])
        rmse = lambda e: float(np.sqrt(np.mean(e * e)))
        # Klatt's predictions get their own best global scale (rate), so the
        # comparison is about the pattern, not about speaking rate.
        kl_shift = float(np.mean(yte - klatt))
        out[kind] = {
            "base": {p: float(np.exp(w[cols.index(p)] + spk_off)) for p in ph},
            "factors": {k: float(np.exp(w[cols.index(k)])) for k in names},
            "n_train": len(train),
            "n_test": len(test),
            "rmse_fit": rmse(yte - pred),
            "rmse_klatt": rmse(yte - klatt - kl_shift),
            "r_fit": float(np.corrcoef(yte, pred)[0, 1]),
            "r_klatt": float(np.corrcoef(yte, klatt)[0, 1]),
            "nat_mean": float(np.mean(np.exp(yte))),
            "klatt_mean": float(np.mean(np.exp(klatt))),
        }

    print(f"# speakers: {','.join(speakers)}; {len(prompts)} prompts; a-set trains, b-set tests\n")
    for kind, name in (("V", "vowels"), ("C", "consonants")):
        o = out[kind]
        print(f"# {name}: n={o['n_train']} train / {o['n_test']} test")
        print(f"#   log-duration RMSE  fitted {o['rmse_fit']:.3f}  vs Klatt rules (rate-matched) {o['rmse_klatt']:.3f}")
        print(f"#   correlation        fitted {o['r_fit']:.3f}  vs Klatt rules {o['r_klatt']:.3f}")
        print(f"#   mean duration      natural {o['nat_mean']:.0f} ms, Klatt rules {o['klatt_mean']:.0f} ms")
    op, np_ = np.array(pauses["ours"]), np.array(pauses["nat"])
    if len(op):
        print(f"# sentence-internal pauses we plan: {len(op)}; speakers paused at {np.mean(np_ > 20) * 100:.0f}% of them, "
              f"mean {np.mean(np_):.0f} ms (ours {np.mean(op):.0f} ms)")
    lines = [
        "//! Generated by tools/fit-durations.py from CMU ARCTIC phone labels",
        f"//! (speakers {', '.join(speakers)}; trained on the arctic_a prompts). Do not edit.",
        "//!",
        "//! dur = base[phone] * product of the factors that apply (see",
        "//! prosody.assignDurationsFitted). Held-out (arctic_b) fit:",
        f"//!   vowels      r = {out['V']['r_fit']:.2f} (Klatt 1979 rules: {out['V']['r_klatt']:.2f})",
        f"//!   consonants  r = {out['C']['r_fit']:.2f} (Klatt 1979 rules: {out['C']['r_klatt']:.2f})",
        "//! CMU ARCTIC: Copyright (c) 2003 Carnegie Mellon University, free for any",
        "//! use; see licenses/CMU-ARCTIC-LICENSE.",
        "",
        'const Phone = @import("phones.zig").Phone;',
        "",
        "pub const VowelFactors = struct {",
        *[f"    {k}: f32," for k in V_FEATS],
        "};",
        "",
        "pub const ConsonantFactors = struct {",
        *[f"    {k}: f32," for k in C_FEATS],
        "};",
        "",
        "/// Milliseconds; 0 = not in the data (use the Klatt value).",
        "pub const base_ms = blk: {",
        "    var t: [Phone.count]f32 = @splat(0);",
        *[f"    t[@intFromEnum(Phone.{p})] = {v:.1f};" for kind in ("V", "C") for p, v in sorted(out[kind]["base"].items())],
        "    break :blk t;",
        "};",
        "",
        "pub const vowel: VowelFactors = .{",
        *[f"    .{k} = {v:.3f}," for k, v in out["V"]["factors"].items()],
        "};",
        "",
        "pub const consonant: ConsonantFactors = .{",
        *[f"    .{k} = {v:.3f}," for k, v in out["C"]["factors"].items()],
        "};",
        "",
        f"/// Speakers paused at {np.mean(np_ > 20) * 100:.0f}% of the comma pauses we plan, {np.mean(np_):.0f} ms on average.",
        f"pub const comma_pause_ms: f32 = {max(np.mean(np_), 0):.0f};",
        "",
    ]
    if "--write" in sys.argv:
        with open(os.path.join(ROOT, "src", "duration_model.zig"), "w") as f:
            f.write("\n".join(lines))
        print("wrote src/duration_model.zig")
    else:
        print("\n".join(lines))

if __name__ == "__main__":
    main()
