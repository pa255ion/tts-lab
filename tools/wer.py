#!/usr/bin/env python3
"""Word error rate of ASR transcripts against reference sentences.

usage: wer.py REFERENCE_LIST TAG...   (reads out/eval/<TAG>_<nn>.txt)
Prints one line per tag: WER and the worst sentence.
"""
import random, re, sys

def words(s):
    s = s.lower().replace("-", " ")
    return re.sub(r"[^a-z' ]", " ", s).split()

def edit(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]

refs = [l.strip() for l in open(sys.argv[1]) if l.strip()]
def ci(per):
    """95% bootstrap interval of the WER, resampling sentences."""
    rng = random.Random(0)
    stats = []
    for _ in range(2000):
        smp = [per[rng.randrange(len(per))] for _ in per]
        stats.append(100 * sum(e for e, _ in smp) / max(1, sum(n for _, n in smp)))
    stats.sort()
    return stats[50], stats[1949]

for tag in sys.argv[2:]:
    errs = total = 0
    per = []
    worst = (0, "", "")
    for i, ref in enumerate(refs):
        try:
            hyp = open(f"out/eval/{tag}_{i:02d}.txt").read().strip()
        except FileNotFoundError:
            hyp = ""
        e = edit(words(ref), words(hyp))
        errs += e
        total += len(words(ref))
        per.append((e, len(words(ref))))
        if e > worst[0]:
            worst = (e, ref, hyp)
    lo, hi = ci(per)
    print(f"{tag:<16} WER {100*errs/total:5.1f}%  (95% CI {lo:4.1f}-{hi:4.1f})   worst: \"{worst[2]}\"")
