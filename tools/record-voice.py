#!/usr/bin/env python3
"""Record the voice script (voice/script.tsv), one WAV per prompt.

    tools/record-voice.py NAME [--sections QVHCAPSR] [--take N] [--style STYLE]
                               [--from ID] [--redo] [--plan]

Files go to data/voice/NAME/STYLE/takeN/ID.wav (16 kHz mono). Prompts that
already have a file are skipped, so you can stop at any time and carry on
later; --redo records them again.

Keys:  SPACE  start recording / stop recording
       ENTER  keep it and go to the next prompt
       r      record this prompt again
       p      play it back
       b      back to the previous prompt
       s      skip this prompt
       q      quit

--plan prints the recommended session. Uses pw-record/pw-play (PipeWire);
the default input device is used (pick it with your desktop's sound settings
or `wpctl set-default`).
"""
import argparse
import array
import math
import os
import select
import signal
import subprocess
import sys
import termios
import time
import tty
import wave

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SCRIPT = os.path.join(ROOT, "voice", "script.tsv")
ARCTIC = os.path.join(ROOT, "data", "arctic", "prompts.tsv")
N_ARCTIC = 100

PLAN = """\
Recommended session (about 45 minutes in all; take breaks whenever you like):

  1. tools/record-voice.py NAME --sections QVHCA --take 1     ~10 min
  2. tools/record-voice.py NAME --sections P --take 1         ~4 min
  3. tools/record-voice.py NAME --sections R                  ~12 min (100 ARCTIC sentences)
  4. tools/record-voice.py NAME --sections S --style neutral  ~2 min
     tools/record-voice.py NAME --sections S --style lively   ~2 min  (animated, enthusiastic)
     tools/record-voice.py NAME --sections S --style calm     ~2 min  (soft, relaxed, lower energy)
     ...any other style you like, by any name.
  5. tools/record-voice.py NAME --sections VHCA --take 2      ~8 min  (on another day, if you can)
     tools/record-voice.py NAME --sections VHCA --take 3
     tools/record-voice.py NAME --sections P --take 2

Steps 1-3 are the minimum that a voice can be fitted from; more takes give
the variation ranges (how much each sound differs from one time to the next).

Tips: same microphone, same distance (a hand's width or two, slightly off to
the side of your mouth so that p's and b's don't pop), a quiet room. Wait half
a beat after pressing SPACE before speaking, and after speaking before
pressing it again. Read at your normal pace and pitch: don't perform unless a
style asks for it. If you stumble, press r.
"""


def load_prompts(sections):
    out = []
    for line in open(SCRIPT, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        pid, sec, text = line.rstrip("\n").split("\t")
        if sec in sections:
            out.append((pid, sec, text))
    if "R" in sections:
        if not os.path.exists(ARCTIC):
            sys.exit("section R needs the ARCTIC prompts: run tools/fetch-arctic.sh first")
        for line in list(open(ARCTIC, encoding="utf-8"))[:N_ARCTIC]:
            pid, text = line.rstrip("\n").split("\t")
            out.append((pid, "R", text))
    return out


def shown(text):
    """{ah-PAH|aa0 p aa1} -> ah-PAH; *word* -> WORD (emphasis)."""
    out, i = [], 0
    while i < len(text):
        c = text[i]
        if c == "{":
            j = text.index("}", i)
            out.append(text[i + 1 : j].split("|")[0])
            i = j + 1
        elif c == "*":
            j = text.index("*", i + 1)
            out.append("\033[1;4m" + text[i + 1 : j].upper() + "\033[0m")
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def check(path):
    """-> list of warnings about a recording."""
    with wave.open(path) as w:
        sr, n = w.getframerate(), w.getnframes()
        x = array.array("h", w.readframes(n))
    if n < sr * 0.3:
        return ["very short: did it record?"]
    warn = []
    peak = max(abs(v) for v in x) / 32768
    if peak > 0.97:
        warn.append(f"clipped (peak {peak:.2f}): move back a little or turn the input gain down")
    hop = sr // 100
    lev = []
    for k in range(0, len(x) - hop, hop):
        seg = x[k : k + hop]
        lev.append(10 * math.log10(sum(v * v for v in seg) / hop / 32768**2 + 1e-12))
    floor = sorted(lev)[len(lev) // 10]
    loud = sorted(lev)[-max(1, len(lev) // 10)]
    if loud < -40:
        warn.append(f"quiet (speech around {loud:.0f} dBFS): move closer or turn the gain up")
    if loud - floor < 25:
        warn.append(f"noisy: speech only {loud - floor:.0f} dB above the background")
    if loud - floor < 25:
        return warn
    if lev[0] > floor + 20 or max(lev[:10]) > loud - 10:
        warn.append("speech right at the start: wait a beat after SPACE")
    if lev[-1] > floor + 20 or max(lev[-10:]) > loud - 10:
        warn.append("speech right at the end: wait a beat before stopping")
    return warn


def getkey():
    return sys.stdin.read(1)


def record(path):
    tmp = path + ".part"
    p = subprocess.Popen(["pw-record", "--rate", "16000", "--channels", "1", "--format", "s16", tmp],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.15)
    t0 = time.time()
    sys.stdout.write("\r  \033[1;31m● recording\033[0m (SPACE to stop) ")
    sys.stdout.flush()
    while True:
        r, _, _ = select.select([sys.stdin], [], [], 0.1)
        if r and getkey() == " ":
            break
        sys.stdout.write(f"\r  \033[1;31m● recording\033[0m {time.time() - t0:4.1f} s (SPACE to stop) ")
        sys.stdout.flush()
    time.sleep(0.1)
    p.send_signal(signal.SIGINT)
    p.wait()
    os.replace(tmp, path)
    sys.stdout.write("\r\033[K")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?")
    ap.add_argument("--sections", default="QVHCAPR")
    ap.add_argument("--take", type=int, default=1)
    ap.add_argument("--style", default="neutral")
    ap.add_argument("--from", dest="start")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    if a.plan or not a.name:
        print(PLAN)
        return
    prompts = load_prompts(set(a.sections))
    if a.start:
        ids = [p[0] for p in prompts]
        prompts = prompts[ids.index(a.start) :]
    d = os.path.join(ROOT, "data", "voice", a.name, a.style, f"take{a.take}")
    os.makedirs(d, exist_ok=True)
    todo = [p for p in prompts if a.redo or not os.path.exists(os.path.join(d, p[0] + ".wav"))]
    print(f"{len(todo)} of {len(prompts)} prompts to record into {os.path.relpath(d, ROOT)}  (style: {a.style})")
    if a.style != "neutral":
        print(f"Read every sentence in a \033[1m{a.style}\033[0m way.")
    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        i = 0
        redone = set()
        while 0 <= i < len(todo):
            pid, sec, text = todo[i]
            path = os.path.join(d, pid + ".wav")
            print(f"\n[{i + 1}/{len(todo)}] {pid}\n\n    \033[1m{shown(text)}\033[0m\n")
            if a.redo and pid not in redone or not os.path.exists(path):
                redone.add(pid)
                sys.stdout.write("  SPACE to start ")
                sys.stdout.flush()
                while (k := getkey()) not in " qsb":
                    pass
                sys.stdout.write("\r\033[K")
                if k == "q":
                    return
                if k == "s":
                    i += 1
                    continue
                if k == "b":
                    i = max(0, i - 1)
                    continue
                record(path)
            while True:
                w = check(path)
                if pid == "Q01":
                    w = [x for x in w if not x.startswith(("quiet", "noisy", "speech right"))]
                print("  " + ("\033[33m" + "\n  ".join("! " + x for x in w) + "\033[0m" if w else "\033[32mok\033[0m"))
                sys.stdout.write("  ENTER next   r redo   p play   b back   q quit ")
                sys.stdout.flush()
                while (k := getkey()) not in "\nrpbq ":
                    pass
                sys.stdout.write("\r\033[K")
                if k == "p":
                    subprocess.run(["pw-play", path])
                    continue
                if k == "r":
                    record(path)
                    continue
                break
            if k == "q":
                return
            if k == "b":
                i = max(0, i - 1)
                continue
            i += 1
        print("\nDone with this part. `tools/record-voice.py --plan` shows the rest.")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
