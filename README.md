# TTS Lab

A toy text-to-speech lab that implements the five classic pre-neural synthesis
methods behind one front end, so you can switch between them and hear (and
measure) the differences:

1. **Formant, classic rules**: impulse source, straight-line formant transitions (MITalk / early Klatt, ~1980)
2. **Formant, refined**: LF glottal source, coarticulation model, jitter, breathiness, formant-tied bursts (Klatt 88 / DECtalk-style refinements)
3. **Diphone + TD-PSOLA** (MBROLA-style)
4. **Unit selection** with Viterbi over target and join costs (Festival multisyn / AT&T style)
5. **Statistical parametric (HMM)**: state Gaussians, MLPG, global variance, cepstral vocoder (HTS-style)

The backend is written in Zig 0.16 and has no dependencies. There is a CLI with
an interactive menu, a C ABI shared library, and a Qt 6 GUI.

See **[NOTES.md](NOTES.md)** for how everything works, the cost profile, the
intelligibility results, and how far formant synthesis can be pushed.

## Quick start

```sh
nix develop                          # zig, ffmpeg, pipewire
tools/fetch-cmudict.sh               # optional, 125k-word dictionary (otherwise ~350 words + letter-to-sound rules)
zig build -Doptimize=ReleaseFast
./zig-out/bin/ttslab                 # interactive menu
```

```sh
ttslab say -e refined "Is this a question?"   # plays when run in a terminal (--no-play to skip)
ttslab compare "The quick brown fox."        # every engine -> out/<engine>.wav
ttslab phonemes "Dr. Smith paid $3.50."      # front-end transcription
ttslab frames -e classic "Turn left."        # formant parameter tracks (CSV)
ttslab bench                                 # cost profile
```

Engines: `classic refined diphone unitsel hmm` (or `1`–`5`). Options:
`--f0 flat|hat|fujisaki --timing klatt|fitted --intonation rules|fitted --pitch X --rate X --range X --formant X --no-psola --modify --no-mlpg --gv`.
`ttslab segs TEXT` prints the planned segments with their prosodic features.
Refined only: `--voice PROFILE --style NAME --variation X --seed N` (see below).

## GUI

```sh
zig build -Doptimize=ReleaseFast
nix develop .#gui -c sh -c 'cmake -S gui -B gui/build -G Ninja && cmake --build gui/build'
gui/build/ttslab-gui
```

## Tests and evaluation

```sh
zig build test
nix develop .#eval -c tools/eval-asr.sh      # whisper.cpp word error rate per engine (see the script for the model download)
tools/spectrogram.sh out/*.wav               # PNG spectrograms (inside nix develop)
AUDIO=1 tools/fetch-arctic.sh                # CMU ARCTIC labels (+20 recordings each) for the next two
nix develop .#eval -c tools/acoustics.py -s out.wav labels.csv ...   # compare with natural speech
nix develop .#eval -c tools/fit-durations.py --write                 # refit src/duration_model.zig
nix develop .#eval -c tools/intonation.py --set b                    # F0 contours vs ARCTIC (AUDIO=60 tools/fetch-arctic.sh)
```

## Fit a voice to your own recordings

The refined formant engine can take a **voice profile**: its vowel and
consonant targets, bandwidths, durations, pitch, pitch range, tempo and voice
quality, fitted to one speaker. Without one it is the stock voice, unchanged.

```sh
zig build -Doptimize=ReleaseFast
AUDIO=300 SPEAKERS="bdl rms jmk" tools/fetch-arctic.sh    # ARCTIC prompts + training audio for the aligner
nix develop .#eval -c tools/align.py train               # once: speaker-independent aligner -> data/models/aligner.npz

tools/record-voice.py --plan                             # the recommended session (~45 min)
tools/record-voice.py NAME --sections QVHCA              # ...and the rest of the plan: data/voice/NAME/STYLE/takeN/ID.wav
nix develop .#eval -c tools/align.py voice NAME          # phone alignment, adapted to the speaker
nix develop .#eval -c tools/fit-voice.py fit NAME        # -> data/voice/NAME/profile.txt
nix develop .#eval -c tools/fit-voice.py score NAME --voice data/voice/NAME/profile.txt   # how close is it?

ttslab say --voice data/voice/NAME/profile.txt "Hello, this is my voice."
ttslab say --voice data/voice/NAME/profile.txt --style lively "What a day!"
ttslab say --voice data/voice/NAME/profile.txt --variation 1 --seed 7 "Say it differently."
```

The script (`voice/script.tsv`) has sustained vowels, h_d words, consonants
between vowels, intonation prompts, 100 ARCTIC sentences and ten sentences to
read in any number of styles. The fit is analysis by synthesis: the engine
speaks the same prompts, both are measured the same way, and the profile
moves until they agree (details in `tools/fit-voice.py` and NOTES §8). A
profile is plain text (`ttslab profile` prints the stock voice in the same
format), so it can be edited by hand. `--variation` (0–1) varies the targets
from token to token as much as your takes did; `{shown|aa0 p aa1}` in any
text gives a word's pronunciation in ARPAbet. Recordings and profiles stay
in `data/voice/`, which git ignores. Front ends can load a profile through
the C API (`ttslab_set_voice`); the GUI has no control for it yet.

## License and credits

MIT, see [LICENSE](LICENSE).

Pronunciations come from the [CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict)
(Copyright © 1993-2015 Carnegie Mellon University, BSD-style licence, full text in
[licenses/CMUdict-LICENSE](licenses/CMUdict-LICENSE)). The built-in word list in
`src/lexicon.zig` is drawn from it; the full dictionary is downloaded by
`tools/fetch-cmudict.sh`, not shipped. The refined engine's duration model
(`src/duration_model.zig`) is fitted to phone labels of the
[CMU ARCTIC](http://festvox.org/cmu_arctic/) databases (Copyright © 2003
Carnegie Mellon University, free for any use; licence in
[licenses/CMU-ARCTIC-LICENSE](licenses/CMU-ARCTIC-LICENSE)); only the fitted
numbers are included, not the data. The letter-to-sound rules are a
re-implementation in the style of Elovitz et al., *Letter-to-Sound Rules for
Automatic Translation of English Text to Phonetics* (Naval Research Laboratory,
1976).
