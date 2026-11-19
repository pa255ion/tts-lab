# TTS Lab — Notes

Notes on pre-neural speech synthesis, written alongside the toy project in this
directory. Sections 1–4 are background; 5+ describe the project, its cost
profile, and how far formant synthesis can be pushed. The log at the bottom is
appended as work goes along.

---

## 1. The pipeline

Almost every non-neural TTS system has the same shape:

```
text ─► 1. normalization ─► 2. pronunciation ─► 3. prosody ─► 4. waveform
        "Dr. 5th $3"        lexicon + rules     pitch, timing,   join or generate
        → "doctor fifth     → /d aa1 k t er0/   loudness         the audio
          three dollars"
```

1. **Text normalization** expands numbers, abbreviations, dates, symbols.
   "St." becomes "Saint" or "Street" depending on context. Unglamorous, and
   the source of many audible errors.
2. **Grapheme-to-phoneme (G2P)**. Look each word up in a pronunciation
   dictionary (CMUdict: ~135k words, ARPAbet phonemes plus stress digits).
   Unknown words go through **letter-to-sound rules**: hand-written
   (NRL rules, Elovitz et al. 1976 — the kind used by early systems) or
   learned (decision trees, joint n-grams). Homographs ("read", "lead")
   need part-of-speech tagging.
3. **Prosody**: decide phrase breaks, how long each phone lasts, and the
   pitch (F0) contour.
4. **Waveform generation**: where the methods differ (section 3).

**Planning happens per utterance, audio can stream.** Intonation needs
lookahead: a yes/no question rises at the end, so you must see the `?` before
you start; pitch declines across a statement, so the slope depends on sentence
length; stress placement depends on syntax. Systems therefore plan a phrase or
sentence at a time, and the actual waveform generation is cheap enough to run
in real time even on 1980s hardware.

## 2. Why diphones, not phonemes

Joining one sample per phoneme sounds awful because of **coarticulation**: a
phone is shaped by its neighbours (the /k/ in "keep" vs "coop"). Most of the
perceptual information is in the *transitions*.

A **diphone** runs from the middle of one phone to the middle of the next.
"cat" = `pau-k  k-ae  ae-t  t-pau`. The cut points sit in the acoustically
stable centres, so joins are less audible, and every transition is a real
recorded one. English needs ~1,500–2,000 diphones. Inventories are recorded
as nonsense words with a monotone pitch, so later pitch modification is small.

## 3. The four waveform approaches

| Approach | Idea | Classic examples |
|---|---|---|
| Formant (rule-based) | No recordings. Glottal source → resonators modelling vocal-tract formants; rules produce parameter tracks | Klatt/DECtalk, MITalk, Votrax, eSpeak |
| Diphone concatenation | One recording of each diphone, joined, then pitch/duration modified with PSOLA | MBROLA, Festival diphone voices |
| Unit selection | Hours of speech; search (Viterbi) for the units that best fit the target and join most smoothly; modify as little as possible | AT&T Natural Voices, Festival multisyn, pre-2017 Siri |
| Statistical parametric (HMM) | Statistical models of acoustic parameters per context; generate smooth trajectories; vocoder renders audio | HTS, early Google/Android voices |

### Formant synthesis
Source–filter model of speech. A source (periodic glottal pulses for voiced
sounds, noise for frication/aspiration) excites a set of second-order
resonators, one per formant. Klatt (1980) used a *cascade* branch (resonators
in series, good for vowels — relative formant amplitudes come out right
automatically) and a *parallel* branch (independent gains, needed for
fricatives and bursts). Every 5 ms a frame of ~40 parameters is sent to the
synthesizer. The intelligence is entirely in the **rules** that create those
frames from a phone string.

### Diphone + PSOLA
TD-PSOLA (pitch-synchronous overlap-add, Moulines & Charpentier 1990):
1. Mark every glottal closure ("pitch marks") in the recording.
2. Cut the signal into two-period Hann-windowed grains centred on the marks.
3. Re-place the grains at the *target* pitch period and overlap-add. The
   spectral envelope (vocal tract) is kept, the periodicity (pitch) changes.
4. Change duration by repeating or skipping grains.
Joins are smoothed because neighbouring grains overlap.

### Unit selection
Each target diphone has many candidates from a large corpus. Two costs:
- **target cost**: how well a candidate's context/pitch/duration/stress
  match what is wanted;
- **join cost**: spectral, pitch and energy discontinuity between consecutive
  candidates (zero if they were adjacent in the corpus).
Viterbi finds the cheapest path. When it picks long runs of adjacent corpus
units, the result is near-natural; when it can't, you hear glitches. Quality
scales with corpus size — and so does memory.

### Statistical parametric (HMM)
Analyse the corpus into frames of vocoder parameters (spectral envelope as
cepstra, F0, aperiodicity). Model each phone-in-context as a few states, each
with Gaussian distributions over static *and delta* features. To synthesize,
pick the state sequence and durations, then solve for the trajectory that is
most likely under both the statics and deltas (**MLPG**, Tokuda et al. 2000).
That makes the output smooth by construction, with no joins at all. Cost:
averaging over-smooths spectra → muffled, "buzzy" vocoder sound.

## 4. Prosody models

- **Durations**: Klatt's rules: `dur = (inherent − min) × factor + min`, with
  factors for stress, phrase-final lengthening, polysyllabic shortening,
  vowel before voiced consonant, clusters, speaking rate.
- **Hat pattern / declination**: baseline slopes down across a phrase, pitch
  jumps up on the first accent and down after the last (MITalk-style rules).
- **Fujisaki model**: `ln F0(t) = ln Fb + Σ phrase-command responses +
  Σ accent-command responses`. Phrase commands are impulses smoothed by a
  critically damped 2nd-order filter (slow arcs); accent commands are steps
  smoothed the same way (local plateaus). Physiologically motivated and gives
  very natural-looking contours from a handful of numbers.
- **ToBI** labels (`H*`, `L+H*`, `L-L%`, `H-H%`) are the usual symbolic layer
  between text analysis and numeric targets.

---

## 5. The toy project

### Layout

```
tts-lab/
├── flake.nix              dev shells: default (zig, ffmpeg), gui (+Qt 6, CMake), eval (+whisper.cpp)
├── build.zig              ttslab CLI, libttslab.so (C ABI), tests
├── include/ttslab.h       C interface used by the GUI
├── src/
│   ├── text.zig           1. normalization: numbers, years, ordinals, $, abbreviations, acronyms
│   ├── lexicon.zig        2. G2P: built-in words / CMUdict, NRL-style letter-to-sound rules
│   ├── prosody.zig        3. phrasing, Klatt duration rules, F0: flat / hat / Fujisaki
│   ├── phones.zig         phone inventory + acoustic facts (formant targets, durations, noise spectra)
│   ├── formant.zig        engine 1+2: Klatt-style cascade/parallel synthesizer, classic + refined rules
│   ├── corpus.zig         "recording studio": formant voice reads prompts; pitch marks; cepstral analysis
│   ├── psola.zig          TD-PSOLA over concatenated pieces
│   ├── diphone.zig        engine 3: ~1,640 diphone inventory + PSOLA
│   ├── unitsel.zig        engine 4: target/join costs, Viterbi, diphone fallback
│   ├── hmm.zig            engine 5: context-dependent state Gaussians, MLPG, GV, cepstral vocoder
│   ├── tts.zig            orchestration, lazy voice builds
│   ├── bench.zig          cost profile (counting allocator, timing)
│   ├── capi.zig           C ABI
│   └── main.zig           CLI + interactive menu
├── gui/                   Qt 6 front end (CMake)
└── tools/                 fetch-cmudict.sh, spectrogram.sh, eval-asr.sh, wer.py
```

### Shared front end
All five engines consume the same `Utterance` (timed phones + F0 per 5 ms
frame), so prosody is held constant when engines are compared, and the F0
model (flat / hat / Fujisaki) can be switched independently of the engine.

### Where the "recordings" come from
There is no human voice in this project. The refined formant engine plays
voice talent: it reads ~1,640 diphone carrier words (`t aa X Y aa`, monotone
110 Hz) and 50 prompt sentences × 3 speaking styles, with per-token random
variation. The data-driven engines only ever see that audio, its phone labels
and its pitch marks (the glottal-closure instants, which a real system would
get from an EGG or an epoch detector). So:

- they inherit the formant voice's timbre, which means the comparison isolates
  what each *method* does (prosody modification, joins, smoothing) and what it
  *costs*, not the naturalness real recordings would bring;
- with real recordings the ranking on naturalness would flip: unit selection
  and diphone voices would sound far more human than any formant voice.

Replacing the voice talent with a human means recording the same prompts,
labelling them (forced alignment, e.g. Montreal Forced Aligner) and
pitch-marking them. The engines would not need to change.

### Building and running

```sh
nix develop                       # zig 0.16, ffmpeg
tools/fetch-cmudict.sh            # optional: 125k-word dictionary
zig build -Doptimize=ReleaseFast
zig build test

./zig-out/bin/ttslab                                 # interactive menu
./zig-out/bin/ttslab say -e refined "Hello there."   # plays in a terminal; TTSLAB_PLAYER=cmd to override
./zig-out/bin/ttslab compare "Is this a question?"   # every engine -> out/*.wav
./zig-out/bin/ttslab say -e diphone --no-psola ...   # raw units
./zig-out/bin/ttslab say -e hmm --no-mlpg ...        # stair-step state means
./zig-out/bin/ttslab say --f0 flat|hat|fujisaki ...
./zig-out/bin/ttslab say --pitch 1.8 --formant 1.17 ...   # "female" voice
./zig-out/bin/ttslab bench                           # cost profile
nix develop .#eval -c tools/eval-asr.sh              # ASR intelligibility
```

## 6. GUI

**Qt 6 Widgets (default, implemented in `gui/`).** The Zig backend is built as
`libttslab.so` with a small C ABI (`include/ttslab.h`), which is the most
robust way to pair Zig with C++. The window has:

- an engine selector, with each engine's options enabled only when they apply
  (PSOLA on/off, unit-selection prosody forcing, MLPG, GV);
- prosody and voice controls (F0 model, pitch, rate, range, vocal-tract
  scale, man/woman/child presets);
- the transcription (stress digits, `*` = pitch accent), timing and
  unit-selection statistics;
- a spectrogram with the *planned* F0 contour drawn over it, and the
  waveform below;
- play, save WAV, "speak with every engine" (renders and plays each in
  turn), and a cost-profile dialog.

Synthesis runs on a worker thread through QtConcurrent, so the few-second
first-use voice builds don't freeze the window.

```sh
zig build -Doptimize=ReleaseFast
nix develop .#gui -c sh -c 'cmake -S gui -B gui/build -G Ninja && cmake --build gui/build'
gui/build/ttslab-gui
```

Alternatives, if Qt is not wanted:

| Option | Pros | Cons |
|---|---|---|
| Qt 6 Widgets (C++) | Native, mature, QAudioSink for audio, QPainter for plots | C++ side, big dependency |
| Dear ImGui + SDL3 (C via cimgui, callable from Zig) | Everything in Zig, immediate mode suits sliders and live plots, SDL does audio | Not native-looking, you draw your own widgets |
| raylib + raygui (C, easy from Zig) | Tiny, one dependency, has audio | Basic widgets |
| Web UI: Zig → wasm + a static HTML page | Runs in a browser, Web Audio, easy plots | Two toolchains, no threads without extra work |
| The built-in terminal menu | No dependency at all | No plots |

For a Zig-only project, Dear ImGui with SDL3 is what I'd pick: the engine
toggles and parameter sliders map naturally onto immediate-mode widgets, and a
live spectrogram is just a texture.

## 7. Cost profile

Measured with `ttslab bench --reps 7` on a desktop (Ryzen 7 5700G, 16
threads for voice builds, synthesis single-threaded), ReleaseFast, 16 kHz
output. The test text is three sentences, 12.1 s of speech. "Synth time"
includes text analysis and prosody, which take under 1 ms. (The first version
of this table, on an i7-1185G7 laptop, had the same ranking at 2–5× these
times.)

| Engine | Build | Recorded audio | Voice data | Synth time | × real time | Peak heap / synth |
|---|---:|---:|---:|---:|---:|---:|
| Formant (classic rules) | — | — | 4.6 KiB | 10.9 ms | 1,104× | 0.96 MiB |
| Formant (refined) | — | — | 4.6 KiB + 0.2 KiB timing model | 16.8 ms | 614× | 0.83 MiB |
| Diphone + TD-PSOLA | 0.26 s | 1,330 s | 8.9 MiB | 4.3 ms | 2,784× | 0.85 MiB |
| Unit selection | 0.63 s | 1,772 s | 32.6 MiB | 5.6 ms | 2,151× | 0.87 MiB |
| Statistical parametric (HMM) | 0.40 s | 442 s | 460 KiB | 95.1 ms | 127× | 1.97 MiB |

The refined engine speaks the same text faster than the others (its fitted
timing, §8), so its time per second of *audio* is about 10% higher than
before the naturalness work (the two extra resonators), not 50%.

Shared by all engines: the lexicon. The built-in list is ~350 words; the full
CMUdict is 126k words and 6.3 MiB in this naive in-memory form (a real system
compresses it to ~1 MB, or keeps only exceptions and leans on letter-to-sound
rules). Peak heap is dominated by the output buffer itself (12 s × 16 kHz ×
4 bytes = 0.77 MB), so working memory is small for every engine.

### What dominates each engine's cost

| Engine | Per-sample work | Per-utterance work | Memory scales with |
|---|---|---|---|
| Formant classic | cascade of 7 resonators + noise bank: roughly 60–90 flops/sample, coefficients recomputed every 5 ms | building 7 parameter tracks: trivial | nothing: a few KB of rule tables |
| Formant refined | + F6, LF pulse evaluation (an `exp` and a `sin`), coefficient updates every 1 ms: roughly 150 flops/sample | dominance model: about 15 gestures × 7 parameters per frame | same |
| Diphone + PSOLA | about 2 overlapping grains per sample: a multiply-add and a window value | one unit lookup per phone | number of units × unit length (≈ 1,600 × 90 ms); **fixed** per voice |
| Unit selection | PSOLA overlap-add only | scoring every candidate for every target diphone, then Viterbi O(N·K²) with cepstral join distances (K = 40 here) | **hours of audio**: grows with corpus size, which is also what quality grows with |
| HMM | the vocoder: 4 FFTs of 512 points per 5 ms frame plus pulse placement, roughly 20 Mflop/s | MLPG: one banded solve per cepstral dimension, O(T) | number of context-clustered states (1,133 here, ~1–2 MB in real HTS voices) |

Observations:

- **Formant synthesis is the smallest, not the fastest.** Its voice is a
  4.6 KB table: 2,000× smaller than the diphone inventory and 7,000× smaller
  than the unit-selection database. But per sample it does more arithmetic
  than PSOLA, which is just copying and windowing recorded samples. All of
  them are far faster than real time: even the refined formant engine needs
  only ~2.5 Mflop/s, which is why DECtalk could run a Klatt synthesizer on a
  1984 TMS32010 DSP.
- **The HMM engine is the slowest here only because of its vocoder.** HTS used
  an MLSA filter (~100 flops/sample) instead of FFT overlap-add, and would be
  roughly formant-speed. The model itself is compact (460 KB) and does not
  grow with corpus size, which is the main advantage of statistical
  parametric synthesis over unit selection.
- **Unit selection's cost is mostly storage.** Our corpus (150 recordings, 7.5
  minutes, plus the diphone fallback) is tiny. Commercial voices used 2–10+
  hours, i.e. hundreds of MB to a few GB, and quality was roughly a function of
  that size. Even so, only 490 of 1,599 diphones occurred in our sentence
  corpus, and 9 of the 37 units in a test sentence had to come from the
  fallback inventory. Coverage is the core problem of the method.
- **Build time** is "recording" plus analysis. In the real world this is
  replaced by a studio session, labelling and pitch-marking, and that step
  dominates the cost of every data-driven voice (weeks, not seconds). A
  formant voice costs instead the *expert* time to write rules.

### Intelligibility (ASR proxy)

`tools/eval-asr.sh` synthesizes 20 held-out sentences (none of them in the
corpus) with each configuration, transcribes them with whisper.cpp
(`base.en`), and computes the word error rate. The 95% intervals come from
bootstrapping over sentences.

| Configuration | WER | 95% CI | First version |
|---|---:|---:|---:|
| Formant classic | 18.3% | 9–29 | 18.3% |
| Formant classic, flat F0 | 14.4% | 6–23 | 14.4% |
| Formant refined | 17.8% | 9–27 | 18.3% |
| Diphone + PSOLA | 15.0% | 8–23 | 32.8% |
| Diphone, raw units (no PSOLA) | 11.7% | 6–18 | 23.9% |
| Unit selection | 15.6% | 7–26 | 14.4% |
| Unit selection, prosody forced by PSOLA | 12.2% | 5–20 | 15.6% |
| HMM (MLPG) | 36.7% | 24–50 | 35.6% |
| HMM without MLPG | 53.3% | 41–67 | 55.0% |
| HMM + global variance | 27.8% | 19–38 | 36.7% |

"First version" is before the naturalness work of §8. The diphone,
unit-selection and HMM voices are *recorded from the refined formant engine*
(§5), so they changed with it; classic did not change at all. A second,
harder set (60 ARCTIC b-set prompts, literary text with rare words, ~550
words per row): classic 22.6%, refined 22.9% before and 25.4% after (95% CIs
about ±5 points).

How to read this. An ASR model is not a listener: it was trained on human
speech, it guesses words from context, and it does not measure naturalness at
all. With ~175 words per row, differences under ~10 points are noise. What
holds up:

- **MLPG is essential.** Without it the HMM engine outputs the stair-step of
  state means, and WER jumps from 36% to 55%. This is exactly why Tokuda's
  parameter generation algorithm made HMM synthesis viable.
- **The concatenative engines inherit their voice talent's quality.** When
  the refined formant voice was brightened (§8), diphone+PSOLA went from the
  second-worst row (32.8%) to among the best (15.0%), with no change to the
  diphone code. A muffled recording is a muffled voice, however good the
  concatenation. This is the voice-talent problem real systems spend most of
  their budget on.
- **GV's effect depends on the data.** With the first voice talent it made no
  difference; with the brighter one it helped (36.7% → 27.8%, CIs overlap).
  It counteracts over-smoothing, but also exaggerates estimation noise.
- **PSOLA's cost shows up as a gap, not a ranking.** Raw diphones (no pitch
  or duration modification) stay ahead of PSOLA-modified ones, and TD-PSOLA
  degrades at large modification factors: the units are monotone 110 Hz and
  the contour spans 85–160 Hz.
- **Classic and refined formant synthesis are indistinguishable by ASR.** The
  refinements improve naturalness, which this metric doesn't see (§8 measures
  that against recordings instead). They also
  fixed specific consonant errors: tying the burst spectrum to the formants
  fixed most /k/→/t/,/p/ confusions, and a release-frication phase fixed
  initial /t/ being heard as /h/. Section 8 has more on this.
- Flat F0 did *not* hurt ASR, and may even have helped. That is a reminder
  that intonation mostly serves naturalness, phrasing and meaning
  (statement vs question), not word recognition.

## 8. How far can formant synthesis be refined?

### The synthesizer is not the bottleneck; the rules are

The decisive experiment is old. Holmes (1973) hand-tuned the parameters of a
parallel formant synthesizer, frame by frame, to copy a natural sentence. The
result was very hard to tell from the original recording. So a
source–filter formant model with a good glottal source can sound essentially
natural: its *representational* ceiling is high. What limits classic formant
TTS is the other half: predicting ~40 parameter tracks every 5 ms from a
phone string with hand-written rules. Every detail of real speech (how F2
bends into a stop, how voicing dies out phrase-finally, how breathiness varies
with stress) must be noticed, formalised and tuned by an expert.

### What this project's "refined" mode adds, and what each change buys

| Refinement | Classic | Refined | Buys |
|---|---|---|---|
| Glottal source | impulse → low-pass | LF model with Rd voice quality | Natural spectral tilt; voice quality can vary (tense ↔ breathy) |
| Voice quality over time | constant | tenser when stressed, laxer and breathier phrase-finally | Phrase endings sound like a person running out of breath, not like a gate closing |
| Periodicity | perfect | jitter, shimmer, 3-sine flutter | Removes the "buzz"; a perfectly periodic source is a strong synthetic cue |
| Aspiration | only for /h/ and VOT | also pitch-synchronous breath noise in voiced speech | Breathiness, softer onsets |
| Coarticulation | straight lines between targets, stop loci | dominance functions: every segment pulls every parameter with its own strength and reach | Vowel-to-vowel coarticulation across consonants, nasality spreading into vowels, context-dependent transitions, all automatic |
| Parameter updates | held for 5 ms | interpolated per sample (except bursts) | No staircase roughness |
| Bursts and frication | fixed noise bank | parallel resonators tied to the current F2..F5 (Klatt 1980) | Velar bursts track the vowel; fixed most /k/ confusions in the ASR test |
| Stop release | burst, then straight into aspiration | burst, then 6–20 ms of decaying frication at the constriction, with the aspiration fading in under it | Initial /t/ stops sounding like /h/ ("Turn" was heard as "Her"); first-word ASR on a /t/-initial set went from 7/20 to 11/20 |
| Vowel reduction, microprosody | partial | unstressed vowels toward schwa; F0 bumps after voiceless consonants | Small naturalness cues |
| Upper spectrum | F4–F6 only | plus two poles at 6.5 and 7.2 kHz (higher-pole correction) | Vowels no longer 25–40 dB too weak at 5–7 kHz; the biggest change in brightness |
| Voice quality baseline | — | tenser Rd, crisper glottal closure (LF return phase ×0.4) | H1–H2 and the 1–4 kHz balance within the range of three natural male voices |
| Noise spectra | flat bypass, fixed shapes | bypass tilted down above 4 kHz; /s/ peak lowered; fricatives through the formant-tied bank | 8 kHz excess of 13 dB removed; less hiss |
| Loudness | per-class constants | voicing amplitude follows F0 (~0.55 dB/semitone) | Accents louder, loudness declines with pitch over a phrase; vowel level spread as in recordings |
| Timing | Klatt (1979) rules and values | same kind of rules, values fitted to 100k phones of read speech | Natural rhythm: vowels and consonants equally long on average, as in recordings, instead of 2:1 |

Two lessons from building this:

- **Refinements interact with intelligibility in non-obvious ways.** My first
  refined version smoothed the voicing amplitude and interpolated burst
  amplitudes per sample. That made it smoother and *less* intelligible: a
  10 ms burst smeared over 5 ms ramps stops being a burst. Real speech
  contains abrupt events, and "smoother" is not always "more natural".
- **Intelligibility saturates early; naturalness doesn't.** Even the classic
  rules get most words across (DECtalk in the 1980s already scored close to
  natural speech on sentence-transcription tests). The remaining gap is
  naturalness, and every increment there costs disproportionate rule-writing
  effort. That is why the field moved to data.

### Measured against recordings

Naturalness needs listeners, but gross departures from natural speech can be
measured. `tools/acoustics.py` computes the same statistics for synthetic and
natural speech: long-term spectrum, per-vowel spectra, harmonics-to-noise
ratio, H1–H2 (a voice-quality measure), obstruent levels relative to vowels,
vowel level spread, F0 movement and segment durations. The reference is CMU
ARCTIC (three US male speakers, bdl, rms and jmk, reading the same prompts;
free for any use; fetched by `tools/fetch-arctic.sh`, not committed), and the
synthetic side speaks the same 20 prompts.

| Measure | Natural (3 speakers) | Refined, before | Refined, now |
|---|---|---:|---:|
| Vowel level at 5–7 kHz, re strongest band | −36 to −45 dB | −62 to −88 dB | −34 to −45 dB |
| Long-term spectrum at 1 / 2 / 4 / 8 kHz, re 500 Hz | −8 to −9 / −10 to −15 / −11 to −12 / −17 to −19 | −12 / −19 / −17 / −12 | −11 / −14 / −14 / −16 |
| H1–H2 in vowels | −3 to +6 dB | +4.0 dB | −2.4 dB |
| Vowel level spread (sd) | 3.3–4.4 dB | 2.5 dB | 3.8 dB |
| Speech time, 20 prompts | 53–58 s | 70.2 s | 56.6 s |
| Mean vowel / consonant duration | 80–84 / 79–88 ms | 147 / 72 ms | 83 / 83 ms |
| Pause at a comma | at ~40% of commas, 56 ms on average | 180 ms, always | 110 ms, always |
| /t/ burst peak, re vowels | −6 to −10 dB | −2.3 dB | −6.8 dB |
| /f/ peak, re vowels | −14 to −15 dB | −22.6 dB | −18.5 dB |
| F0 movement per 10 ms in vowels, mean | 0.26–0.33 st | 0.14 st | 0.20 st |

What the measurements found, in order of size:

1. **A spectral cliff above 4 kHz.** A cascade of DC-normalized resonators
   falls steeply above its last formant, and the real vocal tract has more
   formants than six. Klatt described the fix ("higher-pole correction");
   this project's refined mode lacked it, and vowels were 25–40 dB too weak
   at 5–7 kHz. Two extra resonators fixed it. This is the "muffled" quality
   that formant voices are known for.
2. **Rhythm.** Klatt's inherent durations describe slow, careful speech:
   vowels came out twice as long as consonants, while the recordings have
   them equally long. A duration model with Klatt's structure (phone base
   duration times context factors for stress, accent, phrase-final position,
   word position, clusters, the following consonant) was fitted to ~100,000
   aligned phones (`tools/fit-durations.py`, `src/duration_model.zig`). On
   held-out sentences its predictions correlate with the speakers' durations
   at r = 0.71 for vowels and 0.60 for consonants, against 0.57 and 0.39 for
   Klatt's rules. The numbers are 57 floats: the fitted model is a rule
   system that happened to be written by regression.
3. **Voice quality and the 1–4 kHz balance.** Fant's Rd regression gives a
   slow glottal closure for a male voice, which darkens the spectrum. A
   tenser baseline and a shorter return phase brought H1–H2 and the mid
   frequencies into the natural range.
4. **Noise sources too bright** (frication and bursts had 13 dB too much at
   8 kHz), **flat loudness** (vowel levels varied half as much as in
   recordings), and **consonant levels** off by 4–8 dB for /t f th p/.

ASR did not move (17.8% vs 18.3%; 25.4% vs 22.9% on the harder set, within
noise), which is the point: these are changes a word-error rate cannot see.
One change that it *could* see was dropped: a stronger stressed/unstressed
amplitude contrast (unstressed vowels 3 dB quieter) cost about 3 points on
the 60-sentence set, and the original contrast was kept.

Still missing, by these measures: fast F0 movements (the 90th percentile of
F0 change per 10 ms is 0.37 st against 0.53–0.67 natural), and every
judgement a listener would make. The A/B page in the log entry for
2026-11-18 collects the latter.

### How much further could it go?

1. **More rules.** Better consonant clusters, syllabic consonants ("bottle"),
   glottal stops, flapping (American "butter"), vocal fry at low F0,
   speaker-specific formant targets. Each is a weekend's work and a small
   gain. DECtalk and eSpeak represent decades of this; they are very
   intelligible and clearly synthetic.
2. **Learn the rules from data: the real "in-between".** The fitted duration
   model above is a first, small step: the rule structure stays, the numbers
   come from recordings. The full version keeps the formant
   synthesizer as the vocoder, but predict its parameter tracks with a
   statistical model trained on labelled speech. Formant tracks would be
   extracted from recordings by analysis, or by analysis-by-synthesis fitting.
   That is HMM synthesis with a formant vocoder instead of a cepstral one. In
   this codebase it is a contained change: `hmm.zig` would model `Frame`
   fields instead of cepstra, and `formant.render` would be the vocoder. You
   keep the formant engine's assets (a few hundred KB of model, real-time on
   anything, every parameter interpretable and editable) and gain data-driven
   coarticulation and timing. Recent research does exactly this with neural
   networks ("neural formant synthesis": a network predicts formant-like
   parameters, which stay editable, and a neural vocoder renders them).
3. **Go one level deeper: articulatory synthesis.** Model the tongue, lips and
   glottis and derive the formants physically (e.g. VocalTractLab).
   Coarticulation then follows from physics instead of rules. It is
   expensive, hard to control, and an active research area.

### The trade-offs that make formant synthesis worth refining

Measured here and from the history of these systems:

- **Footprint.** A 4.6 KB rule table versus 9 MB of diphones or 33 MB of
  unit-selection data (hundreds of MB to GB in commercial voices). eSpeak
  covers 100+ languages in a few MB in total. For microcontrollers, embedded
  devices and many-language coverage, nothing else comes close.
- **Total control.** Pitch, speaking rate, vocal-tract length, breathiness,
  tenseness, even singing: all are parameters. In this project `--formant`
  only works for the formant and HMM engines. PSOLA can move pitch but not
  formants, so a "female" diphone voice made by raising pitch sounds like a
  sped-up man. Research stimuli in speech perception are still often made
  with Klatt synthesizers for exactly this reason.
- **Robustness and graceful degradation.** No joins, no coverage holes (unit
  selection was missing 1,109 of 1,599 diphones in its corpus here), no
  glitches when the text is unusual. Speech stays intelligible at very high
  rates: blind screen-reader users run eSpeak-class formant voices at
  400–500+ words per minute, where concatenative voices smear.
- **Latency and predictability.** Output can start as soon as the first
  phrase is planned, with no voice loading and no search.
- **Costs.** Naturalness is capped by the rules, the voice is synthetic in an
  identifiable way, and building a new voice or language needs phonetic
  expertise rather than recordings. Per sample, it is *not* the cheapest
  method: PSOLA concatenation is about 4× cheaper here.

**Bottom line.** Hand-tuned formant synthesis can be made very intelligible
and noticeably more pleasant than 1980s systems, but not natural; that ceiling
is the rules, not the model. If you value its trade-offs (tiny, controllable,
robust, fast), the most promising direction is to keep the formant
synthesizer and learn its control parameters from data. That combines most of
the formant approach's advantages with most of the naturalness gains that
drove the field to data-driven methods.

---

## Log

- 2026-11-12: Project created. Background notes written.
- Front end: tokenizer/normalizer (numbers, years, ordinals, currency,
  abbreviations incl. "St." saint/street, acronyms spelled out), built-in
  lexicon + optional CMUdict (125,644 words loaded), ~300 NRL-style
  letter-to-sound rules, Klatt duration rules, three F0 models.
- Formant engine first light. Levels calibrated by measuring per-segment
  RMS: the classic impulse source was 40 dB too quiet relative to frication.
  First ASR check already read the test sentence almost perfectly.
- Built diphone (1,599 units), corpus (150 recordings), unit selection and
  HMM engines on top of the formant "voice talent".
- Bug: pitch marks. Unvoiced filler marks were inserted between every pair of
  voiced glottal pulses (gap > 120 samples is *every* period at 110 Hz), so
  analysis saw nothing as voiced. Fixed by only filling gaps > 25 ms.
- Bug: clicks at every concatenation join, visible as vertical lines in the
  spectrogram. PSOLA grains near a unit edge were reading the *next* unit's
  samples. Fixed by giving each unit 25 ms of its own context ("pieces").
- ASR harness (whisper.cpp base.en, word error rate, bootstrap CIs).
- Refined formant was initially *less* intelligible than classic. Causes:
  burst amplitudes interpolated per sample (smeared bursts) and a voicing
  smoother blurring onsets; plus a fixed noise bank that can't place velar
  bursts on the following vowel's F2/F3. Fixed with held burst amplitudes and
  a formant-tied parallel branch (Klatt 1980).
- Affricates (/tʃ dʒ/) strengthened: shorter closure, longer and louder
  frication phase, voicing during the frication of /dʒ/.
- HMM vocal-tract warping used `cos` in an O(N·M) loop per frame
  (229 ms for 1 s of audio); a precomputed table brought it to 19 ms.
- Qt 6 GUI built and checked headless (offscreen screenshot).
- Fixed: refined lost sentence-initial /t/ ("Turn left" → "Her left").
  The burst levels matched classic, and short-phrase tests showed classic has
  the same weakness ("Ted" → "Head", "Turn left." → "Earth Lift"); the full
  sentence only survived in classic thanks to context. The cause was the
  release: a 12 ms burst followed directly by breathy aspiration, which a
  listener parses as a click plus /h/. Natural voiceless stops have 10–20 ms
  of frication at the constriction between the burst and the aspiration.
  Refined now synthesizes that (place-dependent length: 6/18/20 ms for
  labial/alveolar/velar; decaying amplitude through the formant-tied burst
  shape; aspiration fading in under it). First-word accuracy on a 20-phrase
  /t/-initial test set went from 7/20 to 11/20; full eval WER 21.7% → 18.3%
  (within noise, but the targeted error is gone). An F1-cutback rule (wider
  B1 and lower F1 during aspiration) was also tried, made no measurable
  difference, and was left out. Classic is unchanged on purpose: it keeps
  the early-rule behaviour as the baseline. "Ted" is still heard as "head".
- 2026-11-18: Naturalness pass on the refined engine, measured against CMU
  ARCTIC recordings (§8 "Measured against recordings"). New tools:
  `tools/acoustics.py` (spectra, HNR, H1–H2, levels, F0, durations for
  synthetic and natural speech alike), `tools/fetch-arctic.sh`,
  `tools/fit-durations.py` and the `ttslab segs` command. Changes, largest
  first: higher-pole correction (F7, F8); a fitted duration model
  (`src/duration_model.zig`, `--timing klatt|fitted`, refined defaults to
  fitted); tenser Rd baseline with a shorter LF return phase; noise path
  tilted above 4 kHz; loudness tied to F0; obstruent level corrections;
  shorter comma pauses (110 ms). ASR unchanged within noise. Every engine
  recorded from the refined voice improved as a side effect (diphone+PSOLA
  32.8% → 15.0% WER). A blind A/B listening page (old vs new refined, eight
  sentences) collects the listener judgements that the metrics cannot make.
- 2026-11-19: Listening result: the new refined voice was preferred in 8 of
  8 blind trials (sides shuffled, 5 of 8 with the new voice as A). One
  listener, so it says nothing about listeners in general, but by a sign
  test that is unlikely to be chance for this listener (p ≈ 0.004).
- 2026-11-19: Intonation pass on the refined engine (`--intonation
  rules|fitted`; refined defaults to fitted, every other engine keeps the
  rules). New tool `tools/intonation.py` measures F0 at the planned accents
  and phrases of 60 ARCTIC prompts, for our audio and for the three
  speakers' recordings (planned phones aligned to their labels), with the
  same pitch tracker for both; `AUDIO=N tools/fetch-arctic.sh` fetches the
  recordings. Fitted on the a-set; b-set (held out):

  | measure                          | rules | fitted | bdl   | rms   | jmk   |
  |----------------------------------|-------|--------|-------|-------|-------|
  | phrase range p90–p10 (st)        | 8.3   | 5.7    | 6.3   | 7.2   | 5.6   |
  | accent peak delay from vowel onset (ms) | 69 | 20 | 20    | 10    | 13    |
  | declination of accent peaks (st/s) | −5.1 | −1.9  | −1.5  | −1.7  | −2.1  |
  | first − last accent (st)         | 7.4   | 2.4    | 2.1   | 3.5   | 3.6   |
  | accent residual sd (st)          | 1.1   | 1.9    | 2.0   | 1.8   | 1.5   |
  | unaccented vowels re p10 (st)    | 2.4   | 2.2    | 2.8   | 3.2   | 2.5   |
  | F0 sd over vowels (st)           | 3.6   | 2.8    | 2.9   | 3.2   | 2.5   |
  | \|dF0\| per 10 ms, p90 (st)        | 0.37  | 0.47   | 0.73  | 0.90  | 0.81  |

  What changed: accent commands start with the syllable onset and end
  halfway through the vowel, rising fast (β 25) and falling slowly (β 3), so
  peaks land early in the vowel and F0 stays up between accents (the rules
  peaked at the end of the vowel and plateaued); downstep 3% per accent
  instead of 15% and a smaller phrase command (the rules declined 2–3× too
  fast); accent sizes vary ±80% per word by a stable hash, since readers
  don't weight accents equally and the text alone doesn't say which; a small
  fall before commas instead of a continuation rise (readers stayed level).
  Also new: `*word*` marks contrastive focus (all engines): the focused
  accent is 1.9× (3× under the rules, whose declination buries it
  otherwise) and falls fast, earlier accents in the phrase are 0.6×, later
  ones 0.25× (post-focus compression), accent sizes don't vary in that
  phrase, the final fall waits for the focus peak, and the word is 10–30%
  longer.
  In the synthesizer, a slow random F0 wander (AR(1), sd 2.5%, 30 ms)
  raised the fast F0 movement, though it still falls short: mid-vowel
  |dF0| p90 0.33 → 0.43 against 0.69–0.80 natural. Some of the natural
  figure is probably tracker noise on real voices. Making the Klatt-style
  flutter 2.5× larger barely moved it, and it was left at its old size.
  Wander is off when the engine records the corpora for the data-driven
  engines: with it, the diphone voice's WER went 15.0% → 19.4%. ASR on the 60 b-set sentences:
  refined 24.5% with the rules, 22.2% fitted (CIs ±5); 20-sentence set:
  refined 16.1% (was 17.8%), diphone, unit selection and HMM unchanged.
- 2026-11-19: Listening result for the intonation pass: no preference
  overall (new 3, old 3, no difference 2; one listener). The listener's
  notes named two faults, both confirmed in the F0: the yes/no question
  rise sat on top of a high nuclear accent ("…tonight?" peaked 9 st above
  the phrase), and a statement could end without a fall ("…not the blue
  one." stayed at 120 Hz where the rules fell to 88 Hz) because "one" got
  its own accent right at the end. Fixes: the nuclear accent of a yes/no
  question is low (0.3×) and the rise smaller (0.5 instead of 0.7), i.e.
  L* H-H% instead of H* plus a rise (now +5 st); the last accent of a
  statement falls quickly (β 8); and "one(s)" after determiner + modifier
  ("the blue one") is a pro-form without an accent, in all engines. b-set
  measures stay within the speakers' range (end of sentence re p10 −0.69 st
  vs −0.93…+0.20; first − last accent 2.5 st); ASR on the 60 b-set
  sentences 19.7% (CI 14.7–24.9; was 22.2%).
