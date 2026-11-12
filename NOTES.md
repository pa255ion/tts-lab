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

Measured with `ttslab bench --reps 7` on a laptop (i7-1185G7, 8 threads for
voice builds, synthesis single-threaded), ReleaseFast, 16 kHz output. The test
text is three sentences, 12.1 s of speech. "Synth time" includes text analysis
and prosody, which take under 1 ms.

| Engine | Build | Recorded audio | Voice data | Synth time | × real time | Peak heap / synth |
|---|---:|---:|---:|---:|---:|---:|
| Formant (classic rules) | — | — | 4.6 KiB | 57.8 ms | 209× | 0.96 MiB |
| Formant (refined) | — | — | 4.6 KiB | 79.2 ms | 153× | 0.96 MiB |
| Diphone + TD-PSOLA | 1.7 s | 1,330 s | 8.9 MiB | 11.0 ms | 1,099× | 0.85 MiB |
| Unit selection | 3.7 s | 1,772 s | 32.6 MiB | 14.1 ms | 865× | 0.88 MiB |
| Statistical parametric (HMM) | 2.0 s | 442 s | 460 KiB | 213 ms | 57× | 1.97 MiB |

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

| Configuration | WER | 95% CI |
|---|---:|---:|
| Formant classic | 18.3% | 9–29 |
| Formant classic, flat F0 | 14.4% | 6–23 |
| Formant refined | 21.7% | 11–34 |
| Diphone + PSOLA | 32.8% | 21–46 |
| Diphone, raw units (no PSOLA) | 23.9% | 15–34 |
| Unit selection | 14.4% | 6–24 |
| Unit selection, prosody forced by PSOLA | 15.6% | 8–25 |
| HMM (MLPG) | 35.6% | 23–48 |
| HMM without MLPG | 55.0% | 42–68 |
| HMM + global variance | 36.7% | 25–48 |

How to read this. An ASR model is not a listener: it was trained on human
speech, it guesses words from context, and it does not measure naturalness at
all. With ~175 words per row, differences under ~10 points are noise. What
holds up:

- **MLPG is essential.** Without it the HMM engine outputs the stair-step of
  state means, and WER jumps from 36% to 55%. This is exactly why Tokuda's
  parameter generation algorithm made HMM synthesis viable.
- **GV does not help intelligibility.** It counteracts over-smoothing (it
  helps perceived crispness), but it also exaggerates estimation noise.
- **Unit selection and formant synthesis are the most intelligible here;
  diphone+PSOLA and HMM the least.** Diphone+PSOLA has to stretch monotone
  110 Hz units to a contour spanning 85–160 Hz, and TD-PSOLA degrades at large
  modification factors. Unit selection mostly avoids modification, and
  forcing its units onto the plan costs little here.
- **Classic and refined formant synthesis are indistinguishable by ASR.** The
  refinements improve naturalness, which this metric doesn't see. They also
  fixed specific consonant errors: tying the burst spectrum to the formants
  fixed most /k/→/t/,/p/ confusions. Section 8 has more on this.
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
| Vowel reduction, microprosody | partial | unstressed vowels toward schwa; F0 bumps after voiceless consonants | Small naturalness cues |

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

### How much further could it go?

1. **More rules.** Better consonant clusters, syllabic consonants ("bottle"),
   glottal stops, flapping (American "butter"), vocal fry at low F0,
   speaker-specific formant targets. Each is a weekend's work and a small
   gain. DECtalk and eSpeak represent decades of this; they are very
   intelligible and clearly synthetic.
2. **Learn the rules from data: the real "in-between".** Keep the formant
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
  method: PSOLA concatenation is 5–7× cheaper here.

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
- Open: refined still sometimes loses sentence-initial /t/ ("Turn" →
  "Her") where classic doesn't; burst/aspiration levels measure the same, so
  the cause is not yet found.
