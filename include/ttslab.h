/* ttslab: C interface to the TTS Lab synthesizer (libttslab.so).
 *
 * Thread safety: one ttslab handle must not be used from two threads at
 * once. Calls may block for a few seconds the first time an engine that
 * needs a voice (diphone, unit selection, HMM) is used; call
 * ttslab_prepare() from a worker thread to build it ahead of time. */
#ifndef TTSLAB_H
#define TTSLAB_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ttslab ttslab;

enum ttslab_engine {
    TTSLAB_FORMANT_CLASSIC = 0,
    TTSLAB_FORMANT_REFINED = 1,
    TTSLAB_DIPHONE = 2,
    TTSLAB_UNIT_SELECTION = 3,
    TTSLAB_HMM = 4
};

enum ttslab_f0_model { TTSLAB_F0_FLAT = 0, TTSLAB_F0_HAT = 1, TTSLAB_F0_FUJISAKI = 2 };

typedef struct {
    int engine;          /* enum ttslab_engine */
    int f0_model;        /* enum ttslab_f0_model */
    float pitch_scale;   /* multiplies the planned F0 */
    float rate;          /* > 1 is faster */
    float pitch_range;   /* size of accents and phrase arcs */
    float formant_scale; /* vocal-tract scaling; formant and HMM engines only */
    int diphone_psola;   /* diphone: modify pitch/duration (1) or raw units (0) */
    int unit_modify;     /* unit selection: force units onto planned prosody */
    int hmm_mlpg;        /* HMM: smooth trajectories with MLPG */
    int hmm_gv;          /* HMM: global-variance postfilter */
} ttslab_options;

typedef struct {
    float *samples;       /* mono, sample_rate Hz, roughly -1..1 */
    size_t n_samples;
    int sample_rate;
    float *f0;            /* planned F0 contour, one value per frame */
    size_t n_f0;
    float f0_frame_rate;  /* frames per second (200) */
    char *transcription;  /* phones with stress digits, '*' = pitch accent */
    double audio_seconds;
    double synth_ms;
    double build_ms;      /* voice build time spent inside this call */
    int units, joins, fallbacks; /* unit-selection statistics */
} ttslab_result;

void ttslab_default_options(ttslab_options *opt);
ttslab *ttslab_create(void);
void ttslab_destroy(ttslab *t);

/* Add CMUdict-format pronunciations; returns words added or -1. */
int ttslab_add_dict(ttslab *t, const char *data, size_t len);
int ttslab_dict_words(ttslab *t);

int ttslab_engine_count(void);
const char *ttslab_engine_name(int engine);
int ttslab_engine_ready(ttslab *t, int engine);
/* Build an engine's voice now; returns seconds spent, or -1 on failure. */
double ttslab_prepare(ttslab *t, int engine);

/* 0 on success. Free the result with ttslab_result_free. */
int ttslab_speak(ttslab *t, const char *text, const ttslab_options *opt, ttslab_result *out);
void ttslab_result_free(ttslab_result *r);

/* Cost profile of all engines as a Markdown table; free with ttslab_string_free. */
char *ttslab_bench(ttslab *t, const ttslab_options *opt, int reps);
void ttslab_string_free(char *s);

/* dB spectrogram: 256 bins (0..8 kHz) per frame, frames every `hop` samples.
 * `out` must hold max_frames * 256 floats. Returns frames written. */
size_t ttslab_spectrogram(const float *samples, size_t n, int hop, float *out, size_t max_frames);

#ifdef __cplusplus
}
#endif
#endif
