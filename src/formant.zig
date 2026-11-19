//! Formant (source–filter) synthesis, in two flavours.
//!
//!  * classic — roughly what a late-70s/early-80s rule system did (MITalk,
//!    early Klatt): impulse-train glottal source, formant tracks made of
//!    straight lines between targets with stop loci, all parameters held
//!    constant for each 5 ms frame.
//!  * refined — the same synthesizer topology with the improvements that came
//!    later (Klatt 1987/KLSYN88, DECtalk, Holmes, and research on
//!    coarticulation): the Liljencrants–Fant (LF) glottal flow-derivative
//!    model with voice-quality (Rd) control, jitter/shimmer/flutter,
//!    pitch-synchronous aspiration and frication, open-phase F1 damping,
//!    a dominance-function coarticulation model (Cohen & Massaro 1993)
//!    instead of straight lines, smooth per-sample parameter interpolation,
//!    stress-dependent amplitude and vowel reduction, microprosody.
//!
//! Synthesizer topology (both modes), after Klatt 1980:
//!
//!   voicing ─┐                                    ┌─ cascade: NP → NZ → F6..F1 ─┐
//!   aspiration ─┴─────────────────────────────────┘                            ├─► out
//!   frication ───────────────────────── parallel: 6 resonators + bypass ───────┘

const std = @import("std");
const dsp = @import("dsp.zig");
const phones = @import("phones.zig");
const Phone = phones.Phone;
const prosody = @import("prosody.zig");
const Utterance = prosody.Utterance;
const Seg = prosody.Seg;

pub const Mode = enum { classic, refined };

pub const Voice = struct {
    /// Multiplies the planned F0.
    pitch_scale: f32 = 1.0,
    /// Multiplies every formant: ~1.0 adult male, ~1.17 adult female, ~1.3 child.
    formant_scale: f32 = 1.0,
    /// Random per-segment target perturbation (0 = none). Used when the
    /// formant engine acts as "voice talent" for the recorded corpora.
    variability: f32 = 0,
    seed: u32 = 1,
};

pub const Frame = struct {
    f0: f32 = 100,
    /// Voicing, aspiration and frication amplitudes (linear).
    av: f32 = 0,
    ah: f32 = 0,
    af: f32 = 0,
    /// Gains of the parallel noise bank (+ bypass), see phones.noise_bank.
    fg: [phones.n_noise]f32 = @splat(0),
    f: [3]f32 = .{ 500, 1500, 2500 },
    b: [3]f32 = .{ 60, 90, 150 },
    /// Nasal zero; 270 = cancels the nasal pole = not nasal.
    fnz: f32 = 270,
    /// LF voice-quality parameter (refined only): ~0.7 tense, 1.1 modal, 2+ breathy.
    rd: f32 = 1.1,
};

pub const samples_per_frame: usize = @intFromFloat(dsp.sample_rate / prosody.frame_rate);

/// Fixed upper formants (Hz, bandwidth).
const f4 = [2]f32{ 3500, 250 };
const f5 = [2]f32{ 4500, 350 };
const f6 = [2]f32{ 5500, 600 };
/// Refined only: two more poles standing in for the vocal tract's formants
/// above F6 (Klatt's "higher-pole correction"). Without them a cascade of
/// DC-normalized resonators falls off a cliff above the last formant, and
/// vowels came out 25-40 dB below natural speech at 5-7 kHz: muffled.
const f7 = [2]f32{ 6500, 800 };
const f8 = [2]f32{ 7200, 1500 };
const nasal_pole = [2]f32{ 270, 100 };

// Level calibration between the voiced (cascade) path and the noise paths;
// tuned by measuring segment RMS (see `ttslab analyze`).
const classic_voice_gain: f32 = 0.55;
const refined_voice_gain: f32 = 0.5;
const asp_gain: f32 = 0.18;
const fric_gain: f32 = 0.36;

pub const Output = struct {
    samples: []f32,
    /// Glottal closure instants (sample index) — the pitch marks a PSOLA
    /// system would otherwise have to estimate.
    gci: []u32,
    /// Whether voicing was on at each GCI.
    gci_voiced: []bool,
};

pub fn synthesize(alloc: std.mem.Allocator, utt: *const Utterance, mode: Mode, voice: Voice) !Output {
    const frames = try makeFrames(alloc, utt, mode, voice);
    defer alloc.free(frames);
    return render(alloc, frames, mode, voice);
}

// ===========================================================================
// Rules: Utterance -> parameter frames
// ===========================================================================

const n_track = 7; // F1 F2 F3 B1 B2 B3 FNZ

const Target = struct {
    s: [n_track]f32, // at segment start
    e: [n_track]f32, // at segment end (differs for diphthongs)
};

fn segTargets(alloc: std.mem.Allocator, segs: []const Seg, voice: Voice) ![]Target {
    const tg = try alloc.alloc(Target, segs.len);
    var rng = dsp.Rng.init(voice.seed *% 2654435761 +% 17);
    for (segs, 0..) |s, i| {
        const inf = s.phone.info();
        var fs = inf.f;
        var fe = inf.f_end;
        if (s.phone.isVowel() and s.stress == 0) {
            // Vowel reduction: unstressed vowels drift toward schwa.
            const k: f32 = if (s.phone == .ah) 1.0 else 0.3;
            for (0..3) |j| {
                fs[j] += (phones.schwa[j] - fs[j]) * k;
                fe[j] += (phones.schwa[j] - fe[j]) * k;
            }
        }
        if (inf.place == .velar and (inf.class == .stop or inf.class == .nasal)) {
            // Velar "pinch": the locus depends on the neighbouring vowel.
            if (neighbourVowel(segs, i)) |nv| {
                const vf2 = nv.info().f[1];
                const f2 = std.math.clamp(@max(vf2, 1300) + 250, 1500, 2400);
                fs[1] = f2;
                fs[2] = f2 + 400;
                fe = fs;
            }
        }
        if (voice.variability > 0) {
            for (0..3) |j| {
                const m = 1 + voice.variability * 0.03 * rng.gauss();
                fs[j] *= m;
                fe[j] *= m;
            }
        }
        tg[i] = .{
            .s = .{ fs[0], fs[1], fs[2], inf.b[0], inf.b[1], inf.b[2], inf.fnz },
            .e = .{ fe[0], fe[1], fe[2], inf.b[0], inf.b[1], inf.b[2], inf.fnz },
        };
    }
    return tg;
}

fn neighbourVowel(segs: []const Seg, i: usize) ?Phone {
    if (i + 1 < segs.len and segs[i + 1].phone.isVowel()) return segs[i + 1].phone;
    if (i > 0 and segs[i - 1].phone.isVowel()) return segs[i - 1].phone;
    return null;
}

pub fn makeFrames(alloc: std.mem.Allocator, utt: *const Utterance, mode: Mode, voice: Voice) ![]Frame {
    const segs = utt.segs;
    const n = utt.f0.len;
    const frames = try alloc.alloc(Frame, n);
    errdefer alloc.free(frames);
    @memset(frames, .{});

    const tg = try segTargets(alloc, segs, voice);
    defer alloc.free(tg);

    // ---- formant / bandwidth / nasal tracks
    var tracks: [n_track][]f32 = undefined;
    for (&tracks) |*t| t.* = try alloc.alloc(f32, n);
    defer for (tracks) |t| alloc.free(t);
    switch (mode) {
        .classic => try classicTracks(alloc, segs, tg, &tracks),
        .refined => try dominanceTracks(alloc, segs, tg, &tracks),
    }

    // Phrase-final region boundaries, for final lengthening of voice quality.
    const pf_start = try alloc.alloc(f32, segs.len);
    defer alloc.free(pf_start);
    const pf_end = try alloc.alloc(f32, segs.len);
    defer alloc.free(pf_end);
    {
        var i: usize = 0;
        while (i < segs.len) {
            if (!segs[i].phrase_final) {
                i += 1;
                continue;
            }
            var j = i;
            while (j < segs.len and segs[j].phrase_final) j += 1;
            for (i..j) |k| {
                pf_start[k] = segs[i].start;
                pf_end[k] = segs[j - 1].end();
            }
            i = j;
        }
    }

    // ---- source parameters
    var si: usize = 0;
    for (frames, 0..) |*fr, fi| {
        const t = @as(f32, @floatFromInt(fi)) / prosody.frame_rate;
        while (si + 1 < segs.len and t >= segs[si].end()) si += 1;
        const s = segs[si];
        for (0..3) |j| {
            fr.f[j] = tracks[j][fi];
            fr.b[j] = tracks[3 + j][fi];
        }
        fr.fnz = tracks[6][fi];
        fr.f0 = utt.f0[fi] * voice.pitch_scale;
        sourceParams(fr, segs, si, t, mode, pf_start[si], pf_end[si]);
        _ = s;
    }

    if (mode == .refined) {
        // Vocal effort rises with pitch: accents get louder, and loudness
        // declines along with F0 over a phrase. ~0.55 dB per semitone, from
        // the voicing amplitude only (noise sources keep their level).
        // Without it vowel levels varied half as much as in recordings.
        var sum: f64 = 0;
        var cnt: usize = 0;
        for (frames) |f| if (f.av > 0.05) {
            sum += @log(f.f0);
            cnt += 1;
        };
        if (cnt > 0) {
            const ref = @exp(@as(f32, @floatCast(sum / @as(f64, @floatFromInt(cnt)))));
            for (frames) |*f| f.av *= std.math.clamp(std.math.pow(f32, f.f0 / ref, 1.1), 0.6, 1.6);
        }
    }

    return frames;
}

fn sourceParams(fr: *Frame, segs: []const Seg, si: usize, t: f32, mode: Mode, pfs: f32, pfe: f32) void {
    const s = segs[si];
    const inf = s.phone.info();
    const refined = mode == .refined;
    const local = t - s.start;
    // Rd baselines are tuned against natural read speech (CMU ARCTIC bdl):
    // H1-H2 about -1 dB in vowels, close to the recordings' -3 dB.
    var rd: f32 = if (refined) 0.8 else 1.1;

    switch (inf.class) {
        .silence => {},
        .vowel, .diphthong => {
            if (refined) {
                fr.av = if (s.stress == 0 or s.function_word) 0.72 else if (s.accent) 1.1 else 1.0;
                rd = if (s.stress == 0 or s.function_word) 0.9 else if (s.accent) 0.6 else 0.7;
            } else {
                fr.av = if (s.stress == 0) 0.85 else 1.0;
            }
        },
        .liquid, .glide => fr.av = if (refined) 0.8 else 0.9,
        .nasal => fr.av = if (refined) 0.8 else 0.9,
        .fricative => {
            fr.af = inf.fric_amp;
            fr.fg = if (refined) tiedShape(s.phone, segs, si, false) else inf.fric;
            // Refined: /f v th/ were 5-10 dB below their level in recordings.
            if (refined and (inf.place == .labial or inf.place == .dental)) fr.af *= 1.8;
            if (inf.voiced) {
                fr.av = 0.45;
                rd = if (refined) 1.1 else 1.5;
            }
            if (refined) {
                // Frication builds up and decays rather than switching.
                const ramp = std.math.clamp(@min(local, s.dur - local) / 0.02, 0.3, 1.0);
                fr.af *= ramp;
            }
        },
        .aspirate => {
            fr.ah = 0.5;
            if (refined) {
                const ramp = std.math.clamp(@min(local, s.dur - local) / 0.015, 0.3, 1.0);
                fr.ah *= ramp;
            }
        },
        .stop, .affricate => {
            const is_aff = inf.class == .affricate;
            const burst_dur: f32 = switch (inf.place) {
                .labial => 0.008,
                .velar => 0.018,
                else => 0.012,
            };
            // Affricates: a short closure, a burst, then a long frication
            // phase that carries most of the perceptual cue.
            const close_end = if (is_aff) s.dur * 0.3 else @max(s.dur - burst_dur, s.dur * 0.5);
            if (inf.voiced) {
                fr.av = if (refined) 0.12 else 0.1;
                rd = if (refined) 1.2 else 1.6;
            }
            if (local >= close_end) {
                if (is_aff and local >= close_end + 0.008) {
                    if (inf.voiced) fr.av = 0.4;
                    fr.af = inf.fric_amp;
                    fr.fg = if (refined) tiedShape(s.phone, segs, si, false) else inf.fric;
                } else {
                    fr.af = if (inf.voiced) 0.4 else 0.7;
                    // Refined, levels against recordings: alveolar bursts
                    // were ~4 dB too strong, labial and voiced velar ones
                    // 3-6 dB too weak.
                    if (refined) {
                        if (inf.place == .alveolar and !inf.voiced) fr.af = 0.5;
                        if (inf.voiced and inf.place != .alveolar) fr.af = 0.6;
                        if (!inf.voiced and inf.place == .labial) fr.af = 1.0;
                    }
                    fr.fg = if (refined) tiedShape(s.phone, segs, si, true) else inf.burst;
                }
            }
        },
    }

    // Aspiration after a released voiceless stop: the following sonorant
    // begins with its formants excited by noise, voicing comes in late (VOT).
    if (s.phone.isSonorant() and si > 0) {
        const prev = segs[si - 1].phone;
        const pinf = prev.info();
        const after_s = si >= 2 and segs[si - 2].phone == .s;
        if (pinf.vot_ms > 0 and !after_s) {
            const vot = @min(pinf.vot_ms / 1000.0, s.dur * 0.5);
            if (local < vot) {
                const k = local / vot;
                fr.ah = 0.45 * (1 - 0.3 * k);
                if (refined) {
                    // Release frication: the burst transient is followed by
                    // turbulence at the constriction (10-20 ms, longest for
                    // alveolars and velars) before the glottal aspiration
                    // takes over. Without it the release is a click
                    // followed by /h/.
                    const fric_s: f32 = switch (pinf.place) {
                        .labial => 0.006,
                        .velar => 0.02,
                        else => 0.018,
                    };
                    if (local < fric_s) {
                        const q = local / fric_s;
                        fr.af = 0.55 * (1 - q);
                        fr.fg = tiedShape(prev, segs, si - 1, true);
                        fr.ah *= 0.4 + 0.6 * q;
                    }
                }
                fr.av = if (refined) fr.av * std.math.clamp((k - 0.7) / 0.3, 0, 1) else 0;
            }
        }
    }

    if (refined) {
        if (s.phrase_final and pfe > pfs and fr.av > 0) {
            // Phrase-final: voice gets laxer and breathier as it fades.
            const prog = std.math.clamp((t - pfs) / (pfe - pfs), 0, 1);
            fr.av *= 1 - 0.35 * prog;
            rd += 0.8 * prog;
        }
        // Breathiness rides on the voiced source.
        if (fr.av > 0.05) fr.ah = @max(fr.ah, 0.06 + 0.06 * @max(rd - 0.7, 0));
        // Microprosody: F0 is perturbed by the consonant before a vowel.
        if (s.phone.isVowel() and si > 0 and local < 0.05) {
            const p = segs[si - 1].phone.info();
            if (segs[si - 1].phone.isObstruent()) {
                const k = 1 - local / 0.05;
                fr.f0 *= if (p.voiced) 1 - 0.04 * k else 1 + 0.07 * k;
            }
        }
        if (s.phone.isObstruent() and inf.voiced) fr.f0 *= 0.95;
    }
    fr.rd = std.math.clamp(rd, 0.4, 2.7);
}

// ---- refined: parallel branch tied to the formants -----------------------
//
// Klatt (1980) excites frication and bursts through parallel resonators at
// the *current* formant frequencies F2..F6 (plus a bypass), with a gain per
// formant per consonant. That matters most for velars, whose burst is a
// compact peak sitting on F2 (back vowels) or F3 (front vowels) of the next
// vowel — a fixed bank can't follow it. Classic mode keeps the simpler fixed
// noise bank. Slots: F2, F3, F4, F5, 5.5 kHz, 6.8 kHz, bypass.

pub const tied_bw = [6]f32{ 200, 300, 400, 600, 1000, 1400 };

fn tiedShape(p: Phone, segs: []const Seg, si: usize, burst: bool) [phones.n_noise]f32 {
    const sib_s = [phones.n_noise]f32{ 0, 0, 0.05, 0.4, 1.0, 0.4, 0.02 };
    const sib_sh = [phones.n_noise]f32{ 0.15, 1.0, 0.5, 0.35, 0.15, 0.05, 0 };
    const labial = [phones.n_noise]f32{ 0, 0, 0, 0, 0.05, 0.1, 1.0 };
    const dental = [phones.n_noise]f32{ 0, 0, 0, 0.05, 0.2, 0.3, 0.8 };
    const inf = p.info();
    if (burst) return switch (inf.place) {
        .labial => .{ 0.1, 0.05, 0, 0, 0, 0, 1.0 },
        .velar => blk: {
            const nv = neighbourVowel(segs, si) orelse .ah;
            break :blk if (nv.info().f[1] < 1600)
                .{ 1.0, 0.4, 0.1, 0, 0, 0, 0.05 }
            else
                .{ 0.3, 1.0, 0.4, 0.1, 0, 0, 0.05 };
        },
        else => .{ 0, 0.1, 0.4, 0.8, 1.0, 0.25, 0.1 },
    };
    return switch (p) {
        .s, .z => sib_s,
        .sh, .zh, .ch, .jh => sib_sh,
        .f, .v => labial,
        .th, .dh => dental,
        else => inf.fric,
    };
}

// ---- classic: straight lines between targets, loci for stops ------------

const Knot = struct { t: f32, v: f32 };

fn classicTracks(alloc: std.mem.Allocator, segs: []const Seg, tg: []const Target, tracks: *[n_track][]f32) !void {
    var knots: std.ArrayList(Knot) = .empty;
    defer knots.deinit(alloc);
    for (0..n_track) |p| {
        knots.clearRetainingCapacity();
        for (segs, 0..) |s, i| {
            const cls = s.phone.info().class;
            switch (cls) {
                .silence => {},
                .aspirate => {
                    // /h/ is the next vowel, whispered.
                    if (i + 1 < segs.len and segs[i + 1].phone != .pau) {
                        try knots.append(alloc, .{ .t = s.start, .v = tg[i + 1].s[p] });
                    }
                },
                .stop, .affricate => {
                    try knots.append(alloc, .{ .t = s.start, .v = locusBlend(segs, tg, i, i -% 1, p) });
                    try knots.append(alloc, .{ .t = s.end(), .v = locusBlend(segs, tg, i, i + 1, p) });
                },
                else => {
                    const tr = @min(0.035, s.dur * 0.3);
                    try knots.append(alloc, .{ .t = s.start + tr, .v = tg[i].s[p] });
                    try knots.append(alloc, .{ .t = s.end() - tr, .v = tg[i].e[p] });
                },
            }
        }
        const out = tracks[p];
        if (knots.items.len == 0) {
            @memset(out, tg[0].s[p]);
            continue;
        }
        var k: usize = 0;
        for (out, 0..) |*v, fi| {
            const t = @as(f32, @floatFromInt(fi)) / prosody.frame_rate;
            while (k + 1 < knots.items.len and knots.items[k + 1].t <= t) k += 1;
            const a = knots.items[k];
            if (t <= a.t or k + 1 >= knots.items.len) {
                v.* = a.v;
                continue;
            }
            const b = knots.items[k + 1];
            const span = b.t - a.t;
            v.* = if (span <= 1e-6) b.v else a.v + (b.v - a.v) * (t - a.t) / span;
        }
    }
}

/// Formant value at the edge of a stop facing segment `j`:
/// locus + k * (neighbour target - locus).
fn locusBlend(segs: []const Seg, tg: []const Target, i: usize, j: usize, p: usize) f32 {
    const locus = tg[i].s[p];
    if (j >= segs.len or !segs[j].phone.isSonorant()) return locus;
    const nb = if (j < i) tg[j].e[p] else tg[j].s[p];
    return switch (p) {
        0 => 250, // F1 is always low at a closure
        1, 2 => locus + 0.5 * (nb - locus),
        else => nb, // bandwidths and nasality: the neighbour's
    };
}

// ---- refined: dominance-function coarticulation -------------------------
//
// Every segment contributes a "gesture" with a target and a dominance
// function D(t) = alpha * exp(-theta * max(0, |t - c| - plateau)). The value
// of a parameter at time t is the dominance-weighted mean of all targets:
//   P(t) = sum D_s(t) T_s / sum D_s(t).
// Strong, short consonants pull briefly and sharply; vowels pull broadly, so
// vowel-to-vowel coarticulation across consonants falls out automatically,
// and nasality spreads into neighbouring vowels because nasals dominate the
// nasal-zero parameter.

const Gesture = struct {
    c: f32,
    plateau: f32,
    theta: f32,
    theta_n: f32,
    a_f1: f32,
    a_f: f32,
    a_n: f32,
    tgt: [n_track]f32,
};

fn dominanceTracks(alloc: std.mem.Allocator, segs: []const Seg, tg: []const Target, tracks: *[n_track][]f32) !void {
    var gs: std.ArrayList(Gesture) = .empty;
    defer gs.deinit(alloc);
    for (segs, 0..) |s, i| {
        const cls = s.phone.info().class;
        const d = s.dur;
        var g: Gesture = .{ .c = s.start + d / 2, .plateau = 0.3 * d, .theta = 1.0 / 0.05, .theta_n = 1.0 / 0.05, .a_f1 = 1, .a_f = 1, .a_n = 0.3, .tgt = tg[i].s };
        switch (cls) {
            .vowel => {},
            .diphthong => {
                var g1 = g;
                g1.c = s.start + 0.2 * d;
                g1.plateau = 0.1 * d;
                try gs.append(alloc, g1);
                g.c = s.start + 0.8 * d;
                g.plateau = 0.1 * d;
                g.tgt = tg[i].e;
            },
            .stop, .affricate => {
                g.plateau = 0.4 * d;
                g.theta = 1.0 / 0.018;
                g.theta_n = g.theta;
                g.a_f1 = 1.0;
                g.a_f = if (cls == .affricate) 0.6 else 0.45;
                g.a_n = 0.5;
            },
            .fricative => {
                g.plateau = 0.35 * d;
                g.theta = 1.0 / 0.025;
                g.theta_n = g.theta;
                g.a_f1 = 0.6;
                g.a_f = 0.6;
                g.a_n = 0.5;
            },
            .nasal => {
                g.plateau = 0.35 * d;
                g.theta = 1.0 / 0.03;
                g.theta_n = 1.0 / 0.04;
                g.a_f1 = 0.8;
                g.a_f = 0.7;
                g.a_n = 1.0;
            },
            .liquid, .glide => {
                g.theta = 1.0 / 0.035;
                g.theta_n = g.theta;
                g.a_f1 = 0.8;
                g.a_f = 0.8;
                g.a_n = 0.5;
            },
            .aspirate, .silence => {
                g.plateau = 0.5 * d;
                g.theta = 1.0 / 0.08;
                g.theta_n = g.theta;
                g.a_f1 = 0.02;
                g.a_f = 0.02;
                g.a_n = 0.02;
            },
        }
        try gs.append(alloc, g);
    }

    var lo: usize = 0;
    const n = tracks[0].len;
    for (0..n) |fi| {
        const t = @as(f32, @floatFromInt(fi)) / prosody.frame_rate;
        while (lo < gs.items.len and gs.items[lo].c < t - 0.5) lo += 1;
        var num: [n_track]f32 = @splat(0);
        var den_f1: f32 = 0;
        var den_f: f32 = 0;
        var den_n: f32 = 0;
        var k = lo;
        while (k < gs.items.len and gs.items[k].c < t + 0.5) : (k += 1) {
            const g = gs.items[k];
            const dist = @max(0, @abs(t - g.c) - g.plateau);
            const e = @exp(-g.theta * dist);
            const en = @exp(-g.theta_n * dist);
            const w1 = g.a_f1 * e;
            const w = g.a_f * e;
            const wn = g.a_n * en;
            num[0] += w1 * g.tgt[0];
            for (1..6) |p| num[p] += w * g.tgt[p];
            num[6] += wn * g.tgt[6];
            den_f1 += w1;
            den_f += w;
            den_n += wn;
        }
        tracks[0][fi] = if (den_f1 > 0) num[0] / den_f1 else 500;
        for (1..6) |p| tracks[p][fi] = if (den_f > 0) num[p] / den_f else tg[0].s[p];
        tracks[6][fi] = if (den_n > 0) num[6] / den_n else 270;
    }
}

// ===========================================================================
// The synthesizer: frames -> samples
// ===========================================================================

/// Liljencrants–Fant glottal flow derivative for one cycle, parameterised by
/// Fant's Rd. Times are in samples; the negative peak (at te) is -1.
/// Return-phase length relative to Fant's Rd regression. The regression gives
/// a slow closure for a male voice at 16 kHz (the source falls ~10 dB/oct
/// above 1 kHz), and vowels came out 6-15 dB darker at 1-4 kHz than natural
/// speech; a crisper closure restores most of that without making the
/// voice pressed (H1-H2 barely moves).
const lf_ta_scale: f32 = 0.4;

/// F0 wander (refined): standard deviation (fraction of F0) and time constant (s).
const wander_sd: f32 = 0.025;
const wander_tau: f32 = 0.03;

const LF = struct {
    tp: f32,
    te: f32,
    ta: f32,
    tc: f32,
    wg: f32,
    eps: f32,
    alpha: f32,
    e0: f32,
    tail: f32,

    fn init(t0: f32, rd_in: f32) LF {
        const rd = std.math.clamp(rd_in, 0.3, 2.7);
        const rap = (-1 + 4.8 * rd) / 100;
        const rkp = (22.4 + 11.8 * rd) / 100;
        const rgp = 1 / (4 * ((0.11 * rd / (0.5 + 1.2 * rkp)) - rap) / rkp);
        const tp = t0 / (2 * rgp);
        const te = @min(tp * (1 + rkp), t0 * 0.97);
        const ta = @max(rap * t0 * lf_ta_scale, 0.5);
        const tc = t0;
        // Return phase: eps * ta = 1 - exp(-eps (tc - te)).
        var eps = 1 / ta;
        for (0..20) |_| eps = (1 - @exp(-eps * (tc - te))) / ta;
        const tail = @exp(-eps * (tc - te));
        const wg = std.math.pi / tp;
        // Area balance (zero net flow over the cycle) fixes alpha.
        const a2 = -(1 / (eps * ta)) * ((1 - tail) / eps - (tc - te) * tail);
        const sn = @sin(wg * te);
        const cs = @cos(wg * te);
        const f = struct {
            fn a1(al: f32, w: f32, tte: f32, s: f32, c: f32) f32 {
                return -(al * s - w * c + w * @exp(-al * tte)) / (s * (al * al + w * w));
            }
        }.a1;
        var lo: f32 = -0.1;
        var hi: f32 = 0.3;
        var flo = f(lo, wg, te, sn, cs) + a2;
        var alpha: f32 = 0;
        if (flo * (f(hi, wg, te, sn, cs) + a2) < 0) {
            for (0..40) |_| {
                const mid = 0.5 * (lo + hi);
                const fm = f(mid, wg, te, sn, cs) + a2;
                if ((fm < 0) == (flo < 0)) {
                    lo = mid;
                    flo = fm;
                } else hi = mid;
            }
            alpha = 0.5 * (lo + hi);
        }
        const e0 = -1 / (@exp(alpha * te) * sn);
        return .{ .tp = tp, .te = te, .ta = ta, .tc = tc, .wg = wg, .eps = eps, .alpha = alpha, .e0 = e0, .tail = tail };
    }

    inline fn eval(l: *const LF, t: f32) f32 {
        if (t < l.te) return l.e0 * @exp(l.alpha * t) * @sin(l.wg * t);
        if (t < l.tc) return -(1 / (l.eps * l.ta)) * (@exp(-l.eps * (t - l.te)) - l.tail);
        return 0;
    }
};

fn lerpFrame(a: Frame, b: Frame, k: f32) Frame {
    var r = a;
    const L = struct {
        inline fn l(x: f32, y: f32, kk: f32) f32 {
            return x + (y - x) * kk;
        }
    }.l;
    r.f0 = L(a.f0, b.f0, k);
    r.av = L(a.av, b.av, k);
    r.ah = L(a.ah, b.ah, k);
    // Frication and burst amplitudes are *held* per frame: interpolating
    // them would smear a 10 ms burst into a soft puff.
    for (0..3) |i| {
        r.f[i] = L(a.f[i], b.f[i], k);
        r.b[i] = L(a.b[i], b.b[i], k);
    }
    r.fnz = L(a.fnz, b.fnz, k);
    r.rd = L(a.rd, b.rd, k);
    return r;
}

pub fn render(alloc: std.mem.Allocator, frames: []const Frame, mode: Mode, voice: Voice) !Output {
    const spf = samples_per_frame;
    const n = (frames.len - 1) * spf;
    const out = try alloc.alloc(f32, n);
    errdefer alloc.free(out);
    var gci: std.ArrayList(u32) = .empty;
    errdefer gci.deinit(alloc);
    var gci_v: std.ArrayList(bool) = .empty;
    errdefer gci_v.deinit(alloc);

    const refined = mode == .refined;
    const fs = voice.formant_scale;
    var rng = dsp.Rng.init(voice.seed *% 747796405 +% 1);

    var r1: dsp.Resonator = .{};
    var r2: dsp.Resonator = .{};
    var r3: dsp.Resonator = .{};
    var r4: dsp.Resonator = .{};
    var r5: dsp.Resonator = .{};
    var r6: dsp.Resonator = .{};
    var r7: dsp.Resonator = .{};
    var r8: dsp.Resonator = .{};
    var np: dsp.Resonator = .{};
    var nz: dsp.AntiResonator = .{};
    // Refined: narrower F4/F5 (measured male values are ~200/250 Hz); with
    // Klatt's wider defaults the 3-5 kHz region came out 6+ dB below
    // natural recordings.
    r4.set(f4[0] * fs, if (refined) 200 else f4[1]);
    r5.set(f5[0] * fs, if (refined) 250 else f5[1]);
    r6.set(f6[0] * fs, f6[1]);
    r7.set(@min(f7[0] * fs, 7600), f7[1]);
    r8.set(@min(f8[0] * fs, 7700), f8[1]);
    np.set(nasal_pole[0] * fs, nasal_pole[1]);

    var byp_lp: f32 = 0;
    const byp_a: f32 = 1 - @exp(-2 * std.math.pi * 4000 / dsp.sample_rate);
    var bank: [phones.noise_bank.len]dsp.Resonator = undefined;
    for (&bank, phones.noise_bank) |*r, nb| r.setPeak(@min(nb.f * fs, 7600), nb.bw * fs);
    if (refined) {
        bank[4].setPeak(@min(5500 * fs, 7600), tied_bw[4]);
        bank[5].setPeak(@min(6800 * fs, 7600), tied_bw[5]);
    }

    // Classic glottal source: impulse -> low-pass "RGP" -> differentiate.
    var rgp: dsp.Resonator = .{};
    rgp.set(0, 100);
    var prev_g: f32 = 0;

    var phase: f32 = 0;
    var period: f32 = dsp.sample_rate / @max(frames[0].f0, 40);
    var lf = LF.init(period, frames[0].rd);
    var cycle_amp: f32 = 1;
    // Slow random F0 wander (AR(1) over glottal cycles), see below.
    var wander: f32 = 0;
    var open = true;
    var b1_open: f32 = 0;
    var t_sec: f32 = 0;
    const update_every: usize = if (refined) 16 else spf;
    var cur: Frame = frames[0];

    for (out, 0..) |*o, i| {
        if (i % update_every == 0) {
            const fi = i / spf;
            cur = if (refined)
                lerpFrame(frames[fi], frames[@min(fi + 1, frames.len - 1)], @as(f32, @floatFromInt(i % spf)) / @as(f32, @floatFromInt(spf)))
            else
                frames[fi];
            r1.set(cur.f[0] * fs, cur.b[0] + b1_open);
            r2.set(cur.f[1] * fs, cur.b[1]);
            r3.set(cur.f[2] * fs, cur.b[2]);
            nz.set(cur.fnz * fs, 100);
            if (refined and cur.af > 0.001) {
                bank[0].setPeak(cur.f[1] * fs, tied_bw[0]);
                bank[1].setPeak(cur.f[2] * fs, tied_bw[1]);
                bank[2].setPeak(f4[0] * fs, tied_bw[2]);
                bank[3].setPeak(f5[0] * fs, tied_bw[3]);
            }
        }
        t_sec += 1 / dsp.sample_rate;

        // ---- glottal cycle bookkeeping
        var impulse: f32 = 0;
        phase += 1;
        if (phase >= period) {
            phase -= period;
            var f0 = @max(cur.f0, 40);
            if (refined) {
                const flutter = (@sin(2 * std.math.pi * 12.7 * t_sec) + @sin(2 * std.math.pi * 7.1 * t_sec) + @sin(2 * std.math.pi * 4.7 * t_sec)) / 3;
                // Wander: natural F0 drifts by a fraction of a semitone
                // within a vowel (|dF0| per 10 ms mid-vowel: p90 0.5-0.7 st
                // in ARCTIC, 0.37 here without it). Period-to-period jitter
                // alone averages out over a few cycles, so this is low-pass
                // noise with a ~30 ms time constant. Not when recording the
                // corpora for the data-driven engines (variability > 0):
                // like a real voice talent, that voice is asked to keep a
                // steady pitch, and the diphone voice lost intelligibility
                // (ASR WER 15% -> 19%) when its units wandered.
                if (voice.variability == 0) {
                    const a = @exp(-period / dsp.sample_rate / wander_tau);
                    wander = a * wander + @sqrt(1 - a * a) * wander_sd * rng.gauss();
                }
                f0 *= 1 + 0.008 * flutter + 0.006 * rng.gauss() + wander; // flutter + jitter + wander
                cycle_amp = std.math.clamp(1 + 0.05 * rng.gauss(), 0.8, 1.2); // shimmer
            }
            period = dsp.sample_rate / f0;
            if (refined) {
                lf = LF.init(period, cur.rd);
                const at = i + @as(usize, @intFromFloat(lf.te));
                if (at < n) {
                    try gci.append(alloc, @intCast(at));
                    try gci_v.append(alloc, cur.av > 0.05);
                }
            } else {
                impulse = 1;
                try gci.append(alloc, @intCast(i));
                try gci_v.append(alloc, cur.av > 0.05);
            }
        }

        var v: f32 = 0;
        if (refined) {
            v = lf.eval(phase) * cycle_amp;
            const now_open = phase < lf.te;
            if (now_open != open) {
                // F1 is damped while the glottis is open (sub-glottal losses).
                open = now_open;
                b1_open = if (open) 40 else 0;
                r1.set(cur.f[0] * fs, cur.b[0] + b1_open);
            }
        } else {
            const g = rgp.step(impulse * 2400);
            v = g - prev_g;
            prev_g = g;
            open = phase < 0.6 * period;
        }

        const voiced = cur.av > 0.05;
        var asp = rng.gauss() * cur.ah * asp_gain;
        if (refined and voiced) asp *= if (open) 1.0 else 0.35;
        const vg = if (refined) refined_voice_gain else classic_voice_gain;
        const glot = v * cur.av * vg + asp;

        var x = nz.step(np.step(glot));
        if (refined) x = r6.step(r7.step(r8.step(x)));
        x = r5.step(x);
        x = r4.step(x);
        x = r3.step(x);
        x = r2.step(x);
        x = r1.step(x);

        var par: f32 = 0;
        if (cur.af > 0.001) {
            var nf = rng.gauss() * cur.af * fric_gain;
            if (refined and voiced) nf *= if (open) 1.0 else 0.5;
            var sign: f32 = 1;
            for (&bank, 0..) |*r, k| {
                par += sign * cur.fg[k] * r.step(nf);
                sign = -sign;
            }
            // The bypass is flat noise; in refined mode it is tilted down
            // above ~4 kHz, as lip/teeth noise and burst spectra are in
            // recordings (it otherwise put 13 dB too much energy at 8 kHz).
            var byp = nf;
            if (refined) {
                byp_lp += byp_a * (nf - byp_lp);
                byp = byp_lp;
            }
            par += cur.fg[phones.n_noise - 1] * byp * 0.25;
        } else {
            for (&bank) |*r| _ = r.step(0);
        }
        o.* = x + par;
    }

    return .{ .samples = out, .gci = try gci.toOwnedSlice(alloc), .gci_voiced = try gci_v.toOwnedSlice(alloc) };
}

pub fn freeOutput(alloc: std.mem.Allocator, o: Output) void {
    alloc.free(o.samples);
    alloc.free(o.gci);
    alloc.free(o.gci_voiced);
}

test "LF pulse integrates to ~zero and peaks at -1" {
    const lf = LF.init(160, 1.1);
    var sum: f32 = 0;
    var min: f32 = 0;
    var t: f32 = 0;
    while (t < 160) : (t += 0.01) {
        const v = lf.eval(t);
        sum += v * 0.01;
        min = @min(min, v);
    }
    try std.testing.expectApproxEqAbs(@as(f32, -1), min, 0.02);
    try std.testing.expect(@abs(sum) < 0.5);
}
