//! The "recording studio" shared by the data-driven engines.
//!
//! Real concatenative and HMM voices are built from hours of a human
//! speaker reading prompts, then labelled (forced alignment) and pitch-marked
//! (from an EGG or an epoch detector). This toy has no human, so the refined
//! formant engine plays the voice talent: it "reads" diphone carrier words and
//! corpus sentences with natural-ish variability, and we keep only the audio,
//! the phone labels and the pitch marks. From here on, the diphone,
//! unit-selection and HMM engines only see recordings — never formant
//! parameters.
//!
//! Consequence (see NOTES.md §7): these engines inherit the formant voice's
//! timbre, so the comparison isolates what each *method* does to prosody,
//! joins and smoothness, plus its cost — not the naturalness a human
//! recording would bring.

const std = @import("std");
const dsp = @import("dsp.zig");
const phones = @import("phones.zig");
const Phone = phones.Phone;
const prosody = @import("prosody.zig");
const formant = @import("formant.zig");
const lexicon = @import("lexicon.zig");

pub const Mark = struct { pos: u32, voiced: bool };

pub const Label = struct {
    phone: Phone,
    stress: u2,
    accent: bool,
    phrase_final: bool,
    start: u32, // samples
    end: u32,

    pub fn mid(l: Label) u32 {
        return (l.start + l.end) / 2;
    }
};

pub const Recording = struct {
    samples: []i16,
    marks: []Mark,
    labels: []Label,

    pub fn bytes(r: Recording) usize {
        return r.samples.len * 2 + r.marks.len * @sizeOf(Mark) + r.labels.len * @sizeOf(Label);
    }

    pub fn deinit(r: Recording, alloc: std.mem.Allocator) void {
        alloc.free(r.samples);
        alloc.free(r.marks);
        alloc.free(r.labels);
    }
};

/// Fixed recording gain, so every recording has the same level (essential
/// for concatenation — real studios fix mic gain for the same reason).
const record_gain: f32 = 0.6;

/// "Record" an utterance with the formant voice.
pub fn record(alloc: std.mem.Allocator, utt: *const prosody.Utterance, seed: u32, variability: f32) !Recording {
    const o = try formant.synthesize(alloc, utt, .refined, .{ .seed = seed, .variability = variability });
    defer formant.freeOutput(alloc, o);

    const samples = try alloc.alloc(i16, o.samples.len);
    errdefer alloc.free(samples);
    for (o.samples, samples) |x, *s| s.* = @intFromFloat(std.math.clamp(x * record_gain, -1, 1) * 32767);

    // Pitch marks: voiced GCIs, plus evenly spaced marks (every 5 ms) through
    // unvoiced stretches so PSOLA has grains everywhere.
    var marks: std.ArrayList(Mark) = .empty;
    errdefer marks.deinit(alloc);
    var last: u32 = 0;
    var first = true;
    for (o.gci, o.gci_voiced) |g, v| {
        if (!v) continue;
        // Only gaps longer than a plausible pitch period (25 ms) are unvoiced.
        if (first or g - last > 400) {
            var p = if (first) 40 else last + 80;
            while (p + 40 < g) : (p += 80) try marks.append(alloc, .{ .pos = p, .voiced = false });
        }
        try marks.append(alloc, .{ .pos = g, .voiced = true });
        last = g;
        first = false;
    }
    var p = if (first) 40 else last + 80;
    while (p + 40 < samples.len) : (p += 80) try marks.append(alloc, .{ .pos = p, .voiced = false });

    const labels = try alloc.alloc(Label, utt.segs.len);
    for (utt.segs, labels) |s, *l| {
        l.* = .{
            .phone = s.phone,
            .stress = s.stress,
            .accent = s.accent,
            .phrase_final = s.phrase_final,
            .start = @intFromFloat(s.start * dsp.sample_rate),
            .end = @min(@as(u32, @intCast(samples.len)), @as(u32, @intFromFloat(s.end() * dsp.sample_rate))),
        };
    }
    return .{ .samples = samples, .marks = try marks.toOwnedSlice(alloc), .labels = labels };
}

/// Build an Utterance directly from a phone list (for carrier words), with a
/// flat F0 and Klatt inherent durations — the monotone, careful style
/// diphone inventories are recorded in.
pub fn carrierUtterance(alloc: std.mem.Allocator, seq: []const Phone, f0: f32) !prosody.Utterance {
    const segs = try alloc.alloc(prosody.Seg, seq.len);
    var t: f32 = 0;
    for (seq, segs) |p, *s| {
        const inf = p.info();
        const d: f32 = if (p == .pau) 0.12 else inf.inh_ms / 1000.0;
        s.* = .{ .phone = p, .stress = if (p.isVowel()) 1 else 0, .start = t, .dur = d };
        t += d;
    }
    const n: usize = @intFromFloat(@ceil(t * prosody.frame_rate) + 1);
    const f0s = try alloc.alloc(f32, n);
    @memset(f0s, f0);
    return .{ .segs = segs, .words = &.{}, .f0 = f0s, .duration = t };
}

// ---------------------------------------------------------------------------
// Acoustic analysis: 5 ms frames of cepstrum, F0, voicing and aperiodicity.
// ---------------------------------------------------------------------------

pub const cep_order = 24;
pub const n_cep = cep_order + 1;
pub const fft_n = 512;
pub const win_len = 400; // 25 ms

pub const Frame = struct {
    cep: [n_cep]f32,
    f0: f32,
    voiced: bool,
    ap: f32,
};

/// Window energy constants for the analysis window, used to calibrate the
/// vocoder so analysis -> resynthesis preserves level.
pub const win_sum: f32 = blk: {
    @setEvalBranchQuota(10000);
    var s: f32 = 0;
    for (0..win_len) |i| s += 0.5 - 0.5 * @cos(2 * std.math.pi * @as(f32, @floatFromInt(i)) / win_len);
    break :blk s;
};
pub const win_sq_sum: f32 = blk: {
    @setEvalBranchQuota(10000);
    var s: f32 = 0;
    for (0..win_len) |i| {
        const w = 0.5 - 0.5 * @cos(2 * std.math.pi * @as(f32, @floatFromInt(i)) / win_len);
        s += w * w;
    }
    break :blk s;
};

pub fn analyze(alloc: std.mem.Allocator, rec: Recording) ![]Frame {
    const spf = 80;
    const n_frames = rec.samples.len / spf + 1;
    const frames = try alloc.alloc(Frame, n_frames);
    var buf: [fft_n]dsp.Complex = undefined;
    var x: [win_len]f32 = undefined;
    var mi: usize = 0;
    for (frames, 0..) |*fr, fi| {
        const center: isize = @intCast(fi * spf);
        for (&x, 0..) |*v, j| {
            const idx = center - win_len / 2 + @as(isize, @intCast(j));
            v.* = if (idx >= 0 and idx < rec.samples.len) @as(f32, @floatFromInt(rec.samples[@intCast(idx)])) / 32767.0 else 0;
        }
        fr.cep = cepstrum(&x, &buf);

        // F0 and voicing from the pitch marks around the frame centre.
        while (mi + 1 < rec.marks.len and rec.marks[mi + 1].pos < center) mi += 1;
        fr.voiced = false;
        fr.f0 = 0;
        if (mi + 1 < rec.marks.len) {
            const a = rec.marks[mi];
            const b = rec.marks[mi + 1];
            if (a.voiced and b.voiced and b.pos > a.pos and b.pos - a.pos < 400) {
                fr.voiced = true;
                fr.f0 = dsp.sample_rate / @as(f32, @floatFromInt(b.pos - a.pos));
            }
        }
        fr.ap = 1;
        if (fr.voiced) {
            // Aperiodicity: 1 - normalised autocorrelation at one period.
            const lag: usize = @intFromFloat(dsp.sample_rate / fr.f0);
            if (lag < win_len / 2) {
                var xy: f32 = 0;
                var xx: f32 = 0;
                var yy: f32 = 0;
                for (0..win_len - lag) |j| {
                    xy += x[j] * x[j + lag];
                    xx += x[j] * x[j];
                    yy += x[j + lag] * x[j + lag];
                }
                const r = xy / @sqrt(xx * yy + 1e-12);
                fr.ap = std.math.clamp(1 - r, 0.02, 1);
            }
        }
    }
    return frames;
}

/// Real cepstrum of a frame (Hann window, 512-point FFT), orders 0..24.
pub fn cepstrum(x: *const [win_len]f32, buf: *[fft_n]dsp.Complex) [n_cep]f32 {
    for (buf, 0..) |*b, i| {
        b.* = if (i < win_len) dsp.Complex.init(x[i] * dsp.hann(win_len, i), 0) else dsp.Complex.init(0, 0);
    }
    dsp.fft(buf, false);
    for (buf) |*b| b.* = dsp.Complex.init(@log(@sqrt(b.re * b.re + b.im * b.im) + 1e-5), 0);
    dsp.fft(buf, true);
    var c: [n_cep]f32 = undefined;
    for (&c, 0..) |*v, i| v.* = buf[i].re / fft_n;
    return c;
}

// ---------------------------------------------------------------------------
// Corpus prompts
// ---------------------------------------------------------------------------

/// Prompt sentences for the unit-selection / HMM corpus. Real corpora
/// (e.g. CMU ARCTIC, ~1100 sentences) are chosen greedily for diphone
/// coverage; this one is small and hand-written.
pub const prompts = [_][]const u8{
    "Hello world, this is a test of the speech synthesizer.",
    "The quick brown fox jumps over the lazy dog.",
    "She sells sea shells by the sea shore.",
    "How are you today?",
    "Is this a question?",
    "Please call me tomorrow morning.",
    "The weather is warm and the sun is high.",
    "We walked along the river to the old bridge.",
    "My friend has a small green garden near the city.",
    "Do you want coffee or tea?",
    "The teacher read a story to the children.",
    "There are seven days in a week.",
    "I think the machine can speak, but it sounds like a robot.",
    "Every word is made of sounds, and every sound has a shape.",
    "Where did you leave the book?",
    "The train leaves the station at nine o'clock.",
    "It was a cold night, so we stayed home.",
    "Thank you very much for the music.",
    "Can you hear the difference between these voices?",
    "A computer can read text and make speech from it.",
    "The red car is faster than the blue car.",
    "Why is the sky blue?",
    "They believe that the system will work.",
    "He found a little black cat under the table.",
    "Open the window and let the light in.",
    "Our family lives in a white house by the sea.",
    "What is your name?",
    "The ship sailed over the water for many hours.",
    "I need milk, bread, and sugar from the store.",
    "This is important, so please listen.",
    "The vowel and the consonant make a syllable.",
    "Speech synthesis uses rules, samples, or models.",
    "Good morning, and welcome to the lab.",
    "The boy and the girl played together in the garden.",
    "Five, six, seven, eight, nine, ten.",
    "Have you ever seen a yellow bird?",
    "The judge came to the church on Friday.",
    "It is usual to measure the vision of a child.",
    "That thin stone is thicker than it looks.",
    "Let's go back to the beginning.",
    "The program was simple, but the problem was hard.",
    "Natural speech is smooth, and the pitch moves with the words.",
    "Each engine makes a different sound.",
    "The neural network learned from the data.",
    "Will it rain tomorrow?",
    "I am happy, but she is sad.",
    "Read the letter again, slowly.",
    "The quality of the voice depends on the model.",
    "Is the engine loud or soft?",
    "The old man told a long story about the sea.",
};

/// The sentence corpus: every prompt read in several styles (pitch, rate,
/// intonation model), recorded and analysed. Shared by unit selection
/// (as its unit database) and the HMM engine (as training data).
pub const Corpus = struct {
    arena: std.heap.ArenaAllocator,
    recs: []Recording,
    frames: [][]Frame,
    audio_seconds: f64 = 0,
    build_seconds: f64 = 0,

    pub const Style = struct { base_f0: f32, rate: f32, model: prosody.F0Model, range: f32 };
    pub const styles = [_]Style{
        .{ .base_f0 = 110, .rate = 1.0, .model = .fujisaki, .range = 1.0 },
        .{ .base_f0 = 100, .rate = 0.9, .model = .hat, .range = 0.9 },
        .{ .base_f0 = 122, .rate = 1.1, .model = .fujisaki, .range = 1.3 },
    };

    pub fn audioBytes(c: *const Corpus) usize {
        var n: usize = 0;
        for (c.recs) |r| n += r.bytes();
        return n;
    }

    pub fn frameBytes(c: *const Corpus) usize {
        var n: usize = 0;
        for (c.frames) |f| n += f.len * @sizeOf(Frame);
        return n;
    }

    pub fn deinit(c: *Corpus) void {
        c.arena.deinit();
    }

    pub fn build(gpa: std.mem.Allocator, lex: *const lexicon.Lexicon, threads: usize) !*Corpus {
        const t0 = dsp.now();
        const c = try gpa.create(Corpus);
        errdefer gpa.destroy(c);
        c.* = .{ .arena = .init(gpa), .recs = &.{}, .frames = &.{} };
        errdefer c.arena.deinit();
        const a = c.arena.allocator();
        const n = prompts.len * styles.len;
        c.recs = try a.alloc(Recording, n);
        c.frames = try a.alloc([]Frame, n);

        const Job = struct {
            fn run(cc: *Corpus, l: *const lexicon.Lexicon, w: usize, nw: usize, err: *?anyerror) void {
                work(cc, l, w, nw) catch |e| {
                    err.* = e;
                };
            }
            fn work(cc: *Corpus, l: *const lexicon.Lexicon, w: usize, nw: usize) !void {
                var scratch = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
                defer scratch.deinit();
                var i = w;
                while (i < cc.recs.len) : (i += nw) {
                    _ = scratch.reset(.retain_capacity);
                    const st = styles[i % styles.len];
                    const utt = try prosody.plan(scratch.allocator(), l, prompts[i / styles.len], .{
                        .f0_model = st.model,
                        .base_f0 = st.base_f0,
                        .rate = st.rate,
                        .pitch_range = st.range,
                    });
                    const rec = try record(std.heap.smp_allocator, &utt, @intCast(1000 + i), 1.0);
                    cc.recs[i] = rec;
                    cc.frames[i] = try analyze(std.heap.smp_allocator, rec);
                }
            }
        };
        const nw = @max(1, @min(threads, 32));
        var errs: [32]?anyerror = @splat(null);
        var hs: [32]?std.Thread = @splat(null);
        for (0..nw) |w| {
            hs[w] = std.Thread.spawn(.{}, Job.run, .{ c, lex, w, nw, &errs[w] }) catch null;
            if (hs[w] == null) Job.run(c, lex, w, nw, &errs[w]);
        }
        for (hs[0..nw]) |h| if (h) |th| th.join();
        for (errs[0..nw]) |e| if (e) |er| return er;

        // Move into the corpus arena so one deinit frees everything.
        for (c.recs, c.frames) |*r, *f| {
            const moved: Recording = .{
                .samples = try a.dupe(i16, r.samples),
                .marks = try a.dupe(Mark, r.marks),
                .labels = try a.dupe(Label, r.labels),
            };
            r.deinit(std.heap.smp_allocator);
            r.* = moved;
            const mf = try a.dupe(Frame, f.*);
            std.heap.smp_allocator.free(f.*);
            f.* = mf;
            c.audio_seconds += @as(f64, @floatFromInt(r.samples.len)) / dsp.sample_rate;
        }
        c.build_seconds = dsp.now() - t0;
        return c;
    }
};

test "carrier record and analyse" {
    const a = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const ar = arena_state.allocator();
    const utt = try carrierUtterance(ar, &.{ .pau, .aa, .s, .t, .aa, .pau }, 110);
    const rec = try record(ar, &utt, 1, 0);
    try std.testing.expect(rec.marks.len > 20);
    const frames = try analyze(ar, rec);
    // Middle of the first /aa/ is voiced at ~110 Hz.
    const fi: usize = (rec.labels[1].mid()) / 80;
    try std.testing.expect(frames[fi].voiced);
    try std.testing.expectApproxEqAbs(@as(f32, 110), frames[fi].f0, 8);
}
