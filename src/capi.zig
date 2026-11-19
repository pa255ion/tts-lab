//! C ABI over the synthesizer, for the Qt GUI (or any other front end).
//! See include/ttslab.h for the documented interface.

const std = @import("std");
const tts = @import("tts.zig");
const prosody = @import("prosody.zig");
const bench = @import("bench.zig");
const dsp = @import("dsp.zig");

const gpa = std.heap.smp_allocator;

pub const Options = extern struct {
    engine: c_int,
    f0_model: c_int,
    pitch_scale: f32,
    rate: f32,
    pitch_range: f32,
    formant_scale: f32,
    diphone_psola: c_int,
    unit_modify: c_int,
    hmm_mlpg: c_int,
    hmm_gv: c_int,
};

pub const Result = extern struct {
    samples: ?[*]f32,
    n_samples: usize,
    sample_rate: c_int,
    f0: ?[*]f32,
    n_f0: usize,
    f0_frame_rate: f32,
    transcription: ?[*:0]u8,
    audio_seconds: f64,
    synth_ms: f64,
    build_ms: f64,
    units: c_int,
    joins: c_int,
    fallbacks: c_int,
};

fn toOptions(o: *const Options) tts.Options {
    return .{
        .engine = @enumFromInt(@as(u8, @intCast(std.math.clamp(o.engine, 0, tts.Engine.count - 1)))),
        .prosody = .{
            .f0_model = @enumFromInt(@as(u2, @intCast(std.math.clamp(o.f0_model, 0, 2)))),
            .rate = o.rate,
            .pitch_range = o.pitch_range,
        },
        .pitch_scale = o.pitch_scale,
        .formant_scale = o.formant_scale,
        .diphone_psola = o.diphone_psola != 0,
        .unit_modify = o.unit_modify != 0,
        .hmm_mlpg = o.hmm_mlpg != 0,
        .hmm_gv = o.hmm_gv != 0,
    };
}

export fn ttslab_default_options(o: *Options) void {
    const d: tts.Options = .{};
    o.* = .{
        .engine = @intFromEnum(d.engine),
        .f0_model = @intFromEnum(d.prosody.f0_model),
        .pitch_scale = d.pitch_scale,
        .rate = d.prosody.rate,
        .pitch_range = d.prosody.pitch_range,
        .formant_scale = d.formant_scale,
        .diphone_psola = @intFromBool(d.diphone_psola),
        .unit_modify = @intFromBool(d.unit_modify),
        .hmm_mlpg = @intFromBool(d.hmm_mlpg),
        .hmm_gv = @intFromBool(d.hmm_gv),
    };
}

export fn ttslab_create() ?*tts.Synth {
    return tts.Synth.init(gpa) catch null;
}

export fn ttslab_destroy(s: ?*tts.Synth) void {
    if (s) |x| x.deinit();
}

/// Add CMUdict-format text; returns the number of words added or -1.
export fn ttslab_add_dict(s: *tts.Synth, data: [*]const u8, len: usize) c_int {
    const n = s.lex.loadCmudict(data[0..len]) catch return -1;
    return @intCast(n);
}

/// Load a voice profile (text, see src/voice_profile.zig) for the refined
/// formant engine; len 0 goes back to the stock voice. 0 on success.
export fn ttslab_set_voice(s: *tts.Synth, data: [*]const u8, len: usize) c_int {
    s.setProfile(data[0..len]) catch return -1;
    return 0;
}

export fn ttslab_dict_words(s: *tts.Synth) c_int {
    return @intCast(s.lex.count());
}

const engine_names = blk: {
    var names: [tts.Engine.count][*:0]const u8 = undefined;
    for (0..tts.Engine.count) |i| {
        const e: tts.Engine = @enumFromInt(i);
        const l = e.label();
        names[i] = l[0..l.len :0];
    }
    break :blk names;
};

export fn ttslab_engine_count() c_int {
    return tts.Engine.count;
}

export fn ttslab_engine_name(i: c_int) [*:0]const u8 {
    if (i < 0 or i >= tts.Engine.count) return "?";
    return engine_names[@intCast(i)];
}

export fn ttslab_engine_ready(s: *tts.Synth, engine: c_int) c_int {
    return @intFromBool(switch (@as(tts.Engine, @enumFromInt(@as(u8, @intCast(std.math.clamp(engine, 0, tts.Engine.count - 1)))))) {
        .formant_classic, .formant_refined => true,
        .diphone => s.diphone_voice != null,
        .unit_selection => s.unit_voice != null,
        .hmm => s.hmm_voice != null,
    });
}

/// Build the voice an engine needs. Returns seconds spent, or -1.
export fn ttslab_prepare(s: *tts.Synth, engine: c_int) f64 {
    const e: tts.Engine = @enumFromInt(@as(u8, @intCast(std.math.clamp(engine, 0, tts.Engine.count - 1))));
    return s.ensure(e) catch -1;
}

export fn ttslab_speak(s: *tts.Synth, text: [*:0]const u8, o: *const Options, out: *Result) c_int {
    out.* = std.mem.zeroes(Result);
    var opt = toOptions(o);
    if (s.profile) |*p| opt.profile = p;
    const r = s.speak(gpa, std.mem.span(text), opt) catch return -1;
    const tr = gpa.dupeZ(u8, r.transcription) catch {
        r.deinit(gpa);
        return -1;
    };
    gpa.free(r.transcription);
    out.* = .{
        .samples = r.samples.ptr,
        .n_samples = r.samples.len,
        .sample_rate = 16000,
        .f0 = r.f0.ptr,
        .n_f0 = r.f0.len,
        .f0_frame_rate = prosody.frame_rate,
        .transcription = tr.ptr,
        .audio_seconds = r.audio_seconds,
        .synth_ms = r.synth_seconds * 1000,
        .build_ms = r.build_seconds * 1000,
        .units = @intCast(r.unit_stats.units),
        .joins = @intCast(r.unit_stats.joins),
        .fallbacks = @intCast(r.unit_stats.fallbacks),
    };
    return 0;
}

export fn ttslab_result_free(r: *Result) void {
    if (r.samples) |p| gpa.free(p[0..r.n_samples]);
    if (r.f0) |p| gpa.free(p[0..r.n_f0]);
    if (r.transcription) |p| gpa.free(std.mem.span(p));
    r.* = std.mem.zeroes(Result);
}

/// Run the cost profile; returns a Markdown table (free with
/// ttslab_string_free) or NULL.
export fn ttslab_bench(s: *tts.Synth, o: *const Options, reps: c_int) ?[*:0]u8 {
    const rows = bench.run(s, gpa, bench.bench_text, @intCast(@max(1, reps)), toOptions(o)) catch return null;
    const md = bench.report(gpa, &rows, s) catch return null;
    defer gpa.free(md);
    const z = gpa.dupeZ(u8, md) catch return null;
    return z.ptr;
}

export fn ttslab_string_free(p: ?[*:0]u8) void {
    if (p) |x| gpa.free(std.mem.span(x));
}

/// Log-magnitude spectrogram: 512-point frames every `hop` samples,
/// 256 bins (0..8 kHz) per frame, in dB, written to `out`
/// (capacity: frames * 256 floats). Returns the number of frames.
export fn ttslab_spectrogram(samples: [*]const f32, n: usize, hop: c_int, out: [*]f32, max_frames: usize) usize {
    const h: usize = @intCast(@max(1, hop));
    const frames = @min(max_frames, n / h + 1);
    var buf: [512]dsp.Complex = undefined;
    for (0..frames) |f| {
        const c: isize = @intCast(f * h);
        for (&buf, 0..) |*b, i| {
            const idx = c - 256 + @as(isize, @intCast(i));
            const x: f32 = if (idx >= 0 and idx < n) samples[@intCast(idx)] else 0;
            b.* = dsp.Complex.init(x * dsp.hann(512, i), 0);
        }
        dsp.fft(&buf, false);
        for (0..256) |k| {
            const m = buf[k].re * buf[k].re + buf[k].im * buf[k].im;
            out[f * 256 + k] = 10 * @log10(m + 1e-10);
        }
    }
    return frames;
}
