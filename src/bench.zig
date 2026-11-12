//! Cost profile: build time, stored voice size, synthesis speed and
//! working memory for each engine.

const std = @import("std");
const dsp = @import("dsp.zig");
const phones = @import("phones.zig");
const tts = @import("tts.zig");
const Engine = tts.Engine;

/// Wraps an allocator and tracks live and peak bytes. Single-threaded use.
pub const Counting = struct {
    child: std.mem.Allocator,
    live: usize = 0,
    peak: usize = 0,
    allocs: usize = 0,

    pub fn allocator(self: *Counting) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }

    pub fn reset(self: *Counting) void {
        self.peak = self.live;
        self.allocs = 0;
    }

    fn bump(self: *Counting, add: usize, sub: usize) void {
        self.live = self.live + add - sub;
        self.peak = @max(self.peak, self.live);
    }

    fn alloc(ctx: *anyopaque, len: usize, al: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *Counting = @ptrCast(@alignCast(ctx));
        const p = self.child.rawAlloc(len, al, ra) orelse return null;
        self.allocs += 1;
        self.bump(len, 0);
        return p;
    }

    fn resize(ctx: *anyopaque, mem: []u8, al: std.mem.Alignment, new_len: usize, ra: usize) bool {
        const self: *Counting = @ptrCast(@alignCast(ctx));
        if (!self.child.rawResize(mem, al, new_len, ra)) return false;
        self.bump(new_len, mem.len);
        return true;
    }

    fn remap(ctx: *anyopaque, mem: []u8, al: std.mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        const self: *Counting = @ptrCast(@alignCast(ctx));
        const p = self.child.rawRemap(mem, al, new_len, ra) orelse return null;
        self.bump(new_len, mem.len);
        return p;
    }

    fn free(ctx: *anyopaque, mem: []u8, al: std.mem.Alignment, ra: usize) void {
        const self: *Counting = @ptrCast(@alignCast(ctx));
        self.child.rawFree(mem, al, ra);
        self.bump(0, mem.len);
    }
};

pub const Row = struct {
    engine: Engine,
    /// Seconds to build the voice from scratch (including dependencies).
    build_s: f64,
    /// Seconds of audio "recorded" to build it.
    recorded_s: f64,
    /// Bytes the voice needs at synthesis time (what would ship on disk).
    voice_bytes: usize,
    synth_s: f64,
    audio_s: f64,
    /// Peak heap bytes allocated during one synthesis (including the output).
    peak_bytes: usize,
    allocs: usize,

    pub fn rtf(r: Row) f64 {
        return r.synth_s / r.audio_s;
    }
};

pub const bench_text =
    "The old lighthouse keeper climbed the stairs every evening. " ++
    "Could you see the ships from the top of the tower? " ++
    "On clear nights, he counted forty seven lights along the coast.";

/// Size of the formant engine's "voice": the rule tables.
pub const formant_rule_bytes = @sizeOf(@TypeOf(phones.table)) + @sizeOf(@TypeOf(phones.noise_bank));

pub fn run(synth: *tts.Synth, gpa: std.mem.Allocator, text: []const u8, reps: usize, base: tts.Options) ![Engine.count]Row {
    var rows: [Engine.count]Row = undefined;
    // Build everything first so build times are attributed cleanly.
    const t_d0 = dsp.now();
    const dv = try synth.diphoneVoice();
    const t_d = dsp.now() - t_d0;
    const t_c0 = dsp.now();
    const sc = try synth.sentenceCorpus();
    const t_c = dsp.now() - t_c0;
    const t_u0 = dsp.now();
    const uv = try synth.unitVoice();
    const t_u = dsp.now() - t_u0;
    const t_h0 = dsp.now();
    const hv = try synth.hmmVoice();
    const t_h = dsp.now() - t_h0;
    // If voices were already built (e.g. from the menu), use their own records.
    const d_build = if (t_d > 0.01) t_d else dv.build_seconds;
    const c_build = if (t_c > 0.01) t_c else sc.build_seconds;
    const u_build = if (t_u > 0.001) t_u else uv.build_seconds;
    const h_build = if (t_h > 0.001) t_h else hv.build_seconds;

    for (0..Engine.count) |ei| {
        const e: Engine = @enumFromInt(ei);
        var opt = base;
        opt.engine = e;
        var counting: Counting = .{ .child = gpa };
        const ca = counting.allocator();
        // Warm-up (also checks it works), then timed runs.
        (try synth.speak(ca, text, opt)).deinit(ca);
        var times: [32]f64 = undefined;
        const n = @min(reps, times.len);
        var audio: f64 = 0;
        counting.reset();
        for (0..n) |k| {
            const r = try synth.speak(ca, text, opt);
            times[k] = r.synth_seconds + r.plan_seconds;
            audio = r.audio_seconds;
            r.deinit(ca);
        }
        std.mem.sort(f64, times[0..n], {}, std.sort.asc(f64));
        const mem = synth.memory();
        rows[ei] = .{
            .engine = e,
            .build_s = switch (e) {
                .formant_classic, .formant_refined => 0,
                .diphone => d_build,
                .unit_selection => d_build + c_build + u_build,
                .hmm => c_build + h_build,
            },
            .recorded_s = switch (e) {
                .formant_classic, .formant_refined => 0,
                .diphone => dv.recorded_seconds,
                .unit_selection => dv.recorded_seconds + sc.audio_seconds,
                .hmm => sc.audio_seconds,
            },
            .voice_bytes = switch (e) {
                .formant_classic, .formant_refined => formant_rule_bytes,
                .diphone => mem.diphone.?,
                // Unit selection needs the corpus audio, its frames (join
                // costs), the unit index and the fallback inventory.
                .unit_selection => mem.corpus_audio.? + mem.corpus_frames.? + mem.unit_index.? + mem.diphone.?,
                .hmm => mem.hmm_models.?,
            },
            .synth_s = times[n / 2],
            .audio_s = audio,
            .peak_bytes = counting.peak,
            .allocs = counting.allocs / n,
        };
    }
    return rows;
}

pub fn fmtBytes(buf: []u8, b: usize) []const u8 {
    const f: f64 = @floatFromInt(b);
    return (if (b < 10 * 1024)
        std.fmt.bufPrint(buf, "{d} B", .{b})
    else if (b < 10 * 1024 * 1024)
        std.fmt.bufPrint(buf, "{d:.1} KiB", .{f / 1024})
    else
        std.fmt.bufPrint(buf, "{d:.1} MiB", .{f / (1024 * 1024)})) catch "?";
}

/// Markdown table of the results.
pub fn report(alloc: std.mem.Allocator, rows: []const Row, synth: *tts.Synth) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    const w = struct {
        fn p(o: *std.ArrayList(u8), a: std.mem.Allocator, comptime f: []const u8, args: anytype) !void {
            const s = try std.fmt.allocPrint(a, f, args);
            defer a.free(s);
            try o.appendSlice(a, s);
        }
    }.p;
    try w(&out, alloc, "| Engine | Build | Recorded audio | Voice data | Synth time | RTF | x real time | Peak heap / synth |\n", .{});
    try w(&out, alloc, "|---|---:|---:|---:|---:|---:|---:|---:|\n", .{});
    for (rows) |r| {
        var b1: [32]u8 = undefined;
        var b2: [32]u8 = undefined;
        try w(&out, alloc, "| {s} | {d:.2} s | {d:.0} s | {s} | {d:.1} ms | {d:.4} | {d:.0}x | {s} |\n", .{
            r.engine.label(),
            r.build_s,
            r.recorded_s,
            fmtBytes(&b1, r.voice_bytes),
            r.synth_s * 1000,
            r.rtf(),
            1 / r.rtf(),
            fmtBytes(&b2, r.peak_bytes),
        });
    }
    const m = synth.memory();
    var b: [32]u8 = undefined;
    try w(&out, alloc, "\nShared front end: lexicon of {d} words, {s}. Audio length of the test text: {d:.2} s. {d} threads for voice builds.\n", .{ m.lexicon_words, fmtBytes(&b, m.lexicon), rows[0].audio_s, synth.threads });
    var b3: [32]u8 = undefined;
    var b4: [32]u8 = undefined;
    var b5: [32]u8 = undefined;
    try w(&out, alloc, "Unit-selection breakdown: corpus audio+labels {s}, analysis frames {s}, unit index {s}; diphones in corpus {d}/{d}. HMM: {d} state models.\n", .{
        fmtBytes(&b3, m.corpus_audio orelse 0),
        fmtBytes(&b4, m.corpus_frames orelse 0),
        fmtBytes(&b5, m.unit_index orelse 0),
        if (synth.unit_voice) |u| u.coverage() else 0,
        m.diphone_units,
        m.hmm_states,
    });
    return out.toOwnedSlice(alloc);
}
