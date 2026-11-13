//! Diphone concatenation (MBROLA / Festival-diphone style).
//!
//! Build: record every phone pair as a nonsense carrier word in a monotone
//! (110 Hz), cut from the middle of the first phone to the middle of the
//! second, keep the audio and pitch marks. ~1,600 units.
//!
//! Synthesize: look up the unit for each adjacent pair of planned phones,
//! lay them end to end, and run TD-PSOLA so each half-phone gets its planned
//! duration and the whole utterance follows the planned F0 contour.
//! With PSOLA off you hear the raw inventory: monotone, inventory timing,
//! butt joins.

const std = @import("std");
const builtin = @import("builtin");
const dsp = @import("dsp.zig");
const phones = @import("phones.zig");
const Phone = phones.Phone;
const prosody = @import("prosody.zig");
const corpus = @import("corpus.zig");
const psola = @import("psola.zig");
const Mark = corpus.Mark;

pub const N = Phone.count;

pub const Unit = struct {
    /// The unit plus `margin` samples of context on each side.
    samples: []i16,
    marks: []Mark,
    /// The unit proper is samples[start..end]; `split` is the phone
    /// boundary. All indices are into `samples`.
    start: u32,
    split: u32,
    end: u32,

    pub fn len(u: Unit) u32 {
        return u.end - u.start;
    }

    pub fn piece(u: Unit) psola.Piece {
        return .{ .buf = u.samples, .start = u.start, .end = u.end, .marks = u.marks };
    }
};

/// Context kept around each unit so PSOLA grains near its edges see the
/// unit's own waveform, not its neighbour's.
const margin: u32 = 400;

pub const Options = struct {
    /// Modify pitch and duration to the plan (TD-PSOLA). Off = raw units.
    psola: bool = true,
};

/// How much of a pause is kept on each side of a unit that touches one.
const pau_keep: u32 = 480; // 30 ms

pub const Voice = struct {
    arena: std.heap.ArenaAllocator,
    units: [N * N]?Unit,
    bytes: usize = 0,
    build_seconds: f64 = 0,
    recorded_seconds: f64 = 0,

    pub fn unit(v: *const Voice, a: Phone, b: Phone) ?Unit {
        return v.units[@as(usize, @intFromEnum(a)) * N + @intFromEnum(b)];
    }

    pub fn count(v: *const Voice) usize {
        var n: usize = 0;
        for (v.units) |u| if (u != null) {
            n += 1;
        };
        return n;
    }

    pub fn deinit(v: *Voice) void {
        v.arena.deinit();
    }

    /// Record and cut the whole inventory, using `threads` workers.
    pub fn build(gpa: std.mem.Allocator, threads: usize) !*Voice {
        const t0 = dsp.now();
        const v = try gpa.create(Voice);
        errdefer gpa.destroy(v);
        v.* = .{ .arena = .init(gpa), .units = @splat(null) };
        errdefer v.arena.deinit();

        const Worker = struct {
            fn run(voice: *Voice, worker: usize, n_workers: usize, results: []?Unit, err: *?anyerror, secs: *f64) void {
                var arena_state = std.heap.ArenaAllocator.init(dsp.heap);
                defer arena_state.deinit();
                var idx = worker;
                while (idx < N * N) : (idx += n_workers) {
                    const a: Phone = @enumFromInt(idx / N);
                    const b: Phone = @enumFromInt(idx % N);
                    if (a == .pau and b == .pau) continue;
                    _ = arena_state.reset(.retain_capacity);
                    const r = recordUnit(arena_state.allocator(), dsp.heap, a, b, @intCast(idx + 1), secs) catch |e| {
                        err.* = e;
                        return;
                    };
                    results[idx] = r;
                }
                _ = voice;
            }
        };
        const n_workers = @max(1, @min(threads, 32));
        var errs: [32]?anyerror = @splat(null);
        var secs: [32]f64 = @splat(0);
        var results: [N * N]?Unit = @splat(null);
        if (builtin.single_threaded) {
            for (0..n_workers) |w| Worker.run(v, w, n_workers, &results, &errs[w], &secs[w]);
        } else {
            var handles: [32]?std.Thread = @splat(null);
            for (0..n_workers) |w| {
                handles[w] = std.Thread.spawn(.{}, Worker.run, .{ v, w, n_workers, &results, &errs[w], &secs[w] }) catch null;
                if (handles[w] == null) Worker.run(v, w, n_workers, &results, &errs[w], &secs[w]);
            }
            for (handles[0..n_workers]) |h| if (h) |th| th.join();
        }
        for (errs[0..n_workers]) |e| if (e) |er| return er;

        // Move units into this voice's arena (workers used the global heap).
        const a = v.arena.allocator();
        for (results, 0..) |r, i| {
            const u = r orelse continue;
            const s = try a.dupe(i16, u.samples);
            const m = try a.dupe(Mark, u.marks);
            dsp.heap.free(u.samples);
            dsp.heap.free(u.marks);
            v.units[i] = .{ .samples = s, .marks = m, .start = u.start, .split = u.split, .end = u.end };
            v.bytes += s.len * 2 + m.len * @sizeOf(Mark) + @sizeOf(Unit);
        }
        for (secs[0..n_workers]) |s| v.recorded_seconds += s;
        v.build_seconds = dsp.now() - t0;
        return v;
    }

    pub fn synthesize(v: *const Voice, alloc: std.mem.Allocator, utt: *const prosody.Utterance, opt: Options) ![]f32 {
        const segs = utt.segs;
        const sr = dsp.sample_rate;

        if (!opt.psola) {
            // Raw concatenation: inventory durations, monotone pitch, butt
            // joins, plus the planned pauses as silence.
            var out: std.ArrayList(f32) = .empty;
            for (0..segs.len - 1) |i| {
                const u = v.unit(segs[i].phone, segs[i + 1].phone) orelse continue;
                if (segs[i].phone == .pau) try out.appendNTimes(alloc, 0, @intFromFloat(@max(0, segs[i].dur * sr - 2 * pau_keep)));
                for (u.samples[u.start..u.end]) |x| try out.append(alloc, @as(f32, @floatFromInt(x)) / 32767.0);
            }
            return out.toOwnedSlice(alloc);
        }

        var pieces: std.ArrayList(psola.Piece) = .empty;
        defer pieces.deinit(alloc);
        var map: std.ArrayList(psola.Knot) = .empty;
        defer map.deinit(alloc);
        var base: f32 = 0;
        for (0..segs.len - 1) |i| {
            const a = segs[i];
            const b = segs[i + 1];
            const u = v.unit(a.phone, b.phone) orelse continue;
            try pieces.append(alloc, u.piece());
            // Target times of this unit's three anchor points: middle of a,
            // the a|b boundary, middle of b. Pause halves keep only
            // `pau_keep` samples of audio; the rest of a planned pause maps
            // onto those samples (PSOLA repeats near-silent grains).
            const t_mid_a = if (a.phone == .pau) a.end() - @as(f32, @floatFromInt(pau_keep)) / sr else a.start + a.dur / 2;
            const t_mid_b = if (b.phone == .pau) b.start + @as(f32, @floatFromInt(pau_keep)) / sr else b.start + b.dur / 2;
            try map.append(alloc, .{ .t = t_mid_a * sr, .a = base });
            try map.append(alloc, .{ .t = a.end() * sr, .a = base + @as(f32, @floatFromInt(u.split - u.start)) });
            base += @floatFromInt(u.len());
            try map.append(alloc, .{ .t = t_mid_b * sr, .a = base - 1 });
        }
        const out_len: usize = @intFromFloat(utt.duration * sr);
        return psola.run(alloc, pieces.items, map.items, out_len, .{ .f0 = utt.f0 });
    }
};

/// Carrier: [pau] (pad) a b (pad) [pau]. Pads give consonants a vowel to
/// attach to and vowels a consonant, so every unit is cut from a natural
/// transition in mid-word context.
fn recordUnit(arena: std.mem.Allocator, keep: std.mem.Allocator, a: Phone, b: Phone, seed: u32, secs: *f64) !Unit {
    var seq: [6]Phone = undefined;
    var n: usize = 0;
    seq[n] = .pau;
    n += 1;
    if (a != .pau) {
        seq[n] = if (a.isVowel()) .t else .aa;
        n += 1;
    }
    const ia = n;
    if (a != .pau) {
        seq[n] = a;
        n += 1;
    } else {
        // `a` is the leading pause itself.
    }
    const ib = n;
    if (b != .pau) {
        seq[n] = b;
        n += 1;
        seq[n] = if (b.isVowel()) .t else .aa;
        n += 1;
    }
    seq[n] = .pau;
    n += 1;
    const utt = try corpus.carrierUtterance(arena, seq[0..n], 110);
    const rec = try corpus.record(arena, &utt, seed, 0.3);
    secs.* += utt.duration;

    const la = if (a == .pau) rec.labels[0] else rec.labels[ia];
    const lb = if (b == .pau) rec.labels[n - 1] else rec.labels[ib];
    const start: u32 = if (a == .pau) la.end - pau_keep else la.mid();
    const end: u32 = if (b == .pau) lb.start + pau_keep else lb.mid();

    const lo = start -| margin;
    const hi = @min(@as(u32, @intCast(rec.samples.len)), end + margin);
    const samples = try keep.dupe(i16, rec.samples[lo..hi]);
    var marks: std.ArrayList(Mark) = .empty;
    for (rec.marks) |m| {
        if (m.pos >= lo and m.pos < hi) try marks.append(keep, .{ .pos = m.pos - lo, .voiced = m.voiced });
    }
    return .{ .samples = samples, .marks = try marks.toOwnedSlice(keep), .start = start - lo, .split = la.end - lo, .end = end - lo };
}
