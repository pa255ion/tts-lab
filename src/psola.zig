//! TD-PSOLA: time-domain pitch-synchronous overlap-add
//! (Moulines & Charpentier 1990).
//!
//! Input: an analysis signal with pitch marks, a monotone time map from
//! target (output) time to analysis time, and optionally a target F0 track.
//! For each synthesis mark we find the analysis mark nearest the mapped
//! time, cut a Hann-windowed grain two analysis periods long centred on it,
//! and add it at the synthesis mark. Synthesis marks are spaced by the
//! *target* period, so pitch changes while the spectral envelope (vocal
//! tract) stays; the time map stretches or compresses duration by repeating
//! or skipping grains. With no target F0 and an identity map this is plain
//! pitch-synchronous resynthesis, which still smooths the joins between
//! concatenated units because neighbouring grains overlap.

const std = @import("std");
const dsp = @import("dsp.zig");
const Mark = @import("corpus.zig").Mark;

/// (target sample, analysis sample) pairs, increasing in both.
pub const Knot = struct { t: f32, a: f32 };

pub const Target = struct {
    /// F0 per 5 ms target frame; null keeps the analysis pitch.
    f0: ?[]const f32 = null,
    /// 80 samples per F0 frame at 16 kHz.
    samples_per_frame: f32 = 80,
};

/// A stretch of source audio to concatenate. Grain centres (pitch marks)
/// come only from [start, end), but grains may read samples outside it from
/// `buf`: that context is what keeps a grain near a unit edge from picking up
/// the *next* unit's waveform (a click).
pub const Piece = struct {
    buf: []const i16,
    start: u32,
    end: u32,
    /// All pitch marks of `buf`, sorted.
    marks: []const Mark,
};

const VMark = struct { v: f32, piece: u32, local: u32, voiced: bool, period: f32 };

fn mapToAnalysis(map: []const Knot, t: f32, hint: *usize) f32 {
    while (hint.* + 1 < map.len and map[hint.* + 1].t <= t) hint.* += 1;
    const a = map[hint.*];
    if (hint.* + 1 >= map.len) return a.a + (t - a.t);
    const b = map[hint.* + 1];
    const span = b.t - a.t;
    if (span <= 1e-6) return b.a;
    return a.a + (b.a - a.a) * (t - a.t) / span;
}

fn nearestMark(marks: []const VMark, a: f32, hint: *usize) usize {
    while (hint.* + 1 < marks.len and marks[hint.* + 1].v <= a) hint.* += 1;
    const k = hint.*;
    if (k + 1 < marks.len and @abs(marks[k + 1].v - a) < @abs(a - marks[k].v)) return k + 1;
    return k;
}

fn localPeriod(marks: []const Mark, k: usize) f32 {
    if (!marks[k].voiced) return 80;
    var lo = k;
    var hi = k;
    if (k > 0 and marks[k - 1].voiced) lo = k - 1;
    if (k + 1 < marks.len and marks[k + 1].voiced) hi = k + 1;
    if (hi == lo) return 80;
    const p = @as(f32, @floatFromInt(marks[hi].pos - marks[lo].pos)) / @as(f32, @floatFromInt(hi - lo));
    return std.math.clamp(p, 32, 400);
}

/// Total length of the virtual analysis signal (pieces end to end).
pub fn analysisLength(pieces: []const Piece) f32 {
    var n: f32 = 0;
    for (pieces) |p| n += @floatFromInt(p.end - p.start);
    return n;
}

/// `map` goes from target samples to *virtual analysis* samples: the
/// pieces laid end to end.
pub fn run(
    alloc: std.mem.Allocator,
    pieces: []const Piece,
    map: []const Knot,
    out_len: usize,
    target: Target,
) ![]f32 {
    const y = try alloc.alloc(f32, out_len);
    @memset(y, 0);
    var vm: std.ArrayList(VMark) = .empty;
    defer vm.deinit(alloc);
    var base: f32 = 0;
    for (pieces, 0..) |p, pi| {
        for (p.marks, 0..) |m, k| {
            if (m.pos < p.start or m.pos >= p.end) continue;
            try vm.append(alloc, .{
                .v = base + @as(f32, @floatFromInt(m.pos - p.start)),
                .piece = @intCast(pi),
                .local = m.pos,
                .voiced = m.voiced,
                .period = localPeriod(p.marks, k),
            });
        }
        base += @floatFromInt(p.end - p.start);
    }
    if (vm.items.len == 0 or map.len == 0) return y;

    var map_hint: usize = 0;
    var mark_hint: usize = 0;
    var t: f32 = 0;
    var rng = dsp.Rng.init(12345);
    while (t < @as(f32, @floatFromInt(out_len))) {
        const a = mapToAnalysis(map, t, &map_hint);
        const m = vm.items[nearestMark(vm.items, a, &mark_hint)];
        const pa = m.period;
        var step = pa;
        if (m.voiced) {
            if (target.f0) |f0| {
                const fi: usize = @min(f0.len - 1, @as(usize, @intFromFloat(t / target.samples_per_frame)));
                if (f0[fi] > 30) step = dsp.sample_rate / f0[fi];
            }
        }
        // Grain: Hann window, two analysis periods long, centred on the mark.
        const buf = pieces[m.piece].buf;
        const half: isize = @intFromFloat(pa);
        const centre: isize = @intCast(m.local);
        const tc: isize = @intFromFloat(t);
        // Repeated unvoiced grains would sound buzzy (a periodic copy of
        // noise); flipping their sign at random breaks the periodicity.
        const sign: f32 = if (!m.voiced and rng.next() & 1 == 0) -1.0 / 32767.0 else 1.0 / 32767.0;
        var j: isize = -half;
        while (j < half) : (j += 1) {
            const src = centre + j;
            const dst = tc + j;
            if (src < 0 or src >= buf.len or dst < 0 or dst >= out_len) continue;
            const w = 0.5 + 0.5 * @cos(std.math.pi * @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(half)));
            y[@intCast(dst)] += sign * w * @as(f32, @floatFromInt(buf[@intCast(src)]));
        }
        t += step;
    }
    return y;
}

test "identity resynthesis keeps a sine" {
    const a = std.testing.allocator;
    const n = 4000;
    const x = try a.alloc(i16, n);
    defer a.free(x);
    for (x, 0..) |*v, i| v.* = @intFromFloat(16000 * @sin(2 * std.math.pi * 100 * @as(f32, @floatFromInt(i)) / 16000));
    var marks: [25]Mark = undefined;
    for (&marks, 0..) |*m, i| m.* = .{ .pos = @intCast(i * 160), .voiced = true };
    const pieces = [_]Piece{.{ .buf = x, .start = 0, .end = n, .marks = &marks }};
    const map = [_]Knot{ .{ .t = 0, .a = 0 }, .{ .t = n, .a = n } };
    const y = try run(a, &pieces, &map, n, .{});
    defer a.free(y);
    // Away from the edges the output should equal the input.
    for (800..3200) |i| try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(x[i])) / 32767.0, y[i], 0.05);
}
