//! Unit selection (Hunt & Black 1996; AT&T NextGen, Festival multisyn).
//!
//! Every adjacent phone pair in the sentence corpus is a candidate diphone
//! unit. For a target utterance, each position has many candidates; Viterbi
//! picks the sequence minimising
//!     sum target_cost(target_i, cand_i) + sum join_cost(cand_i-1, cand_i)
//! where the join cost is zero for units that were neighbours in the corpus,
//! so the search prefers long contiguous stretches of real recording.
//! Diphones the corpus lacks fall back to the diphone inventory (as real
//! hybrid systems do). By default the chosen units are not modified (only
//! pitch-synchronously overlap-added at the joins); optionally PSOLA forces
//! them onto the planned prosody.

const std = @import("std");
const dsp = @import("dsp.zig");
const phones = @import("phones.zig");
const Phone = phones.Phone;
const prosody = @import("prosody.zig");
const corpus = @import("corpus.zig");
const diphone = @import("diphone.zig");
const psola = @import("psola.zig");
const Mark = corpus.Mark;

const N = Phone.count;

pub const Options = struct {
    /// Force the selected units onto the planned F0/durations with PSOLA.
    modify_prosody: bool = false,
    /// Candidates kept per position after target-cost pruning.
    beam: usize = 40,
};

pub const Cand = struct {
    rec: u32,
    seg: u32, // label index of the first phone
    start: u32,
    split: u32,
    end: u32,
    f0: [3]f32,
    dur: [2]f32,
    stress: [2]u2,
    accent: [2]bool,
    pfinal: bool,
    prev: Phone,
    next: Phone,
};

pub const Stats = struct {
    units: usize = 0,
    fallbacks: usize = 0,
    /// Number of joins between units that were not corpus neighbours.
    joins: usize = 0,
};

pub const Voice = struct {
    arena: std.heap.ArenaAllocator,
    corpus: *const corpus.Corpus,
    fallback: *const diphone.Voice,
    cands: []Cand,
    index: [N * N][]u32,
    bytes: usize = 0,
    build_seconds: f64 = 0,

    pub fn deinit(v: *Voice) void {
        v.arena.deinit();
    }

    pub fn coverage(v: *const Voice) usize {
        var n: usize = 0;
        for (v.index) |l| if (l.len > 0) {
            n += 1;
        };
        return n;
    }

    pub fn build(gpa: std.mem.Allocator, c: *const corpus.Corpus, fb: *const diphone.Voice) !*Voice {
        const t0 = dsp.now();
        const v = try gpa.create(Voice);
        errdefer gpa.destroy(v);
        v.* = .{ .arena = .init(gpa), .corpus = c, .fallback = fb, .cands = &.{}, .index = @splat(&.{}) };
        errdefer v.arena.deinit();
        const a = v.arena.allocator();

        var cands: std.ArrayList(Cand) = .empty;
        var lists: [N * N]std.ArrayList(u32) = @splat(.empty);
        for (c.recs, c.frames, 0..) |rec, frames, ri| {
            const L = rec.labels;
            for (0..L.len - 1) |i| {
                const la = L[i];
                const lb = L[i + 1];
                if (la.phone == .pau and lb.phone == .pau) continue;
                const start = la.mid();
                const end = lb.mid();
                const cand: Cand = .{
                    .rec = @intCast(ri),
                    .seg = @intCast(i),
                    .start = start,
                    .split = la.end,
                    .end = end,
                    .f0 = .{ f0At(frames, start), f0At(frames, la.end), f0At(frames, end) },
                    .dur = .{ @as(f32, @floatFromInt(la.end - start)) / dsp.sample_rate, @as(f32, @floatFromInt(end - la.end)) / dsp.sample_rate },
                    .stress = .{ la.stress, lb.stress },
                    .accent = .{ la.accent, lb.accent },
                    .pfinal = lb.phrase_final,
                    .prev = if (i > 0) L[i - 1].phone else .pau,
                    .next = if (i + 2 < L.len) L[i + 2].phone else .pau,
                };
                const key = @as(usize, @intFromEnum(la.phone)) * N + @intFromEnum(lb.phone);
                try lists[key].append(a, @intCast(cands.items.len));
                try cands.append(a, cand);
            }
        }
        v.cands = cands.items;
        for (&lists, 0..) |*l, k| v.index[k] = l.items;
        v.bytes = cands.items.len * (@sizeOf(Cand) + 4);
        v.build_seconds = dsp.now() - t0;
        return v;
    }

    pub fn synthesize(v: *const Voice, alloc: std.mem.Allocator, utt: *const prosody.Utterance, opt: Options, stats: *Stats) ![]f32 {
        const segs = utt.segs;
        const n_pos = segs.len - 1;
        const sr = dsp.sample_rate;

        // ---- candidate lists with target costs
        const Node = struct { cand: ?u32, tcost: f32, cost: f32 = 0, back: u32 = 0 };
        const lattice = try alloc.alloc([]Node, n_pos);
        defer {
            for (lattice) |l| alloc.free(l);
            alloc.free(lattice);
        }
        for (0..n_pos) |i| {
            const tgt = targetOf(utt, i);
            const key = @as(usize, @intFromEnum(segs[i].phone)) * N + @intFromEnum(segs[i + 1].phone);
            const ids = v.index[key];
            if (ids.len == 0) {
                lattice[i] = try alloc.alloc(Node, 1);
                lattice[i][0] = .{ .cand = null, .tcost = 0 };
                continue;
            }
            var tmp = try alloc.alloc(Node, ids.len);
            for (ids, 0..) |id, k| tmp[k] = .{ .cand = id, .tcost = targetCost(&tgt, &v.cands[id]) };
            std.mem.sort(Node, tmp, {}, struct {
                fn lt(_: void, x: Node, y: Node) bool {
                    return x.tcost < y.tcost;
                }
            }.lt);
            const keep = @min(opt.beam, tmp.len);
            lattice[i] = try alloc.realloc(tmp, keep);
        }

        // ---- Viterbi
        for (lattice[0]) |*nd| nd.cost = nd.tcost;
        for (1..n_pos) |i| {
            for (lattice[i]) |*nd| {
                var best: f32 = std.math.inf(f32);
                var arg: u32 = 0;
                for (lattice[i - 1], 0..) |pv, k| {
                    const c = pv.cost + v.joinCost(pv.cand, nd.cand);
                    if (c < best) {
                        best = c;
                        arg = @intCast(k);
                    }
                }
                nd.cost = best + nd.tcost;
                nd.back = arg;
            }
        }
        const path = try alloc.alloc(?u32, n_pos);
        defer alloc.free(path);
        {
            var bi: u32 = 0;
            var best: f32 = std.math.inf(f32);
            for (lattice[n_pos - 1], 0..) |nd, k| if (nd.cost < best) {
                best = nd.cost;
                bi = @intCast(k);
            };
            var i = n_pos;
            while (i > 0) {
                i -= 1;
                path[i] = lattice[i][bi].cand;
                bi = lattice[i][bi].back;
            }
        }

        // ---- concatenate
        var pieces: std.ArrayList(psola.Piece) = .empty;
        defer pieces.deinit(alloc);
        var map: std.ArrayList(psola.Knot) = .empty;
        defer map.deinit(alloc);
        stats.* = .{ .units = n_pos };
        var base: f32 = 0;
        for (0..n_pos) |i| {
            var piece: psola.Piece = undefined;
            var split: u32 = 0;
            if (path[i]) |id| {
                const c = v.cands[id];
                const rec = v.corpus.recs[c.rec];
                piece = .{ .buf = rec.samples, .start = c.start, .end = c.end, .marks = rec.marks };
                split = c.split - c.start;
                if (i > 0) {
                    const contiguous = if (path[i - 1]) |pid|
                        v.cands[pid].rec == c.rec and v.cands[pid].seg + 1 == c.seg
                    else
                        false;
                    if (!contiguous) stats.joins += 1;
                }
            } else {
                stats.fallbacks += 1;
                if (i > 0) stats.joins += 1;
                const u = v.fallback.unit(segs[i].phone, segs[i + 1].phone) orelse continue;
                piece = u.piece();
                split = u.split - u.start;
            }
            try pieces.append(alloc, piece);
            const len: f32 = @floatFromInt(piece.end - piece.start);
            if (opt.modify_prosody) {
                const a = segs[i];
                const b = segs[i + 1];
                try map.append(alloc, .{ .t = (a.start + a.dur / 2) * sr, .a = base });
                try map.append(alloc, .{ .t = a.end() * sr, .a = base + @as(f32, @floatFromInt(split)) });
                try map.append(alloc, .{ .t = (b.start + b.dur / 2) * sr, .a = base + len - 1 });
            }
            base += len;
        }
        if (opt.modify_prosody) {
            const out_len: usize = @intFromFloat(utt.duration * sr);
            return psola.run(alloc, pieces.items, map.items, out_len, .{ .f0 = utt.f0 });
        }
        // Unmodified: identity map, pitch-synchronous overlap-add only.
        const map_id = [_]psola.Knot{ .{ .t = 0, .a = 0 }, .{ .t = base, .a = base } };
        return psola.run(alloc, pieces.items, &map_id, @intFromFloat(base), .{});
    }

    fn joinCost(v: *const Voice, prev: ?u32, cur: ?u32) f32 {
        const p_id = prev orelse return 1.0;
        const c_id = cur orelse return 1.0;
        const p = v.cands[p_id];
        const c = v.cands[c_id];
        if (p.rec == c.rec and p.seg + 1 == c.seg) return 0;
        const fa = v.corpus.frames[p.rec][@min(p.end / 80, v.corpus.frames[p.rec].len - 1)];
        const fb = v.corpus.frames[c.rec][@min(c.start / 80, v.corpus.frames[c.rec].len - 1)];
        var d: f32 = 0;
        for (1..13) |k| {
            const e = fa.cep[k] - fb.cep[k];
            d += e * e;
        }
        var cost = 0.4 + 2.0 * @sqrt(d) + 0.5 * @abs(fa.cep[0] - fb.cep[0]);
        if (p.f0[2] > 0 and c.f0[0] > 0) cost += 4.0 * @abs(@log(p.f0[2] / c.f0[0]));
        return cost;
    }
};

fn f0At(frames: []const corpus.Frame, sample: u32) f32 {
    const fi = @min(frames.len - 1, sample / 80);
    return if (frames[fi].voiced) frames[fi].f0 else 0;
}

const TargetSpec = struct {
    f0: [3]f32,
    dur: [2]f32,
    stress: [2]u2,
    accent: [2]bool,
    pfinal: bool,
    prev: Phone,
    next: Phone,
    voiced: [2]bool,
};

fn targetOf(utt: *const prosody.Utterance, i: usize) TargetSpec {
    const a = utt.segs[i];
    const b = utt.segs[i + 1];
    return .{
        .f0 = .{ utt.f0At(a.start + a.dur / 2), utt.f0At(a.end()), utt.f0At(b.start + b.dur / 2) },
        .dur = .{ a.dur / 2, b.dur / 2 },
        .stress = .{ a.stress, b.stress },
        .accent = .{ a.accent, b.accent },
        .pfinal = b.phrase_final,
        .prev = if (i > 0) utt.segs[i - 1].phone else .pau,
        .next = if (i + 2 < utt.segs.len) utt.segs[i + 2].phone else .pau,
        .voiced = .{ a.phone.info().voiced, b.phone.info().voiced },
    };
}

fn ctxCost(want: Phone, got: Phone) f32 {
    if (want == got) return 0;
    if (phones.broad(want) == phones.broad(got)) return 0.5;
    return 1.5;
}

fn targetCost(t: *const TargetSpec, c: *const Cand) f32 {
    var cost: f32 = 0;
    cost += ctxCost(t.prev, c.prev) + ctxCost(t.next, c.next);
    for (0..2) |k| {
        if (t.stress[k] != c.stress[k]) cost += 1.0;
        if (t.accent[k] != c.accent[k]) cost += 0.5;
        if (t.dur[k] > 0 and c.dur[k] > 0) cost += 1.0 * @abs(@log(c.dur[k] / t.dur[k]));
    }
    if (t.pfinal != c.pfinal) cost += 0.7;
    for (0..3) |k| {
        if (c.f0[k] > 0 and t.f0[k] > 0) cost += 3.0 * @abs(@log(c.f0[k] / t.f0[k]));
    }
    return cost;
}
