//! Statistical parametric synthesis, HTS-style (Zen, Tokuda & Black 2009),
//! reduced to its essentials.
//!
//! Training (from the sentence corpus):
//!   * every phone is split into 3 states (begin / middle / end);
//!   * each state, in context (left and right broad phone class), gets a
//!     diagonal Gaussian over 25 cepstra and their deltas, plus mean
//!     aperiodicity, voicing probability and duration. Contexts with too
//!     little data back off to the context-independent state — a stand-in for
//!     HTS's decision-tree clustering. (No Baum–Welch: the corpus labels are
//!     exact, so states are a uniform split of each phone.)
//!
//! Synthesis:
//!   * state sequence and durations from the planned phones;
//!   * MLPG (maximum-likelihood parameter generation, Tokuda et al. 2000):
//!     solve for the cepstral trajectory c that maximises the likelihood of
//!     both statics and deltas, i.e. (W' S^-1 W) c = W' S^-1 mu — a banded
//!     linear system. Without MLPG you get the stair-step of state means;
//!   * optional global-variance (GV) postfilter, the classic remedy for
//!     over-smoothing (Toda & Tokuda 2007);
//!   * vocoder: per frame, the cepstrum becomes a minimum-phase impulse
//!     response, excited by a pulse train (voiced) and/or shaped noise
//!     (aperiodicity) — a simple relative of the MLSA / STRAIGHT / WORLD
//!     vocoders.

const std = @import("std");
const dsp = @import("dsp.zig");
const phones = @import("phones.zig");
const Phone = phones.Phone;
const prosody = @import("prosody.zig");
const corpus = @import("corpus.zig");

const M = corpus.n_cep;
const D = 2 * M; // static + delta
const n_states = 3;
const NB = phones.n_broad;
const n_full = Phone.count * n_states * NB * NB;
const n_back = Phone.count * n_states;
/// Minimum frames for a context-dependent state to be used.
const min_frames: f32 = 12;

pub const Options = struct {
    mlpg: bool = true,
    gv: bool = false,
    formant_scale: f32 = 1.0,
};

const State = struct {
    n: f32,
    mean: [D]f32,
    variance: [D]f32,
    ap: f32,
    voiced: f32,
    dur: f32,
};

const Acc = struct {
    n: f64 = 0,
    s: [D]f64 = @splat(0),
    ss: [D]f64 = @splat(0),
    ap: f64 = 0,
    voiced: f64 = 0,
    occ: f64 = 0,

    fn add(a: *Acc, v: *const [D]f32, ap: f32, voiced: bool) void {
        a.n += 1;
        for (0..D) |d| {
            a.s[d] += v[d];
            a.ss[d] += v[d] * v[d];
        }
        a.ap += ap;
        if (voiced) a.voiced += 1;
    }
};

pub const Voice = struct {
    arena: std.heap.ArenaAllocator,
    full: []?State,
    back: []?State,
    /// Average per-utterance variance of each cepstral dimension (for GV).
    gv: [M]f32,
    models: usize = 0,
    bytes: usize = 0,
    build_seconds: f64 = 0,

    pub fn deinit(v: *Voice) void {
        v.arena.deinit();
    }

    pub fn train(gpa: std.mem.Allocator, c: *const corpus.Corpus) !*Voice {
        const t0 = dsp.now();
        const v = try gpa.create(Voice);
        errdefer gpa.destroy(v);
        v.* = .{ .arena = .init(gpa), .full = &.{}, .back = &.{}, .gv = @splat(0) };
        errdefer v.arena.deinit();
        const a = v.arena.allocator();

        const accs_full = try gpa.alloc(Acc, n_full);
        defer gpa.free(accs_full);
        @memset(accs_full, .{});
        const accs_back = try gpa.alloc(Acc, n_back);
        defer gpa.free(accs_back);
        @memset(accs_back, .{});
        var global: Acc = .{};
        var gv_sum: [M]f64 = @splat(0);

        for (c.recs, c.frames) |rec, frames| {
            // Utterance-level variance per dimension, for GV.
            var us: [M]f64 = @splat(0);
            var uss: [M]f64 = @splat(0);
            for (frames) |f| for (0..M) |d| {
                us[d] += f.cep[d];
                uss[d] += f.cep[d] * f.cep[d];
            };
            const nf: f64 = @floatFromInt(frames.len);
            for (0..M) |d| gv_sum[d] += uss[d] / nf - (us[d] / nf) * (us[d] / nf);

            const L = rec.labels;
            for (L, 0..) |lab, li| {
                const f_start = lab.start / 80;
                const f_end = @min(frames.len, lab.end / 80);
                if (f_end <= f_start) continue;
                const lb = phones.broad(if (li > 0) L[li - 1].phone else .pau);
                const rb = phones.broad(if (li + 1 < L.len) L[li + 1].phone else .pau);
                const len = f_end - f_start;
                for (0..n_states) |st| {
                    const s0 = f_start + len * st / n_states;
                    const s1 = f_start + len * (st + 1) / n_states;
                    const kf = fullKey(lab.phone, st, lb, rb);
                    const kb = backKey(lab.phone, st);
                    if (s1 > s0) {
                        accs_full[kf].occ += 1;
                        accs_back[kb].occ += 1;
                    }
                    for (s0..s1) |fi| {
                        var vec: [D]f32 = undefined;
                        const prev = frames[if (fi > 0) fi - 1 else fi].cep;
                        const next = frames[@min(fi + 1, frames.len - 1)].cep;
                        for (0..M) |d| {
                            vec[d] = frames[fi].cep[d];
                            vec[M + d] = 0.5 * (next[d] - prev[d]);
                        }
                        accs_full[kf].add(&vec, frames[fi].ap, frames[fi].voiced);
                        accs_back[kb].add(&vec, frames[fi].ap, frames[fi].voiced);
                        global.add(&vec, 0, false);
                    }
                }
            }
        }
        for (0..M) |d| v.gv[d] = @floatCast(gv_sum[d] / @as(f64, @floatFromInt(c.recs.len)));

        var floor: [D]f32 = undefined;
        for (0..D) |d| {
            const m = global.s[d] / global.n;
            floor[d] = @floatCast(@max(1e-6, 0.01 * (global.ss[d] / global.n - m * m)));
        }
        v.full = try a.alloc(?State, n_full);
        v.back = try a.alloc(?State, n_back);
        for (accs_full, v.full) |acc, *out| out.* = finish(acc, &floor, 0);
        for (accs_back, v.back) |acc, *out| out.* = finish(acc, &floor, 0);
        for (v.full) |s| if (s) |x| if (x.n >= min_frames) {
            v.models += 1;
        };
        for (v.back) |s| if (s != null) {
            v.models += 1;
        };
        v.bytes = v.models * @sizeOf(State);
        v.build_seconds = dsp.now() - t0;
        return v;
    }

    fn lookup(v: *const Voice, p: Phone, st: usize, lb: phones.Broad, rb: phones.Broad) State {
        if (v.full[fullKey(p, st, lb, rb)]) |s| if (s.n >= min_frames) return s;
        if (v.back[backKey(p, st)]) |s| return s;
        // Unseen phone: fall back to silence.
        return v.back[backKey(.pau, st)] orelse unreachable;
    }

    pub fn synthesize(v: *const Voice, alloc: std.mem.Allocator, utt: *const prosody.Utterance, opt: Options) ![]f32 {
        const segs = utt.segs;
        const T: usize = @intFromFloat(@round(utt.duration * prosody.frame_rate));
        const states = try alloc.alloc(State, T);
        defer alloc.free(states);

        // ---- state sequence with durations proportional to the models'
        for (segs, 0..) |s, i| {
            const f0i: usize = @intFromFloat(@round(s.start * prosody.frame_rate));
            const f1i: usize = @min(T, @as(usize, @intFromFloat(@round(s.end() * prosody.frame_rate))));
            if (f1i <= f0i) continue;
            const lb = phones.broad(if (i > 0) segs[i - 1].phone else .pau);
            const rb = phones.broad(if (i + 1 < segs.len) segs[i + 1].phone else .pau);
            var st_models: [n_states]State = undefined;
            var total: f32 = 0;
            for (0..n_states) |st| {
                st_models[st] = v.lookup(s.phone, st, lb, rb);
                total += st_models[st].dur;
            }
            const len: f32 = @floatFromInt(f1i - f0i);
            var acc: f32 = 0;
            var st: usize = 0;
            for (f0i..f1i) |fi| {
                const pos = @as(f32, @floatFromInt(fi - f0i)) + 0.5;
                while (st + 1 < n_states and pos > (acc + st_models[st].dur) / total * len) {
                    acc += st_models[st].dur;
                    st += 1;
                }
                states[fi] = st_models[st];
            }
        }

        // ---- parameter generation
        const cep = try alloc.alloc([M]f32, T);
        defer alloc.free(cep);
        if (opt.mlpg) {
            try mlpg(alloc, states, cep);
        } else {
            for (states, cep) |s, *c| for (0..M) |d| {
                c[d] = s.mean[d];
            };
        }
        if (opt.gv) globalVariance(cep, &v.gv);

        // ---- vocoder
        return vocode(alloc, utt, states, cep, opt.formant_scale);
    }
};

fn fullKey(p: Phone, st: usize, lb: phones.Broad, rb: phones.Broad) usize {
    return ((@as(usize, @intFromEnum(p)) * n_states + st) * NB + @intFromEnum(lb)) * NB + @intFromEnum(rb);
}

fn backKey(p: Phone, st: usize) usize {
    return @as(usize, @intFromEnum(p)) * n_states + st;
}

fn finish(acc: Acc, floor: *const [D]f32, _: u8) ?State {
    if (acc.n < 1) return null;
    var s: State = .{
        .n = @floatCast(acc.n),
        .mean = undefined,
        .variance = undefined,
        .ap = @floatCast(acc.ap / acc.n),
        .voiced = @floatCast(acc.voiced / acc.n),
        .dur = @floatCast(acc.n / @max(1, acc.occ)),
    };
    for (0..D) |d| {
        const m = acc.s[d] / acc.n;
        s.mean[d] = @floatCast(m);
        s.variance[d] = @max(floor[d], @as(f32, @floatCast(acc.ss[d] / acc.n - m * m)));
    }
    return s;
}

/// Solve (W' S^-1 W) c = W' S^-1 mu per dimension. With a 3-point delta
/// window the matrix is symmetric pentadiagonal; banded Cholesky solves it
/// in O(T).
fn mlpg(alloc: std.mem.Allocator, states: []const State, out: [][M]f32) !void {
    const T = states.len;
    const d0 = try alloc.alloc(f64, T);
    defer alloc.free(d0);
    const d1 = try alloc.alloc(f64, T);
    defer alloc.free(d1);
    const d2 = try alloc.alloc(f64, T);
    defer alloc.free(d2);
    const b = try alloc.alloc(f64, T);
    defer alloc.free(b);
    for (0..M) |d| {
        @memset(d0, 0);
        @memset(d1, 0);
        @memset(d2, 0);
        @memset(b, 0);
        for (states, 0..) |s, t| {
            const ps = 1.0 / @as(f64, s.variance[d]);
            d0[t] += ps;
            b[t] += ps * s.mean[d];
            if (t >= 1 and t + 1 < T) {
                const pd = 1.0 / @as(f64, s.variance[M + d]);
                const md: f64 = s.mean[M + d];
                d0[t - 1] += 0.25 * pd;
                d0[t + 1] += 0.25 * pd;
                d2[t - 1] -= 0.25 * pd;
                b[t - 1] -= 0.5 * pd * md;
                b[t + 1] += 0.5 * pd * md;
            }
        }
        // Banded Cholesky: A = L L', L lower with two sub-diagonals.
        // Reuse d0 -> L[t][t], d1 -> L[t][t-1], d2 -> L[t][t-2] (shifted).
        const l0 = d0;
        const l1 = d1; // l1[t] = L[t][t-1]
        const a2 = try alloc.dupe(f64, d2); // A[t][t+2]
        defer alloc.free(a2);
        const l2 = d2; // l2[t] = L[t][t-2]
        // A[t][t+1] is zero for this window; keep the general form anyway.
        for (0..T) |t| {
            l2[t] = if (t >= 2) a2[t - 2] / l0[t - 2] else 0;
            l1[t] = if (t >= 1) (0 - (if (t >= 2) l2[t] * l1[t - 1] else 0)) / l0[t - 1] else 0;
            const diag = l0[t] - l1[t] * l1[t] - l2[t] * l2[t];
            l0[t] = @sqrt(@max(diag, 1e-12));
        }
        // Forward, then backward substitution.
        for (0..T) |t| {
            var y = b[t];
            if (t >= 1) y -= l1[t] * b[t - 1];
            if (t >= 2) y -= l2[t] * b[t - 2];
            b[t] = y / l0[t];
        }
        var t = T;
        while (t > 0) {
            t -= 1;
            var x = b[t];
            if (t + 1 < T) x -= l1[t + 1] * b[t + 1];
            if (t + 2 < T) x -= l2[t + 2] * b[t + 2];
            b[t] = x / l0[t];
        }
        for (out, 0..) |*o, i| o[d] = @floatCast(b[i]);
    }
}

/// Scale each cepstral trajectory around its mean so its variance matches
/// the average variance of natural (training) utterances.
fn globalVariance(cep: [][M]f32, gv: *const [M]f32) void {
    const n: f32 = @floatFromInt(cep.len);
    for (1..M) |d| {
        var s: f32 = 0;
        var ss: f32 = 0;
        for (cep) |c| {
            s += c[d];
            ss += c[d] * c[d];
        }
        const m = s / n;
        const var_gen = ss / n - m * m;
        if (var_gen <= 1e-9) continue;
        const k = std.math.clamp(@sqrt(gv[d] / var_gen), 1.0, 2.5);
        for (cep) |*c| c[d] = m + (c[d] - m) * k;
    }
}

const NF = corpus.fft_n;

/// cos(2 pi k n / NF) for k = 0..NF/2, n = 0..M-1.
const cos_table: [NF / 2 + 1][M]f32 = blk: {
    @setEvalBranchQuota(200000);
    var t: [NF / 2 + 1][M]f32 = undefined;
    for (0..NF / 2 + 1) |k| {
        for (0..M) |n| t[k][n] = @cos(2 * std.math.pi * @as(f32, @floatFromInt((k * n) % NF)) / NF);
    }
    break :blk t;
};

fn warpCepstrum(c: *[M]f32, scale: f32) void {
    // log spectrum E(k), k = 0..NF/2, then E'(k) = E(k / scale), then back.
    var e: [NF / 2 + 1]f32 = undefined;
    for (&e, 0..) |*v, k| {
        var acc = c[0];
        for (1..M) |n| acc += 2 * c[n] * cos_table[k][n];
        v.* = acc;
    }
    var w: [NF / 2 + 1]f32 = undefined;
    for (&w, 0..) |*v, k| {
        const src = @as(f32, @floatFromInt(k)) / scale;
        const i: usize = @intFromFloat(@floor(src));
        if (i >= NF / 2) {
            v.* = e[NF / 2];
        } else {
            const fr = src - @as(f32, @floatFromInt(i));
            v.* = e[i] * (1 - fr) + e[i + 1] * fr;
        }
    }
    for (0..M) |n| {
        var acc = w[0] + w[NF / 2] * (if (n % 2 == 0) @as(f32, 1) else -1);
        for (1..NF / 2) |k| acc += 2 * w[k] * cos_table[k][n];
        c[n] = acc / NF;
    }
}

/// Minimum-phase spectrum from a real cepstrum (fold, FFT, exp).
fn minPhaseSpectrum(c: *const [M]f32, h: *[NF]dsp.Complex) void {
    for (h, 0..) |*v, i| {
        v.* = dsp.Complex.init(if (i == 0) c[0] else if (i < M) 2 * c[i] else 0, 0);
    }
    dsp.fft(h, false);
    for (h) |*v| {
        const mag = @exp(v.re);
        v.* = dsp.Complex.init(mag * @cos(v.im), mag * @sin(v.im));
    }
}

fn vocode(alloc: std.mem.Allocator, utt: *const prosody.Utterance, states: []const State, cep: [][M]f32, formant_scale: f32) ![]f32 {
    const hop = 80;
    const T = states.len;
    const out_len = T * hop;
    const y = try alloc.alloc(f32, out_len + NF);
    @memset(y, 0);
    var spec: [NF]dsp.Complex = undefined;
    var ir: [NF]dsp.Complex = undefined;
    var nbuf: [NF]dsp.Complex = undefined;
    var rng = dsp.Rng.init(4242);
    var next_pulse: f32 = 0;
    const noise_gain = 1 / @sqrt(corpus.win_sq_sum);

    for (0..T) |t| {
        var c = cep[t];
        if (formant_scale != 1.0) warpCepstrum(&c, formant_scale);
        minPhaseSpectrum(&c, &spec);
        ir = spec;
        dsp.fft(&ir, true);

        const voiced = states[t].voiced > 0.5;
        const ap = std.math.clamp(states[t].ap, 0, 1);
        const f0 = @max(utt.f0At(@as(f32, @floatFromInt(t)) / prosody.frame_rate), 40);
        const period = dsp.sample_rate / f0;

        // Pulses falling in this frame's hop.
        const hop_end: f32 = @floatFromInt((t + 1) * hop);
        while (next_pulse < hop_end) {
            if (voiced) {
                const amp = period / corpus.win_sum * @sqrt(1 - ap);
                const p: usize = @intFromFloat(next_pulse);
                for (0..NF) |k| {
                    if (p + k >= y.len) break;
                    y[p + k] += amp * ir[k].re / NF;
                }
                next_pulse += period;
            } else {
                next_pulse = hop_end;
            }
        }

        // Shaped noise: a Hann-windowed white segment (2 hops), filtered by
        // the frame's spectrum, overlap-added.
        const ng = noise_gain * (if (voiced) @sqrt(ap) else 1.0);
        for (&nbuf, 0..) |*v, i| {
            v.* = if (i < 2 * hop) dsp.Complex.init(rng.gauss() * ng * dsp.hann(2 * hop, i), 0) else dsp.Complex.init(0, 0);
        }
        dsp.fft(&nbuf, false);
        for (&nbuf, spec) |*v, h| v.* = v.mul(h);
        dsp.fft(&nbuf, true);
        const start = @as(isize, @intCast(t * hop)) - hop;
        for (0..NF) |k| {
            const idx = start + @as(isize, @intCast(k));
            if (idx < 0 or idx >= y.len) continue;
            y[@intCast(idx)] += nbuf[k].re / NF;
        }
    }
    return alloc.realloc(y, out_len);
}

test "mlpg reproduces constant means" {
    const a = std.testing.allocator;
    var states: [20]State = undefined;
    for (&states) |*s| {
        s.* = .{ .n = 1, .mean = @splat(0), .variance = @splat(1), .ap = 0, .voiced = 1, .dur = 1 };
        s.mean[0] = 2.5;
    }
    const out = try a.alloc([M]f32, 20);
    defer a.free(out);
    try mlpg(a, &states, out);
    for (out) |o| try std.testing.expectApproxEqAbs(@as(f32, 2.5), o[0], 1e-3);
}
