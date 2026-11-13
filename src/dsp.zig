//! Shared signal-processing building blocks.

const std = @import("std");
const builtin = @import("builtin");

pub const sample_rate: f32 = 16000;
pub const sr_int: usize = 16000;

/// Klatt's digital resonator: a two-pole filter with unity gain at DC.
///   y[n] = A x[n] + B y[n-1] + C y[n-2]
pub const Resonator = struct {
    a: f32 = 1,
    b: f32 = 0,
    c: f32 = 0,
    y1: f32 = 0,
    y2: f32 = 0,

    pub fn set(r: *Resonator, f: f32, bw: f32) void {
        const t = 1.0 / sample_rate;
        r.c = -@exp(-2 * std.math.pi * bw * t);
        r.b = 2 * @exp(-std.math.pi * bw * t) * @cos(2 * std.math.pi * f * t);
        r.a = 1 - r.b - r.c;
    }

    /// Same poles, but gain normalised to ~1 at the centre frequency instead
    /// of DC. Used for the parallel branch where each resonator has its own
    /// amplitude control.
    pub fn setPeak(r: *Resonator, f: f32, bw: f32) void {
        r.set(f, bw);
        // Peak gain of the DC-normalised resonator is roughly f / bw for
        // narrow bands; dividing it out keeps the parallel gains meaningful.
        r.a *= @min(1.0, bw / @max(f, 1.0)) * 1.0;
    }

    pub inline fn step(r: *Resonator, x: f32) f32 {
        const y = r.a * x + r.b * r.y1 + r.c * r.y2;
        r.y2 = r.y1;
        r.y1 = y;
        return y;
    }
};

/// Klatt's antiresonator (two zeros), the inverse of a resonator.
pub const AntiResonator = struct {
    a: f32 = 1,
    b: f32 = 0,
    c: f32 = 0,
    x1: f32 = 0,
    x2: f32 = 0,

    pub fn set(r: *AntiResonator, f: f32, bw: f32) void {
        var tmp: Resonator = .{};
        tmp.set(f, bw);
        r.a = 1 / tmp.a;
        r.b = -tmp.b / tmp.a;
        r.c = -tmp.c / tmp.a;
    }

    pub inline fn step(r: *AntiResonator, x: f32) f32 {
        const y = r.a * x + r.b * r.x1 + r.c * r.x2;
        r.x2 = r.x1;
        r.x1 = x;
        return y;
    }
};

/// Small, fast, deterministic noise source (xorshift32).
pub const Rng = struct {
    s: u32,

    pub fn init(seed: u32) Rng {
        return .{ .s = if (seed == 0) 0x9E3779B9 else seed };
    }

    pub inline fn next(r: *Rng) u32 {
        var x = r.s;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        r.s = x;
        return x;
    }

    /// Uniform in [-1, 1).
    pub inline fn uniform(r: *Rng) f32 {
        return @as(f32, @floatFromInt(r.next() >> 8)) / 8388608.0 - 1.0;
    }

    /// Approximately Gaussian, unit variance (sum of 4 uniforms).
    pub inline fn gauss(r: *Rng) f32 {
        return (r.uniform() + r.uniform() + r.uniform() + r.uniform()) * 0.8660254;
    }
};

pub fn hann(n: usize, i: usize) f32 {
    return 0.5 - 0.5 * @cos(2 * std.math.pi * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n)));
}

pub const Complex = std.math.Complex(f32);

/// In-place iterative radix-2 FFT. `inverse` computes the unscaled inverse.
pub fn fft(x: []Complex, inverse: bool) void {
    const n = x.len;
    std.debug.assert(std.math.isPowerOfTwo(n));
    var j: usize = 0;
    for (1..n) |i| {
        var bit = n >> 1;
        while (j & bit != 0) : (bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std.mem.swap(Complex, &x[i], &x[j]);
    }
    var len: usize = 2;
    while (len <= n) : (len <<= 1) {
        const ang = 2 * std.math.pi / @as(f32, @floatFromInt(len)) * @as(f32, if (inverse) 1 else -1);
        const wl = Complex.init(@cos(ang), @sin(ang));
        var i: usize = 0;
        while (i < n) : (i += len) {
            var w = Complex.init(1, 0);
            for (0..len / 2) |k| {
                const u = x[i + k];
                const v = x[i + k + len / 2].mul(w);
                x[i + k] = u.add(v);
                x[i + k + len / 2] = u.sub(v);
                w = w.mul(wl);
            }
        }
    }
}

/// Peak-normalise to `peak`, returning the gain applied.
pub fn normalize(x: []f32, peak: f32) f32 {
    var m: f32 = 0;
    for (x) |v| m = @max(m, @abs(v));
    if (m < 1e-9) return 1;
    const g = peak / m;
    for (x) |*v| v.* *= g;
    return g;
}

/// RMS normalise to `target` (with a peak limit so nothing clips).
pub fn normalizeRms(x: []f32, target: f32) void {
    var e: f64 = 0;
    for (x) |v| e += v * v;
    if (x.len == 0 or e <= 0) return;
    const rms: f32 = @floatCast(@sqrt(e / @as(f64, @floatFromInt(x.len))));
    var g = target / rms;
    var m: f32 = 0;
    for (x) |v| m = @max(m, @abs(v));
    if (m * g > 0.98) g = 0.98 / m;
    for (x) |*v| v.* *= g;
}

/// Heap for work done outside an arena: thread-safe where there are threads,
/// the wasm allocator in single-threaded builds (e.g. wasm32-freestanding).
pub const heap = if (builtin.single_threaded) (if (builtin.cpu.arch.isWasm()) std.heap.wasm_allocator else std.heap.page_allocator) else std.heap.smp_allocator;

/// Monotonic clock in seconds (Linux), so the library needs no std.Io.
/// Elsewhere (wasm) timings read as 0; callers time externally.
pub fn now() f64 {
    if (builtin.os.tag != .linux) return 0;
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

test "fft roundtrip" {
    var x: [16]Complex = undefined;
    for (&x, 0..) |*v, i| v.* = Complex.init(@floatFromInt(i), 0);
    fft(&x, false);
    fft(&x, true);
    for (x, 0..) |v, i| try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(i)), v.re / 16, 1e-4);
}
