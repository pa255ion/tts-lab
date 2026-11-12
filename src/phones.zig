//! The phone inventory: ARPAbet (as used by CMUdict) plus a pause symbol.
//!
//! Each phone carries the acoustic "facts" the formant rules need: formant
//! targets (male voice, roughly Klatt 1980 / Peterson & Barney), inherent and
//! minimum durations (Klatt 1979), place of articulation (for stop loci) and
//! the spectral shape of any frication noise. The concatenative and HMM
//! engines only use the names and classes; their acoustics come from data.

const std = @import("std");

pub const Phone = enum(u8) {
    pau,
    // monophthong vowels
    aa, ae, ah, ao, eh, er, ih, iy, uh, uw,
    // diphthongs
    aw, ay, ey, ow, oy,
    // stops
    b, d, g, p, t, k,
    // affricates
    ch, jh,
    // fricatives
    f, v, th, dh, s, z, sh, zh, hh,
    // nasals
    m, n, ng,
    // liquids & glides
    l, r, w, y,

    pub const count = @typeInfo(Phone).@"enum".fields.len;

    pub fn info(p: Phone) *const Info {
        return &table[@intFromEnum(p)];
    }

    pub fn name(p: Phone) []const u8 {
        return @tagName(p);
    }

    pub fn isVowel(p: Phone) bool {
        const c = p.info().class;
        return c == .vowel or c == .diphthong;
    }

    pub fn isSonorant(p: Phone) bool {
        return switch (p.info().class) {
            .vowel, .diphthong, .nasal, .liquid, .glide => true,
            else => false,
        };
    }

    pub fn isObstruent(p: Phone) bool {
        return switch (p.info().class) {
            .stop, .affricate, .fricative => true,
            else => false,
        };
    }

    /// Parse an ARPAbet symbol ("AA", "aa1", "TH"). The stress digit, if any,
    /// is returned separately.
    pub fn parse(sym: []const u8) ?struct { phone: Phone, stress: u2 } {
        var s = sym;
        var stress: u2 = 0;
        if (s.len > 0 and s[s.len - 1] >= '0' and s[s.len - 1] <= '2') {
            stress = @intCast(s[s.len - 1] - '0');
            s = s[0 .. s.len - 1];
        }
        if (s.len == 0 or s.len > 3) return null;
        var buf: [3]u8 = undefined;
        for (s, 0..) |c, i| buf[i] = std.ascii.toLower(c);
        const lower = buf[0..s.len];
        inline for (@typeInfo(Phone).@"enum".fields) |f| {
            if (std.mem.eql(u8, f.name, lower)) return .{ .phone = @enumFromInt(f.value), .stress = stress };
        }
        return null;
    }
};

pub const Class = enum { silence, vowel, diphthong, stop, affricate, fricative, aspirate, nasal, liquid, glide };

pub const Place = enum { none, labial, dental, alveolar, postalveolar, velar, glottal };

/// Centre frequencies of the fixed parallel noise bank (Hz). Frication and
/// burst spectra are expressed as gains on these, plus a flat bypass path.
pub const noise_bank = [_]struct { f: f32, bw: f32 }{
    .{ .f = 1800, .bw = 350 },
    .{ .f = 2500, .bw = 400 },
    .{ .f = 3300, .bw = 500 },
    .{ .f = 4300, .bw = 700 },
    .{ .f = 5500, .bw = 1000 },
    .{ .f = 6800, .bw = 1400 },
};
pub const n_noise = noise_bank.len + 1; // + bypass

pub const Info = struct {
    class: Class,
    voiced: bool,
    place: Place = .none,
    /// Formant targets F1..F3 (Hz). For diphthongs, the onset target.
    f: [3]f32,
    /// Offglide target for diphthongs; equal to `f` otherwise.
    f_end: [3]f32,
    /// Bandwidths B1..B3 (Hz).
    b: [3]f32 = .{ 60, 90, 150 },
    /// Klatt inherent / minimum duration (ms).
    inh_ms: f32,
    min_ms: f32,
    /// Overall frication amplitude (0..1) and its spectral shape.
    fric_amp: f32 = 0,
    fric: [n_noise]f32 = .{ 0, 0, 0, 0, 0, 0, 0 },
    /// Burst spectrum for stops/affricates.
    burst: [n_noise]f32 = .{ 0, 0, 0, 0, 0, 0, 0 },
    /// Nasal zero frequency (Hz). Equal to the nasal pole (270) = no nasality.
    fnz: f32 = 270,
    /// Voice-onset time after release, voiceless stops only (ms).
    vot_ms: f32 = 0,
};

fn v(f1: f32, f2: f32, f3: f32, inh: f32, min: f32) Info {
    return .{ .class = .vowel, .voiced = true, .f = .{ f1, f2, f3 }, .f_end = .{ f1, f2, f3 }, .inh_ms = inh, .min_ms = min };
}

fn dip(a: [3]f32, e: [3]f32, inh: f32, min: f32) Info {
    return .{ .class = .diphthong, .voiced = true, .f = a, .f_end = e, .inh_ms = inh, .min_ms = min };
}

const s_shape = [n_noise]f32{ 0, 0, 0.04, 0.25, 1.0, 0.7, 0.02 };
const sh_shape = [n_noise]f32{ 0.35, 1.0, 0.6, 0.3, 0.12, 0.04, 0 };
const f_shape = [n_noise]f32{ 0, 0, 0, 0, 0.1, 0.15, 1.0 };
const th_shape = [n_noise]f32{ 0, 0, 0, 0.1, 0.25, 0.3, 0.8 };
const lab_burst = [n_noise]f32{ 0, 0, 0, 0, 0, 0, 1.0 };
const alv_burst = [n_noise]f32{ 0, 0, 0.15, 0.6, 1.0, 0.6, 0.1 };
const vel_burst = [n_noise]f32{ 1.0, 0.7, 0.35, 0.1, 0.02, 0, 0.05 };

pub const table = blk: {
    var t: [Phone.count]Info = undefined;
    const set = struct {
        fn f(tt: *[Phone.count]Info, p: Phone, i: Info) void {
            tt[@intFromEnum(p)] = i;
        }
    }.f;
    set(&t, .pau, .{ .class = .silence, .voiced = false, .f = .{ 500, 1500, 2500 }, .f_end = .{ 500, 1500, 2500 }, .inh_ms = 200, .min_ms = 200 });

    set(&t, .iy, v(310, 2020, 2960, 155, 55));
    set(&t, .ih, v(400, 1800, 2570, 135, 40));
    set(&t, .eh, v(530, 1680, 2500, 150, 70));
    set(&t, .ae, v(620, 1660, 2430, 230, 80));
    set(&t, .aa, v(700, 1220, 2600, 240, 100));
    set(&t, .ao, v(600, 990, 2570, 240, 100));
    set(&t, .uh, v(450, 1100, 2350, 160, 60));
    set(&t, .uw, v(350, 1250, 2200, 210, 70));
    set(&t, .ah, v(620, 1220, 2550, 140, 60));
    set(&t, .er, v(470, 1270, 1540, 180, 80));

    set(&t, .ey, dip(.{ 480, 1720, 2520 }, .{ 330, 2200, 2600 }, 190, 100));
    set(&t, .ay, dip(.{ 660, 1200, 2550 }, .{ 400, 1880, 2500 }, 250, 150));
    set(&t, .ow, dip(.{ 540, 1100, 2300 }, .{ 450, 900, 2300 }, 220, 80));
    set(&t, .aw, dip(.{ 640, 1230, 2550 }, .{ 420, 940, 2350 }, 260, 100));
    set(&t, .oy, dip(.{ 550, 960, 2400 }, .{ 360, 1820, 2450 }, 280, 150));

    // Stops: formant values here are the *loci* used for transitions.
    set(&t, .b, .{ .class = .stop, .voiced = true, .place = .labial, .f = .{ 200, 900, 2100 }, .f_end = .{ 200, 900, 2100 }, .inh_ms = 85, .min_ms = 60, .burst = lab_burst });
    set(&t, .p, .{ .class = .stop, .voiced = false, .place = .labial, .f = .{ 200, 900, 2100 }, .f_end = .{ 200, 900, 2100 }, .inh_ms = 90, .min_ms = 50, .burst = lab_burst, .vot_ms = 45 });
    set(&t, .d, .{ .class = .stop, .voiced = true, .place = .alveolar, .f = .{ 200, 1700, 2600 }, .f_end = .{ 200, 1700, 2600 }, .inh_ms = 75, .min_ms = 50, .burst = alv_burst });
    set(&t, .t, .{ .class = .stop, .voiced = false, .place = .alveolar, .f = .{ 200, 1700, 2600 }, .f_end = .{ 200, 1700, 2600 }, .inh_ms = 75, .min_ms = 50, .burst = alv_burst, .vot_ms = 55 });
    set(&t, .g, .{ .class = .stop, .voiced = true, .place = .velar, .f = .{ 200, 1990, 2850 }, .f_end = .{ 200, 1990, 2850 }, .inh_ms = 80, .min_ms = 60, .burst = vel_burst });
    set(&t, .k, .{ .class = .stop, .voiced = false, .place = .velar, .f = .{ 200, 1990, 2850 }, .f_end = .{ 200, 1990, 2850 }, .inh_ms = 80, .min_ms = 60, .burst = vel_burst, .vot_ms = 65 });

    set(&t, .ch, .{ .class = .affricate, .voiced = false, .place = .postalveolar, .f = .{ 300, 1840, 2750 }, .f_end = .{ 300, 1840, 2750 }, .inh_ms = 70, .min_ms = 50, .fric_amp = 0.8, .fric = sh_shape, .burst = alv_burst });
    set(&t, .jh, .{ .class = .affricate, .voiced = true, .place = .postalveolar, .f = .{ 260, 1840, 2750 }, .f_end = .{ 260, 1840, 2750 }, .inh_ms = 70, .min_ms = 50, .fric_amp = 0.55, .fric = sh_shape, .burst = alv_burst });

    set(&t, .f, .{ .class = .fricative, .voiced = false, .place = .labial, .f = .{ 340, 1100, 2080 }, .f_end = .{ 340, 1100, 2080 }, .inh_ms = 100, .min_ms = 80, .fric_amp = 0.14, .fric = f_shape });
    set(&t, .v, .{ .class = .fricative, .voiced = true, .place = .labial, .f = .{ 220, 1100, 2080 }, .f_end = .{ 220, 1100, 2080 }, .inh_ms = 60, .min_ms = 40, .fric_amp = 0.08, .fric = f_shape });
    set(&t, .th, .{ .class = .fricative, .voiced = false, .place = .dental, .f = .{ 320, 1290, 2540 }, .f_end = .{ 320, 1290, 2540 }, .inh_ms = 90, .min_ms = 60, .fric_amp = 0.12, .fric = th_shape });
    set(&t, .dh, .{ .class = .fricative, .voiced = true, .place = .dental, .f = .{ 270, 1290, 2540 }, .f_end = .{ 270, 1290, 2540 }, .inh_ms = 50, .min_ms = 30, .fric_amp = 0.05, .fric = th_shape });
    set(&t, .s, .{ .class = .fricative, .voiced = false, .place = .alveolar, .f = .{ 320, 1390, 2530 }, .f_end = .{ 320, 1390, 2530 }, .inh_ms = 105, .min_ms = 60, .fric_amp = 0.5, .fric = s_shape });
    set(&t, .z, .{ .class = .fricative, .voiced = true, .place = .alveolar, .f = .{ 240, 1390, 2530 }, .f_end = .{ 240, 1390, 2530 }, .inh_ms = 75, .min_ms = 40, .fric_amp = 0.3, .fric = s_shape });
    set(&t, .sh, .{ .class = .fricative, .voiced = false, .place = .postalveolar, .f = .{ 300, 1840, 2750 }, .f_end = .{ 300, 1840, 2750 }, .inh_ms = 105, .min_ms = 80, .fric_amp = 0.9, .fric = sh_shape });
    set(&t, .zh, .{ .class = .fricative, .voiced = true, .place = .postalveolar, .f = .{ 300, 1840, 2750 }, .f_end = .{ 300, 1840, 2750 }, .inh_ms = 70, .min_ms = 40, .fric_amp = 0.35, .fric = sh_shape });
    // /h/ takes the formants of the following vowel; its own values are unused.
    set(&t, .hh, .{ .class = .aspirate, .voiced = false, .place = .glottal, .f = .{ 500, 1500, 2500 }, .f_end = .{ 500, 1500, 2500 }, .inh_ms = 80, .min_ms = 20 });

    set(&t, .m, .{ .class = .nasal, .voiced = true, .place = .labial, .f = .{ 480, 1270, 2130 }, .f_end = .{ 480, 1270, 2130 }, .b = .{ 40, 300, 300 }, .inh_ms = 70, .min_ms = 60, .fnz = 450 });
    set(&t, .n, .{ .class = .nasal, .voiced = true, .place = .alveolar, .f = .{ 480, 1340, 2470 }, .f_end = .{ 480, 1340, 2470 }, .b = .{ 40, 300, 300 }, .inh_ms = 60, .min_ms = 50, .fnz = 450 });
    set(&t, .ng, .{ .class = .nasal, .voiced = true, .place = .velar, .f = .{ 480, 2000, 2900 }, .f_end = .{ 480, 2000, 2900 }, .b = .{ 40, 300, 300 }, .inh_ms = 95, .min_ms = 60, .fnz = 450 });

    set(&t, .l, .{ .class = .liquid, .voiced = true, .place = .alveolar, .f = .{ 310, 1050, 2880 }, .f_end = .{ 310, 1050, 2880 }, .b = .{ 50, 100, 280 }, .inh_ms = 80, .min_ms = 40 });
    set(&t, .r, .{ .class = .liquid, .voiced = true, .place = .postalveolar, .f = .{ 310, 1060, 1380 }, .f_end = .{ 310, 1060, 1380 }, .b = .{ 70, 100, 120 }, .inh_ms = 80, .min_ms = 30 });
    set(&t, .w, .{ .class = .glide, .voiced = true, .place = .labial, .f = .{ 290, 610, 2150 }, .f_end = .{ 290, 610, 2150 }, .b = .{ 50, 80, 60 }, .inh_ms = 80, .min_ms = 60 });
    set(&t, .y, .{ .class = .glide, .voiced = true, .place = .postalveolar, .f = .{ 260, 2070, 3020 }, .f_end = .{ 260, 2070, 3020 }, .b = .{ 40, 250, 500 }, .inh_ms = 80, .min_ms = 40 });
    break :blk t;
};

/// Schwa: the target an unstressed AH (and, partially, other reduced
/// vowels) moves toward.
pub const schwa = [3]f32{ 500, 1500, 2500 };

/// Coarse class used for context-dependent modelling (HMM) and unit
/// selection context matching.
pub const Broad = enum(u3) { pau, vowel, sonorant, stop, fricative };
pub const n_broad = 5;

pub fn broad(p: Phone) Broad {
    return switch (p.info().class) {
        .silence => .pau,
        .vowel, .diphthong => .vowel,
        .nasal, .liquid, .glide => .sonorant,
        .stop, .affricate => .stop,
        .fricative, .aspirate => .fricative,
    };
}

test "parse" {
    const r = Phone.parse("AA1").?;
    try std.testing.expectEqual(Phone.aa, r.phone);
    try std.testing.expectEqual(@as(u2, 1), r.stress);
    try std.testing.expectEqual(Phone.th, Phone.parse("TH").?.phone);
    try std.testing.expect(Phone.parse("XX") == null);
}
