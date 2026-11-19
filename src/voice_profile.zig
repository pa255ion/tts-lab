//! A voice profile: the refined formant engine's acoustic targets, fitted to
//! one speaker's recordings (tools/fit-voice.py), as a small text file.
//!
//! Everything in it overrides a default from phones.zig or prosody.Options;
//! anything left out keeps the default, so an empty profile is the stock
//! voice. Format, one item per line, `#` starts a comment:
//!
//!   name   NAME
//!   phone  iy  f 302 2210 2880  [fe F1 F2 F3]  [b B1 B2 B3]  [sd S1 S2 S3]  [dur X]  [amp X]
//!   style  neutral  f0 112  range 1.0  rate 1.0  rd 0.0  breath 1.0
//!
//! `f`/`fe`: formant targets at the start/end (Hz; for stops, the loci).
//! `b`: bandwidths. `sd`: token-to-token spread of F1-F3 (Hz), used by
//! --variation. `dur`: duration multiplier. `amp`: frication/burst level
//! multiplier. A style sets the prosody (`f0`: the planner's base F0,
//! `range`: pitch range factor, `rate`: speaking rate) and the voice source
//! (`rd`: added to the LF Rd, + = laxer/breathier, - = tenser; `breath`:
//! breath-noise multiplier). The first style is the default.
//!
//! Parsed from bytes, so a profile can be embedded or handed over by a
//! front end (no file system needed: works on wasm).

const std = @import("std");
const phones = @import("phones.zig");
const Phone = phones.Phone;

pub const PhoneEntry = struct {
    f: ?[3]f32 = null,
    f_end: ?[3]f32 = null,
    b: ?[3]f32 = null,
    sd: [3]f32 = .{ 0, 0, 0 },
    dur: f32 = 1,
    amp: f32 = 1,
};

pub const Style = struct {
    name: []const u8,
    base_f0: ?f32 = null,
    pitch_range: f32 = 1,
    rate: f32 = 1,
    rd: f32 = 0,
    breath: f32 = 1,
};

pub const Profile = struct {
    name: []const u8 = "",
    phone: [Phone.count]PhoneEntry = @splat(.{}),
    styles: []const Style = &.{},

    /// The named style, or the first one; null if the profile has none.
    pub fn style(p: *const Profile, name: ?[]const u8) ?*const Style {
        if (name) |n| for (p.styles) |*s| if (std.mem.eql(u8, s.name, n)) return s;
        return if (p.styles.len > 0) &p.styles[0] else null;
    }

    pub fn entry(p: *const Profile, ph: Phone) *const PhoneEntry {
        return &p.phone[@intFromEnum(ph)];
    }
};

pub const ParseError = error{ BadLine, OutOfMemory };

/// Parse a profile. Strings and the style list are allocated in `arena`.
/// On error, `line_no` (if given) is set to the offending line.
pub fn parse(arena: std.mem.Allocator, src: []const u8, line_no: ?*usize) ParseError!Profile {
    var p: Profile = .{};
    var styles: std.ArrayList(Style) = .empty;
    var lines = std.mem.splitScalar(u8, src, '\n');
    var ln: usize = 0;
    while (lines.next()) |raw| {
        ln += 1;
        if (line_no) |l| l.* = ln;
        const line = if (std.mem.indexOfScalar(u8, raw, '#')) |h| raw[0..h] else raw;
        var it = std.mem.tokenizeAny(u8, line, " \t\r");
        const kw = it.next() orelse continue;
        if (std.mem.eql(u8, kw, "name")) {
            p.name = try arena.dupe(u8, it.rest());
        } else if (std.mem.eql(u8, kw, "phone")) {
            const ph = (Phone.parse(it.next() orelse return error.BadLine) orelse return error.BadLine).phone;
            var e: PhoneEntry = .{};
            while (it.next()) |key| {
                if (std.mem.eql(u8, key, "f")) {
                    e.f = try three(&it);
                } else if (std.mem.eql(u8, key, "fe")) {
                    e.f_end = try three(&it);
                } else if (std.mem.eql(u8, key, "b")) {
                    e.b = try three(&it);
                } else if (std.mem.eql(u8, key, "sd")) {
                    e.sd = try three(&it);
                } else if (std.mem.eql(u8, key, "dur")) {
                    e.dur = std.math.clamp(try num(&it), 0.3, 3);
                } else if (std.mem.eql(u8, key, "amp")) {
                    e.amp = std.math.clamp(try num(&it), 0, 8);
                } else if (std.mem.eql(u8, key, "n")) {
                    _ = try num(&it); // token count, informational
                } else return error.BadLine;
            }
            p.phone[@intFromEnum(ph)] = e;
        } else if (std.mem.eql(u8, kw, "style")) {
            var s: Style = .{ .name = try arena.dupe(u8, it.next() orelse return error.BadLine) };
            while (it.next()) |key| {
                const v = try num(&it);
                if (std.mem.eql(u8, key, "f0")) {
                    s.base_f0 = std.math.clamp(v, 40, 500);
                } else if (std.mem.eql(u8, key, "range")) {
                    s.pitch_range = std.math.clamp(v, 0, 4);
                } else if (std.mem.eql(u8, key, "rate")) {
                    s.rate = std.math.clamp(v, 0.4, 2.5);
                } else if (std.mem.eql(u8, key, "rd")) {
                    s.rd = std.math.clamp(v, -1, 2);
                } else if (std.mem.eql(u8, key, "breath")) {
                    s.breath = std.math.clamp(v, 0, 5);
                } else return error.BadLine;
            }
            try styles.append(arena, s);
        } else return error.BadLine;
    }
    p.styles = styles.items;
    return p;
}

fn num(it: anytype) ParseError!f32 {
    const t = it.next() orelse return error.BadLine;
    const v = std.fmt.parseFloat(f32, t) catch return error.BadLine;
    if (!std.math.isFinite(v)) return error.BadLine;
    return v;
}

fn three(it: anytype) ParseError![3]f32 {
    var out: [3]f32 = undefined;
    for (&out) |*o| o.* = std.math.clamp(try num(it), 20, 7000);
    return out;
}

test "parse a profile" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const p = try parse(a,
        \\# test
        \\name test voice
        \\phone iy f 300 2200 2900 sd 20 80 100 n 41
        \\phone ey f 450 1800 2500 fe 320 2250 2700 dur 1.1
        \\style neutral f0 118 range 1.2
        \\style lively f0 130 range 1.6 rate 1.1 rd -0.1
    , null);
    try std.testing.expectEqualStrings("test voice", p.name);
    try std.testing.expectEqual(@as(f32, 2200), p.entry(.iy).f.?[1]);
    try std.testing.expectEqual(@as(f32, 2250), p.entry(.ey).f_end.?[1]);
    try std.testing.expect(p.entry(.aa).f == null);
    try std.testing.expectEqual(@as(f32, 118), p.style(null).?.base_f0.?);
    try std.testing.expectEqual(@as(f32, 1.1), p.style("lively").?.rate);
    var ln: usize = 0;
    try std.testing.expectError(error.BadLine, parse(a, "name x\nphone zz f 1 2 3\n", &ln));
    try std.testing.expectEqual(@as(usize, 2), ln);
}
