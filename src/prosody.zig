//! Stage 3: from tokens to a timed, pitched phone sequence (an Utterance).
//!
//! The whole utterance is planned before any audio is made: durations need
//! to know where phrases end, and the F0 contour needs to know whether the
//! sentence is a question and how long it is. Every waveform engine consumes
//! the same Utterance, so prosody is held constant when engines are compared.

const std = @import("std");
const phones = @import("phones.zig");
const Phone = phones.Phone;
const text = @import("text.zig");
const lexicon = @import("lexicon.zig");

/// Parameter frame rate shared by all engines: one frame every 5 ms.
pub const frame_rate: f32 = 200;

pub const F0Model = enum {
    /// Constant pitch: the "robot" baseline.
    flat,
    /// MITalk-style rules: declining baseline and topline, rise at the first
    /// accent, fall (or rise) after the last.
    hat,
    /// Fujisaki command-response model.
    fujisaki,
};

pub const Options = struct {
    f0_model: F0Model = .fujisaki,
    base_f0: f32 = 110,
    /// Scales the size of accents and phrase arcs.
    pitch_range: f32 = 1.0,
    /// > 1 is faster.
    rate: f32 = 1.0,
};

pub const Seg = struct {
    phone: Phone,
    stress: u2 = 0,
    start: f32 = 0,
    dur: f32 = 0,
    /// Index into Utterance.words; null for pauses.
    word: ?u16 = null,
    /// Carries a pitch accent (primary-stressed vowel of a content word).
    accent: bool = false,
    phrase_final: bool = false,
    function_word: bool = false,

    pub fn end(s: Seg) f32 {
        return s.start + s.dur;
    }
};

pub const Word = struct {
    text: []const u8,
    source: lexicon.Lexicon.Source,
};

pub const Utterance = struct {
    segs: []Seg,
    words: []Word,
    /// F0 in Hz, one value per frame, defined everywhere (engines decide
    /// where voicing happens).
    f0: []f32,
    duration: f32,

    pub fn f0At(u: *const Utterance, t: f32) f32 {
        const x = t * frame_rate;
        if (x <= 0) return u.f0[0];
        const i: usize = @intFromFloat(x);
        if (i + 1 >= u.f0.len) return u.f0[u.f0.len - 1];
        const fr = x - @as(f32, @floatFromInt(i));
        return u.f0[i] * (1 - fr) + u.f0[i + 1] * fr;
    }

    /// "hh ah0 l ow1 | w er1 l d" style transcription, for display.
    pub fn transcription(u: *const Utterance, a: std.mem.Allocator) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        var last_word: ?u16 = null;
        for (u.segs) |s| {
            if (s.phone == .pau) {
                if (out.items.len > 0 and out.items[out.items.len - 1] != ' ') try out.append(a, ' ');
                if (out.items.len > 0) try out.appendSlice(a, "/ ");
                last_word = null;
                continue;
            }
            if (last_word != null and s.word != last_word) try out.appendSlice(a, "| ");
            last_word = s.word;
            try out.appendSlice(a, s.phone.name());
            if (s.phone.isVowel()) try out.append(a, '0' + @as(u8, s.stress));
            if (s.accent) try out.append(a, '*');
            try out.append(a, ' ');
        }
        while (out.items.len > 0 and (out.items[out.items.len - 1] == ' ' or out.items[out.items.len - 1] == '/')) out.items.len -= 1;
        return out.items;
    }
};

const Phrase = struct {
    first_seg: usize,
    last_seg: usize, // inclusive, last non-pause seg
    kind: text.Punct,
    rising: bool,
    first_in_sentence: bool,
};

/// Plan an utterance. Everything is allocated in `arena`.
pub fn plan(arena: std.mem.Allocator, lex: *const lexicon.Lexicon, input: []const u8, opt: Options) !Utterance {
    const toks = try text.tokenize(arena, input);

    var segs: std.ArrayList(Seg) = .empty;
    var words: std.ArrayList(Word) = .empty;
    var phrases: std.ArrayList(Phrase) = .empty;

    try segs.append(arena, .{ .phone = .pau, .dur = 0.08 });

    var phrase_start: usize = segs.items.len;
    var sentence_first_word: ?[]const u8 = null;
    var first_in_sentence = true;

    for (toks.items, 0..) |tok, ti| {
        switch (tok) {
            .word, .letter => {
                var src: lexicon.Lexicon.Source = .spelled;
                const w_text: []const u8 = switch (tok) {
                    .word => |w| w,
                    .letter => |l| try std.fmt.allocPrint(arena, "{c}", .{l}),
                    else => unreachable,
                };
                const pron = switch (tok) {
                    .word => |w| try lex.pronounce(arena, w, &src),
                    .letter => |l| lexicon.letterName(l),
                    else => unreachable,
                };
                if (pron.len == 0) continue;
                if (sentence_first_word == null) sentence_first_word = w_text;
                const func = tok == .word and lexicon.isFunctionWord(w_text);
                const wi: u16 = @intCast(words.items.len);
                try words.append(arena, .{ .text = w_text, .source = src });
                for (pron) |p| {
                    try segs.append(arena, .{
                        .phone = p.phone,
                        .stress = if (p.phone.isVowel()) p.stress else 0,
                        .word = wi,
                        .function_word = func,
                        .accent = !func and p.phone.isVowel() and p.stress == 1,
                    });
                }
            },
            .punct => |p| {
                if (p == .none) continue;
                const is_last = ti + 1 == toks.items.len;
                try closePhrase(arena, &segs, &phrases, phrase_start, p, sentence_first_word, first_in_sentence, is_last);
                phrase_start = segs.items.len;
                if (p == .comma) {
                    first_in_sentence = false;
                } else {
                    first_in_sentence = true;
                    sentence_first_word = null;
                }
            },
        }
    }
    if (segs.items.len > phrase_start) {
        try closePhrase(arena, &segs, &phrases, phrase_start, .period, sentence_first_word, first_in_sentence, true);
    }
    // Make sure we end on a short pause.
    if (segs.items[segs.items.len - 1].phone != .pau) try segs.append(arena, .{ .phone = .pau, .dur = 0.15 });

    // A multi-syllable word with only secondary stress still needs one accent.
    ensureOneAccentPerContentWord(segs.items);

    assignDurations(segs.items, opt.rate);

    var t: f32 = 0;
    for (segs.items) |*s| {
        s.start = t;
        t += s.dur;
    }
    const n_frames: usize = @intFromFloat(@ceil(t * frame_rate) + 1);
    const f0 = try arena.alloc(f32, n_frames);
    switch (opt.f0_model) {
        .flat => @memset(f0, opt.base_f0),
        .hat => hatContour(f0, segs.items, phrases.items, opt),
        .fujisaki => fujisakiContour(f0, segs.items, phrases.items, opt),
    }

    return .{ .segs = segs.items, .words = words.items, .f0 = f0, .duration = t };
}

fn closePhrase(
    arena: std.mem.Allocator,
    segs: *std.ArrayList(Seg),
    phrases: *std.ArrayList(Phrase),
    start: usize,
    kind: text.Punct,
    sentence_first_word: ?[]const u8,
    first_in_sentence: bool,
    is_last: bool,
) !void {
    if (segs.items.len <= start) return;
    const last = segs.items.len - 1;
    if (segs.items[last].phone == .pau) return;
    // Yes/no questions rise; wh-questions fall like statements.
    const wh = if (sentence_first_word) |w| lexicon.isWhWord(w) else false;
    try phrases.append(arena, .{
        .first_seg = start,
        .last_seg = last,
        .kind = kind,
        .rising = kind == .question and !wh,
        .first_in_sentence = first_in_sentence,
    });
    // Phrase-final lengthening domain: last vowel of the phrase onward.
    var i = last + 1;
    while (i > start) {
        i -= 1;
        segs.items[i].phrase_final = true;
        if (segs.items[i].phone.isVowel()) break;
    }
    const pause: f32 = switch (kind) {
        .comma => 0.18,
        else => if (is_last) 0.25 else 0.4,
    };
    try segs.append(arena, .{ .phone = .pau, .dur = pause });
}

fn ensureOneAccentPerContentWord(segs: []Seg) void {
    var i: usize = 0;
    while (i < segs.len) {
        const w = segs[i].word orelse {
            i += 1;
            continue;
        };
        var j = i;
        var has = false;
        while (j < segs.len and segs[j].word == w) : (j += 1) has = has or segs[j].accent;
        if (!has and !segs[i].function_word) {
            for (segs[i..j]) |*s| if (s.phone.isVowel() and s.stress > 0) {
                s.accent = true;
                break;
            };
        }
        i = j;
    }
}

// ---------------------------------------------------------------------------
// Durations — a compact version of Klatt's (1979) rules:
//   dur = (inherent - minimum) * PRCNT + minimum
// ---------------------------------------------------------------------------

fn assignDurations(segs: []Seg, rate: f32) void {
    for (segs, 0..) |*s, i| {
        if (s.phone == .pau) continue;
        const inf = s.phone.info();
        var prc: f32 = 1.0;
        var min = inf.min_ms;
        if (s.phone.isVowel()) {
            if (s.stress == 0 or s.function_word) {
                prc *= 0.6;
                min *= 0.5;
            } else if (s.stress == 2) prc *= 0.85;
            const sylls = syllablesInWord(segs, i);
            if (sylls > 1) prc *= 0.85;
            if (!isLastVowelOfWord(segs, i)) prc *= 0.85;
            if (nextConsonant(segs, i)) |c| {
                const ci = c.info();
                if (!ci.voiced and (ci.class == .stop or ci.class == .affricate)) {
                    prc *= if (s.phrase_final) 0.7 else 0.85;
                } else if (ci.voiced and s.phrase_final) prc *= 1.2;
            }
            if (s.phrase_final) prc *= 1.4;
            if (s.accent) prc *= 1.1;
        } else {
            const prev_c = i > 0 and !segs[i - 1].phone.isVowel() and segs[i - 1].phone != .pau;
            const next_c = i + 1 < segs.len and !segs[i + 1].phone.isVowel() and segs[i + 1].phone != .pau;
            if (prev_c or next_c) prc *= 0.7;
            if (s.phrase_final) prc *= 1.3;
            if (s.function_word) prc *= 0.8;
        }
        const ms = (inf.inh_ms - min) * prc + min;
        s.dur = ms / 1000.0 / rate;
    }
    for (segs) |*s| if (s.phone == .pau) {
        s.dur /= rate;
    };
}

fn syllablesInWord(segs: []const Seg, i: usize) usize {
    const w = segs[i].word orelse return 0;
    var n: usize = 0;
    for (segs) |s| if (s.word == w and s.phone.isVowel()) {
        n += 1;
    };
    return n;
}

fn isLastVowelOfWord(segs: []const Seg, i: usize) bool {
    var j = i + 1;
    while (j < segs.len and segs[j].word == segs[i].word) : (j += 1) {
        if (segs[j].phone.isVowel()) return false;
    }
    return true;
}

fn nextConsonant(segs: []const Seg, i: usize) ?Phone {
    if (i + 1 >= segs.len) return null;
    const p = segs[i + 1].phone;
    if (p == .pau or p.isVowel()) return null;
    return p;
}

// ---------------------------------------------------------------------------
// F0: hat pattern
// ---------------------------------------------------------------------------

fn hatContour(f0: []f32, segs: []const Seg, phrases: []const Phrase, opt: Options) void {
    const base = opt.base_f0;
    const range = opt.pitch_range;
    @memset(f0, base * 0.95);
    for (phrases) |ph| {
        const t0 = segs[ph.first_seg].start;
        const t1 = segs[ph.last_seg].end();
        var first_acc: ?usize = null;
        var last_acc: ?usize = null;
        for (ph.first_seg..ph.last_seg + 1) |i| if (segs[i].accent) {
            if (first_acc == null) first_acc = i;
            last_acc = i;
        };
        const start_level: f32 = if (ph.first_in_sentence) 1.0 else 0.95;
        const fa = first_acc orelse ph.first_seg;
        const la = last_acc orelse ph.last_seg;
        const rise_t = segs[fa].start;
        const fall_t = segs[la].start + segs[la].dur * 0.4;
        const prev = if (ph.first_seg > 0) ph.first_seg - 1 else 0;
        const fr0: usize = @intFromFloat(segs[prev].start * frame_rate);
        const fr1: usize = @min(f0.len, @as(usize, @intFromFloat(segs[ph.last_seg + 1].end() * frame_rate)));
        for (fr0..fr1) |fi| {
            const t = @as(f32, @floatFromInt(fi)) / frame_rate;
            const prog = std.math.clamp((t - t0) / @max(0.01, t1 - t0), 0, 1);
            const baseline = base * (start_level - 0.15 * prog);
            const topline = baseline * (1 + 0.3 * range);
            var v: f32 = baseline;
            if (t >= rise_t - 0.06 and t < rise_t) {
                v = baseline + (topline - baseline) * (t - (rise_t - 0.06)) / 0.06;
            } else if (t >= rise_t and t <= fall_t) {
                v = topline;
            } else if (t > fall_t) {
                const k = std.math.clamp((t - fall_t) / 0.15, 0, 1);
                if (ph.rising) {
                    v = topline * (1 - k) + base * (1 + 0.55 * range) * k;
                    if (t > t1) v = base * (1 + 0.55 * range);
                } else {
                    const end_v = baseline * (if (ph.kind == .comma) @as(f32, 1.0) else 0.9);
                    v = topline * (1 - k) + end_v * k;
                }
            }
            f0[fi] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// F0: Fujisaki command-response model
//   ln F0(t) = ln Fb + sum Ap Gp(t - T0) + sum Aa [Ga(t - T1) - Ga(t - T2)]
// ---------------------------------------------------------------------------

const fuji_alpha: f32 = 3.0;
const fuji_beta: f32 = 20.0;
const fuji_gamma: f32 = 0.9;

fn gp(t: f32) f32 {
    if (t <= 0) return 0;
    return fuji_alpha * fuji_alpha * t * @exp(-fuji_alpha * t);
}

fn ga(t: f32) f32 {
    if (t <= 0) return 0;
    return @min(1 - (1 + fuji_beta * t) * @exp(-fuji_beta * t), fuji_gamma);
}

const Cmd = struct { t: f32, t2: f32 = 0, amp: f32, accent: bool };

fn fujisakiContour(f0: []f32, segs: []const Seg, phrases: []const Phrase, opt: Options) void {
    var cmds: [256]Cmd = undefined;
    var n: usize = 0;
    const range = opt.pitch_range;
    for (phrases) |ph| {
        if (n + 3 > cmds.len) break;
        const t0 = segs[ph.first_seg].start;
        const t1 = segs[ph.last_seg].end();
        cmds[n] = .{ .t = t0 - 0.25, .amp = range * (if (ph.first_in_sentence) @as(f32, 0.4) else 0.25), .accent = false };
        n += 1;
        var k: usize = 0;
        var last_acc: ?usize = null;
        for (ph.first_seg..ph.last_seg + 1) |i| if (segs[i].accent) {
            last_acc = i;
        };
        const boost: f32 = if (ph.kind == .exclaim) 1.4 else 1.0;
        for (ph.first_seg..ph.last_seg + 1) |i| {
            if (!segs[i].accent or n + 2 > cmds.len) continue;
            const nuclear = i == last_acc.?;
            var amp = 0.32 * std.math.pow(f32, 0.85, @floatFromInt(k)) * boost;
            if (nuclear and !ph.rising) amp *= 0.9;
            cmds[n] = .{ .t = segs[i].start - 0.04, .t2 = segs[i].end() - 0.02, .amp = amp * range, .accent = true };
            n += 1;
            k += 1;
        }
        if (ph.rising) {
            // Final rise: a strong accent command from the last vowel onward.
            var lv = ph.last_seg;
            while (lv > ph.first_seg and !segs[lv].phone.isVowel()) lv -= 1;
            cmds[n] = .{ .t = segs[lv].start, .t2 = t1 + 0.4, .amp = 0.7 * range, .accent = true };
            n += 1;
        } else if (ph.kind != .comma) {
            // Final lowering.
            cmds[n] = .{ .t = t1 - 0.35, .amp = -0.25 * range, .accent = false };
            n += 1;
        } else {
            // Continuation rise before a comma.
            var lv = ph.last_seg;
            while (lv > ph.first_seg and !segs[lv].phone.isVowel()) lv -= 1;
            cmds[n] = .{ .t = segs[lv].start + 0.02, .t2 = t1 + 0.1, .amp = 0.15 * range, .accent = true };
            n += 1;
        }
    }
    const fb = opt.base_f0 * 0.8;
    const ln_fb = @log(fb);
    for (f0, 0..) |*v, fi| {
        const t = @as(f32, @floatFromInt(fi)) / frame_rate;
        var acc: f32 = ln_fb;
        for (cmds[0..n]) |c| {
            if (c.accent) acc += c.amp * (ga(t - c.t) - ga(t - c.t2)) else acc += c.amp * gp(t - c.t);
        }
        v.* = @exp(acc);
    }
}

test "plan a sentence" {
    var lex = try lexicon.Lexicon.init(std.testing.allocator);
    defer lex.deinit();
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    inline for (.{ F0Model.flat, F0Model.hat, F0Model.fujisaki }) |m| {
        const u = try plan(a, &lex, "Hello world, is this a question?", .{ .f0_model = m });
        try std.testing.expect(u.duration > 1.0 and u.duration < 4.0);
        // Rising question: the pitch ends well above the nuclear accent.
        if (m != .flat) {
            var lv = u.segs.len - 1;
            while (!u.segs[lv].accent) lv -= 1;
            const last_v = u.segs[lv];
            try std.testing.expect(u.f0At(u.segs[u.segs.len - 2].end()) > u.f0At(last_v.start) * 1.1);
        }
    }
}
