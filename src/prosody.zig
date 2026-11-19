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
const dm = @import("duration_model.zig");

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

pub const Timing = enum {
    /// Klatt's (1979) duration rules with his inherent/minimum durations.
    klatt,
    /// The same kind of context factors, with values fitted to natural read
    /// speech (CMU ARCTIC, tools/fit-durations.py).
    fitted,
};

pub const Intonation = enum {
    /// Hand-set Fujisaki commands (the original rules).
    rules,
    /// The same model with command timing and sizes fitted to natural read
    /// speech (CMU ARCTIC, tools/intonation.py): accent peaks early in the
    /// vowel, milder downstep and declination, accents of unequal size, and
    /// almost no rise before commas.
    fitted,
};

pub const Options = struct {
    f0_model: F0Model = .fujisaki,
    /// null: rules (tts.Synth picks the fitted values for the refined
    /// formant engine). Applies to the Fujisaki model only.
    intonation: ?Intonation = null,
    /// null: Klatt rules (tts.Synth picks the fitted model for the refined
    /// formant engine).
    timing: ?Timing = null,
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
    /// Inside `*...*`: contrastive focus.
    emphasis: bool = false,

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
    const timing = opt.timing orelse .klatt;

    var segs: std.ArrayList(Seg) = .empty;
    var words: std.ArrayList(Word) = .empty;
    var phrases: std.ArrayList(Phrase) = .empty;

    try segs.append(arena, .{ .phone = .pau, .dur = 0.08 });

    var phrase_start: usize = segs.items.len;
    var sentence_first_word: ?[]const u8 = null;
    var first_in_sentence = true;
    var emph = false;

    for (toks.items, 0..) |tok, ti| {
        switch (tok) {
            .emph => emph = !emph,
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
                const w0 = segs.items.len;
                for (pron) |p| {
                    try segs.append(arena, .{
                        .phone = p.phone,
                        .stress = if (p.phone.isVowel()) p.stress else 0,
                        .word = wi,
                        .function_word = func and !emph,
                        .accent = (!func or emph) and p.phone.isVowel() and p.stress == 1,
                        .emphasis = emph,
                    });
                }
                if (emph) accentStrongest(segs.items[w0..]);
            },
            .punct => |p| {
                if (p == .none) continue;
                const is_last = ti + 1 == toks.items.len;
                try closePhrase(arena, &segs, &phrases, phrase_start, p, sentence_first_word, first_in_sentence, is_last, timing);
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
        try closePhrase(arena, &segs, &phrases, phrase_start, .period, sentence_first_word, first_in_sentence, true, timing);
    }
    // Make sure we end on a short pause.
    if (segs.items[segs.items.len - 1].phone != .pau) try segs.append(arena, .{ .phone = .pau, .dur = 0.15 });

    // A multi-syllable word with only secondary stress still needs one accent.
    ensureOneAccentPerContentWord(segs.items);
    deaccentProformOne(segs.items, words.items);

    switch (timing) {
        .klatt => assignDurations(segs.items, opt.rate),
        .fitted => assignDurationsFitted(segs.items, opt.rate),
    }
    // Focus lengthens the word, its accented vowel most.
    for (segs.items) |*s| if (s.emphasis) {
        s.dur *= if (s.accent) 1.3 else 1.1;
    };

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
        .fujisaki => switch (opt.intonation orelse .rules) {
            .rules => fujisakiContour(f0, segs.items, phrases.items, opt),
            .fitted => fujisakiFitted(f0, segs.items, phrases.items, opt),
        },
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
    timing: Timing,
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
        // Fitted timing: speakers in the data made a pause at only ~40% of
        // commas (56 ms on average); a synthetic voice keeps a short one,
        // since it has no other cue for the phrase boundary.
        .comma => if (timing == .fitted) 0.11 else 0.18,
        else => if (is_last) 0.25 else 0.4,
    };
    try segs.append(arena, .{ .phone = .pau, .dur = pause });
}

/// An emphasized word must carry an accent even if it has no primary stress
/// ("*the* answer"): put it on its most stressed vowel.
fn accentStrongest(word: []Seg) void {
    var best: ?*Seg = null;
    for (word) |*s| {
        if (s.accent) return;
        if (s.phone.isVowel() and (best == null or s.stress == 1 or (s.stress == 2 and best.?.stress == 0))) {
            if (best == null or s.stress > 0) best = s;
        }
    }
    if (best) |b| b.accent = true;
}

/// "the blue one", "a big one": after determiner + modifier, "one(s)" is a
/// pro-form that takes no accent (the modifier carries it). Elsewhere
/// ("chapter one", "one day") it is a number and keeps its accent.
fn deaccentProformOne(segs: []Seg, words: []const Word) void {
    const dets = [_][]const u8{ "the", "a", "an", "this", "that", "these", "those", "which", "another", "every", "each", "my", "your", "his", "her", "our", "their", "its" };
    for (segs) |*s| {
        const w = s.word orelse continue;
        if (w < 2 or s.emphasis) continue;
        const t = words[w].text;
        if (!std.ascii.eqlIgnoreCase(t, "one") and !std.ascii.eqlIgnoreCase(t, "ones")) continue;
        const d = words[w - 2].text;
        for (dets) |x| if (std.ascii.eqlIgnoreCase(d, x)) {
            s.accent = false;
            s.function_word = true;
        };
    }
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

// ---------------------------------------------------------------------------
// Durations — the same kind of rules with values fitted to recordings
// (src/duration_model.zig, generated by tools/fit-durations.py):
//   dur = base[phone] * product of the context factors that apply
// The factors mirror the Python feature extraction exactly.
// ---------------------------------------------------------------------------

fn assignDurationsFitted(segs: []Seg, rate: f32) void {
    const V = dm.vowel;
    const C = dm.consonant;
    for (segs, 0..) |*s, i| {
        if (s.phone == .pau) continue;
        const inf = s.phone.info();
        var ms = dm.base_ms[@intFromEnum(s.phone)];
        if (ms == 0) ms = inf.inh_ms * 0.7;
        const nxt: Phone = if (i + 1 < segs.len) segs[i + 1].phone else .pau;
        const prv: Phone = if (i > 0) segs[i - 1].phone else .pau;
        if (s.phone.isVowel()) {
            if (s.stress == 0) ms *= V.unstressed;
            if (s.stress == 2) ms *= V.secondary;
            if (s.function_word) ms *= V.function_word;
            if (s.accent) ms *= V.accent;
            if (s.phrase_final) ms *= V.phrase_final;
            if (syllablesInWord(segs, i) > 1) ms *= V.polysyllabic;
            if (!isLastVowelOfWord(segs, i)) ms *= V.not_word_final;
            const ni = nxt.info();
            const vless_stop = !ni.voiced and (ni.class == .stop or ni.class == .affricate);
            const voiced_obs = ni.voiced and (ni.class == .stop or ni.class == .affricate or ni.class == .fricative);
            const sonorant_c = ni.class == .nasal or ni.class == .liquid or ni.class == .glide;
            if (vless_stop) ms *= V.next_vless_stop;
            if (voiced_obs) ms *= V.next_voiced_obs;
            if (sonorant_c) ms *= V.next_sonorant;
            if (s.phrase_final and (voiced_obs or sonorant_c)) ms *= V.final_next_voiced;
        } else {
            const isC = struct {
                fn f(p: Phone) bool {
                    return p != .pau and !p.isVowel();
                }
            }.f;
            if (isC(prv) or isC(nxt)) ms *= C.cluster;
            if (s.phrase_final) ms *= C.phrase_final;
            if (s.function_word) ms *= C.function_word;
            if (i == 0 or segs[i - 1].word != s.word) ms *= C.word_initial;
            if (i + 1 >= segs.len or segs[i + 1].word != s.word) ms *= C.word_final;
            if (nxt.isVowel() and segs[i + 1].stress == 1) ms *= C.stressed_onset;
        }
        s.dur = @max(ms, 20) / 1000.0 / rate;
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
    return gaB(t, fuji_beta);
}

fn gaB(t: f32, beta: f32) f32 {
    if (t <= 0) return 0;
    return @min(1 - (1 + beta * t) * @exp(-beta * t), fuji_gamma);
}

/// Contrastive focus (`*word*`): the focused accent is boosted by `boost`, accents
/// before it in the phrase are reduced, and accents after it are almost
/// flattened (post-focus compression). 1 when the phrase has no focus.
fn focusScale(segs: []const Seg, ph: Phrase, i: usize, boost: f32) f32 {
    var focus: ?usize = null;
    for (ph.first_seg..ph.last_seg + 1) |k| if (segs[k].accent and segs[k].emphasis) {
        focus = k;
        break;
    };
    const f = focus orelse return 1;
    if (segs[i].emphasis) return boost;
    return if (i < f) 0.6 else 0.25;
}

const Cmd = struct { t: f32, t2: f32 = 0, amp: f32, accent: bool, fall: f32 = 0 };

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
            // The rules' steep downstep and final lowering bury a late
            // focus unless it is boosted more.
            amp *= focusScale(segs, ph, i, 3.0);
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

// The fitted variant. Measured on 60 ARCTIC a-set prompts x 3 speakers
// (tools/intonation.py), the rules put accent peaks at the end of the vowel
// (natural: 10-30 ms after onset), declined 2-3x too fast (accent amplitude
// fell 15% per accent and a large phrase command decayed under it), made all
// accents nearly the same size, and rose 2 st before commas where readers
// stay level.

/// Fitted command values (amplitudes in ln F0 units), chosen so that the
/// measures of tools/intonation.py on the a-set fall within the range of the
/// three speakers; checked on the b-set.
const Fit = struct {
    /// Phrase commands, sentence-initial and after a comma.
    phrase_first: f32 = 0.28,
    phrase_later: f32 = 0.15,
    accent: f32 = 0.26,
    /// Each later accent in a phrase is this much smaller (rules: 0.85).
    downstep: f32 = 0.97,
    /// Accent command onset re the start of the syllable onset (s).
    t1: f32 = 0.0,
    /// Accent command offset re vowel onset, as a fraction of the vowel.
    t2_frac: f32 = 0.5,
    /// Rise and fall rates: accents rise fast and fall slowly, which keeps
    /// F0 up between accents as in the recordings.
    beta: f32 = 25,
    beta_fall: f32 = 3,
    /// Accent size varies by +- this fraction per word (wordJitter).
    vary: f32 = 0.8,
    /// Phrase commands at the end: sentence-final lowering, and a smaller
    /// fall before a comma (rules: a continuation rise).
    final_low: f32 = -0.2,
    comma: f32 = -0.1,
    /// Focus (`*word*`): size of the focused accent.
    focus: f32 = 1.9,
    /// Fall rate after the last accent of a statement.
    beta_final: f32 = 8,
    /// Yes/no questions: nuclear accent scale and final rise. ARCTIC has
    /// too few questions to fit these; set by ear.
    question_nuclear: f32 = 0.3,
    question_rise: f32 = 0.5,
};

/// Start of the syllable whose vowel is segs[i]: back over the consonants
/// before it within the word (at most two), so the accent rise begins
/// with the onset, as it does in speech.
fn syllableOnset(segs: []const Seg, i: usize) f32 {
    var k = i;
    var n: usize = 0;
    while (k > 0 and n < 2) {
        const p = segs[k - 1];
        if (p.phone == .pau or p.phone.isVowel() or p.word != segs[i].word) break;
        k -= 1;
        n += 1;
    }
    return segs[k].start;
}

/// Stable pseudo-random number in [-1, 1] for a word position: readers
/// don't give every accent the same weight, and which ones they favour
/// can't be predicted from the text alone.
fn wordJitter(segs: []const Seg, i: usize) f32 {
    var h: u32 = 2166136261;
    var k = i;
    while (k > 0 and segs[k - 1].word == segs[i].word) k -= 1;
    while (k < segs.len and segs[k].word == segs[i].word) : (k += 1) {
        h = (h ^ @intFromEnum(segs[k].phone)) *% 16777619;
    }
    h = (h ^ @as(u32, @intCast(i))) *% 16777619;
    h ^= h >> 13;
    h *%= 0x5bd1e995;
    h ^= h >> 15;
    return @as(f32, @floatFromInt(h & 0xffff)) / 32767.5 - 1;
}

fn fujisakiFitted(f0: []f32, segs: []const Seg, phrases: []const Phrase, opt: Options) void {
    const P: Fit = .{};
    var cmds: [256]Cmd = undefined;
    var n: usize = 0;
    const range = opt.pitch_range;
    for (phrases) |ph| {
        if (n + 3 > cmds.len) break;
        const t0 = segs[ph.first_seg].start;
        const t1 = segs[ph.last_seg].end();
        cmds[n] = .{ .t = t0 - 0.25, .amp = range * (if (ph.first_in_sentence) P.phrase_first else P.phrase_later), .accent = false };
        n += 1;
        var k: usize = 0;
        const boost: f32 = if (ph.kind == .exclaim) 1.4 else 1.0;
        // With a focus in the phrase, the focus alone decides prominence:
        // no random accent sizes, and the final fall waits for its peak.
        var last_acc: usize = ph.last_seg + 1;
        for (ph.first_seg..ph.last_seg + 1) |i| if (segs[i].accent) {
            last_acc = i;
        };
        var focus_end: f32 = 0;
        for (ph.first_seg..ph.last_seg + 1) |i| if (segs[i].accent and segs[i].emphasis) {
            focus_end = segs[i].start + P.t2_frac * segs[i].dur;
        };
        for (ph.first_seg..ph.last_seg + 1) |i| {
            if (!segs[i].accent or n + 2 > cmds.len) continue;
            var amp = P.accent * std.math.pow(f32, P.downstep, @floatFromInt(k)) * boost;
            if (focus_end == 0) amp *= 1 + P.vary * wordJitter(segs, i);
            amp *= focusScale(segs, ph, i, P.focus);
            const on = syllableOnset(segs, i) + P.t1;
            const off = segs[i].start + P.t2_frac * segs[i].dur;
            // After a focus F0 drops quickly and stays low (post-focus compression).
            var fall = if (segs[i].emphasis) P.beta else P.beta_fall;
            if (i == last_acc and ph.kind != .comma) {
                if (ph.rising) {
                    // Yes/no question: a low nuclear accent, then the rise
                    // (L* H-H%); a high accent under the rise overshoots.
                    amp *= P.question_nuclear;
                } else {
                    // Declarative nucleus: the fall to the end is quick (H* L-L%).
                    fall = P.beta_final;
                }
            }
            cmds[n] = .{ .t = on, .t2 = @max(off, on + 0.05), .amp = amp * range, .accent = true, .fall = fall };
            n += 1;
            k += 1;
        }
        if (ph.rising) {
            var lv = ph.last_seg;
            while (lv > ph.first_seg and !segs[lv].phone.isVowel()) lv -= 1;
            cmds[n] = .{ .t = segs[lv].start, .t2 = t1 + 0.4, .amp = P.question_rise * range, .accent = true, .fall = P.beta };
            n += 1;
        } else if (ph.kind != .comma) {
            cmds[n] = .{ .t = @max(t1 - 0.35, focus_end), .amp = P.final_low * range, .accent = false };
            n += 1;
        } else {
            // Readers mostly fall to a low-mid level before a comma.
            cmds[n] = .{ .t = t1 - 0.35, .amp = P.comma * range, .accent = false };
            n += 1;
        }
    }
    const ln_fb = @log(opt.base_f0 * 0.8);
    for (f0, 0..) |*v, fi| {
        const t = @as(f32, @floatFromInt(fi)) / frame_rate;
        var acc: f32 = ln_fb;
        for (cmds[0..n]) |c| {
            if (c.accent) {
                // Rise with beta to the level reached at T2, then fall
                // back at the command's own rate.
                const top = gaB(@min(t, c.t2) - c.t, P.beta);
                acc += c.amp * (if (t <= c.t2) top else top * (1 - gaB(t - c.t2, c.fall) / fuji_gamma));
            } else acc += c.amp * gp(t - c.t);
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

test "emphasis moves the main prominence" {
    var lex = try lexicon.Lexicon.init(std.testing.allocator);
    defer lex.deinit();
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    // Fitted only: under the rules' steep declination a late focus rises
    // well above its surroundings but not above the sentence's first accent.
    inline for (.{Intonation.fitted}) |into| {
        const u = try plan(a, &lex, "Mary gave Tom the *book*.", .{ .intonation = into });
        var peak = [2]f32{ 0, 0 }; // first word, focused word
        for (u.segs) |s| if (s.accent) {
            const k: usize = if (s.emphasis) 1 else if (s.word == 0) 0 else continue;
            peak[k] = @max(peak[k], u.f0At(s.start + s.dur * 0.5));
        };
        try std.testing.expect(peak[1] > peak[0]);
    }
}
