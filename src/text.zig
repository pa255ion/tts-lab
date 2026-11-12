//! Stage 1: text normalization and tokenization.
//!
//! Turns raw text into a flat list of tokens: words (lower-case, letters and
//! apostrophes only), single letters to be spelled out, and punctuation that
//! matters for prosody. Numbers, currency, ordinals, years and a handful of
//! abbreviations are expanded into words here, so that later stages only ever
//! see pronounceable words.

const std = @import("std");

pub const Punct = enum { none, comma, period, question, exclaim };

pub const Token = union(enum) {
    word: []const u8,
    letter: u8,
    punct: Punct,
};

pub const Tokens = std.ArrayList(Token);

/// All allocations go into `arena`; the caller frees it as a whole.
pub fn tokenize(arena: std.mem.Allocator, text: []const u8) !Tokens {
    var out: Tokens = .empty;
    var i: usize = 0;
    while (i < text.len) {
        const c = text[i];
        if (std.ascii.isAlphabetic(c)) {
            const start = i;
            while (i < text.len and (std.ascii.isAlphabetic(text[i]) or text[i] == '\'')) i += 1;
            const raw = text[start..i];
            // An abbreviation keeps its trailing period from ending the sentence.
            const has_dot = i < text.len and text[i] == '.';
            if (has_dot) {
                if (abbreviation(raw, nextWordCapitalized(text, i + 1))) |exp| {
                    i += 1;
                    try appendWords(arena, &out, exp);
                    continue;
                }
            }
            try appendWord(arena, &out, raw);
        } else if (std.ascii.isDigit(c) or (c == '$' and i + 1 < text.len and std.ascii.isDigit(text[i + 1]))) {
            i = try number(arena, &out, text, i);
        } else {
            switch (c) {
                ',', ';', ':' => try out.append(arena, .{ .punct = .comma }),
                '.' => try out.append(arena, .{ .punct = .period }),
                '?' => try out.append(arena, .{ .punct = .question }),
                '!' => try out.append(arena, .{ .punct = .exclaim }),
                '%' => try appendWords(arena, &out, "percent"),
                '&' => try appendWords(arena, &out, "and"),
                '+' => try appendWords(arena, &out, "plus"),
                '=' => try appendWords(arena, &out, "equals"),
                '@' => try appendWords(arena, &out, "at"),
                else => {},
            }
            i += 1;
        }
    }
    return out;
}

fn nextWordCapitalized(text: []const u8, from: usize) bool {
    var j = from;
    while (j < text.len and text[j] == ' ') j += 1;
    return j < text.len and std.ascii.isUpper(text[j]);
}

fn abbreviation(raw: []const u8, next_cap: bool) ?[]const u8 {
    var buf: [8]u8 = undefined;
    if (raw.len > buf.len) return null;
    const l = std.ascii.lowerString(&buf, raw);
    const map = [_]struct { []const u8, []const u8 }{
        .{ "dr", "doctor" },   .{ "mr", "mister" },     .{ "mrs", "missus" },
        .{ "ms", "miz" },      .{ "prof", "professor" }, .{ "etc", "et cetera" },
        .{ "vs", "versus" },   .{ "jr", "junior" },     .{ "sr", "senior" },
        .{ "ave", "avenue" },  .{ "approx", "approximately" },
    };
    for (map) |m| if (std.mem.eql(u8, l, m[0])) return m[1];
    // "St. Louis" vs "Main St." — the classic context-dependent expansion.
    if (std.mem.eql(u8, l, "st")) return if (next_cap) "saint" else "street";
    return null;
}

fn appendWords(arena: std.mem.Allocator, out: *Tokens, words: []const u8) !void {
    var it = std.mem.tokenizeScalar(u8, words, ' ');
    while (it.next()) |w| try out.append(arena, .{ .word = w });
}

fn appendWord(arena: std.mem.Allocator, out: *Tokens, raw: []const u8) !void {
    // All-caps words without vowels are acronyms: spell them ("TTS", "NRL").
    if (raw.len >= 2 and raw.len <= 5 and isAllUpper(raw) and !hasVowel(raw)) {
        for (raw) |c| try out.append(arena, .{ .letter = std.ascii.toLower(c) });
        return;
    }
    if (raw.len == 1 and std.ascii.isUpper(raw[0]) and raw[0] != 'A' and raw[0] != 'I') {
        try out.append(arena, .{ .letter = std.ascii.toLower(raw[0]) });
        return;
    }
    const w = try arena.alloc(u8, raw.len);
    _ = std.ascii.lowerString(w, raw);
    try out.append(arena, .{ .word = w });
}

fn isAllUpper(s: []const u8) bool {
    for (s) |c| if (!std.ascii.isUpper(c)) return false;
    return true;
}

fn hasVowel(s: []const u8) bool {
    for (s) |c| switch (std.ascii.toLower(c)) {
        'a', 'e', 'i', 'o', 'u', 'y' => return true,
        else => {},
    };
    return false;
}

// ---------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------

const ones = [_][]const u8{ "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen" };
const tens = [_][]const u8{ "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };
const ordinal_ones = [_][]const u8{ "zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth" };

fn number(arena: std.mem.Allocator, out: *Tokens, text: []const u8, start: usize) !usize {
    var i = start;
    const currency = text[i] == '$';
    if (currency) i += 1;
    const int_start = i;
    while (i < text.len and (std.ascii.isDigit(text[i]) or (text[i] == ',' and i + 3 < text.len and std.ascii.isDigit(text[i + 1])))) i += 1;
    var digits: std.ArrayList(u8) = .empty;
    for (text[int_start..i]) |c| if (c != ',') try digits.append(arena, c);
    const value = std.fmt.parseInt(u64, digits.items, 10) catch 0;

    var frac: ?[]const u8 = null;
    if (i + 1 < text.len and text[i] == '.' and std.ascii.isDigit(text[i + 1])) {
        const fs = i + 1;
        i += 1;
        while (i < text.len and std.ascii.isDigit(text[i])) i += 1;
        frac = text[fs..i];
    }

    // Ordinal suffix: 1st, 2nd, 3rd, 4th.
    if (i + 1 < text.len and frac == null and !currency) {
        const suf = text[i .. i + 2];
        if (std.ascii.eqlIgnoreCase(suf, "st") or std.ascii.eqlIgnoreCase(suf, "nd") or
            std.ascii.eqlIgnoreCase(suf, "rd") or std.ascii.eqlIgnoreCase(suf, "th"))
        {
            try spellInt(arena, out, value, true);
            return i + 2;
        }
    }

    const len = i - int_start;
    const is_year = !currency and frac == null and len == 4 and
        ((value >= 1100 and value <= 1999) or (value >= 2010 and value <= 2099));
    if (is_year) {
        // 1984 -> nineteen eighty four; 1905 -> nineteen oh five; 1900 -> nineteen hundred
        const hi = value / 100;
        const lo = value % 100;
        try spellInt(arena, out, hi, false);
        if (lo == 0) {
            try appendWords(arena, out, "hundred");
        } else {
            if (lo < 10) try appendWords(arena, out, "oh");
            try spellInt(arena, out, lo, false);
        }
    } else {
        try spellInt(arena, out, value, false);
    }

    if (frac) |f| {
        if (currency) {
            const cents = std.fmt.parseInt(u64, f[0..@min(f.len, 2)], 10) catch 0;
            try appendWords(arena, out, if (value == 1) "dollar" else "dollars");
            if (cents > 0) {
                try appendWords(arena, out, "and");
                try spellInt(arena, out, if (f.len == 1) cents * 10 else cents, false);
                try appendWords(arena, out, if (cents == 1) "cent" else "cents");
            }
            return i;
        }
        try appendWords(arena, out, "point");
        for (f) |d| try appendWords(arena, out, ones[d - '0']);
    }
    if (currency and frac == null) try appendWords(arena, out, if (value == 1) "dollar" else "dollars");
    return i;
}

fn spellInt(arena: std.mem.Allocator, out: *Tokens, n: u64, ordinal: bool) !void {
    if (n == 0) return appendWords(arena, out, if (ordinal) "zeroth" else "zero");
    const scales = [_]struct { u64, []const u8 }{
        .{ 1_000_000_000_000, "trillion" }, .{ 1_000_000_000, "billion" },
        .{ 1_000_000, "million" },          .{ 1_000, "thousand" },
    };
    var rest = n;
    for (scales) |s| {
        if (rest >= s[0]) {
            try spellBelow1000(arena, out, rest / s[0], false);
            rest %= s[0];
            if (rest == 0 and ordinal) {
                try appendWords(arena, out, try std.fmt.allocPrint(arena, "{s}th", .{s[1]}));
                return;
            }
            try appendWords(arena, out, s[1]);
        }
    }
    if (rest > 0) try spellBelow1000(arena, out, rest, ordinal);
}

fn spellBelow1000(arena: std.mem.Allocator, out: *Tokens, n: u64, ordinal: bool) !void {
    var rest = n;
    if (rest >= 100) {
        try appendWords(arena, out, ones[rest / 100]);
        rest %= 100;
        if (rest == 0) return appendWords(arena, out, if (ordinal) "hundredth" else "hundred");
        try appendWords(arena, out, "hundred");
    }
    if (rest >= 20) {
        const t = rest / 10;
        const o = rest % 10;
        if (o == 0) {
            if (ordinal) {
                // twenty -> twentieth
                const base = tens[t];
                return appendWords(arena, out, try std.fmt.allocPrint(arena, "{s}ieth", .{base[0 .. base.len - 1]}));
            }
            return appendWords(arena, out, tens[t]);
        }
        try appendWords(arena, out, tens[t]);
        rest = o;
    }
    if (rest > 0) try appendWords(arena, out, if (ordinal) ordinal_ones[rest] else ones[rest]);
}

test "numbers" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const toks = try tokenize(a, "Dr. Smith paid $3.50 on the 21st in 1984.");
    var s: std.ArrayList(u8) = .empty;
    for (toks.items) |t| switch (t) {
        .word => |w| {
            try s.appendSlice(a, w);
            try s.append(a, ' ');
        },
        .punct => try s.appendSlice(a, ". "),
        .letter => |l| try s.append(a, l),
    };
    try std.testing.expectEqualStrings("doctor smith paid three dollars and fifty cents on the twenty first in nineteen eighty four . ", s.items);
}
