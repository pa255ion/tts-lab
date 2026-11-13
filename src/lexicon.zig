//! Stage 2: grapheme-to-phoneme conversion.
//!
//! Three sources, tried in order:
//!   1. a pronunciation dictionary — the full CMUdict if it was loaded
//!      (`tools/fetch-cmudict.sh`), otherwise a ~350-word built-in list;
//!   2. letter-to-sound rules in the style of the NRL rules (Elovitz, Johnson,
//!      McHugh & Shore, 1976), the method early synthesizers used for
//!      everything not in their (small) exception dictionary;
//!   3. letter names, for tokens the normalizer decided to spell out.

const std = @import("std");
const Phone = @import("phones.zig").Phone;

pub const PS = struct { phone: Phone, stress: u2 = 0 };

pub const Lexicon = struct {
    arena: std.heap.ArenaAllocator,
    map: std.StringHashMapUnmanaged([]const PS) = .empty,
    rules: []const Rule,
    /// Bytes of pronunciation data held (keys + phone strings), for the
    /// cost profile.
    data_bytes: usize = 0,
    cmudict_loaded: bool = false,

    pub fn init(gpa: std.mem.Allocator) !Lexicon {
        var self: Lexicon = .{ .arena = .init(gpa), .rules = &.{} };
        errdefer self.arena.deinit();
        self.rules = try parseRules(self.arena.allocator());
        _ = try self.addDictText(builtin_dict);
        return self;
    }

    pub fn deinit(self: *Lexicon) void {
        self.arena.deinit();
    }

    /// Parse CMUdict-format text ("WORD  P1 P2 ..." per line, ";;;" or "#"
    /// comments, alternates as "WORD(2)" which are skipped). Returns the
    /// number of entries added.
    pub fn addDictText(self: *Lexicon, data: []const u8) !usize {
        const a = self.arena.allocator();
        var added: usize = 0;
        var lines = std.mem.splitScalar(u8, data, '\n');
        var buf: std.ArrayList(PS) = .empty;
        while (lines.next()) |raw_line| {
            var line = std.mem.trim(u8, raw_line, " \t\r");
            if (std.mem.indexOfScalar(u8, line, '#')) |h| line = std.mem.trim(u8, line[0..h], " \t");
            if (line.len == 0 or std.mem.startsWith(u8, line, ";;;")) continue;
            var it = std.mem.tokenizeAny(u8, line, " \t");
            const head = it.next() orelse continue;
            if (std.mem.indexOfScalar(u8, head, '(') != null) continue;
            buf.clearRetainingCapacity();
            var ok = true;
            while (it.next()) |sym| {
                const p = Phone.parse(sym) orelse {
                    ok = false;
                    break;
                };
                try buf.append(a, .{ .phone = p.phone, .stress = p.stress });
            }
            if (!ok or buf.items.len == 0) continue;
            const key = try a.alloc(u8, head.len);
            _ = std.ascii.lowerString(key, head);
            const gop = try self.map.getOrPut(a, key);
            if (!gop.found_existing) {
                gop.value_ptr.* = try a.dupe(PS, buf.items);
                self.data_bytes += key.len + buf.items.len * @sizeOf(PS) + @sizeOf([]const PS) + 16;
                added += 1;
            }
        }
        return added;
    }

    pub fn loadCmudict(self: *Lexicon, data: []const u8) !usize {
        const n = try self.addDictText(data);
        self.cmudict_loaded = true;
        return n;
    }

    pub fn count(self: *const Lexicon) usize {
        return self.map.count();
    }

    pub const Source = enum { dictionary, rules, spelled };

    /// Pronounce a lower-case word. `arena` receives any rule-generated
    /// pronunciation.
    pub fn pronounce(self: *const Lexicon, arena: std.mem.Allocator, word: []const u8, source: *Source) ![]const PS {
        if (self.map.get(word)) |p| {
            source.* = .dictionary;
            return p;
        }
        // Possessive / contracted "'s": pronounce the stem, add the suffix.
        if (std.mem.endsWith(u8, word, "'s") and word.len > 2) {
            var inner: Source = .dictionary;
            const stem = try self.pronounce(arena, word[0 .. word.len - 2], &inner);
            source.* = inner;
            const last = stem[stem.len - 1].phone;
            const suffix: []const PS = switch (last) {
                .s, .z, .sh, .zh, .ch, .jh => &.{ .{ .phone = .ih }, .{ .phone = .z } },
                .p, .t, .k, .f, .th => &.{.{ .phone = .s }},
                else => &.{.{ .phone = .z }},
            };
            return std.mem.concat(arena, PS, &.{ stem, suffix });
        }
        var clean: std.ArrayList(u8) = .empty;
        for (word) |c| if (std.ascii.isAlphabetic(c)) try clean.append(arena, c);
        if (clean.items.len == 0) return &.{};
        // A lone letter that isn't a word ("x", "y" in "y equals 2x") is said by name.
        if (clean.items.len == 1) {
            source.* = .spelled;
            return letterName(clean.items[0]);
        }
        source.* = .rules;
        return letterToSound(arena, self.rules, clean.items);
    }
};

pub fn letterName(c: u8) []const PS {
    const i = std.ascii.toLower(c) -% 'a';
    if (i >= 26) return &.{};
    return letter_names[i];
}

const letter_names = blk: {
    const src = [26][]const u8{ "EY1", "B IY1", "S IY1", "D IY1", "IY1", "EH1 F", "JH IY1", "EY1 CH", "AY1", "JH EY1", "K EY1", "EH1 L", "EH1 M", "EH1 N", "OW1", "P IY1", "K Y UW1", "AA1 R", "EH1 S", "T IY1", "Y UW1", "V IY1", "D AH1 B AH0 L Y UW0", "EH1 K S", "W AY1", "Z IY1" };
    var out: [26][]const PS = undefined;
    for (src, 0..) |s, i| out[i] = comptimePron(s);
    break :blk out;
};

fn comptimePron(comptime s: []const u8) []const PS {
    comptime {
        @setEvalBranchQuota(100000);
        var arr: []const PS = &.{};
        var it = std.mem.tokenizeScalar(u8, s, ' ');
        while (it.next()) |sym| {
            const p = Phone.parse(sym).?;
            arr = arr ++ [_]PS{.{ .phone = p.phone, .stress = p.stress }};
        }
        const final = arr[0..arr.len].*;
        return &final;
    }
}

/// Words that never carry a pitch accent and are usually reduced.
pub fn isFunctionWord(w: []const u8) bool {
    const list = [_][]const u8{ "the", "a", "an", "and", "of", "to", "in", "is", "it", "that", "for", "on", "with", "as", "was", "he", "she", "they", "we", "you", "i", "me", "my", "your", "his", "her", "our", "their", "be", "are", "were", "been", "have", "has", "had", "do", "does", "did", "at", "by", "from", "or", "but", "if", "so", "than", "then", "can", "could", "will", "would", "should", "may", "might", "must", "shall", "its", "it's", "this", "these", "those", "there", "am", "i'm", "into", "about", "them", "him", "us", "who", "what", "which", "when", "where", "how", "why" };
    for (list) |f| if (std.mem.eql(u8, f, w)) return true;
    return false;
}

pub fn isWhWord(w: []const u8) bool {
    const list = [_][]const u8{ "what", "where", "when", "why", "who", "how", "which", "whose", "whom" };
    for (list) |f| if (std.mem.eql(u8, f, w)) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Letter-to-sound rules
// ---------------------------------------------------------------------------
//
// Each rule is "left[MATCH]right=PHONES". The word is upper-cased and padded
// with spaces; for the letter at the cursor, the first rule whose MATCH,
// left and right contexts all fit wins, the cursor advances past MATCH and
// PHONES are emitted. Context symbols (from the NRL report):
//
//   ' '  word boundary         #  one or more vowels     :  zero or more consonants
//   ^    exactly one consonant .  one voiced consonant   +  front vowel (E I Y)
//   %    suffix (ER E ES ED ING ELY)                     &  sibilant (S C G Z X J CH SH)
//   @    T S R D L Z N J TH CH SH
//
// AX is schwa: emitted as AH and never stressed.

const Rule = struct { left: []const u8, match: []const u8, right: []const u8, out: []const PS };

const rule_text =
    \\ [A] =AX
    \\ [ARE] =AA R
    \\ [AR]O=AX R
    \\[AR]#=EH R
    \\ ^[AS]#=EY S
    \\[A]WA=AX
    \\[AW]=AO
    \\ :[ANY]=EH N IY
    \\[A]^+#=EY
    \\#:[ALLY]=AX L IY
    \\ [AL]#=AX L
    \\[AGAIN]=AX G EH N
    \\#:[AG]E=IH JH
    \\[A]^+:#=AE
    \\ :[A]^+ =EY
    \\[A]^%=EY
    \\ [ARR]=AX R
    \\[ARR]=AE R
    \\ :[AR] =AA R
    \\[AR] =ER
    \\[AR]=AA R
    \\[AIR]=EH R
    \\[AI]=EY
    \\[AY]=EY
    \\[AU]=AO
    \\#:[AL] =AX L
    \\#:[ALS] =AX L Z
    \\[ALK]=AO K
    \\[AL]^=AO L
    \\ :[ABLE]=EY B AX L
    \\[ABLE]=AX B AX L
    \\[ANG]+=EY N JH
    \\[A]=AE
    \\ [BE]^#=B IH
    \\[BEING]=B IY IH NG
    \\ [BOTH] =B OW TH
    \\ [BUS]#=B IH Z
    \\[BUIL]=B IH L
    \\[B]=B
    \\ [CH]^=K
    \\^E[CH]=K
    \\[CH]=CH
    \\ S[CI]#=S AY
    \\[CI]A=SH
    \\[CI]O=SH
    \\[CI]EN=SH
    \\[C]+=S
    \\[CK]=K
    \\[COM]%=K AH M
    \\[C]=K
    \\#:[DED] =D IH D
    \\.E[D] =D
    \\#^:E[D] =T
    \\ [DE]^#=D IH
    \\ [DO] =D UW
    \\ [DOES]=D AH Z
    \\ [DOING]=D UW IH NG
    \\ [DOW]=D AW
    \\[DU]A=JH UW
    \\[D]=D
    \\#:[E] =
    \\' :^[E] =
    \\ :[E] =IY
    \\#[ED] =D
    \\#:[E]D =
    \\[EV]ER=EH V
    \\[E]^%=IY
    \\[ERI]#=IY R IY
    \\[ERI]=EH R IH
    \\#:[ER]#=ER
    \\[ER]#=EH R
    \\[ER]=ER
    \\ [EVEN]=IY V EH N
    \\#:[E]W=
    \\@[EW]=UW
    \\[EW]=Y UW
    \\[E]O=IY
    \\#:&[ES] =IH Z
    \\#:[E]S =
    \\#:[ELY] =L IY
    \\#:[EMENT]=M EH N T
    \\[EFUL]=F UH L
    \\[EE]=IY
    \\[EARN]=ER N
    \\ [EAR]^=ER
    \\[EAD]=EH D
    \\#:[EA] =IY AX
    \\[EA]SU=EH
    \\[EA]=IY
    \\[EIGH]=EY
    \\[EI]=IY
    \\ [EYE]=AY
    \\[EY]=IY
    \\[EU]=Y UW
    \\[E]=EH
    \\[FUL]=F UH L
    \\[F]=F
    \\[GIV]=G IH V
    \\ [G]I^=G
    \\[GE]T=G EH
    \\SU[GGES]=G JH EH S
    \\[GG]=G
    \\ B#[G]=G
    \\[G]+=JH
    \\[GREAT]=G R EY T
    \\#[GH]=
    \\[G]=G
    \\ [HAV]=HH AE V
    \\ [HERE]=HH IY R
    \\ [HOUR]=AW ER
    \\[HOW]=HH AW
    \\[H]#=HH
    \\[H]=
    \\ [IN]=IH N
    \\ [I] =AY
    \\[IN]D=AY N
    \\[IER]=IY ER
    \\#:R[IED] =IY D
    \\[IED] =AY D
    \\[IEN]=IY EH N
    \\[IE]T=AY EH
    \\ :[I]%=AY
    \\[I]%=IY
    \\[IE]=IY
    \\[I]^+:#=IH
    \\[IR]#=AY R
    \\[IZ]%=AY Z
    \\[IS]%=AY Z
    \\[I]D%=AY
    \\+^[I]^+=IH
    \\[I]T%=AY
    \\#^:[I]^+=IH
    \\[I]^+=AY
    \\[IR]=ER
    \\[IGH]=AY
    \\[ILD]=AY L D
    \\[IGN] =AY N
    \\[IGN]^=AY N
    \\[IGN]%=AY N
    \\[IQUE]=IY K
    \\[I]=IH
    \\[J]=JH
    \\ [K]N=
    \\[K]=K
    \\[LO]C#=L OW
    \\L[L]=
    \\#^:[L]%=AX L
    \\[LEAD]=L IY D
    \\[L]=L
    \\[MOV]=M UW V
    \\[M]=M
    \\E[NG]+=N JH
    \\[NG]R=NG G
    \\[NG]#=NG G
    \\[NGL]%=NG G AX L
    \\[NG]=NG
    \\[NK]=NG K
    \\ [NOW] =N AW
    \\[N]=N
    \\[OF] =AX V
    \\[OROUGH]=ER OW
    \\#:[OR] =ER
    \\#:[ORS] =ER Z
    \\[OR]=AO R
    \\ [ONE]=W AH N
    \\[OW]=OW
    \\ [OVER]=OW V ER
    \\[OV]=AH V
    \\[O]^%=OW
    \\[O]^EN=OW
    \\[O]^I#=OW
    \\[OL]D=OW L
    \\[OUGHT]=AO T
    \\[OUGH]=AH F
    \\ [OU]=AW
    \\H[OU]S#=AW
    \\[OUS]=AX S
    \\[OUR]=AO R
    \\[OULD]=UH D
    \\^[OU]^L=AH
    \\[OUP]=UW P
    \\[OU]=AW
    \\[OY]=OY
    \\[OING]=OW IH NG
    \\[OI]=OY
    \\[OOR]=AO R
    \\[OOK]=UH K
    \\[OOD]=UH D
    \\[OO]=UW
    \\[O]E=OW
    \\[O] =OW
    \\[OA]=OW
    \\ [ONLY]=OW N L IY
    \\ [ONCE]=W AH N S
    \\[ON'T]=OW N T
    \\C[O]N=AA
    \\[O]NG=AO
    \\ :^[O]N=AH
    \\I[ON]=AX N
    \\#:[ON] =AX N
    \\#^[ON]=AX N
    \\[O]ST =OW
    \\[OF]^=AO F
    \\[OTHER]=AH DH ER
    \\[OSS] =AO S
    \\#^:[OM]=AH M
    \\[O]=AA
    \\[PH]=F
    \\[PEOP]=P IY P
    \\[POW]=P AW
    \\[PUT] =P UH T
    \\[P]=P
    \\[QUAR]=K W AO R
    \\[QU]=K W
    \\[Q]=K
    \\ [RE]^#=R IY
    \\[R]=R
    \\[SH]=SH
    \\#[SION]=ZH AX N
    \\[SOME]=S AH M
    \\#[SUR]#=ZH ER
    \\[SUR]#=SH ER
    \\#[SU]#=ZH UW
    \\#[SSU]#=SH UW
    \\#[SED] =Z D
    \\#[S]#=Z
    \\[SAID]=S EH D
    \\^[SION]=SH AX N
    \\[S]S=
    \\.[S] =Z
    \\#:.E[S] =Z
    \\#^:##[S] =Z
    \\#^:#[S] =S
    \\U[S] =S
    \\ :#[S] =Z
    \\ [SCH]=S K
    \\[S]C+=
    \\#[SM]=Z M
    \\#[SN]'=Z AX N
    \\[S]=S
    \\ [THE] =DH AX
    \\[TO] =T UW
    \\[THAT] =DH AE T
    \\ [THIS] =DH IH S
    \\ [THEY]=DH EY
    \\ [THERE]=DH EH R
    \\[THER]=DH ER
    \\[THEIR]=DH EH R
    \\ [THAN] =DH AE N
    \\ [THEM] =DH EH M
    \\[THESE] =DH IY Z
    \\ [THEN]=DH EH N
    \\[THROUGH]=TH R UW
    \\[THOSE]=DH OW Z
    \\[THOUGH] =DH OW
    \\ [THUS]=DH AH S
    \\[TH]=TH
    \\#:[TED] =T IH D
    \\S[TI]#N=CH
    \\[TI]O=SH
    \\[TI]A=SH
    \\[TIEN]=SH AX N
    \\[TUR]#=CH ER
    \\[TU]A=CH UW
    \\ [TWO]=T UW
    \\[T]=T
    \\ [UN]I=Y UW N
    \\ [UN]=AH N
    \\ [UPON]=AX P AO N
    \\@[UR]#=UH R
    \\[UR]#=Y UH R
    \\[UR]=ER
    \\[U]^ =AH
    \\[U]^^=AH
    \\[UY]=AY
    \\ G[U]#=
    \\G[U]%=
    \\G[U]#=W
    \\#N[U]=Y UW
    \\@[U]=UW
    \\[U]=Y UW
    \\[VIEW]=V Y UW
    \\[V]=V
    \\ [WERE]=W ER
    \\[WA]S=W AA
    \\[WA]T=W AA
    \\[WHERE]=W EH R
    \\[WHAT]=W AA T
    \\[WHOL]=HH OW L
    \\[WHO]=HH UW
    \\[WH]=W
    \\[WAR]=W AO R
    \\[WOR]^=W ER
    \\[WR]=R
    \\[W]=W
    \\[X]=K S
    \\[YOUNG]=Y AH NG
    \\ [YOU]=Y UW
    \\ [YES]=Y EH S
    \\ [Y]=Y
    \\#^:[Y] =IY
    \\#^:[Y]I=IY
    \\ :[Y] =AY
    \\ :[Y]#=AY
    \\ :[Y]^+:#=IH
    \\ :[Y]^#=AY
    \\[Y]=IH
    \\[Z]=Z
;

/// Stress value used to mark AX (schwa) while rules are applied; replaced by
/// 0 when stress is assigned.
const schwa_mark: u2 = 3;

fn parseRules(a: std.mem.Allocator) ![]const Rule {
    var rules: std.ArrayList(Rule) = .empty;
    var lines = std.mem.splitScalar(u8, rule_text, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const lb = std.mem.indexOfScalar(u8, line, '[') orelse return error.BadRule;
        const rb = std.mem.indexOfScalar(u8, line, ']') orelse return error.BadRule;
        const eq = std.mem.lastIndexOfScalar(u8, line, '=') orelse return error.BadRule;
        var out: std.ArrayList(PS) = .empty;
        var it = std.mem.tokenizeScalar(u8, line[eq + 1 ..], ' ');
        while (it.next()) |sym| {
            if (std.mem.eql(u8, sym, "AX")) {
                try out.append(a, .{ .phone = .ah, .stress = schwa_mark });
            } else {
                const p = Phone.parse(sym) orelse return error.BadRule;
                try out.append(a, .{ .phone = p.phone });
            }
        }
        try rules.append(a, .{ .left = line[0..lb], .match = line[lb + 1 .. rb], .right = line[rb + 1 .. eq], .out = out.items });
    }
    return rules.items;
}

fn isVowelCh(c: u8) bool {
    return switch (c) {
        'A', 'E', 'I', 'O', 'U', 'Y' => true,
        else => false,
    };
}
fn isConsCh(c: u8) bool {
    return std.ascii.isUpper(c) and !isVowelCh(c);
}
fn isVoicedCons(c: u8) bool {
    return switch (c) {
        'B', 'D', 'V', 'G', 'J', 'L', 'M', 'N', 'R', 'W', 'Z' => true,
        else => false,
    };
}
fn isFront(c: u8) bool {
    return c == 'E' or c == 'I' or c == 'Y';
}

fn matchLeft(pat: []const u8, w: []const u8, pos: usize) bool {
    var j: isize = @as(isize, @intCast(pos)) - 1;
    var k = pat.len;
    while (k > 0) {
        k -= 1;
        const c = pat[k];
        const at: u8 = if (j >= 0) w[@intCast(j)] else ' ';
        switch (c) {
            ' ' => {
                if (std.ascii.isAlphabetic(at)) return false;
                j -= 1;
            },
            '#' => {
                if (!isVowelCh(at)) return false;
                while (j >= 0 and isVowelCh(w[@intCast(j)])) j -= 1;
            },
            ':' => while (j >= 0 and isConsCh(w[@intCast(j)])) {
                j -= 1;
            },
            '^' => {
                if (!isConsCh(at)) return false;
                j -= 1;
            },
            '.' => {
                if (!isVoicedCons(at)) return false;
                j -= 1;
            },
            '+' => {
                if (!isFront(at)) return false;
                j -= 1;
            },
            '&' => {
                if (at == 'H' and j >= 1 and (w[@intCast(j - 1)] == 'C' or w[@intCast(j - 1)] == 'S')) {
                    j -= 2;
                } else if (std.mem.indexOfScalar(u8, "SCGZXJ", at) != null) {
                    j -= 1;
                } else return false;
            },
            '@' => {
                if (at == 'H' and j >= 1 and std.mem.indexOfScalar(u8, "TCS", w[@intCast(j - 1)]) != null) {
                    j -= 2;
                } else if (std.mem.indexOfScalar(u8, "TSRDLZNJ", at) != null) {
                    j -= 1;
                } else return false;
            },
            else => {
                if (at != c) return false;
                j -= 1;
            },
        }
    }
    return true;
}

fn matchRight(pat: []const u8, w: []const u8, pos: usize) bool {
    var j = pos;
    for (pat) |c| {
        const at: u8 = if (j < w.len) w[j] else ' ';
        switch (c) {
            ' ' => {
                if (std.ascii.isAlphabetic(at)) return false;
                j += 1;
            },
            '#' => {
                if (!isVowelCh(at)) return false;
                while (j < w.len and isVowelCh(w[j])) j += 1;
            },
            ':' => while (j < w.len and isConsCh(w[j])) {
                j += 1;
            },
            '^' => {
                if (!isConsCh(at)) return false;
                j += 1;
            },
            '.' => {
                if (!isVoicedCons(at)) return false;
                j += 1;
            },
            '+' => {
                if (!isFront(at)) return false;
                j += 1;
            },
            '%' => {
                const sufs = [_][]const u8{ "ING", "ELY", "ER", "ES", "ED", "E" };
                var found = false;
                for (sufs) |s| {
                    if (std.mem.startsWith(u8, w[j..], s)) {
                        j += s.len;
                        found = true;
                        break;
                    }
                }
                if (!found) return false;
            },
            '&' => {
                if (std.mem.startsWith(u8, w[j..], "CH") or std.mem.startsWith(u8, w[j..], "SH")) {
                    j += 2;
                } else if (std.mem.indexOfScalar(u8, "SCGZXJ", at) != null) {
                    j += 1;
                } else return false;
            },
            '@' => {
                if (std.mem.indexOfScalar(u8, "TSRDLZNJ", at) != null) {
                    j += 1;
                } else return false;
            },
            else => {
                if (at != c) return false;
                j += 1;
            },
        }
    }
    return true;
}

pub fn letterToSound(arena: std.mem.Allocator, rules: []const Rule, word: []const u8) ![]const PS {
    // Pad with spaces so boundary contexts work at both ends.
    const w = try arena.alloc(u8, word.len + 2);
    w[0] = ' ';
    w[w.len - 1] = ' ';
    for (word, 0..) |c, i| w[i + 1] = std.ascii.toUpper(c);

    var out: std.ArrayList(PS) = .empty;
    var pos: usize = 1;
    while (pos < w.len - 1) {
        var matched = false;
        for (rules) |r| {
            if (r.match[0] != w[pos]) continue;
            if (!std.mem.startsWith(u8, w[pos..], r.match)) continue;
            if (!matchLeft(r.left, w, pos)) continue;
            if (!matchRight(r.right, w, pos + r.match.len)) continue;
            try out.appendSlice(arena, r.out);
            pos += r.match.len;
            matched = true;
            break;
        }
        if (!matched) pos += 1;
    }
    assignStress(word, out.items);
    return out.items;
}

/// Rule output has no stress. English default: stress the first full vowel,
/// skipping common unstressed prefixes.
fn assignStress(word: []const u8, ps: []PS) void {
    const prefixes = [_][]const u8{ "a", "be", "de", "re", "con", "com", "pre", "pro", "dis", "ex", "un", "in", "im", "for" };
    var skip: usize = 0;
    for (prefixes) |p| {
        if (word.len > p.len + 3 and std.ascii.startsWithIgnoreCase(word, p)) {
            skip = 1;
            break;
        }
    }
    var vowels: usize = 0;
    var stressed = false;
    for (ps) |*x| {
        if (!x.phone.isVowel()) continue;
        if (x.stress == schwa_mark) {
            x.stress = 0;
            continue;
        }
        if (!stressed and vowels >= skip) {
            x.stress = 1;
            stressed = true;
        } else x.stress = 0;
        vowels += 1;
    }
    if (!stressed) for (ps) |*x| if (x.phone.isVowel()) {
        x.stress = 1;
        break;
    };
}

// ---------------------------------------------------------------------------
// Built-in dictionary: common words, numbers, and the demo vocabulary.
//
// Transcriptions follow, and are partly drawn from, the CMU Pronouncing
// Dictionary: Copyright (C) 1993-2015 Carnegie Mellon University. All rights
// reserved. Used under its BSD-style licence; see licenses/CMUdict-LICENSE
// for the full text, conditions and disclaimer.
// ---------------------------------------------------------------------------

pub const builtin_dict =
    \\the DH AH0
    \\a AH0
    \\an AE1 N
    \\and AH0 N D
    \\of AH1 V
    \\to T UW1
    \\in IH0 N
    \\is IH1 Z
    \\it IH1 T
    \\that DH AE1 T
    \\for F AO1 R
    \\on AA1 N
    \\with W IH1 DH
    \\as AE1 Z
    \\was W AA1 Z
    \\he HH IY1
    \\she SH IY1
    \\they DH EY1
    \\we W IY1
    \\you Y UW1
    \\i AY1
    \\me M IY1
    \\my M AY1
    \\your Y AO1 R
    \\his HH IH1 Z
    \\her HH ER1
    \\him HH IH1 M
    \\them DH EH1 M
    \\us AH1 S
    \\our AW1 ER0
    \\their DH EH1 R
    \\this DH IH1 S
    \\these DH IY1 Z
    \\those DH OW1 Z
    \\be B IY1
    \\am AE1 M
    \\are AA1 R
    \\were W ER1
    \\been B IH1 N
    \\have HH AE1 V
    \\has HH AE1 Z
    \\had HH AE1 D
    \\do D UW1
    \\does D AH1 Z
    \\did D IH1 D
    \\not N AA1 T
    \\no N OW1
    \\yes Y EH1 S
    \\can K AE1 N
    \\could K UH1 D
    \\will W IH1 L
    \\would W UH1 D
    \\should SH UH1 D
    \\may M EY1
    \\might M AY1 T
    \\must M AH1 S T
    \\shall SH AE1 L
    \\at AE1 T
    \\by B AY1
    \\from F R AH1 M
    \\up AH1 P
    \\down D AW1 N
    \\out AW1 T
    \\over OW1 V ER0
    \\under AH1 N D ER0
    \\about AH0 B AW1 T
    \\into IH1 N T UW0
    \\through TH R UW1
    \\after AE1 F T ER0
    \\before B IH0 F AO1 R
    \\or AO1 R
    \\but B AH1 T
    \\if IH1 F
    \\so S OW1
    \\than DH AE1 N
    \\then DH EH1 N
    \\when W EH1 N
    \\where W EH1 R
    \\what W AH1 T
    \\why W AY1
    \\who HH UW1
    \\how HH AW1
    \\which W IH1 CH
    \\there DH EH1 R
    \\here HH IY1 R
    \\all AO1 L
    \\some S AH1 M
    \\any EH1 N IY0
    \\many M EH1 N IY0
    \\much M AH1 CH
    \\more M AO1 R
    \\most M OW1 S T
    \\very V EH1 R IY0
    \\just JH AH1 S T
    \\only OW1 N L IY0
    \\also AO1 L S OW0
    \\well W EH1 L
    \\now N AW1
    \\new N UW1
    \\old OW1 L D
    \\good G UH1 D
    \\great G R EY1 T
    \\small S M AO1 L
    \\big B IH1 G
    \\long L AO1 NG
    \\little L IH1 T AH0 L
    \\zero Z IH1 R OW0
    \\one W AH1 N
    \\two T UW1
    \\three TH R IY1
    \\four F AO1 R
    \\five F AY1 V
    \\six S IH1 K S
    \\seven S EH1 V AH0 N
    \\eight EY1 T
    \\nine N AY1 N
    \\ten T EH1 N
    \\eleven IH0 L EH1 V AH0 N
    \\twelve T W EH1 L V
    \\thirteen TH ER1 T IY1 N
    \\fourteen F AO1 R T IY1 N
    \\fifteen F IH0 F T IY1 N
    \\twenty T W EH1 N T IY0
    \\thirty TH ER1 T IY0
    \\forty F AO1 R T IY0
    \\fifty F IH1 F T IY0
    \\sixty S IH1 K S T IY0
    \\hundred HH AH1 N D R AH0 D
    \\thousand TH AW1 Z AH0 N D
    \\million M IH1 L Y AH0 N
    \\first F ER1 S T
    \\second S EH1 K AH0 N D
    \\third TH ER1 D
    \\point P OY1 N T
    \\dollar D AA1 L ER0
    \\dollars D AA1 L ER0 Z
    \\cent S EH1 N T
    \\cents S EH1 N T S
    \\percent P ER0 S EH1 N T
    \\hello HH AH0 L OW1
    \\world W ER1 L D
    \\speech S P IY1 CH
    \\voice V OY1 S
    \\synthesis S IH1 N TH AH0 S AH0 S
    \\synthesizer S IH1 N TH AH0 S AY2 Z ER0
    \\text T EH1 K S T
    \\computer K AH0 M P Y UW1 T ER0
    \\sound S AW1 N D
    \\sounds S AW1 N D Z
    \\word W ER1 D
    \\words W ER1 D Z
    \\say S EY1
    \\said S EH1 D
    \\speak S P IY1 K
    \\talk T AO1 K
    \\hear HH IY1 R
    \\listen L IH1 S AH0 N
    \\quick K W IH1 K
    \\brown B R AW1 N
    \\fox F AA1 K S
    \\jumps JH AH1 M P S
    \\lazy L EY1 Z IY0
    \\dog D AO1 G
    \\cat K AE1 T
    \\time T AY1 M
    \\day D EY1
    \\night N AY1 T
    \\year Y IH1 R
    \\today T AH0 D EY1
    \\tomorrow T AH0 M AA1 R OW2
    \\morning M AO1 R N IH0 NG
    \\people P IY1 P AH0 L
    \\man M AE1 N
    \\woman W UH1 M AH0 N
    \\child CH AY1 L D
    \\water W AO1 T ER0
    \\house HH AW1 S
    \\home HH OW1 M
    \\car K AA1 R
    \\city S IH1 T IY0
    \\way W EY1
    \\thing TH IH1 NG
    \\things TH IH1 NG Z
    \\think TH IH1 NG K
    \\know N OW1
    \\see S IY1
    \\look L UH1 K
    \\come K AH1 M
    \\go G OW1
    \\get G EH1 T
    \\make M EY1 K
    \\take T EY1 K
    \\give G IH1 V
    \\want W AA1 N T
    \\like L AY1 K
    \\use Y UW1 Z
    \\work W ER1 K
    \\find F AY1 N D
    \\tell T EH1 L
    \\ask AE1 S K
    \\feel F IY1 L
    \\try T R AY1
    \\leave L IY1 V
    \\call K AO1 L
    \\need N IY1 D
    \\play P L EY1
    \\run R AH1 N
    \\move M UW1 V
    \\live L IH1 V
    \\believe B IH0 L IY1 V
    \\please P L IY1 Z
    \\thank TH AE1 NG K
    \\thanks TH AE1 NG K S
    \\sorry S AA1 R IY0
    \\right R AY1 T
    \\left L EH1 F T
    \\name N EY1 M
    \\number N AH1 M B ER0
    \\music M Y UW1 Z IH0 K
    \\machine M AH0 SH IY1 N
    \\human HH Y UW1 M AH0 N
    \\natural N AE1 CH ER0 AH0 L
    \\robot R OW1 B AA2 T
    \\question K W EH1 S CH AH0 N
    \\answer AE1 N S ER0
    \\test T EH1 S T
    \\example IH0 G Z AE1 M P AH0 L
    \\sentence S EH1 N T AH0 N S
    \\weather W EH1 DH ER0
    \\rain R EY1 N
    \\sun S AH1 N
    \\happy HH AE1 P IY0
    \\sad S AE1 D
    \\beautiful B Y UW1 T AH0 F AH0 L
    \\important IH0 M P AO1 R T AH0 N T
    \\different D IH1 F ER0 AH0 N T
    \\interesting IH1 N T R AH0 S T IH0 NG
    \\really R IH1 L IY0
    \\never N EH1 V ER0
    \\always AO1 L W EY2 Z
    \\every EH1 V R IY0
    \\each IY1 CH
    \\other AH1 DH ER0
    \\another AH0 N AH1 DH ER0
    \\same S EY1 M
    \\because B IH0 K AO1 Z
    \\while W AY1 L
    \\again AH0 G EH1 N
    \\still S T IH1 L
    \\even IY1 V IH0 N
    \\back B AE1 K
    \\last L AE1 S T
    \\next N EH1 K S T
    \\open OW1 P AH0 N
    \\close K L OW1 Z
    \\light L AY1 T
    \\green G R IY1 N
    \\red R EH1 D
    \\blue B L UW1
    \\white W AY1 T
    \\black B L AE1 K
    \\table T EY1 B AH0 L
    \\chair CH EH1 R
    \\book B UH1 K
    \\door D AO1 R
    \\window W IH1 N D OW0
    \\phone F OW1 N
    \\station S T EY1 SH AH0 N
    \\train T R EY1 N
    \\formant F AO1 R M AH0 N T
    \\diphone D AY1 F OW0 N
    \\tone T OW1 N
    \\pitch P IH1 CH
    \\doctor D AA1 K T ER0
    \\smith S M IH1 TH
    \\paid P EY1 D
    \\mister M IH1 S T ER0
    \\saint S EY1 N T
    \\street S T R IY1 T
    \\its IH1 T S
    \\i'm AY1 M
    \\don't D OW1 N T
    \\can't K AE1 N T
    \\it's IH1 T S
    \\that's DH AE1 T S
    \\let's L EH1 T S
    \\oh OW1
    \\okay OW2 K EY1
    \\welcome W EH1 L K AH0 M
    \\goodbye G UH2 D B AY1
    \\love L AH1 V
    \\life L AY1 F
    \\mind M AY1 N D
    \\hand HH AE1 N D
    \\eye AY1
    \\head HH EH1 D
    \\face F EY1 S
    \\place P L EY1 S
    \\part P AA1 R T
    \\problem P R AA1 B L AH0 M
    \\system S IH1 S T AH0 M
    \\program P R OW1 G R AE2 M
    \\early ER1 L IY0
    \\late L EY1 T
    \\fast F AE1 S T
    \\slow S L OW1
    \\high HH AY1
    \\low L OW1
    \\loud L AW1 D
    \\soft S AO1 F T
    \\together T AH0 G EH1 DH ER0
    \\english IH1 NG G L IH0 SH
    \\language L AE1 NG G W AH0 JH
    \\read R IY1 D
    \\write R AY1 T
    \\letter L EH1 T ER0
    \\near N IY1 R
    \\far F AA1 R
    \\walk W AO1 K
    \\walked W AO1 K T
    \\river R IH1 V ER0
    \\bridge B R IH1 JH
    \\garden G AA1 R D AH0 N
    \\yellow Y EH1 L OW0
    \\orange AO1 R AH0 N JH
    \\morning M AO1 R N IH0 NG
    \\evening IY1 V N IH0 NG
    \\family F AE1 M AH0 L IY0
    \\friend F R EH1 N D
    \\friends F R EH1 N D Z
    \\school S K UW1 L
    \\teacher T IY1 CH ER0
    \\money M AH1 N IY0
    \\road R OW1 D
    \\ship SH IH1 P
    \\sea S IY1
    \\fish F IH1 SH
    \\bird B ER1 D
    \\tree T R IY1
    \\stone S T OW1 N
    \\cold K OW1 L D
    \\warm W AO1 R M
    \\hot HH AA1 T
    \\coffee K AA1 F IY0
    \\tea T IY1
    \\bread B R EH1 D
    \\milk M IH1 L K
    \\sugar SH UH1 G ER0
    \\kitchen K IH1 CH AH0 N
    \\window W IH1 N D OW0
    \\paper P EY1 P ER0
    \\picture P IH1 K CH ER0
    \\story S T AO1 R IY0
    \\simple S IH1 M P AH0 L
    \\early ER1 L IY0
    \\quiet K W AY1 AH0 T
    \\began B IH0 G AE1 N
    \\found F AW1 N D
    \\went W EH1 N T
    \\came K EY1 M
    \\took T UH1 K
    \\made M EY1 D
    \\saw S AO1
    \\knew N UW1
    \\thought TH AO1 T
    \\heard HH ER1 D
    \\told T OW1 L D
    \\gave G EY1 V
    \\kept K EH1 P T
    \\left L EH1 F T
    \\stood S T UH1 D
    \\ran R AE1 N
    \\sat S AE1 T
    \\wrote R OW1 T
    \\spoke S P OW1 K
    \\yesterday Y EH1 S T ER0 D EY2
    \\monday M AH1 N D EY2
    \\friday F R AY1 D EY2
    \\january JH AE1 N Y UW0 EH2 R IY0
    \\minutes M IH1 N AH0 T S
    \\hours AW1 ER0 Z
    \\seconds S EH1 K AH0 N D Z
    \\clock K L AA1 K
    \\o'clock AH0 K L AA1 K
    \\early ER1 L IY0
    \\noise N OY1 Z
    \\boy B OY1
    \\girl G ER1 L
    \\joy JH OY1
    \\measure M EH1 ZH ER0
    \\vision V IH1 ZH AH0 N
    \\usual Y UW1 ZH AH0 W AH0 L
    \\judge JH AH1 JH
    \\church CH ER1 CH
    \\thin TH IH1 N
    \\thick TH IH1 K
    \\vowel V AW1 AH0 L
    \\vowels V AW1 AH0 L Z
    \\consonant K AA1 N S AH0 N AH0 N T
    \\rule R UW1 L
    \\rules R UW1 L Z
    \\model M AA1 D AH0 L
    \\data D EY1 T AH0
    \\neural N UH1 R AH0 L
    \\network N EH1 T W ER2 K
    \\research R IY0 S ER1 CH
    \\engine EH1 N JH AH0 N
    \\engines EH1 N JH AH0 N Z
    \\samples S AE1 M P AH0 L Z
    \\sample S AE1 M P AH0 L
    \\memory M EH1 M ER0 IY0
    \\quality K W AA1 L AH0 T IY0
    \\smooth S M UW1 DH
    \\intonation IH2 N T AH0 N EY1 SH AH0 N
;

test "lexicon basics" {
    var lex = try Lexicon.init(std.testing.allocator);
    defer lex.deinit();
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    var src: Lexicon.Source = .dictionary;
    const p = try lex.pronounce(arena_state.allocator(), "hello", &src);
    try std.testing.expectEqual(Lexicon.Source.dictionary, src);
    try std.testing.expectEqual(Phone.hh, p[0].phone);
    // "phonics" is not built in: goes through rules. PH -> F.
    const q = try lex.pronounce(arena_state.allocator(), "phonics", &src);
    try std.testing.expectEqual(Lexicon.Source.rules, src);
    try std.testing.expectEqual(Phone.f, q[0].phone);
    // Lone letters are names ("x" -> EH K S), but "a" stays the word.
    const x = try lex.pronounce(arena_state.allocator(), "x", &src);
    try std.testing.expectEqual(Lexicon.Source.spelled, src);
    try std.testing.expectEqual(Phone.eh, x[0].phone);
    _ = try lex.pronounce(arena_state.allocator(), "a", &src);
    try std.testing.expectEqual(Lexicon.Source.dictionary, src);
}
