//! Top level: text in, samples out, through any of the five engines.
//! Voices for the data-driven engines are built lazily on first use.

const std = @import("std");
const dsp = @import("dsp.zig");
const lexicon = @import("lexicon.zig");
const prosody = @import("prosody.zig");
const formant = @import("formant.zig");
const corpus = @import("corpus.zig");
const diphone = @import("diphone.zig");
const unitsel = @import("unitsel.zig");
const hmm = @import("hmm.zig");
pub const voice_profile = @import("voice_profile.zig");

pub const Engine = enum(u8) {
    formant_classic,
    formant_refined,
    diphone,
    unit_selection,
    hmm,

    pub const count = @typeInfo(Engine).@"enum".fields.len;

    pub fn label(e: Engine) []const u8 {
        return switch (e) {
            .formant_classic => "Formant (classic rules)",
            .formant_refined => "Formant (refined)",
            .diphone => "Diphone + TD-PSOLA",
            .unit_selection => "Unit selection",
            .hmm => "Statistical parametric (HMM)",
        };
    }

    pub fn short(e: Engine) []const u8 {
        return switch (e) {
            .formant_classic => "classic",
            .formant_refined => "refined",
            .diphone => "diphone",
            .unit_selection => "unitsel",
            .hmm => "hmm",
        };
    }

    pub fn parse(s: []const u8) ?Engine {
        inline for (@typeInfo(Engine).@"enum".fields) |f| {
            const e: Engine = @enumFromInt(f.value);
            if (std.mem.eql(u8, s, e.short()) or std.mem.eql(u8, s, f.name)) return e;
        }
        if (s.len == 1 and s[0] >= '1' and s[0] < '1' + count) return @enumFromInt(s[0] - '1');
        return null;
    }

    pub fn supportsFormantScale(e: Engine) bool {
        return e == .formant_classic or e == .formant_refined or e == .hmm;
    }
};

pub const Options = struct {
    engine: Engine = .formant_refined,
    prosody: prosody.Options = .{},
    /// Multiplies the planned pitch (all engines).
    pitch_scale: f32 = 1.0,
    /// Vocal-tract scaling; formant and HMM engines only.
    formant_scale: f32 = 1.0,
    diphone_psola: bool = true,
    unit_modify: bool = false,
    hmm_mlpg: bool = true,
    hmm_gv: bool = false,
    /// A voice fitted to a speaker (refined formant engine only; see
    /// voice_profile.zig). null: the stock voice.
    profile: ?*const voice_profile.Profile = null,
    /// The profile's speaking style (null: its first).
    style: ?[]const u8 = null,
    /// 0 = the speaker's mean targets, 1 = their measured token-to-token
    /// spread (needs a profile fitted from several takes).
    variation: f32 = 0,
    seed: u32 = 1,

    /// The formant engines' voice for these options.
    pub fn formantVoice(o: Options) formant.Voice {
        var v: formant.Voice = .{ .formant_scale = o.formant_scale };
        if (o.activeProfile()) |pr| {
            v.profile = pr;
            v.variation = o.variation;
            v.seed = o.seed;
            if (pr.style(o.style)) |st| {
                v.rd_offset = st.rd;
                v.breath = st.breath;
            }
        }
        return v;
    }

    /// The profile, if it applies to the chosen engine.
    pub fn activeProfile(o: Options) ?*const voice_profile.Profile {
        return if (o.engine == .formant_refined) o.profile else null;
    }
};

pub const Result = struct {
    samples: []f32,
    transcription: []u8,
    f0: []f32,
    audio_seconds: f64,
    plan_seconds: f64,
    synth_seconds: f64,
    /// Voice build time spent during this call (first use of an engine).
    build_seconds: f64,
    rule_words: usize,
    unit_stats: unitsel.Stats = .{},

    pub fn deinit(r: Result, alloc: std.mem.Allocator) void {
        alloc.free(r.samples);
        alloc.free(r.transcription);
        alloc.free(r.f0);
    }
};

pub const Synth = struct {
    gpa: std.mem.Allocator,
    lex: lexicon.Lexicon,
    threads: usize,
    diphone_voice: ?*diphone.Voice = null,
    sentence_corpus: ?*corpus.Corpus = null,
    unit_voice: ?*unitsel.Voice = null,
    hmm_voice: ?*hmm.Voice = null,
    /// A profile loaded by a front end (capi: ttslab_set_voice).
    profile: ?voice_profile.Profile = null,
    profile_arena: ?std.heap.ArenaAllocator = null,

    pub fn init(gpa: std.mem.Allocator) !*Synth {
        const s = try gpa.create(Synth);
        errdefer gpa.destroy(s);
        s.* = .{
            .gpa = gpa,
            .lex = try lexicon.Lexicon.init(gpa),
            .threads = if (@import("builtin").single_threaded) 1 else std.Thread.getCpuCount() catch 4,
        };
        return s;
    }

    pub fn deinit(s: *Synth) void {
        if (s.profile_arena) |*a| a.deinit();
        if (s.hmm_voice) |v| {
            v.deinit();
            s.gpa.destroy(v);
        }
        if (s.unit_voice) |v| {
            v.deinit();
            s.gpa.destroy(v);
        }
        if (s.sentence_corpus) |c| {
            c.deinit();
            s.gpa.destroy(c);
        }
        if (s.diphone_voice) |v| {
            v.deinit();
            s.gpa.destroy(v);
        }
        s.lex.deinit();
        s.gpa.destroy(s);
    }

    /// Build whatever `engine` needs; returns the seconds spent.
    pub fn ensure(s: *Synth, engine: Engine) !f64 {
        const t0 = dsp.now();
        switch (engine) {
            .formant_classic, .formant_refined => {},
            .diphone => _ = try s.diphoneVoice(),
            .unit_selection => _ = try s.unitVoice(),
            .hmm => _ = try s.hmmVoice(),
        }
        return dsp.now() - t0;
    }

    pub fn diphoneVoice(s: *Synth) !*diphone.Voice {
        if (s.diphone_voice == null) s.diphone_voice = try diphone.Voice.build(s.gpa, s.threads);
        return s.diphone_voice.?;
    }

    pub fn sentenceCorpus(s: *Synth) !*corpus.Corpus {
        if (s.sentence_corpus == null) s.sentence_corpus = try corpus.Corpus.build(s.gpa, &s.lex, s.threads);
        return s.sentence_corpus.?;
    }

    pub fn unitVoice(s: *Synth) !*unitsel.Voice {
        if (s.unit_voice == null) {
            const c = try s.sentenceCorpus();
            const d = try s.diphoneVoice();
            s.unit_voice = try unitsel.Voice.build(s.gpa, c, d);
        }
        return s.unit_voice.?;
    }

    pub fn hmmVoice(s: *Synth) !*hmm.Voice {
        if (s.hmm_voice == null) s.hmm_voice = try hmm.Voice.train(s.gpa, try s.sentenceCorpus());
        return s.hmm_voice.?;
    }

    /// Load (or, with empty text, clear) the profile kept in the Synth.
    pub fn setProfile(s: *Synth, src: []const u8) !void {
        if (s.profile_arena) |*a| a.deinit();
        s.profile_arena = null;
        s.profile = null;
        if (src.len == 0) return;
        var arena = std.heap.ArenaAllocator.init(s.gpa);
        errdefer arena.deinit();
        const p = try voice_profile.parse(arena.allocator(), src, null);
        s.profile_arena = arena;
        s.profile = p;
    }

    pub fn plan(s: *Synth, arena: std.mem.Allocator, text: []const u8, opt: Options) !prosody.Utterance {
        var po = opt.prosody;
        var dur_scale: [@import("phones.zig").Phone.count]f32 = undefined;
        if (opt.activeProfile()) |pr| {
            // The speaker's pitch, range and tempo, on top of the user's
            // own settings (which default to 1).
            if (pr.style(opt.style)) |st| {
                if (st.base_f0) |f| po.base_f0 = f * (po.base_f0 / 110);
                po.pitch_range *= st.pitch_range;
                po.rate *= st.rate;
            }
            for (&dur_scale, pr.phone) |*d, e| d.* = e.dur;
            po.dur_scale = &dur_scale;
        }
        po.base_f0 *= opt.pitch_scale;
        // The refined formant engine uses the data-fitted timing; the others
        // keep Klatt's rules (their voices were recorded with them).
        if (po.timing == null) po.timing = if (opt.engine == .formant_refined) .fitted else .klatt;
        return prosody.plan(arena, &s.lex, text, po);
    }

    /// Synthesize `text`. The result is allocated with `alloc`.
    pub fn speak(s: *Synth, alloc: std.mem.Allocator, text: []const u8, opt: Options) !Result {
        const build_s = try s.ensure(opt.engine);
        var arena_state = std.heap.ArenaAllocator.init(alloc);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        const t0 = dsp.now();
        const utt = try s.plan(arena, text, opt);
        const t1 = dsp.now();
        var stats: unitsel.Stats = .{};
        const samples: []f32 = switch (opt.engine) {
            .formant_classic, .formant_refined => blk: {
                const mode: formant.Mode = if (opt.engine == .formant_classic) .classic else .refined;
                const o = try formant.synthesize(alloc, &utt, mode, opt.formantVoice());
                alloc.free(o.gci);
                alloc.free(o.gci_voiced);
                break :blk o.samples;
            },
            .diphone => try s.diphone_voice.?.synthesize(alloc, &utt, .{ .psola = opt.diphone_psola }),
            .unit_selection => try s.unit_voice.?.synthesize(alloc, &utt, .{ .modify_prosody = opt.unit_modify }, &stats),
            .hmm => try s.hmm_voice.?.synthesize(alloc, &utt, .{ .mlpg = opt.hmm_mlpg, .gv = opt.hmm_gv, .formant_scale = opt.formant_scale }),
        };
        const t2 = dsp.now();
        dsp.normalizeRms(samples, 0.08);

        var rule_words: usize = 0;
        for (utt.words) |w| if (w.source == .rules) {
            rule_words += 1;
        };
        return .{
            .samples = samples,
            .transcription = try alloc.dupe(u8, try utt.transcription(arena)),
            .f0 = try alloc.dupe(f32, utt.f0),
            .audio_seconds = @as(f64, @floatFromInt(samples.len)) / dsp.sample_rate,
            .plan_seconds = t1 - t0,
            .synth_seconds = t2 - t1,
            .build_seconds = build_s,
            .rule_words = rule_words,
            .unit_stats = stats,
        };
    }

    pub const Memory = struct {
        lexicon: usize,
        lexicon_words: usize,
        diphone: ?usize,
        diphone_units: usize,
        corpus_audio: ?usize,
        corpus_frames: ?usize,
        unit_index: ?usize,
        hmm_models: ?usize,
        hmm_states: usize,
    };

    pub fn memory(s: *const Synth) Memory {
        return .{
            .lexicon = s.lex.data_bytes,
            .lexicon_words = s.lex.count(),
            .diphone = if (s.diphone_voice) |v| v.bytes else null,
            .diphone_units = if (s.diphone_voice) |v| v.count() else 0,
            .corpus_audio = if (s.sentence_corpus) |c| c.audioBytes() else null,
            .corpus_frames = if (s.sentence_corpus) |c| c.frameBytes() else null,
            .unit_index = if (s.unit_voice) |v| v.bytes else null,
            .hmm_models = if (s.hmm_voice) |v| v.bytes else null,
            .hmm_states = if (s.hmm_voice) |v| v.models else 0,
        };
    }
};
