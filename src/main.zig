//! ttslab CLI.
//!
//!   ttslab                       interactive menu
//!   ttslab say [opts] TEXT       synthesize to a WAV and play it
//!   ttslab compare [opts] TEXT   every engine -> out/<engine>.wav
//!   ttslab phonemes TEXT         show the front end's transcription
//!   ttslab segs TEXT             planned segments and prosodic features (CSV)
//!   ttslab profile               the stock voice as a voice profile (template)
//!   ttslab bench [--reps N]      cost profile (markdown)
//!
//! Options: -e/--engine classic|refined|diphone|unitsel|hmm (or 1-5)
//!          --f0 flat|hat|fujisaki  --timing klatt|fitted  --pitch X  --rate X  --range X  --formant X
//!          --no-psola  --modify  --no-mlpg  --gv
//!          --voice PROFILE  --style NAME  --variation X  --seed N   (refined engine; tools/fit-voice.py)
//!          -o/--out FILE  --play/--no-play  --dict FILE  --no-dict  --threads N
//!
//! Playback uses pw-play, paplay or aplay (first that works); set
//! TTSLAB_PLAYER to use another command (the WAV path is appended).
//! say/compare play by default when stdout is a terminal.

const std = @import("std");
const tts = @import("tts.zig");
const phones = @import("phones.zig");
const prosody = @import("prosody.zig");
const wav = @import("wav.zig");
const bench = @import("bench.zig");
const dsp = @import("dsp.zig");
const formant = @import("formant.zig");

var stdout_buf: [8192]u8 = undefined;
var stdout_w: std.Io.File.Writer = undefined;
var io: std.Io = undefined;
var player_env: ?[]const u8 = null;
var player_announced = false;

fn print(comptime fmt: []const u8, args: anytype) void {
    stdout_w.interface.print(fmt, args) catch return;
    stdout_w.interface.flush() catch return;
}

const Cli = struct {
    opt: tts.Options = .{},
    out: []const u8 = "out/say.wav",
    /// null: play if stdout is a terminal.
    play: ?bool = null,
    dict: ?[]const u8 = "data/cmudict.dict",
    dict_explicit: bool = false,
    reps: usize = 5,
    threads: ?usize = null,
    voice: ?[]const u8 = null,
    positional: std.ArrayList([]const u8) = .empty,
};

fn parseF(s: []const u8) !f32 {
    return std.fmt.parseFloat(f32, s) catch {
        std.debug.print("not a number: {s}\n", .{s});
        return error.BadArgs;
    };
}

fn parseArgs(a: std.mem.Allocator, args: []const [:0]const u8) !Cli {
    var c: Cli = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const x = args[i];
        const next = struct {
            fn f(as: []const [:0]const u8, j: *usize) ![]const u8 {
                j.* += 1;
                if (j.* >= as.len) {
                    std.debug.print("{s} needs a value\n", .{as[j.* - 1]});
                    return error.BadArgs;
                }
                return as[j.*];
            }
        }.f;
        if (std.mem.eql(u8, x, "-e") or std.mem.eql(u8, x, "--engine")) {
            const v = try next(args, &i);
            c.opt.engine = tts.Engine.parse(v) orelse {
                std.debug.print("unknown engine: {s}\n", .{v});
                return error.BadArgs;
            };
        } else if (std.mem.eql(u8, x, "--f0")) {
            const v = try next(args, &i);
            c.opt.prosody.f0_model = std.meta.stringToEnum(prosody.F0Model, v) orelse return error.BadArgs;
        } else if (std.mem.eql(u8, x, "--pitch")) {
            c.opt.pitch_scale = try parseF(try next(args, &i));
        } else if (std.mem.eql(u8, x, "--timing")) {
            const v = try next(args, &i);
            c.opt.prosody.timing = std.meta.stringToEnum(prosody.Timing, v) orelse return error.BadArgs;
        } else if (std.mem.eql(u8, x, "--rate")) {
            c.opt.prosody.rate = try parseF(try next(args, &i));
        } else if (std.mem.eql(u8, x, "--range")) {
            c.opt.prosody.pitch_range = try parseF(try next(args, &i));
        } else if (std.mem.eql(u8, x, "--formant")) {
            c.opt.formant_scale = try parseF(try next(args, &i));
        } else if (std.mem.eql(u8, x, "--no-psola")) {
            c.opt.diphone_psola = false;
        } else if (std.mem.eql(u8, x, "--modify")) {
            c.opt.unit_modify = true;
        } else if (std.mem.eql(u8, x, "--no-mlpg")) {
            c.opt.hmm_mlpg = false;
        } else if (std.mem.eql(u8, x, "--gv")) {
            c.opt.hmm_gv = true;
        } else if (std.mem.eql(u8, x, "-o") or std.mem.eql(u8, x, "--out")) {
            c.out = try next(args, &i);
        } else if (std.mem.eql(u8, x, "--play")) {
            c.play = true;
        } else if (std.mem.eql(u8, x, "--no-play")) {
            c.play = false;
        } else if (std.mem.eql(u8, x, "--dict")) {
            c.dict = try next(args, &i);
            c.dict_explicit = true;
        } else if (std.mem.eql(u8, x, "--no-dict")) {
            c.dict = null;
        } else if (std.mem.eql(u8, x, "--reps")) {
            c.reps = std.fmt.parseInt(usize, try next(args, &i), 10) catch return error.BadArgs;
        } else if (std.mem.eql(u8, x, "--voice")) {
            c.voice = try next(args, &i);
        } else if (std.mem.eql(u8, x, "--style")) {
            c.opt.style = try next(args, &i);
        } else if (std.mem.eql(u8, x, "--variation")) {
            c.opt.variation = try parseF(try next(args, &i));
        } else if (std.mem.eql(u8, x, "--seed")) {
            c.opt.seed = std.fmt.parseInt(u32, try next(args, &i), 10) catch return error.BadArgs;
        } else if (std.mem.eql(u8, x, "--threads")) {
            c.threads = std.fmt.parseInt(usize, try next(args, &i), 10) catch return error.BadArgs;
        } else {
            try c.positional.append(a, x);
        }
    }
    return c;
}

pub fn main(init: std.process.Init) u8 {
    io = init.io;
    player_env = init.environ_map.get("TTSLAB_PLAYER");
    stdout_w = std.Io.File.stdout().writer(io, &stdout_buf);
    run(init) catch |err| {
        if (err != error.BadArgs) std.debug.print("error: {t}\n", .{err});
        return 1;
    };
    return 0;
}

fn run(init: std.process.Init) !void {
    const gpa = std.heap.smp_allocator;
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    var cli = try parseArgs(arena, argv[1..]);

    var synth = try tts.Synth.init(gpa);
    defer synth.deinit();
    if (cli.threads) |t| synth.threads = t;
    if (cli.dict) |path| loadDict(synth, path, !cli.dict_explicit);
    if (cli.voice) |path| {
        const data = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20)) catch {
            std.debug.print("could not read voice profile {s}\n", .{path});
            return error.BadArgs;
        };
        defer gpa.free(data);
        var ln: usize = 0;
        var tmp = std.heap.ArenaAllocator.init(gpa);
        defer tmp.deinit();
        _ = tts.voice_profile.parse(tmp.allocator(), data, &ln) catch {
            std.debug.print("{s}:{d}: not a voice profile line\n", .{ path, ln });
            return error.BadArgs;
        };
        try synth.setProfile(data);
        cli.opt.profile = &synth.profile.?;
        const st = synth.profile.?.style(cli.opt.style);
        std.debug.print("voice profile {s} ({s}), style {s}\n", .{ path, synth.profile.?.name, if (st) |x| x.name else "-" });
        if (cli.opt.style) |want| if (st == null or !std.mem.eql(u8, st.?.name, want)) std.debug.print("  (no style \"{s}\" in the profile: using the first)\n", .{want});
        if (cli.opt.engine != .formant_refined) std.debug.print("  (voice profiles apply to the refined formant engine only)\n", .{});
    }

    const interactive = std.Io.File.stdout().isTty(io) catch false;
    const play = cli.play orelse interactive;
    const pos = cli.positional.items;
    const cmd = if (pos.len > 0) pos[0] else "menu";
    const rest = std.mem.join(arena, " ", if (pos.len > 1) pos[1..] else &.{}) catch "";

    if (std.mem.eql(u8, cmd, "menu")) return menu(synth, &cli);
    if (std.mem.eql(u8, cmd, "say")) {
        try speakAndSave(synth, rest, cli.opt, cli.out);
        if (play) playFile(cli.out);
        return;
    }
    if (std.mem.eql(u8, cmd, "compare")) return compare(synth, rest, cli.opt, play);
    if (std.mem.eql(u8, cmd, "batch")) return batch(synth, arena, pos[1..], &cli);
    if (std.mem.eql(u8, cmd, "phonemes")) {
        var as = std.heap.ArenaAllocator.init(gpa);
        defer as.deinit();
        const u = try synth.plan(as.allocator(), rest, cli.opt);
        print("{s}\n", .{try u.transcription(as.allocator())});
        for (u.words) |w| if (w.source == .rules) print("  (letter-to-sound rules: {s})\n", .{w.text});
        return;
    }
    if (std.mem.eql(u8, cmd, "profile")) {
        // The stock refined voice written as a voice profile: a template for
        // hand edits, and the starting point of tools/fit-voice.py.
        print("# tts-lab voice profile (see src/voice_profile.zig)\nname stock\n", .{});
        for (0..phones.Phone.count) |i| {
            const ph: phones.Phone = @enumFromInt(i);
            if (ph == .pau) continue;
            const inf = ph.info();
            print("phone {s} f {d:.0} {d:.0} {d:.0}", .{ ph.name(), inf.f[0], inf.f[1], inf.f[2] });
            if (inf.class == .diphthong) print(" fe {d:.0} {d:.0} {d:.0}", .{ inf.f_end[0], inf.f_end[1], inf.f_end[2] });
            print(" b {d:.0} {d:.0} {d:.0}\n", .{ inf.b[0], inf.b[1], inf.b[2] });
        }
        print("style neutral f0 110 range 1 rate 1 rd 0 breath 1\n", .{});
        return;
    }
    if (std.mem.eql(u8, cmd, "segs")) {
        // The planned segments with their prosodic features, as CSV.
        var as = std.heap.ArenaAllocator.init(gpa);
        defer as.deinit();
        const u = try synth.plan(as.allocator(), rest, cli.opt);
        print("phone,stress,word,function_word,accent,phrase_final,dur_ms\n", .{});
        for (u.segs) |sg| {
            const w: i32 = if (sg.word) |x| @intCast(x) else -1;
            print("{s},{d},{d},{d},{d},{d},{d:.1}\n", .{ sg.phone.name(), sg.stress, w, @intFromBool(sg.function_word), @intFromBool(sg.accent), @intFromBool(sg.phrase_final), sg.dur * 1000 });
        }
        return;
    }
    if (std.mem.eql(u8, cmd, "frames")) {
        // Formant parameter tracks as CSV (one row per 5 ms frame).
        var as = std.heap.ArenaAllocator.init(gpa);
        defer as.deinit();
        const u = try synth.plan(as.allocator(), rest, cli.opt);
        const mode: formant.Mode = if (cli.opt.engine == .formant_classic) .classic else .refined;
        const frames = try formant.makeFrames(as.allocator(), &u, mode, cli.opt.formantVoice());
        print("t_ms,phone,f0,av,ah,af,F1,F2,F3,B1,FNZ,rd\n", .{});
        var si: usize = 0;
        for (frames, 0..) |f, i| {
            const t = @as(f32, @floatFromInt(i)) / prosody.frame_rate;
            while (si + 1 < u.segs.len and t >= u.segs[si].end()) si += 1;
            print("{d},{s},{d:.0},{d:.2},{d:.2},{d:.2},{d:.0},{d:.0},{d:.0},{d:.0},{d:.0},{d:.2}\n", .{ i * 5, u.segs[si].phone.name(), f.f0, f.av, f.ah, f.af, f.f[0], f.f[1], f.f[2], f.b[0], f.fnz, f.rd });
        }
        return;
    }
    if (std.mem.eql(u8, cmd, "bench")) {
        const text = if (rest.len > 0) rest else bench.bench_text;
        print("Building all voices and timing {d} runs per engine...\n", .{cli.reps});
        const rows = try bench.run(synth, gpa, text, cli.reps, cli.opt);
        const md = try bench.report(gpa, &rows, synth);
        defer gpa.free(md);
        print("{s}", .{md});
        return;
    }
    std.debug.print("unknown command: {s} (try: menu, say, compare, phonemes, bench)\n", .{cmd});
    return error.BadArgs;
}

fn loadDict(synth: *tts.Synth, path: []const u8, is_default: bool) void {
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, synth.gpa, .limited(64 << 20)) catch {
        if (!is_default) std.debug.print("could not read dictionary {s}\n", .{path});
        return;
    };
    defer synth.gpa.free(data);
    const n = synth.lex.loadCmudict(data) catch 0;
    std.debug.print("loaded {d} words from {s}\n", .{ n, path });
}

fn speakAndSave(synth: *tts.Synth, text: []const u8, opt: tts.Options, path: []const u8) !void {
    if (opt.engine != .formant_classic and opt.engine != .formant_refined and !built(synth, opt.engine)) {
        print("(building the {s} voice on first use...)\n", .{opt.engine.label()});
    }
    const r = try synth.speak(synth.gpa, text, opt);
    defer r.deinit(synth.gpa);
    try saveWav(r.samples, path);
    print("{s}\n", .{r.transcription});
    if (r.build_seconds > 0.01) print("  voice build: {d:.2} s\n", .{r.build_seconds});
    print("  {s}: {d:.2} s of audio in {d:.1} ms (plan {d:.2} ms) -> {s}\n", .{ opt.engine.label(), r.audio_seconds, r.synth_seconds * 1000, r.plan_seconds * 1000, path });
    if (opt.engine == .unit_selection) print("  units {d}, non-contiguous joins {d}, inventory fallbacks {d}\n", .{ r.unit_stats.units, r.unit_stats.joins, r.unit_stats.fallbacks });
    if (opt.formant_scale != 1.0 and !opt.engine.supportsFormantScale()) print("  (formant scaling is not possible with this engine: ignored)\n", .{});
}

fn built(synth: *tts.Synth, e: tts.Engine) bool {
    return switch (e) {
        .formant_classic, .formant_refined => true,
        .diphone => synth.diphone_voice != null,
        .unit_selection => synth.unit_voice != null,
        .hmm => synth.hmm_voice != null,
    };
}

fn saveWav(samples: []const f32, path: []const u8) !void {
    if (std.fs.path.dirname(path)) |d| std.Io.Dir.cwd().createDirPath(io, d) catch {};
    const bytes = try wav.encode(std.heap.smp_allocator, samples, 16000);
    defer std.heap.smp_allocator.free(bytes);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
}

fn compare(synth: *tts.Synth, text: []const u8, base: tts.Options, play: bool) !void {
    for (0..tts.Engine.count) |i| {
        var opt = base;
        opt.engine = @enumFromInt(i);
        var buf: [128]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "out/{s}.wav", .{opt.engine.short()});
        try speakAndSave(synth, text, opt, path);
        if (play) playFile(path);
    }
}

/// `batch LIST TAG`: synthesize every line of LIST with the chosen engine
/// to out/eval/<TAG>_<n>.wav (used by tools/eval-asr.sh).
fn batch(synth: *tts.Synth, arena: std.mem.Allocator, args: []const []const u8, cli: *Cli) !void {
    if (args.len < 2) {
        std.debug.print("usage: ttslab batch LIST TAG [options]\n", .{});
        return error.BadArgs;
    }
    const data = try std.Io.Dir.cwd().readFileAlloc(io, args[0], arena, .limited(1 << 20));
    var lines = std.mem.splitScalar(u8, data, '\n');
    var n: usize = 0;
    var total_synth: f64 = 0;
    var total_audio: f64 = 0;
    while (lines.next()) |l| {
        const line = std.mem.trim(u8, l, " \r\t");
        if (line.len == 0) continue;
        const r = try synth.speak(synth.gpa, line, cli.opt);
        defer r.deinit(synth.gpa);
        // A TAG with a slash is a path prefix; then the planned segments are
        // written next to each WAV too (tools/fit-voice.py reads them).
        const prefix = std.mem.indexOfScalar(u8, args[1], '/') != null;
        const path = if (prefix)
            try std.fmt.allocPrint(arena, "{s}_{d:0>3}.wav", .{ args[1], n })
        else
            try std.fmt.allocPrint(arena, "out/eval/{s}_{d:0>2}.wav", .{ args[1], n });
        try saveWav(r.samples, path);
        if (prefix) {
            var as = std.heap.ArenaAllocator.init(synth.gpa);
            defer as.deinit();
            const u = try synth.plan(as.allocator(), line, cli.opt);
            var csv: std.ArrayList(u8) = .empty;
            try csv.appendSlice(as.allocator(), "phone,stress,word,function_word,accent,phrase_final,start_ms,dur_ms\n");
            for (u.segs) |sg| {
                const w: i32 = if (sg.word) |x| @intCast(x) else -1;
                try csv.print(as.allocator(), "{s},{d},{d},{d},{d},{d},{d:.1},{d:.1}\n", .{ sg.phone.name(), sg.stress, w, @intFromBool(sg.function_word), @intFromBool(sg.accent), @intFromBool(sg.phrase_final), sg.start * 1000, sg.dur * 1000 });
            }
            const cpath = try std.fmt.allocPrint(arena, "{s}_{d:0>3}.csv", .{ args[1], n });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = cpath, .data = csv.items });
        }
        total_synth += r.synth_seconds;
        total_audio += r.audio_seconds;
        n += 1;
    }
    print("{s}: {d} files, {d:.1} s audio, synth {d:.1} ms total\n", .{ args[1], n, total_audio, total_synth * 1000 });
}

fn playFile(path: []const u8) void {
    const gpa = std.heap.smp_allocator;
    if (player_env) |cmd| {
        // User-chosen player: split on spaces, append the file.
        var argv: std.ArrayList([]const u8) = .empty;
        defer argv.deinit(gpa);
        var it = std.mem.tokenizeScalar(u8, cmd, ' ');
        while (it.next()) |a| argv.append(gpa, a) catch return;
        argv.append(gpa, path) catch return;
        if (!tryPlayer(argv.items)) print("  (TTSLAB_PLAYER={s} failed)\n", .{cmd});
        return;
    }
    const players = [_][]const u8{ "pw-play", "paplay", "aplay" };
    for (players) |p| {
        if (tryPlayer(&.{ p, path })) return;
    }
    print("  (no audio player worked: tried pw-play, paplay, aplay. Set TTSLAB_PLAYER, or open {s} yourself)\n", .{path});
}

/// Run a player to completion. Returns false (after printing why) if it
/// could not be started or exited with an error, so the next one is tried.
fn tryPlayer(argv: []const []const u8) bool {
    const gpa = std.heap.smp_allocator;
    const r = std.process.run(gpa, io, .{ .argv = argv, .stderr_limit = .limited(4096), .stdout_limit = .limited(4096) }) catch |e| {
        if (e != error.FileNotFound) print("  ({s}: {t})\n", .{ argv[0], e });
        return false;
    };
    defer gpa.free(r.stdout);
    defer gpa.free(r.stderr);
    const ok = switch (r.term) {
        .exited => |code| code == 0,
        else => false,
    };
    if (!ok) {
        print("  ({s} failed: {s})\n", .{ argv[0], std.mem.trim(u8, r.stderr, " \n") });
        return false;
    }
    if (!player_announced) {
        print("  (played with {s}; set TTSLAB_PLAYER to use another player)\n", .{argv[0]});
        player_announced = true;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Interactive menu
// ---------------------------------------------------------------------------

fn menu(synth: *tts.Synth, cli: *Cli) !void {
    var in_buf: [4096]u8 = undefined;
    var stdin = std.Io.File.stdin().reader(io, &in_buf);
    var opt = cli.opt;
    var last_text: []u8 = try synth.gpa.dupe(u8, "Hello! This is a toy speech synthesizer. Can you hear the difference?");
    defer synth.gpa.free(last_text);
    var play = true;

    while (true) {
        print(
            \\
            \\================ TTS Lab ================
            \\ Engine : [{d}] {s}
            \\ Prosody: F0 model {t}, pitch x{d:.2}, rate x{d:.2}, range x{d:.2}, formant x{d:.2}
            \\ Options: diphone PSOLA {s} | unit-sel modify prosody {s} | HMM MLPG {s}, GV {s} | play {s}
            \\-----------------------------------------
            \\  1-5   choose engine:  1 classic formant  2 refined formant
            \\                        3 diphone+PSOLA    4 unit selection   5 HMM
            \\  f     cycle F0 model (flat / hat / fujisaki)
            \\  p X   pitch scale     r X  rate     g X  pitch range     v X  formant (vocal tract) scale
            \\  o     toggle the current engine's option
            \\  a     speak last text with all engines (out/<engine>.wav)
            \\  s     speak last text again        x  show transcription
            \\  m     toggle playback              b  benchmark (cost profile)
            \\  q     quit
            \\ Anything else is spoken. Last: "{s}"
            \\>
        , .{
            @intFromEnum(opt.engine) + 1,                   opt.engine.label(),
            opt.prosody.f0_model,                           opt.pitch_scale,
            opt.prosody.rate,                               opt.prosody.pitch_range,
            opt.formant_scale,                              onOff(opt.diphone_psola),
            onOff(opt.unit_modify),                         onOff(opt.hmm_mlpg),
            onOff(opt.hmm_gv),                              onOff(play),
            last_text,
        });
        const line_raw = (stdin.interface.takeDelimiter('\n') catch null) orelse return;
        const line = std.mem.trim(u8, line_raw, " \t\r");
        if (line.len == 0) continue;

        var it = std.mem.tokenizeScalar(u8, line, ' ');
        const head = it.next().?;
        const arg = it.rest();
        const single = head.len == 1 and arg.len == 0;

        if (single and head[0] >= '1' and head[0] <= '5') {
            opt.engine = @enumFromInt(head[0] - '1');
            if (!built(synth, opt.engine)) {
                print("building the {s} voice...\n", .{opt.engine.label()});
                const s = synth.ensure(opt.engine) catch |e| {
                    print("build failed: {t}\n", .{e});
                    continue;
                };
                print("built in {d:.2} s\n", .{s});
            }
            continue;
        }
        if (single and head[0] == 'q') return;
        if (single and head[0] == 'f') {
            opt.prosody.f0_model = switch (opt.prosody.f0_model) {
                .flat => .hat,
                .hat => .fujisaki,
                .fujisaki => .flat,
            };
            continue;
        }
        if (single and head[0] == 'm') {
            play = !play;
            continue;
        }
        if (single and head[0] == 'o') {
            switch (opt.engine) {
                .diphone => opt.diphone_psola = !opt.diphone_psola,
                .unit_selection => opt.unit_modify = !opt.unit_modify,
                .hmm => {
                    // Cycle: MLPG -> MLPG+GV -> no MLPG.
                    if (opt.hmm_mlpg and !opt.hmm_gv) {
                        opt.hmm_gv = true;
                    } else if (opt.hmm_mlpg) {
                        opt.hmm_mlpg = false;
                        opt.hmm_gv = false;
                    } else opt.hmm_mlpg = true;
                },
                else => print("(the formant engines' option is the engine itself: 1 = classic, 2 = refined)\n", .{}),
            }
            continue;
        }
        if (head.len == 1 and (head[0] == 'p' or head[0] == 'r' or head[0] == 'g' or head[0] == 'v') and arg.len > 0) {
            const val = std.fmt.parseFloat(f32, std.mem.trim(u8, arg, " ")) catch {
                print("not a number\n", .{});
                continue;
            };
            switch (head[0]) {
                'p' => opt.pitch_scale = std.math.clamp(val, 0.3, 3),
                'r' => opt.prosody.rate = std.math.clamp(val, 0.3, 4),
                'g' => opt.prosody.pitch_range = std.math.clamp(val, 0, 3),
                else => opt.formant_scale = std.math.clamp(val, 0.6, 1.6),
            }
            continue;
        }
        if (single and head[0] == 'x') {
            var as = std.heap.ArenaAllocator.init(synth.gpa);
            defer as.deinit();
            const u = try synth.plan(as.allocator(), last_text, opt);
            print("{s}\n", .{try u.transcription(as.allocator())});
            continue;
        }
        if (single and head[0] == 'a') {
            compare(synth, last_text, opt, play) catch |e| print("failed: {t}\n", .{e});
            continue;
        }
        if (single and head[0] == 'b') {
            const rows = bench.run(synth, synth.gpa, bench.bench_text, cli.reps, opt) catch |e| {
                print("bench failed: {t}\n", .{e});
                continue;
            };
            const md = try bench.report(synth.gpa, &rows, synth);
            defer synth.gpa.free(md);
            print("{s}", .{md});
            continue;
        }
        if (!(single and head[0] == 's')) {
            synth.gpa.free(last_text);
            last_text = try synth.gpa.dupe(u8, line);
        }
        speakAndSave(synth, last_text, opt, "out/menu.wav") catch |e| {
            print("synthesis failed: {t}\n", .{e});
            continue;
        };
        if (play) playFile("out/menu.wav");
    }
}

fn onOff(b: bool) []const u8 {
    return if (b) "on" else "off";
}
