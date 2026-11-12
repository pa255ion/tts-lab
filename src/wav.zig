//! 16-bit mono WAV encoding.

const std = @import("std");

pub fn encode(alloc: std.mem.Allocator, samples: []const f32, sample_rate: u32) ![]u8 {
    const data_len: u32 = @intCast(samples.len * 2);
    const buf = try alloc.alloc(u8, 44 + data_len);
    const W = struct {
        fn u32le(b: []u8, v: u32) void {
            std.mem.writeInt(u32, b[0..4], v, .little);
        }
        fn u16le(b: []u8, v: u16) void {
            std.mem.writeInt(u16, b[0..2], v, .little);
        }
    };
    @memcpy(buf[0..4], "RIFF");
    W.u32le(buf[4..], 36 + data_len);
    @memcpy(buf[8..16], "WAVEfmt ");
    W.u32le(buf[16..], 16);
    W.u16le(buf[20..], 1); // PCM
    W.u16le(buf[22..], 1); // mono
    W.u32le(buf[24..], sample_rate);
    W.u32le(buf[28..], sample_rate * 2);
    W.u16le(buf[32..], 2);
    W.u16le(buf[34..], 16);
    @memcpy(buf[36..40], "data");
    W.u32le(buf[40..], data_len);
    for (samples, 0..) |s, i| {
        const v: i16 = @intFromFloat(std.math.clamp(s, -1, 1) * 32767);
        std.mem.writeInt(i16, buf[44 + i * 2 ..][0..2], v, .little);
    }
    return buf;
}
