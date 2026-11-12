#!/usr/bin/env bash
# Render a spectrogram PNG for each WAV given: foo.wav -> foo.png
# Run inside `nix develop`. LD_LIBRARY_PATH is cleared because a global one
# pointing at a newer glibc's libraries breaks nixpkgs' ffmpeg.
set -euo pipefail
for f in "$@"; do
  env -u LD_LIBRARY_PATH ffmpeg -loglevel error -y -i "$f" \
    -lavfi "showspectrumpic=s=${SPEC_SIZE:-1400x400}:legend=0:scale=log:fscale=lin:color=intensity:gain=1" \
    "${f%.wav}.png"
done
