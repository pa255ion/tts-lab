#!/usr/bin/env bash
# Intelligibility check: synthesize tools/eval-sentences.txt with each
# configuration, transcribe with whisper.cpp, report word error rate.
# Run inside `nix develop .#eval` (needs data/models/ggml-base.en.bin:
#   curl -L -o data/models/ggml-base.en.bin \
#     https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin)
#
# A crude proxy: an ASR model trained on human speech is not a listener, and
# it is good at guessing words from context. Use it to compare engines, not
# as an absolute score.
set -euo pipefail
cd "$(dirname "$0")/.."
LIST=${LIST:-tools/eval-sentences.txt}
BIN=./zig-out/bin/ttslab
MODEL=${MODEL:-data/models/ggml-base.en.bin}
mkdir -p out/eval
rm -f out/eval/*.wav out/eval/*.txt

configs=(
  "classic|-e classic"
  "classic_flat|-e classic --f0 flat"
  "refined|-e refined"
  "diphone|-e diphone"
  "diphone_raw|-e diphone --no-psola"
  "unitsel|-e unitsel"
  "unitsel_modify|-e unitsel --modify"
  "hmm|-e hmm"
  "hmm_nomlpg|-e hmm --no-mlpg"
  "hmm_gv|-e hmm --gv"
)
[ -n "${CONFIGS:-}" ] && IFS=, read -ra configs <<< "$CONFIGS"
tags=()
for c in "${configs[@]}"; do
  tag=${c%%|*}; opts=${c#*|}
  # shellcheck disable=SC2086
  $BIN batch "$LIST" "$tag" $opts 2>/dev/null
  tags+=("$tag")
done
export MODEL
ls out/eval/*.wav | xargs -P "${JOBS:-2}" -I{} sh -c \
  'env -u LD_LIBRARY_PATH whisper-cli -m "$MODEL" -t 4 -f "$1" -nt -np -l en 2>/dev/null > "${1%.wav}.txt"' _ {}
python3 tools/wer.py "$LIST" "${tags[@]}"
