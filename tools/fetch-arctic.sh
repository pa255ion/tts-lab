#!/usr/bin/env bash
# Download the CMU ARCTIC phone labels and prompts (no audio) for three US
# male speakers to data/arctic/, for tools/fit-durations.py.
# ARCTIC is free for any use under CMU's licence (data/arctic/COPYING);
# nothing from it is committed, only the numbers fitted from it.
# AUDIO=1 also fetches the first 20 recordings per speaker (tools/acoustics.py);
# AUDIO=N the first N (tools/align.py trains on 300 per speaker).
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/arctic
base=http://festvox.org/cmu_arctic/cmu_arctic
mkdir -p "$D"
curl -fsSL -o "$D/txt.done.data" "$base/cmu_us_bdl_arctic/etc/txt.done.data"
curl -fsSL -o "$D/COPYING" "$base/cmu_us_bdl_arctic/COPYING"
sed -nE 's/^\( (arctic_[ab][0-9]+) "(.*)" \)$/\1\t\2/p' "$D/txt.done.data" > "$D/prompts.tsv"
for spk in ${SPEAKERS:-bdl rms jmk}; do
  mkdir -p "$D/$spk"
  cut -f1 "$D/prompts.tsv" | while read -r id; do
    [ -s "$D/$spk/$id.lab" ] || echo "$base/cmu_us_${spk}_arctic/lab/$id.lab $D/$spk/$id.lab"
    n=${AUDIO:-0}; [ "$n" = 1 ] && n=20
    if [ "$n" -gt 0 ] && [[ $id == arctic_a* ]] && [ "$((10#${id#arctic_a}))" -le "$n" ] && [ ! -s "$D/$spk/$id.wav" ]; then
      echo "$base/cmu_us_${spk}_arctic/wav/$id.wav $D/$spk/$id.wav"
    fi
  done
done | xargs -r -P 6 -n 2 sh -c 'curl -fsS -m 30 --retry 2 -o "$1" "$0"'
echo "$(wc -l < "$D/prompts.tsv") prompts; labels: $(ls "$D"/*/*.lab | wc -l)"
