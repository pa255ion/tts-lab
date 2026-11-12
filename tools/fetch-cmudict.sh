#!/usr/bin/env bash
# Download the CMU Pronouncing Dictionary (BSD-style licence, ~135k words)
# to data/cmudict.dict, where ttslab picks it up automatically.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
curl -fL --retry 3 -o data/cmudict.dict \
  https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict
wc -l data/cmudict.dict
