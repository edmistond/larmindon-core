#!/usr/bin/env bash
# Regenerates spanish_speech.wav, the non-English fixture for Nemotron 3.5.
#
# macOS-only (`say`), but the output is committed, so the fixture itself works
# everywhere. Regenerate only when changing the script: a different macOS
# release may render the same text differently, which changes the goldens.
#
# usage: bash testdata/make_spanish.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
voice="Mónica"

if ! say -v '?' | grep -q "^${voice} "; then
  echo "voice ${voice} (es_ES) is not installed" >&2
  exit 1
fi

# Declaratives, a question and an exclamation, so the punctuation reset sees
# every terminator the model emits for Spanish.
text="Buenos días a todos. Hoy vamos a hablar del tiempo en Madrid. \
Esta mañana hacía frío, pero por la tarde la temperatura subió hasta veinte grados. \
¿Alguien sabe si va a llover el fin de semana? \
Espero que no, porque quiero ir al parque con mi familia. \
¡Qué ganas tengo de ver el sol!"

# 16 kHz mono 16-bit PCM: what replay_wav reads, and no resampling on the way in.
say -v "$voice" --file-format=WAVE --data-format=LEI16@16000 \
  -o "$here/spanish_speech.wav" "$text"
echo "wrote $here/spanish_speech.wav"
