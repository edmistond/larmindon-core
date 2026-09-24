#!/usr/bin/env bash
# ASR behaviour-identity gate.
#
# Replays a fixed WAV through the real engine (real resampler, AGC, Silero VAD
# and Nemotron model) and diffs against goldens captured before the ASR backend
# refactor. `processing_loop` has no other test seam, so this is the only thing
# that catches a transcript regression.
#
#   fixture 1  ordinary speech           -> main transcribe path
#   fixture 2  same audio, empty-reset=1 -> mid-speech stuck-decoder reset and
#                                           its replay path, checked via
#                                           diagnostics (the transcript alone is
#                                           identical to fixture 1)
#
#   optional   Nemotron 3.5 (multilingual) goldens, run only when a 3.5
#              model is given with --model-ml or LARMINDON_MODEL_ML:
#                baseline speech with en-US and with auto
#                Spanish speech with es-ES and with auto
#
# Fixtures 1 and 2 need the English Nemotron model: the one named by
# ~/.config/larmindon/settings.json, or --model. python3 is needed for the
# diagnostics dump. See README.md.
#
# usage: bash testdata/check_regression.sh [--model PATH] [--model-ml PATH]
set -uo pipefail

model=""
model_ml="${LARMINDON_MODEL_ML:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --model) model="${2:?--model needs a path}"; shift 2 ;;
    --model-ml) model_ml="${2:?--model-ml needs a path}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
crate="$(dirname "$here")"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$crate" || exit 1

fail=0
# --provider is pinned, not inherited. The harness starts from your saved
# settings, so without this the gate silently runs whatever asr_provider
# happens to be set to — and diffs Nemotron goldens against another backend's
# output. That is a confusing failure precisely when you are least expecting it.
run() {
  cargo run --release --quiet --example replay_wav -- \
    "$@" --provider nemotron 2>/dev/null
}
# The English fixtures pin --model only when given one, so a bare run still
# uses the saved model_path exactly as before.
run_en() {
  if [ -n "$model" ]; then run "$@" --model "$model"; else run "$@"; fi
}
# Goldens start at the EMISSIONS banner; earlier lines are machine-specific
# stdout (settings path, model path).
emissions() { sed -n '/=== EMISSIONS/,$p' "$1"; }

if ! cargo build --release --quiet --example replay_wav 2>&1; then
  echo "BUILD FAILED"
  exit 1
fi

echo "=== fixture 1: baseline speech ==="
run_en "$here/baseline_speech.wav" > "$work/baseline.txt"
if diff -u "$here/golden_baseline.txt" <(emissions "$work/baseline.txt") > "$work/d1"; then
  echo "  PASS"
else
  echo "  FAIL"; head -40 "$work/d1"; fail=1
fi

echo "=== fixture 2: mid-speech reset + replay ==="
run_en "$here/baseline_speech.wav" --empty-reset 1 --diag "$work/reset.sqlite" \
  > "$work/reset.txt"
if diff -u "$here/golden_reset.txt" <(emissions "$work/reset.txt") > "$work/d2"; then
  echo "  PASS (transcript)"
else
  echo "  FAIL (transcript)"; head -40 "$work/d2"; fail=1
fi

# python3 first: macOS and most Linux installs have no bare `python`. CRs are
# stripped because Python on Windows prints CRLF and the golden is LF.
py="$(command -v python3 || command -v python)"
if "${py:-python3}" "$here/dump_diag.py" "$work/reset.sqlite" 2>"$work/pyerr" \
  | tr -d '\r' > "$work/reset_diag.txt"; then
  if diff -u "$here/golden_reset_diag.txt" "$work/reset_diag.txt" > "$work/d3"; then
    echo "  PASS (diagnostics)"
  else
    echo "  FAIL (diagnostics)"; head -40 "$work/d3"; fail=1
  fi
else
  echo "  SKIP (diagnostics) - could not run dump_diag.py:"; cat "$work/pyerr"
fi

# Multilingual goldens. Opt-in, so the gate still runs on a machine that only
# has the English model. The mode line is checked first: pointing --model-ml at
# the English model would otherwise fail every diff in a way that looks like a
# 3.5 regression.
if [ -n "$model_ml" ]; then
  ml_fixture() { # <label> <wav> <lang> <golden>
    echo "=== multilingual: $1 ($3) ==="
    run "$here/$2" --model "$model_ml" --lang "$3" > "$work/ml.txt"
    if ! grep -q '^Nemotron mode: multilingual' "$work/ml.txt"; then
      echo "  FAIL - $model_ml did not load as a multilingual model"; fail=1
      return
    fi
    if diff -u "$here/$4" <(emissions "$work/ml.txt") > "$work/dml"; then
      echo "  PASS"
    else
      echo "  FAIL"; head -40 "$work/dml"; fail=1
    fi
  }
  ml_fixture "baseline speech" baseline_speech.wav en-US golden_ml_baseline_en-US.txt
  ml_fixture "baseline speech" baseline_speech.wav auto golden_ml_baseline_auto.txt
  ml_fixture "Spanish speech" spanish_speech.wav es-ES golden_ml_spanish_es-ES.txt
  ml_fixture "Spanish speech" spanish_speech.wav auto golden_ml_spanish_auto.txt
else
  echo "=== multilingual: SKIP (pass --model-ml or set LARMINDON_MODEL_ML) ==="
fi

echo
if [ $fail -eq 0 ]; then
  echo "ALL REGRESSION GATES PASS"
else
  echo "REGRESSION DETECTED"
fi
exit $fail
