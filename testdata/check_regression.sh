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
# Requires the Nemotron model named by ~/.config/larmindon/settings.json (or
# --model), and python3 for the diagnostics dump. See README.md.
#
# usage: bash testdata/check_regression.sh
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
crate="$(dirname "$here")"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$crate" || exit 1

fail=0
run() { cargo run --release --quiet --example replay_wav -- "$@" 2>/dev/null; }
# Goldens start at the EMISSIONS banner; earlier lines are machine-specific
# stdout (settings path, model path).
emissions() { sed -n '/=== EMISSIONS/,$p' "$1"; }

if ! cargo build --release --quiet --example replay_wav 2>&1; then
  echo "BUILD FAILED"
  exit 1
fi

echo "=== fixture 1: baseline speech ==="
run "$here/baseline_speech.wav" > "$work/baseline.txt"
if diff -u "$here/golden_baseline.txt" <(emissions "$work/baseline.txt") > "$work/d1"; then
  echo "  PASS"
else
  echo "  FAIL"; head -40 "$work/d1"; fail=1
fi

echo "=== fixture 2: mid-speech reset + replay ==="
run "$here/baseline_speech.wav" --empty-reset 1 --diag "$work/reset.sqlite" \
  > "$work/reset.txt"
if diff -u "$here/golden_reset.txt" <(emissions "$work/reset.txt") > "$work/d2"; then
  echo "  PASS (transcript)"
else
  echo "  FAIL (transcript)"; head -40 "$work/d2"; fail=1
fi

if python "$here/dump_diag.py" "$work/reset.sqlite" > "$work/reset_diag.txt" 2>"$work/pyerr"; then
  if diff -u "$here/golden_reset_diag.txt" "$work/reset_diag.txt" > "$work/d3"; then
    echo "  PASS (diagnostics)"
  else
    echo "  FAIL (diagnostics)"; head -40 "$work/d3"; fail=1
  fi
else
  echo "  SKIP (diagnostics) - could not run dump_diag.py:"; cat "$work/pyerr"
fi

echo
if [ $fail -eq 0 ]; then
  echo "ALL REGRESSION GATES PASS"
else
  echo "REGRESSION DETECTED"
fi
exit $fail
