"""Dump the drain-invariant columns of a diagnostics DB for regression diffing.

Timing and scheduling columns (uptime_ms, inference_ms, iteration_ms,
drain_samples, drain_audio_ms, vad_ms, resample_ms, asr_buf_len,
speech_duration_ms) vary run to run and are deliberately excluded. What remains
is reproducible because chunking is anchored at speech onset, so drain
boundaries do not change which samples land in which chunk.

usage: python dump_diag.py <diag.sqlite>
"""

import sqlite3
import sys

EVENTS = """
    SELECT event_type, chunk_num, text_empty, text_preview, vad_state, chunk_source
    FROM events WHERE session_id = ? ORDER BY id
"""

VAD_EVENTS = """
    SELECT event_type, consecutive_empty, chunks_since_decoder_reset,
           audio_ms_since_decoder_reset, replay_chunks, replay_audio_ms,
           replay_nonempty_chunks
    FROM vad_events WHERE session_id = ? ORDER BY id
"""


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    conn = sqlite3.connect(sys.argv[1])
    row = conn.execute("SELECT max(id) FROM sessions").fetchone()
    if not row or row[0] is None:
        print("no sessions in diagnostics DB", file=sys.stderr)
        return 1
    # The replay harness runs a warm-up session first; the last one is the real
    # pass.
    session_id = row[0]

    out = []
    for label, query in (("events", EVENTS), ("vad_events", VAD_EVENTS)):
        rows = list(conn.execute(query, (session_id,)))
        out.append(f"=== {label} ({len(rows)}) ===")
        out.extend(repr(r) for r in rows)

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
