//! Feed a 16 kHz mono float32 WAV file through the Nemotron engine and print
//! the segment updates. Verification harness for the SpeechEngine extraction:
//!
//! ```sh
//! say -o /tmp/test.wav --data-format=LEF32@16000 "Hello world."
//! cargo run -p larmindon-engine-nemotron --example feed_wav -- \
//!     ~/projects/nemotron-03-2026 /tmp/test.wav
//! ```

use larmindon_core::diagnostics::DiagSink;
use larmindon_core::engine::registry::EngineFactory;
use larmindon_core::engine::SessionContext;
use larmindon_engine_nemotron::NemotronFactory;

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect("usage: feed_wav <model_dir> <wav_file>");
    let wav_path = args.next().expect("usage: feed_wav <model_dir> <wav_file>");

    let samples = read_f32_wav(&wav_path);
    println!(
        "Read {} samples ({:.1}s at 16kHz) from {}",
        samples.len(),
        samples.len() as f64 / 16000.0,
        wav_path
    );

    let factory = NemotronFactory;
    let mut config = factory.default_config();
    config["model_path"] = model_path.into();

    let mut engine = factory.create(&config).expect("create engine");
    engine
        .begin_session(
            SessionContext {
                diag: DiagSink::disabled(),
            },
            &config,
        )
        .expect("begin_session");

    let mut transcript = String::new();
    engine.on_speech_start();
    // Feed in VAD-frame-sized pieces like the real processing loop does.
    for frame in samples.chunks(512) {
        for update in engine.feed(frame).expect("feed") {
            println!(
                "  segment {} (final={}): {:?}",
                update.segment_id, update.is_final, update.text
            );
            transcript.push_str(&update.text);
        }
    }
    for update in engine.on_speech_end().expect("on_speech_end") {
        println!(
            "  segment {} (final={}): {:?}",
            update.segment_id, update.is_final, update.text
        );
        transcript.push_str(&update.text);
    }
    engine.end_session().expect("end_session");

    println!("\nTranscript:{}", transcript);
}

/// Minimal RIFF/WAVE reader for the LEF32 mono files `say` produces.
fn read_f32_wav(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read wav file");
    assert_eq!(&bytes[0..4], b"RIFF", "not a RIFF file");
    assert_eq!(&bytes[8..12], b"WAVE", "not a WAVE file");

    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if chunk_id == b"data" {
            let data = &bytes[pos + 8..pos + 8 + chunk_len];
            return data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
        }
        pos += 8 + chunk_len + (chunk_len & 1);
    }
    panic!("no data chunk found in {}", path);
}
