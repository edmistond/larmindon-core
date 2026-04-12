use ndarray::{s, Array1, Array2, ArrayD, IxDyn};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

const VAD_SAMPLE_RATE: u32 = 16_000;
const VAD_CHUNK_SIZE: usize = 512;
const VAD_CONTEXT_SIZE: usize = 64;

// ---------------------------------------------------------------------------
// Silero VAD model wrapper (v5, opset 16)
// ---------------------------------------------------------------------------

struct SileroModel {
    session: Session,
    state: ArrayD<f32>,
    context: Array2<f32>,
}

impl SileroModel {
    fn new(model_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            state: ArrayD::<f32>::zeros(IxDyn(&[2, 1, 128])),
            context: Array2::<f32>::zeros((1, VAD_CONTEXT_SIZE)),
        })
    }

    fn reset(&mut self) {
        self.state = ArrayD::<f32>::zeros(IxDyn(&[2, 1, 128]));
        self.context = Array2::<f32>::zeros((1, VAD_CONTEXT_SIZE));
    }

    /// Feed a 512-sample frame and return the speech probability [0.0, 1.0].
    fn process_frame(&mut self, frame: &[f32]) -> Result<f32, Box<dyn std::error::Error>> {
        assert_eq!(frame.len(), VAD_CHUNK_SIZE);

        // Build input: [1, context_size + chunk_size]
        let total_len = VAD_CONTEXT_SIZE + VAD_CHUNK_SIZE;
        let mut input = Array2::<f32>::zeros((1, total_len));
        input
            .slice_mut(s![.., 0..VAD_CONTEXT_SIZE])
            .assign(&self.context);
        for (j, &sample) in frame.iter().enumerate() {
            input[[0, VAD_CONTEXT_SIZE + j]] = sample;
        }

        let sr_array = Array1::<i64>::from_elem(1, VAD_SAMPLE_RATE as i64);

        let input_tensor = Tensor::from_array(input.clone())?;
        let state_tensor = Tensor::from_array(self.state.clone())?;
        let sr_tensor = Tensor::from_array(sr_array)?;

        let inputs = ort::inputs![input_tensor, state_tensor, sr_tensor];
        let outputs = self.session.run(inputs)?;

        // Update state
        let state_key = if outputs.contains_key("stateN") {
            "stateN"
        } else {
            "state"
        };
        let (state_shape, state_data) = outputs[state_key].try_extract_tensor::<f32>()?;
        let shape_usize: Vec<usize> = state_shape.iter().map(|&d| d as usize).collect();
        self.state = ArrayD::<f32>::from_shape_vec(IxDyn(&shape_usize), state_data.to_vec())?;

        // Update context: last 64 samples of the input chunk
        let new_ctx: Vec<f32> = frame[VAD_CHUNK_SIZE - VAD_CONTEXT_SIZE..].to_vec();
        self.context = Array2::from_shape_vec((1, VAD_CONTEXT_SIZE), new_ctx)?;

        // Extract speech probability
        let output_key = if outputs.contains_key("output") {
            "output"
        } else {
            outputs
                .iter()
                .next()
                .map(|(name, _)| name)
                .unwrap_or("output")
        };
        let (_shape, output_data) = outputs[output_key].try_extract_tensor::<f32>()?;
        Ok(output_data[0])
    }
}

// ---------------------------------------------------------------------------
// Pre-speech ring buffer
// ---------------------------------------------------------------------------

pub struct PreSpeechRingBuffer {
    data: Vec<f32>,
    pub(crate) capacity: usize,
    write_pos: usize,
    len: usize,
}

impl PreSpeechRingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            capacity,
            write_pos: 0,
            len: 0,
        }
    }

    pub fn push_slice(&mut self, samples: &[f32]) {
        for &s in samples {
            self.data[self.write_pos] = s;
            self.write_pos = (self.write_pos + 1) % self.capacity;
        }
        self.len = (self.len + samples.len()).min(self.capacity);
    }

    /// Drain all buffered samples in chronological order, reset the buffer.
    pub fn drain_all(&mut self) -> Vec<f32> {
        if self.len == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(self.len);
        let start = if self.len < self.capacity {
            0
        } else {
            self.write_pos
        };
        for i in 0..self.len {
            out.push(self.data[(start + i) % self.capacity]);
        }
        self.len = 0;
        self.write_pos = 0;
        out
    }
}

// ---------------------------------------------------------------------------
// VAD state machine (model-independent, testable with synthetic probabilities)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadState {
    Silence,
    Speech,
}

#[derive(Debug)]
pub enum VadDecision {
    /// Frame is silence; was buffered in ring buffer.
    Silence,
    /// Speech just started; ring buffer has been drained into `pre_speech_samples`.
    SpeechStarted { pre_speech_samples: Vec<f32> },
    /// Speech continues; frame should be appended to ASR buffer.
    SpeechContinues,
    /// Speech just ended; frame should be appended, then flush + reset.
    SpeechEnded,
}

/// Pure state machine for VAD decisions, decoupled from the ONNX model.
/// Takes speech probabilities as input and produces decisions based on
/// threshold hysteresis and frame counting.
pub struct VadStateMachine {
    ring_buffer: PreSpeechRingBuffer,
    state: VadState,
    threshold_start: f32,
    threshold_end: f32,
    min_silence_frames: u32,
    min_speech_frames: u32,
    speech_frame_count: u32,
    silence_frame_count: u32,
}

impl VadStateMachine {
    pub fn new(
        threshold_start: f32,
        threshold_end: f32,
        min_silence_duration_ms: u32,
        min_speech_duration_ms: u32,
        pre_speech_ms: usize,
    ) -> Self {
        let pre_speech_samples = VAD_SAMPLE_RATE as usize * pre_speech_ms / 1000;
        let frame_ms = (VAD_CHUNK_SIZE as f32 / VAD_SAMPLE_RATE as f32 * 1000.0) as u32; // 32ms

        Self {
            ring_buffer: PreSpeechRingBuffer::new(pre_speech_samples),
            state: VadState::Silence,
            threshold_start,
            threshold_end,
            min_silence_frames: min_silence_duration_ms / frame_ms,
            min_speech_frames: min_speech_duration_ms / frame_ms,
            speech_frame_count: 0,
            silence_frame_count: 0,
        }
    }

    pub fn state(&self) -> VadState {
        self.state
    }

    /// Reset the state machine for reuse. Does NOT reset any external model state.
    pub fn reset(&mut self) {
        self.ring_buffer = PreSpeechRingBuffer::new(self.ring_buffer.capacity);
        self.state = VadState::Silence;
        self.speech_frame_count = 0;
        self.silence_frame_count = 0;
    }

    /// Update tunable parameters without resetting state.
    pub fn update_params(
        &mut self,
        threshold_start: f32,
        threshold_end: f32,
        min_silence_duration_ms: u32,
        min_speech_duration_ms: u32,
    ) {
        let frame_ms = (VAD_CHUNK_SIZE as f32 / VAD_SAMPLE_RATE as f32 * 1000.0) as u32;
        self.threshold_start = threshold_start;
        self.threshold_end = threshold_end;
        self.min_silence_frames = min_silence_duration_ms / frame_ms;
        self.min_speech_frames = min_speech_duration_ms / frame_ms;
    }

    /// Process a frame given a speech probability. Returns the decision.
    /// The `frame` samples are only used for ring buffer storage during silence.
    pub fn process(&mut self, prob: f32, frame: &[f32]) -> VadDecision {
        let active_threshold = match self.state {
            VadState::Silence => self.threshold_start,
            VadState::Speech => self.threshold_end,
        };
        let is_speech = prob >= active_threshold;

        match self.state {
            VadState::Silence => {
                if is_speech {
                    self.speech_frame_count += 1;
                    self.silence_frame_count = 0;

                    if self.speech_frame_count >= self.min_speech_frames {
                        self.state = VadState::Speech;
                        let pre_speech = self.ring_buffer.drain_all();
                        VadDecision::SpeechStarted {
                            pre_speech_samples: pre_speech,
                        }
                    } else {
                        // Not enough consecutive speech frames yet; buffer it
                        self.ring_buffer.push_slice(frame);
                        VadDecision::Silence
                    }
                } else {
                    // Leaky decrement: a single sub-threshold frame costs one
                    // frame of progress rather than resetting the entire counter.
                    // This tolerates brief probability dips during onset (unvoiced
                    // consonants, inter-phoneme gaps) that are common in noisy audio.
                    self.speech_frame_count = self.speech_frame_count.saturating_sub(1);
                    self.silence_frame_count += 1;
                    self.ring_buffer.push_slice(frame);
                    VadDecision::Silence
                }
            }
            VadState::Speech => {
                if is_speech {
                    self.speech_frame_count += 1;
                    self.silence_frame_count = 0;
                    VadDecision::SpeechContinues
                } else {
                    self.silence_frame_count += 1;
                    self.speech_frame_count = 0;

                    if self.silence_frame_count >= self.min_silence_frames {
                        self.state = VadState::Silence;
                        VadDecision::SpeechEnded
                    } else {
                        // Brief pause; keep treating as speech
                        VadDecision::SpeechContinues
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// VadProcessor: combines model + state machine
// ---------------------------------------------------------------------------

pub struct VadProcessor {
    model: SileroModel,
    state_machine: VadStateMachine,
}

impl VadProcessor {
    pub fn new(
        model_path: &Path,
        threshold_start: f32,
        threshold_end: f32,
        min_silence_duration_ms: u32,
        min_speech_duration_ms: u32,
        pre_speech_ms: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let model = SileroModel::new(model_path)?;
        let state_machine = VadStateMachine::new(
            threshold_start,
            threshold_end,
            min_silence_duration_ms,
            min_speech_duration_ms,
            pre_speech_ms,
        );

        Ok(Self {
            model,
            state_machine,
        })
    }

    pub fn state(&self) -> VadState {
        self.state_machine.state()
    }

    /// Reset all state for reuse between sessions.
    /// Clears the model's internal ONNX state, ring buffer, and state machine counters.
    pub fn reset(&mut self) {
        self.model.reset();
        self.state_machine.reset();
    }

    /// Update tunable parameters without reloading the model.
    pub fn update_params(
        &mut self,
        threshold_start: f32,
        threshold_end: f32,
        min_silence_duration_ms: u32,
        min_speech_duration_ms: u32,
    ) {
        self.state_machine.update_params(
            threshold_start,
            threshold_end,
            min_silence_duration_ms,
            min_speech_duration_ms,
        );
    }

    /// Process a single 512-sample frame. Returns the decision and the speech probability.
    pub fn process_frame(
        &mut self,
        frame: &[f32],
    ) -> Result<(VadDecision, f32), Box<dyn std::error::Error>> {
        let prob = self.model.process_frame(frame)?;
        let decision = self.state_machine.process(prob, frame);
        Ok((decision, prob))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Dummy frame for state machine tests (contents don't matter for decision logic)
    fn dummy_frame() -> Vec<f32> {
        vec![0.0; VAD_CHUNK_SIZE]
    }

    // Helper: create a state machine with typical defaults
    fn default_state_machine() -> VadStateMachine {
        VadStateMachine::new(
            0.5, // threshold_start
            0.3, // threshold_end
            500, // min_silence_duration_ms (~15 frames at 32ms)
            250, // min_speech_duration_ms (~7 frames at 32ms)
            500, // pre_speech_ms
        )
    }

    // -----------------------------------------------------------------------
    // PreSpeechRingBuffer tests
    // -----------------------------------------------------------------------

    #[test]
    fn ring_buffer_empty_drain() {
        let mut buf = PreSpeechRingBuffer::new(10);
        assert!(buf.drain_all().is_empty());
    }

    #[test]
    fn ring_buffer_push_fewer_than_capacity() {
        let mut buf = PreSpeechRingBuffer::new(10);
        buf.push_slice(&[1.0, 2.0, 3.0]);
        let drained = buf.drain_all();
        assert_eq!(drained, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn ring_buffer_push_exactly_capacity() {
        let mut buf = PreSpeechRingBuffer::new(4);
        buf.push_slice(&[1.0, 2.0, 3.0, 4.0]);
        let drained = buf.drain_all();
        assert_eq!(drained, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn ring_buffer_push_overflow_wraparound() {
        let mut buf = PreSpeechRingBuffer::new(4);
        buf.push_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let drained = buf.drain_all();
        // Oldest samples (1.0, 2.0) should be evicted
        assert_eq!(drained, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn ring_buffer_drain_resets_state() {
        let mut buf = PreSpeechRingBuffer::new(4);
        buf.push_slice(&[1.0, 2.0]);
        let _ = buf.drain_all();
        // After drain, buffer should be empty
        assert!(buf.drain_all().is_empty());
    }

    #[test]
    fn ring_buffer_push_after_drain() {
        let mut buf = PreSpeechRingBuffer::new(4);
        buf.push_slice(&[1.0, 2.0]);
        let _ = buf.drain_all();
        buf.push_slice(&[10.0, 20.0]);
        let drained = buf.drain_all();
        assert_eq!(drained, vec![10.0, 20.0]);
    }

    #[test]
    fn ring_buffer_multiple_pushes_before_drain() {
        let mut buf = PreSpeechRingBuffer::new(6);
        buf.push_slice(&[1.0, 2.0]);
        buf.push_slice(&[3.0, 4.0]);
        buf.push_slice(&[5.0, 6.0]);
        let drained = buf.drain_all();
        assert_eq!(drained, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn ring_buffer_wraparound_preserves_chronological_order() {
        let mut buf = PreSpeechRingBuffer::new(3);
        buf.push_slice(&[1.0, 2.0, 3.0]); // fills exactly
        buf.push_slice(&[4.0]); // wraps: [4.0, 2.0, 3.0], write_pos=1
        let drained = buf.drain_all();
        assert_eq!(drained, vec![2.0, 3.0, 4.0]);
    }

    // -----------------------------------------------------------------------
    // VadStateMachine tests
    // -----------------------------------------------------------------------

    #[test]
    fn silence_stays_silence_below_threshold() {
        let mut sm = default_state_machine();
        let frame = dummy_frame();
        for _ in 0..20 {
            let decision = sm.process(0.1, &frame);
            assert!(matches!(decision, VadDecision::Silence));
        }
        assert_eq!(sm.state(), VadState::Silence);
    }

    #[test]
    fn speech_onset_after_min_speech_frames() {
        let mut sm = default_state_machine();
        let frame = dummy_frame();
        // min_speech_duration_ms=250 at 32ms/frame = 7 frames
        let mut started = false;
        for i in 0..20 {
            let decision = sm.process(0.9, &frame);
            match decision {
                VadDecision::SpeechStarted { .. } => {
                    started = true;
                    // Should happen at frame 7 (0-indexed: i=6)
                    assert_eq!(i, 6, "Speech should start at frame 7");
                    break;
                }
                VadDecision::Silence => {} // still accumulating
                _ => panic!("Unexpected decision during onset: {:?}", decision),
            }
        }
        assert!(started, "Speech should have started");
        assert_eq!(sm.state(), VadState::Speech);
    }

    #[test]
    fn leaky_decrement_tolerates_brief_dip() {
        let mut sm = default_state_machine();
        let frame = dummy_frame();
        // Feed 5 speech frames, then 1 silence, then more speech
        for _ in 0..5 {
            sm.process(0.9, &frame);
        }
        assert_eq!(sm.state(), VadState::Silence); // not enough yet (need 7)

        // One sub-threshold frame: leaky decrement reduces counter by 1 (5->4)
        sm.process(0.1, &frame);
        assert_eq!(sm.state(), VadState::Silence);

        // Now feed 3 more speech frames (counter: 4+3=7 >= min_speech_frames)
        let mut started = false;
        for _ in 0..3 {
            let decision = sm.process(0.9, &frame);
            if matches!(decision, VadDecision::SpeechStarted { .. }) {
                started = true;
            }
        }
        assert!(started, "Speech should start despite brief dip");
    }

    #[test]
    fn leaky_decrement_does_not_go_negative() {
        let mut sm = default_state_machine();
        let frame = dummy_frame();
        // Feed silence frames — counter should stay at 0
        for _ in 0..10 {
            sm.process(0.1, &frame);
        }
        // Now feed exactly min_speech_frames of speech
        let mut started = false;
        for _ in 0..7 {
            if matches!(sm.process(0.9, &frame), VadDecision::SpeechStarted { .. }) {
                started = true;
            }
        }
        assert!(started);
    }

    #[test]
    fn speech_continues_above_threshold() {
        let mut sm = default_state_machine();
        let frame = dummy_frame();
        // Trigger speech start
        for _ in 0..7 {
            sm.process(0.9, &frame);
        }
        assert_eq!(sm.state(), VadState::Speech);

        // Continued speech
        for _ in 0..10 {
            let decision = sm.process(0.9, &frame);
            assert!(matches!(decision, VadDecision::SpeechContinues));
        }
    }

    #[test]
    fn brief_pause_during_speech_continues() {
        let mut sm = default_state_machine();
        let frame = dummy_frame();
        // Trigger speech
        for _ in 0..7 {
            sm.process(0.9, &frame);
        }
        assert_eq!(sm.state(), VadState::Speech);

        // Brief pause: a few frames below end threshold (need 15 for SpeechEnded)
        for _ in 0..5 {
            let decision = sm.process(0.1, &frame);
            assert!(
                matches!(decision, VadDecision::SpeechContinues),
                "Brief pause should not end speech"
            );
        }
        assert_eq!(sm.state(), VadState::Speech);
    }

    #[test]
    fn speech_ends_after_min_silence_frames() {
        let mut sm = default_state_machine();
        let frame = dummy_frame();
        // Trigger speech
        for _ in 0..7 {
            sm.process(0.9, &frame);
        }
        assert_eq!(sm.state(), VadState::Speech);

        // min_silence_duration_ms=500 at 32ms/frame = 15 frames
        let mut ended = false;
        for i in 0..20 {
            let decision = sm.process(0.1, &frame);
            if matches!(decision, VadDecision::SpeechEnded) {
                ended = true;
                assert_eq!(i, 14, "Speech should end at frame 15 (0-indexed: 14)");
                break;
            }
        }
        assert!(ended, "Speech should have ended");
        assert_eq!(sm.state(), VadState::Silence);
    }

    #[test]
    fn threshold_hysteresis() {
        let mut sm = default_state_machine();
        let frame = dummy_frame();
        // Trigger speech (needs prob >= 0.5)
        for _ in 0..7 {
            sm.process(0.9, &frame);
        }
        assert_eq!(sm.state(), VadState::Speech);

        // Prob=0.4: above end threshold (0.3) but below start threshold (0.5)
        // Should continue as speech due to hysteresis
        for _ in 0..20 {
            let decision = sm.process(0.4, &frame);
            assert!(
                matches!(decision, VadDecision::SpeechContinues),
                "0.4 is above end threshold (0.3), should continue speech"
            );
        }
        assert_eq!(sm.state(), VadState::Speech);
    }

    #[test]
    fn update_params_changes_behavior() {
        let mut sm = default_state_machine();
        let frame = dummy_frame();
        // With default threshold_start=0.5, prob=0.3 is silence
        sm.process(0.3, &frame);
        assert_eq!(sm.state(), VadState::Silence);

        // Lower threshold_start to 0.2
        sm.update_params(0.2, 0.1, 500, 250);

        // Now prob=0.3 should count as speech
        for _ in 0..7 {
            sm.process(0.3, &frame);
        }
        assert_eq!(sm.state(), VadState::Speech);
    }

    #[test]
    fn speech_started_includes_pre_speech_samples() {
        let mut sm = VadStateMachine::new(0.5, 0.3, 500, 96, 100);
        // min_speech_duration_ms=96 at 32ms/frame = 3 frames
        // pre_speech_ms=100 at 16kHz = 1600 samples

        let silence_frame = vec![0.1; VAD_CHUNK_SIZE];
        let speech_frame = vec![0.9; VAD_CHUNK_SIZE];

        // Push some silence frames into ring buffer
        sm.process(0.1, &silence_frame);
        sm.process(0.1, &silence_frame);

        // Now trigger speech
        sm.process(0.9, &speech_frame); // frame 1, buffered
        sm.process(0.9, &speech_frame); // frame 2, buffered
        let decision = sm.process(0.9, &speech_frame); // frame 3 -> SpeechStarted

        match decision {
            VadDecision::SpeechStarted { pre_speech_samples } => {
                // Ring buffer capacity is 1600 samples (100ms at 16kHz).
                // We pushed 2 silence frames (1024 samples) + 2 speech frames
                // that were buffered while accumulating (1024 more) = 2048 total.
                // Since 2048 > capacity 1600, the buffer wraps and we get 1600.
                assert_eq!(pre_speech_samples.len(), 1600);
            }
            _ => panic!("Expected SpeechStarted, got {:?}", decision),
        }
    }

    #[test]
    fn reset_clears_all_state() {
        let mut sm = default_state_machine();
        let frame = dummy_frame();
        // Get into speech state
        for _ in 0..7 {
            sm.process(0.9, &frame);
        }
        assert_eq!(sm.state(), VadState::Speech);

        sm.reset();
        assert_eq!(sm.state(), VadState::Silence);

        // Should need full min_speech_frames again
        let mut count = 0;
        for _ in 0..7 {
            if matches!(sm.process(0.9, &frame), VadDecision::SpeechStarted { .. }) {
                break;
            }
            count += 1;
        }
        assert_eq!(count, 6, "After reset, should need 7 frames again");
    }

    // -----------------------------------------------------------------------
    // VadProcessor integration tests (require ONNX model)
    // -----------------------------------------------------------------------

    #[test]
    #[ignore] // Requires silero_vad.onnx model file
    fn vad_processor_silence_stays_silent() {
        let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/models/silero_vad.onnx");
        let mut vad = VadProcessor::new(Path::new(model_path), 0.5, 0.3, 500, 250, 500)
            .expect("Failed to load VAD model");

        let silence = vec![0.0; VAD_CHUNK_SIZE];
        for _ in 0..30 {
            let (decision, prob) = vad.process_frame(&silence).unwrap();
            assert!(
                matches!(decision, VadDecision::Silence),
                "Silence input should produce Silence decision, got {:?} (prob={:.3})",
                decision,
                prob,
            );
        }
    }

    #[test]
    #[ignore] // Requires silero_vad.onnx model file
    fn vad_processor_tone_triggers_speech() {
        let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/models/silero_vad.onnx");
        let mut vad = VadProcessor::new(Path::new(model_path), 0.5, 0.3, 500, 250, 500)
            .expect("Failed to load VAD model");

        // Generate a 440Hz sine wave at 16kHz sample rate
        let tone: Vec<f32> = (0..VAD_CHUNK_SIZE)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();

        let mut saw_speech = false;
        for _ in 0..50 {
            let (decision, _prob) = vad.process_frame(&tone).unwrap();
            if matches!(
                decision,
                VadDecision::SpeechStarted { .. } | VadDecision::SpeechContinues
            ) {
                saw_speech = true;
                break;
            }
        }
        // Note: a pure tone may or may not trigger the VAD model.
        // This test verifies the pipeline works end-to-end, not that the
        // model classifies tones as speech.
        println!(
            "Tone VAD test: saw_speech={} (model may not classify tones as speech)",
            saw_speech
        );
    }

    #[test]
    #[ignore] // Requires silero_vad.onnx model file
    fn vad_processor_reset_returns_to_silence() {
        let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/models/silero_vad.onnx");
        let mut vad = VadProcessor::new(Path::new(model_path), 0.5, 0.3, 500, 250, 500)
            .expect("Failed to load VAD model");

        // Process some frames
        let frame = vec![0.0; VAD_CHUNK_SIZE];
        for _ in 0..10 {
            let _ = vad.process_frame(&frame);
        }

        vad.reset();
        assert_eq!(vad.state(), VadState::Silence);
    }
}
