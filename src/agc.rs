//! Automatic Gain Control (AGC) for normalizing audio levels.
//!
//! Inserted between the resampler and VAD stages to ensure both the VAD
//! and Parakeet ASR see consistent RMS regardless of mic distance, source
//! volume, or talker differences.

/// Small value to avoid division by zero in gain calculation.
const EPSILON: f32 = 1e-10;

/// Time constant for RMS energy tracking EMA (~1 second).
/// Slow enough to track level trends, not transients.
const RMS_TIME_CONSTANT_S: f32 = 1.0;

/// Maximum attenuation in dB (fixed internal constant).
/// Limits how much the AGC can reduce gain for loud signals.
const MAX_ATTENUATION_DB: f32 = 20.0;

/// Automatic Gain Control processor.
///
/// Normalizes audio level so downstream VAD and ASR see consistent RMS.
/// Operates on f32 PCM samples at a given sample rate.
pub struct AgcProcessor {
    // Config (precomputed linear/alpha values)
    enabled: bool,
    target_rms: f32,
    max_gain: f32,
    min_gain: f32,
    rms_alpha: f32,
    attack_alpha: f32,
    release_alpha: f32,

    // State
    rms_sq_ema: f32,
    current_gain: f32,
    sample_rate: f32,
}

impl AgcProcessor {
    pub fn new(
        enabled: bool,
        target_rms_dbfs: f32,
        max_gain_db: f32,
        attack_ms: f32,
        release_ms: f32,
        sample_rate: f32,
    ) -> Self {
        Self {
            enabled,
            target_rms: db_to_linear(target_rms_dbfs),
            max_gain: db_to_linear(max_gain_db),
            min_gain: 1.0 / db_to_linear(MAX_ATTENUATION_DB),
            rms_alpha: time_constant_to_alpha(RMS_TIME_CONSTANT_S, sample_rate),
            attack_alpha: time_constant_to_alpha(attack_ms / 1000.0, sample_rate),
            release_alpha: time_constant_to_alpha(release_ms / 1000.0, sample_rate),
            rms_sq_ema: 0.0,
            current_gain: 1.0,
            sample_rate,
        }
    }

    /// Process audio samples in-place.
    ///
    /// RMS tracking runs continuously even when disabled, so re-enabling
    /// resumes from the current level estimate without re-learning.
    /// When disabled, `current_gain` ramps smoothly toward 1.0 for
    /// click-free transitions.
    pub fn process(&mut self, samples: &mut [f32]) {
        for sample in samples.iter_mut() {
            let x = *sample;

            // Always track RMS (even when disabled)
            self.rms_sq_ema = self.rms_alpha * x * x + (1.0 - self.rms_alpha) * self.rms_sq_ema;

            if self.enabled {
                let rms_estimate = self.rms_sq_ema.sqrt();

                // Compute and clamp target gain
                let target_gain = (self.target_rms / rms_estimate.max(EPSILON))
                    .clamp(self.min_gain, self.max_gain);
                // TODO: VAD gating — only update target_gain during speech

                // Asymmetric smoothing: fast attack (gain down), slow release (gain up)
                let alpha = if target_gain < self.current_gain {
                    self.attack_alpha
                } else {
                    self.release_alpha
                };
                self.current_gain += alpha * (target_gain - self.current_gain);
            } else {
                // Ramp toward unity for click-free disable transition
                if (self.current_gain - 1.0).abs() < 1e-6 {
                    self.current_gain = 1.0; // Snap for exact passthrough
                } else {
                    let alpha = if 1.0 < self.current_gain {
                        self.attack_alpha
                    } else {
                        self.release_alpha
                    };
                    self.current_gain += alpha * (1.0 - self.current_gain);
                }
            }

            *sample = x * self.current_gain;
        }
    }

    /// Current applied gain in dB (useful for a future UI meter).
    pub fn current_gain_db(&self) -> f32 {
        linear_to_db(self.current_gain)
    }

    /// Update tunable parameters without resetting internal state.
    /// Recomputes alpha values from the current sample rate.
    pub fn update_params(
        &mut self,
        enabled: bool,
        target_rms_dbfs: f32,
        max_gain_db: f32,
        attack_ms: f32,
        release_ms: f32,
    ) {
        self.enabled = enabled;
        self.target_rms = db_to_linear(target_rms_dbfs);
        self.max_gain = db_to_linear(max_gain_db);
        self.attack_alpha = time_constant_to_alpha(attack_ms / 1000.0, self.sample_rate);
        self.release_alpha = time_constant_to_alpha(release_ms / 1000.0, self.sample_rate);
    }

    /// Reset state for session boundaries. Configuration is preserved.
    pub fn reset(&mut self) {
        self.rms_sq_ema = 0.0;
        self.current_gain = 1.0;
    }
}

/// Convert dBFS (amplitude) to linear scale.
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Convert linear amplitude to dB.
fn linear_to_db(linear: f32) -> f32 {
    20.0 * linear.max(EPSILON).log10()
}

/// Convert a time constant (seconds) to per-sample EMA alpha.
///
/// `α = 1 - exp(-1 / (τ × sample_rate))`
fn time_constant_to_alpha(time_constant_s: f32, sample_rate: f32) -> f32 {
    if time_constant_s <= 0.0 {
        return 1.0;
    }
    1.0 - (-1.0 / (time_constant_s * sample_rate)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const SR_16K: f32 = 16000.0;
    const SR_48K: f32 = 48000.0;

    fn make_sine(freq: f32, amplitude: f32, duration_s: f32, sample_rate: f32) -> Vec<f32> {
        let n = (duration_s * sample_rate) as usize;
        (0..n)
            .map(|i| amplitude * (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        let sum_sq: f32 = samples.iter().map(|x| x * x).sum();
        (sum_sq / samples.len() as f32).sqrt()
    }

    fn default_agc(enabled: bool) -> AgcProcessor {
        AgcProcessor::new(enabled, -20.0, 30.0, 20.0, 400.0, SR_16K)
    }

    // --- Passthrough when disabled ---

    #[test]
    fn passthrough_when_disabled() {
        let mut agc = default_agc(false);
        let original = make_sine(440.0, 0.5, 0.5, SR_16K);
        let mut processed = original.clone();
        agc.process(&mut processed);

        for (i, (orig, proc_)) in original.iter().zip(processed.iter()).enumerate() {
            assert_eq!(*orig, *proc_, "Sample {} differs: {} vs {}", i, orig, proc_);
        }
    }

    // --- No discontinuity on disable/re-enable ---

    #[test]
    fn disable_reenable_no_discontinuity() {
        let mut agc = default_agc(true);

        // Process a loud signal to move gain away from 1.0
        let mut loud = make_sine(440.0, 0.8, 2.0, SR_16K);
        agc.process(&mut loud);

        let gain_before_disable = agc.current_gain;
        assert!(
            (gain_before_disable - 1.0).abs() > 0.01,
            "Gain should have moved from 1.0, got {}",
            gain_before_disable
        );

        // Disable and process — gain should ramp toward 1.0
        agc.update_params(false, -20.0, 30.0, 20.0, 400.0);
        let mut transition = make_sine(440.0, 0.8, 0.1, SR_16K);
        agc.process(&mut transition);

        // Check no large jump between consecutive samples
        let max_delta: f32 = transition
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0, f32::max);

        // A 440Hz sine at 16kHz has max sample-to-sample delta of
        // ~amplitude * 2π * 440/16000. With gain ramping, allow generous margin.
        assert!(
            max_delta < 0.4,
            "Discontinuity too large: max_delta={}",
            max_delta
        );

        // Re-enable and continue processing
        agc.update_params(true, -20.0, 30.0, 20.0, 400.0);
        let mut after = make_sine(440.0, 0.8, 0.1, SR_16K);
        agc.process(&mut after);

        let max_delta_after: f32 = after
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0, f32::max);
        assert!(
            max_delta_after < 0.4,
            "Discontinuity after re-enable: max_delta={}",
            max_delta_after
        );
    }

    // --- Convergence ---

    #[test]
    fn convergence_constant_amplitude() {
        let mut agc = default_agc(true);
        let target_rms = db_to_linear(-20.0);

        // Input: sine at amplitude 0.5 → RMS ≈ 0.354
        let amplitude = 0.5;
        let input_rms = amplitude / 2.0_f32.sqrt();
        let expected_gain = target_rms / input_rms;

        let mut signal = make_sine(440.0, amplitude, 5.0, SR_16K);
        agc.process(&mut signal);

        let error_pct = ((agc.current_gain - expected_gain) / expected_gain).abs() * 100.0;
        assert!(
            error_pct < 5.0,
            "Gain {:.4} should be within 5% of expected {:.4} (error: {:.1}%)",
            agc.current_gain,
            expected_gain,
            error_pct
        );
    }

    // --- Max-gain clamp ---

    #[test]
    fn max_gain_clamp_near_silence() {
        let max_gain_db = 30.0;
        let max_gain_linear = db_to_linear(max_gain_db);
        let mut agc = default_agc(true);

        // Near-silent input (RMS < -60 dBFS)
        let mut silence: Vec<f32> = vec![1e-5; (SR_16K * 3.0) as usize];
        agc.process(&mut silence);

        assert!(
            agc.current_gain <= max_gain_linear * 1.01, // 1% tolerance
            "Gain {:.1} dB exceeds max_gain {} dB",
            agc.current_gain_db(),
            max_gain_db
        );
    }

    // --- Attack / release asymmetry ---

    #[test]
    fn attack_faster_than_release() {
        let attack_ms = 20.0;
        let release_ms = 400.0;

        // Test attack: start quiet, step to loud
        let mut agc1 = AgcProcessor::new(true, -20.0, 30.0, attack_ms, release_ms, SR_16K);
        let mut quiet1 = make_sine(440.0, 0.05, 5.0, SR_16K);
        agc1.process(&mut quiet1);
        let gain_before_attack = agc1.current_gain;

        let mut loud1 = make_sine(440.0, 0.5, 1.0, SR_16K);
        agc1.process(&mut loud1);
        let gain_after_attack = agc1.current_gain;

        // Test release: start loud, step to quiet
        let mut agc2 = AgcProcessor::new(true, -20.0, 30.0, attack_ms, release_ms, SR_16K);
        let mut loud2 = make_sine(440.0, 0.5, 5.0, SR_16K);
        agc2.process(&mut loud2);
        let gain_before_release = agc2.current_gain;

        let mut quiet2 = make_sine(440.0, 0.05, 1.0, SR_16K);
        agc2.process(&mut quiet2);
        let gain_after_release = agc2.current_gain;

        // Compute relative movement in each direction
        let attack_movement = ((gain_after_attack - gain_before_attack) / gain_before_attack).abs();
        let release_movement =
            ((gain_after_release - gain_before_release) / gain_before_release).abs();

        // Attack (gain reduction) should produce more movement than release
        // (gain increase) in the same time window
        assert!(
            attack_movement > release_movement,
            "Attack movement ({:.3}) should exceed release movement ({:.3})",
            attack_movement,
            release_movement
        );
    }

    // --- Rate independence ---

    #[test]
    fn rate_independence() {
        let target_db = -20.0;
        let max_gain_db = 30.0;
        let attack_ms = 20.0;
        let release_ms = 400.0;
        let amplitude = 0.3;
        let duration_s = 3.0;

        let mut agc_16k =
            AgcProcessor::new(true, target_db, max_gain_db, attack_ms, release_ms, SR_16K);
        let mut agc_48k =
            AgcProcessor::new(true, target_db, max_gain_db, attack_ms, release_ms, SR_48K);

        let mut signal_16k = make_sine(440.0, amplitude, duration_s, SR_16K);
        let mut signal_48k = make_sine(440.0, amplitude, duration_s, SR_48K);

        agc_16k.process(&mut signal_16k);
        agc_48k.process(&mut signal_48k);

        // Both should converge to the same gain (same physical time constants)
        let error_pct =
            ((agc_16k.current_gain - agc_48k.current_gain) / agc_16k.current_gain).abs() * 100.0;
        assert!(
            error_pct < 2.0,
            "Rate-dependent gain divergence: 16k={:.4}, 48k={:.4} ({:.1}% error)",
            agc_16k.current_gain,
            agc_48k.current_gain,
            error_pct
        );
    }

    // --- Integration: level variation ---

    #[test]
    fn output_rms_tracks_target() {
        let target_db = -20.0;
        let target_rms = db_to_linear(target_db);
        let mut agc = default_agc(true);

        // Quiet segment at -30 dBFS, then loud segment at -10 dBFS.
        // After settling, output RMS of last 1 second should be near target.
        let quiet_amp = db_to_linear(-30.0) * 2.0_f32.sqrt();
        let loud_amp = db_to_linear(-10.0) * 2.0_f32.sqrt();

        let mut quiet_segment = make_sine(440.0, quiet_amp, 5.0, SR_16K);
        agc.process(&mut quiet_segment);

        let last_1s = &quiet_segment[quiet_segment.len() - SR_16K as usize..];
        let output_rms_quiet = rms(last_1s);
        let error_db_quiet = (20.0 * (output_rms_quiet / target_rms).log10()).abs();
        assert!(
            error_db_quiet < 2.0,
            "Quiet segment: output RMS {:.4} ({:.1} dBFS), target {:.4} ({} dBFS), error {:.1} dB",
            output_rms_quiet,
            20.0 * output_rms_quiet.log10(),
            target_rms,
            target_db,
            error_db_quiet
        );

        let mut loud_segment = make_sine(440.0, loud_amp, 5.0, SR_16K);
        agc.process(&mut loud_segment);

        let last_1s = &loud_segment[loud_segment.len() - SR_16K as usize..];
        let output_rms_loud = rms(last_1s);
        let error_db_loud = (20.0 * (output_rms_loud / target_rms).log10()).abs();
        assert!(
            error_db_loud < 2.0,
            "Loud segment: output RMS {:.4} ({:.1} dBFS), target {:.4} ({} dBFS), error {:.1} dB",
            output_rms_loud,
            20.0 * output_rms_loud.log10(),
            target_rms,
            target_db,
            error_db_loud
        );
    }

    // --- Helper function tests ---

    #[test]
    fn db_linear_roundtrip() {
        for db in [-40.0_f32, -20.0, -6.0, 0.0, 6.0, 20.0] {
            let linear = db_to_linear(db);
            let back = linear_to_db(linear);
            assert!(
                (back - db).abs() < 0.01,
                "Roundtrip failed: {} dB -> {} linear -> {} dB",
                db,
                linear,
                back
            );
        }
    }

    #[test]
    fn time_constant_to_alpha_zero_is_instant() {
        assert_eq!(time_constant_to_alpha(0.0, SR_16K), 1.0);
    }

    #[test]
    fn time_constant_to_alpha_large_is_slow() {
        let alpha = time_constant_to_alpha(10.0, SR_16K);
        assert!(alpha < 0.001, "Large time constant should give small alpha");
    }

    #[test]
    fn current_gain_db_at_unity() {
        let agc = default_agc(false);
        assert!((agc.current_gain_db() - 0.0).abs() < 0.01);
    }
}
