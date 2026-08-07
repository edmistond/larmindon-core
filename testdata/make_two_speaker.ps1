# Generates the two-speaker fixtures for validating diarization end to end.
#
# Uses SAPI with SSML voice switching, so both speakers land in ONE continuous
# stream rather than concatenated files — turn boundaries then look like a real
# conversation to the diarizer instead of like hard cuts.
#
#   powershell -ExecutionPolicy Bypass -File testdata/make_two_speaker.ps1
#   powershell -ExecutionPolicy Bypass -File testdata/make_two_speaker.ps1 -Short
#
# Output is 16 kHz mono 16-bit PCM, which is what replay_wav's reader accepts
# and what the pipeline runs at natively (so no resampling in the way).
#
# THIS SCRIPT IS WINDOWS-ONLY (SAPI), BUT ITS OUTPUT IS COMMITTED. That is
# deliberate: the fixtures have to be usable on macOS and Linux, where there is
# no SAPI, so the generator being platform-specific must not make the testing
# platform-specific. Regenerate only on Windows, and only when changing the
# script.
#
# CAVEAT: synthetic voices are a weak test of diarization ACCURACY — the models
# are trained on human speech. These fixtures validate that a speaker label flows
# wire -> accumulator -> TranscriptUpdate -> UI. If speakers come back merged,
# suspect the fixture before suspecting the code, and confirm with real
# two-person audio.

param(
    # Four turns instead of ten, for fast iteration on the first live runs.
    # At 1x realtime a full run costs its own duration, so failing fast matters.
    [switch]$Short,
    [string]$Out,
    [string]$VoiceA = "Microsoft David Desktop",
    [string]$VoiceB = "Microsoft Zira Desktop"
)

if (-not $Out) {
    $Out = if ($Short) { "$PSScriptRoot/two_speaker_short.wav" }
           else { "$PSScriptRoot/two_speaker.wav" }
}

Add-Type -AssemblyName System.Speech

# Alternating turns. Deliberately includes: sentence terminators (the durable
# lane flushes on these), a decimal and an abbreviation (which must NOT split —
# see ends_sentence), a question mark, and turns long enough that a turn spans
# several responses.
$turns = @(
    @($VoiceA, "Good morning. Thanks for making time today, I know your schedule has been packed this week."),
    @($VoiceB, "Of course, happy to help. I had a look at the numbers you sent over last night."),
    @($VoiceA, "And what did you think? I was worried the third quarter looked weaker than it should."),
    @($VoiceB, "It is weaker, but not alarmingly so. Revenue came in at 3.14 million, which is about four percent under plan."),
    @($VoiceA, "Four percent. That is recoverable if the pipeline holds up through December."),
    @($VoiceB, "That was my read as well. The bigger question is whether the renewal rate stabilises, e.g. whether the enterprise accounts stay put."),
    @($VoiceA, "Right. Those twelve accounts are most of the exposure. Have we heard anything from the two that went quiet?"),
    @($VoiceB, "One came back yesterday and wants to renew early. The other is still deciding, and I would not push them."),
    @($VoiceA, "Agreed, pushing would be a mistake there. Let us give them another two weeks before we follow up."),
    @($VoiceB, "Sounds good. I will put together a short summary and send it round before the board call on Thursday.")
)

if ($Short) {
    # Keep the first four turns: two per speaker, one speaker change, and the
    # decimal that must not split a sentence.
    $turns = $turns[0..3]
}

$ssml = New-Object System.Text.StringBuilder
[void]$ssml.Append('<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">')
foreach ($turn in $turns) {
    $voice = $turn[0]
    # Escape for XML: the text is ours, but this keeps the script safe to edit.
    $text = [System.Security.SecurityElement]::Escape($turn[1])
    [void]$ssml.Append("<voice name=""$voice"">$text</voice>")
    # A natural inter-turn gap. Long enough to be a turn boundary, short enough
    # that the VAD does not close the session's speech segment for good.
    [void]$ssml.Append('<break time="450ms"/>')
}
[void]$ssml.Append('</speak>')

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

$installed = $synth.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }
foreach ($v in @($VoiceA, $VoiceB)) {
    if ($installed -notcontains $v) {
        Write-Error "Voice '$v' is not installed. Available: $($installed -join ', ')"
        exit 1
    }
}

$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono
)

$synth.SetOutputToWaveFile($Out, $format)
$synth.SpeakSsml($ssml.ToString())
$synth.SetOutputToNull()
$synth.Dispose()

$bytes = (Get-Item $Out).Length
# 44-byte header, 2 bytes per sample, 16000 samples per second.
$seconds = [math]::Round(($bytes - 44) / 2 / 16000, 1)
Write-Host "Wrote $Out - $bytes bytes, ~$seconds s, $($turns.Count) turns across 2 voices"
