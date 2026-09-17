from __future__ import annotations

import wave
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from openmed.ambient import voice_privacy as voice_privacy_module
from openmed.ambient.voice_privacy import McAdamsVoiceAnonymizer, anonymize_wav_file
from openmed.core.capabilities import MissingOptionalDependencyError

_SAMPLE_RATE = 16_000


def _sine_pcm16(
    *, seconds: float, freq_hz: float, sample_rate: int, channels: int = 1
) -> bytes:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = (0.3 * np.sin(2 * np.pi * freq_hz * t) * 32767).astype(np.int16)
    if channels > 1:
        tone = np.repeat(tone, channels)
    return tone.tobytes()


def _write_wav(path: Path, pcm16: bytes, *, sample_rate: int, channels: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16)


class TestMcAdamsVoiceAnonymizerValidation:
    def test_raises_actionable_error_when_numpy_missing(self) -> None:
        with mock.patch.object(
            voice_privacy_module,
            "require_backend",
            side_effect=MissingOptionalDependencyError(
                package="numpy",
                feature="Voiceprint anonymization",
                extra="voice-privacy",
            ),
        ):
            with pytest.raises(MissingOptionalDependencyError, match="voice-privacy"):
                McAdamsVoiceAnonymizer()

    @pytest.mark.parametrize("coefficient", [0.0, -0.5, 1.6, 2.0])
    def test_rejects_invalid_mcadams_coefficient(self, coefficient: float) -> None:
        with pytest.raises(ValueError, match="mcadams_coefficient"):
            McAdamsVoiceAnonymizer(mcadams_coefficient=coefficient)

    def test_rejects_hop_larger_than_window(self) -> None:
        with pytest.raises(ValueError, match="hop_ms"):
            McAdamsVoiceAnonymizer(window_ms=10.0, hop_ms=20.0)

    @pytest.mark.parametrize("order", [1, 5, 50, 100])
    def test_rejects_out_of_range_lpc_order(self, order: int) -> None:
        with pytest.raises(ValueError, match="lpc_order"):
            McAdamsVoiceAnonymizer(lpc_order=order)


class TestMcAdamsVoiceAnonymizerSignal:
    def test_output_length_matches_input_for_mono(self) -> None:
        anonymizer = McAdamsVoiceAnonymizer()
        pcm16 = _sine_pcm16(seconds=1.0, freq_hz=180.0, sample_rate=_SAMPLE_RATE)

        anonymized = anonymizer.anonymize_pcm16(
            pcm16, sample_rate=_SAMPLE_RATE, channels=1
        )

        assert len(anonymized) == len(pcm16)

    def test_output_length_matches_input_for_stereo(self) -> None:
        anonymizer = McAdamsVoiceAnonymizer()
        pcm16 = _sine_pcm16(
            seconds=0.5, freq_hz=150.0, sample_rate=_SAMPLE_RATE, channels=2
        )

        anonymized = anonymizer.anonymize_pcm16(
            pcm16, sample_rate=_SAMPLE_RATE, channels=2
        )

        assert len(anonymized) == len(pcm16)

    def test_does_not_crash_on_silence(self) -> None:
        anonymizer = McAdamsVoiceAnonymizer()
        silence = np.zeros(_SAMPLE_RATE, dtype=np.int16).tobytes()

        anonymized = anonymizer.anonymize_pcm16(
            silence, sample_rate=_SAMPLE_RATE, channels=1
        )

        assert len(anonymized) == len(silence)

    def test_warping_changes_the_signal(self) -> None:
        pcm16 = _sine_pcm16(seconds=1.0, freq_hz=180.0, sample_rate=_SAMPLE_RATE)

        unwarped = McAdamsVoiceAnonymizer(mcadams_coefficient=1.0).anonymize_pcm16(
            pcm16, sample_rate=_SAMPLE_RATE, channels=1
        )
        warped = McAdamsVoiceAnonymizer(mcadams_coefficient=0.7).anonymize_pcm16(
            pcm16, sample_rate=_SAMPLE_RATE, channels=1
        )

        unwarped_arr = np.frombuffer(unwarped, dtype=np.int16).astype(np.float64)
        warped_arr = np.frombuffer(warped, dtype=np.int16).astype(np.float64)
        assert not np.allclose(unwarped_arr, warped_arr, atol=50)

    def test_coefficient_one_preserves_most_signal_energy(self) -> None:
        pcm16 = _sine_pcm16(seconds=1.0, freq_hz=180.0, sample_rate=_SAMPLE_RATE)
        original = np.frombuffer(pcm16, dtype=np.int16).astype(np.float64)

        reconstructed = McAdamsVoiceAnonymizer(mcadams_coefficient=1.0).anonymize_pcm16(
            pcm16, sample_rate=_SAMPLE_RATE, channels=1
        )
        reconstructed_arr = np.frombuffer(reconstructed, dtype=np.int16).astype(
            np.float64
        )

        correlation = np.corrcoef(original, reconstructed_arr)[0, 1]
        assert correlation > 0.9


class TestAnonymizeWavFile:
    def test_round_trips_wav_metadata_and_frame_count(self, tmp_path: Path) -> None:
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        pcm16 = _sine_pcm16(seconds=0.75, freq_hz=200.0, sample_rate=_SAMPLE_RATE)
        _write_wav(input_path, pcm16, sample_rate=_SAMPLE_RATE, channels=1)

        anonymize_wav_file(input_path, output_path, mcadams_coefficient=0.75)

        with wave.open(str(output_path), "rb") as handle:
            assert handle.getframerate() == _SAMPLE_RATE
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getnframes() == int(0.75 * _SAMPLE_RATE)

    def test_rejects_non_16bit_input(self, tmp_path: Path) -> None:
        input_path = tmp_path / "float.wav"
        with wave.open(str(input_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(1)
            handle.setframerate(_SAMPLE_RATE)
            handle.writeframes(b"\x00" * _SAMPLE_RATE)

        with pytest.raises(ValueError, match="16-bit PCM"):
            anonymize_wav_file(input_path, tmp_path / "out.wav")
