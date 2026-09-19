from __future__ import annotations

import struct
import wave
from pathlib import Path
from unittest import mock

import pytest

from openmed.ambient import sources as sources_module
from openmed.ambient.sources import (
    AudioChunk,
    MicrophoneAudioSource,
    WavFileAudioSource,
)
from openmed.core.capabilities import MissingOptionalDependencyError


def _write_silent_wav(
    path: Path, *, seconds: float, sample_rate: int, channels: int
) -> None:
    frame_count = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        silence = struct.pack("<h", 0) * channels
        handle.writeframes(silence * frame_count)


class TestWavFileAudioSource:
    def test_reads_metadata_from_header(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "sample.wav"
        _write_silent_wav(wav_path, seconds=1.0, sample_rate=16_000, channels=1)

        source = WavFileAudioSource(wav_path)

        assert source.sample_rate == 16_000
        assert source.channels == 1

    def test_chunks_cover_the_whole_file_and_mark_the_last_chunk(
        self, tmp_path: Path
    ) -> None:
        wav_path = tmp_path / "sample.wav"
        _write_silent_wav(wav_path, seconds=2.5, sample_rate=16_000, channels=1)

        source = WavFileAudioSource(wav_path, chunk_seconds=1.0)
        chunks = list(source.chunks())

        assert len(chunks) == 3
        assert all(isinstance(chunk, AudioChunk) for chunk in chunks)
        assert chunks[-1].is_last is True
        assert all(not chunk.is_last for chunk in chunks[:-1])
        total_frames = sum(len(chunk.pcm16) // 2 for chunk in chunks)
        assert total_frames == int(2.5 * 16_000)

    def test_chunk_timestamps_are_chunk_start_times(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "sample.wav"
        _write_silent_wav(wav_path, seconds=2.5, sample_rate=16_000, channels=1)

        source = WavFileAudioSource(wav_path, chunk_seconds=1.0)
        timestamps = [chunk.timestamp for chunk in source.chunks()]

        assert timestamps == pytest.approx([0.0, 1.0, 2.0])

    def test_rejects_non_16bit_wav(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "float.wav"
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(1)
            handle.setframerate(16_000)
            handle.writeframes(b"\x00" * 16_000)

        with pytest.raises(ValueError, match="16-bit PCM"):
            WavFileAudioSource(wav_path)

    def test_rejects_non_positive_chunk_seconds(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "sample.wav"
        _write_silent_wav(wav_path, seconds=1.0, sample_rate=16_000, channels=1)

        with pytest.raises(ValueError, match="chunk_seconds"):
            WavFileAudioSource(wav_path, chunk_seconds=0)


class TestMicrophoneAudioSource:
    def test_raises_actionable_error_when_sounddevice_missing(self) -> None:
        with mock.patch.object(
            sources_module,
            "require_backend",
            side_effect=MissingOptionalDependencyError(
                package="sounddevice", feature="Live microphone capture", extra="mic"
            ),
        ):
            with pytest.raises(MissingOptionalDependencyError, match="mic"):
                MicrophoneAudioSource()

    def test_rejects_non_positive_chunk_seconds(self) -> None:
        with mock.patch.object(sources_module, "require_backend", return_value=None):
            with pytest.raises(ValueError, match="chunk_seconds"):
                MicrophoneAudioSource(chunk_seconds=0)
