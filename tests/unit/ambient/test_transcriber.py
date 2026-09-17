from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List
from unittest import mock

import pytest

from openmed.ambient import transcriber as transcriber_module
from openmed.ambient.sources import AudioChunk
from openmed.ambient.transcriber import WHISPER_SAMPLE_RATE, FasterWhisperTranscriber
from openmed.core.capabilities import MissingOptionalDependencyError


@dataclass
class _FakeSegment:
    text: str
    start: float
    end: float
    no_speech_prob: float = 0.0


class _FakeWhisperModel:
    def __init__(self, segments: List[_FakeSegment]):
        self._segments = segments
        self.calls: list = []

    def transcribe(self, samples, language=None, vad_filter=True):
        self.calls.append(
            {"language": language, "vad_filter": vad_filter, "samples": samples}
        )
        return iter(self._segments), object()


def _silent_pcm16(*, seconds: float, sample_rate: int, channels: int) -> bytes:
    frame_count = int(seconds * sample_rate)
    return (struct.pack("<h", 0) * channels) * frame_count


def _build_transcriber() -> FasterWhisperTranscriber:
    with mock.patch.object(transcriber_module, "require_backend", return_value=None):
        return FasterWhisperTranscriber()


class TestFasterWhisperTranscriber:
    def test_raises_actionable_error_when_faster_whisper_missing(self) -> None:
        with mock.patch.object(
            transcriber_module,
            "require_backend",
            side_effect=MissingOptionalDependencyError(
                package="faster-whisper",
                feature="Streaming speech-to-text transcription",
                extra="speech",
            ),
        ):
            with pytest.raises(MissingOptionalDependencyError, match="speech"):
                FasterWhisperTranscriber()

    def test_rejects_non_16khz_chunks(self) -> None:
        transcriber = _build_transcriber()
        chunk = AudioChunk(
            pcm16=_silent_pcm16(seconds=1.0, sample_rate=8_000, channels=1),
            sample_rate=8_000,
            channels=1,
            timestamp=0.0,
        )

        with pytest.raises(ValueError, match="16kHz"):
            transcriber.transcribe_chunk(chunk)

    def test_transcribe_chunk_offsets_segments_by_chunk_timestamp(self) -> None:
        transcriber = _build_transcriber()
        fake_model = _FakeWhisperModel(
            [
                _FakeSegment(text=" Patient reports fever. ", start=0.0, end=1.2),
                _FakeSegment(text="  ", start=1.2, end=1.4),  # whitespace-only, dropped
            ]
        )
        transcriber._model = fake_model

        chunk = AudioChunk(
            pcm16=_silent_pcm16(
                seconds=1.0, sample_rate=WHISPER_SAMPLE_RATE, channels=1
            ),
            sample_rate=WHISPER_SAMPLE_RATE,
            channels=1,
            timestamp=10.0,
        )

        segments = transcriber.transcribe_chunk(chunk)

        assert len(segments) == 1
        assert segments[0].text == "Patient reports fever."
        assert segments[0].start == pytest.approx(10.0)
        assert segments[0].end == pytest.approx(11.2)

    def test_transcribe_chunk_downmixes_stereo_to_mono(self) -> None:
        transcriber = _build_transcriber()
        fake_model = _FakeWhisperModel([_FakeSegment(text="hello", start=0.0, end=0.5)])
        transcriber._model = fake_model

        chunk = AudioChunk(
            pcm16=_silent_pcm16(
                seconds=0.5, sample_rate=WHISPER_SAMPLE_RATE, channels=2
            ),
            sample_rate=WHISPER_SAMPLE_RATE,
            channels=2,
            timestamp=0.0,
        )

        transcriber.transcribe_chunk(chunk)

        samples = fake_model.calls[0]["samples"]
        assert samples.ndim == 1
        assert len(samples) == int(0.5 * WHISPER_SAMPLE_RATE)
