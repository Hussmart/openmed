from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, List
from unittest import mock

from openmed.ambient import pipeline as pipeline_module
from openmed.ambient.pipeline import AmbientRedactionPipeline
from openmed.ambient.sources import AudioChunk
from openmed.ambient.transcriber import TranscriptSegment
from openmed.core.pii import DeidentificationResult, PIIEntity


@dataclass
class _FakeAudioSource:
    chunk_texts: List[str]
    sample_rate: int = 16_000
    channels: int = 1

    def chunks(self) -> Iterator[AudioChunk]:
        for index, _ in enumerate(self.chunk_texts):
            yield AudioChunk(
                pcm16=b"\x00\x00",
                sample_rate=self.sample_rate,
                channels=self.channels,
                timestamp=float(index),
                is_last=index == len(self.chunk_texts) - 1,
            )


class _FakeTranscriber:
    def __init__(self, texts: List[str]):
        self._texts = texts
        self._index = 0

    def transcribe_chunk(self, chunk: AudioChunk) -> List[TranscriptSegment]:
        text = self._texts[self._index]
        self._index += 1
        return [
            TranscriptSegment(
                text=text, start=chunk.timestamp, end=chunk.timestamp + 1.0
            )
        ]


def _fake_deid_result(text: str, redacted: str) -> DeidentificationResult:
    entity = PIIEntity(
        text="John Doe", label="PERSON", confidence=0.95, start=8, end=16
    )
    return DeidentificationResult(
        original_text=text,
        deidentified_text=redacted,
        pii_entities=[entity],
        method="mask",
        timestamp=datetime.now(timezone.utc),
    )


class TestAmbientRedactionPipeline:
    def test_stream_deidentifies_every_transcribed_segment(self) -> None:
        texts = ["Patient John Doe has a fever.", "No PHI in this one."]
        fake_transcriber = _FakeTranscriber(texts)
        source = _FakeAudioSource(chunk_texts=texts)
        pipeline = AmbientRedactionPipeline(transcriber=fake_transcriber)

        with mock.patch.object(
            pipeline_module,
            "deidentify",
            side_effect=lambda text, **_: _fake_deid_result(
                text, text.replace("John Doe", "[PERSON]")
            ),
        ) as mock_deidentify:
            results = list(pipeline.stream(source))

        assert len(results) == 2
        assert mock_deidentify.call_count == 2
        assert results[0].redacted_text == "Patient [PERSON] has a fever."
        assert results[0].to_dict()["pii_categories"] == ["PERSON"]
        assert results[1].redacted_text == "No PHI in this one."

    def test_stream_passes_configured_deid_options(self) -> None:
        texts = ["hello"]
        fake_transcriber = _FakeTranscriber(texts)
        source = _FakeAudioSource(chunk_texts=texts)
        pipeline = AmbientRedactionPipeline(
            transcriber=fake_transcriber,
            deid_method="hash",
            deid_model_name="OpenMed/custom-pii-model",
            deid_confidence_threshold=0.9,
            lang="es",
        )

        with mock.patch.object(
            pipeline_module,
            "deidentify",
            return_value=_fake_deid_result("hello", "hello"),
        ) as mock_deidentify:
            list(pipeline.stream(source))

        _, kwargs = mock_deidentify.call_args
        assert kwargs["method"] == "hash"
        assert kwargs["model_name"] == "OpenMed/custom-pii-model"
        assert kwargs["confidence_threshold"] == 0.9
        assert kwargs["lang"] == "es"

    def test_stream_omits_model_name_when_not_configured(self) -> None:
        texts = ["hello"]
        pipeline = AmbientRedactionPipeline(transcriber=_FakeTranscriber(texts))
        source = _FakeAudioSource(chunk_texts=texts)

        with mock.patch.object(
            pipeline_module,
            "deidentify",
            return_value=_fake_deid_result("hello", "hello"),
        ) as mock_deidentify:
            list(pipeline.stream(source))

        _, kwargs = mock_deidentify.call_args
        assert "model_name" not in kwargs
