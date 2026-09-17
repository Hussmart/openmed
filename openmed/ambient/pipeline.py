"""Chunk-by-chunk orchestration: audio -> transcript -> de-identified text.

:class:`AmbientRedactionPipeline` is deliberately thin. It does not duplicate
any PII-detection heuristics: every redaction decision is delegated to
:func:`openmed.core.pii.deidentify`, the same engine every other de-identification
surface in OpenMed (CLI, REST service, batch jobs) already uses. This module's
only job is wiring a live or replayed audio stream through a transcriber and
into that existing engine, one segment at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

from openmed.core.config import OpenMedConfig
from openmed.core.pii import DeidentificationMethod, DeidentificationResult, deidentify

from .sources import AudioSource
from .transcriber import FasterWhisperTranscriber, TranscriptSegment

__all__ = ["AmbientRedactionPipeline", "RedactedSegment"]


@dataclass(frozen=True)
class RedactedSegment:
    """One transcribed speech segment paired with its de-identification result."""

    segment: TranscriptSegment
    deid_result: DeidentificationResult

    @property
    def redacted_text(self) -> str:
        """The de-identified transcript text for this segment."""

        return self.deid_result.deidentified_text

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible summary (never the raw transcript)."""

        return {
            "start": self.segment.start,
            "end": self.segment.end,
            "redacted_text": self.redacted_text,
            "pii_entity_count": len(self.deid_result.pii_entities),
            "pii_categories": sorted(
                {
                    entity.entity_type or entity.label
                    for entity in self.deid_result.pii_entities
                }
            ),
        }


class AmbientRedactionPipeline:
    """Stream audio through transcription and de-identification, chunk by chunk.

    Example:
        >>> from openmed.ambient import AmbientRedactionPipeline, WavFileAudioSource
        >>> pipeline = AmbientRedactionPipeline()  # doctest: +SKIP
        >>> source = WavFileAudioSource("visit.wav")  # doctest: +SKIP
        >>> for redacted in pipeline.stream(source):  # doctest: +SKIP
        ...     print(redacted.redacted_text)
    """

    def __init__(
        self,
        *,
        model_size: str = "small.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
        deid_method: DeidentificationMethod = "mask",
        deid_model_name: Optional[str] = None,
        deid_confidence_threshold: float = 0.7,
        lang: str = "en",
        config: Optional[OpenMedConfig] = None,
        transcriber: Optional[FasterWhisperTranscriber] = None,
    ) -> None:
        self._transcriber = transcriber or FasterWhisperTranscriber(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
        )
        self._deid_method = deid_method
        self._deid_model_name = deid_model_name
        self._deid_confidence_threshold = deid_confidence_threshold
        self._lang = lang
        self._config = config

    def stream(self, audio_source: AudioSource) -> Iterator[RedactedSegment]:
        """Yield one :class:`RedactedSegment` per decoded speech segment.

        Each segment is de-identified independently as it arrives, so a
        consumer sees redacted text within one chunk's transcription latency
        of it being spoken -- it never has to wait for the whole stream to end.
        """

        for chunk in audio_source.chunks():
            for segment in self._transcriber.transcribe_chunk(chunk):
                deid_kwargs: Dict[str, Any] = {
                    "method": self._deid_method,
                    "confidence_threshold": self._deid_confidence_threshold,
                    "lang": self._lang,
                }
                if self._deid_model_name is not None:
                    deid_kwargs["model_name"] = self._deid_model_name
                if self._config is not None:
                    deid_kwargs["config"] = self._config
                result = deidentify(segment.text, **deid_kwargs)
                yield RedactedSegment(segment=segment, deid_result=result)
