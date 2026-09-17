"""Streaming speech-to-text transcription for the ambient redaction pipeline.

Wraps ``faster-whisper`` (CTranslate2 Whisper) behind the same lazy-import,
``require_backend``-guarded pattern used by the rest of OpenMed's optional
backends (see :mod:`openmed.core.capabilities`), so importing this module never
requires ``faster-whisper`` to be installed -- only *using* the transcriber
does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from openmed.core.capabilities import require_backend

from .sources import AudioChunk

__all__ = ["FasterWhisperTranscriber", "TranscriptSegment", "WHISPER_SAMPLE_RATE"]

#: faster-whisper's feature extractor is trained on 16kHz mono audio; both
#: bundled :mod:`openmed.ambient.sources` sources default to this rate.
WHISPER_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class TranscriptSegment:
    """One decoded speech segment, timestamped against the audio stream."""

    text: str
    start: float
    end: float
    no_speech_prob: float = 0.0


class FasterWhisperTranscriber:
    """Chunk-at-a-time wrapper around a ``faster_whisper.WhisperModel``.

    The underlying model is loaded lazily on first use (not at construction),
    so building a :class:`FasterWhisperTranscriber` is cheap and safe to do
    even before the model checkpoint has been downloaded.

    Example:
        >>> transcriber = FasterWhisperTranscriber(model_size="small.en")  # doctest: +SKIP
        >>> transcriber.transcribe_chunk(chunk)  # doctest: +SKIP
        [TranscriptSegment(text='...', start=0.0, end=2.4, no_speech_prob=0.01)]
    """

    def __init__(
        self,
        *,
        model_size: str = "small.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
        vad_filter: bool = True,
        download_root: Optional[str] = None,
    ) -> None:
        require_backend("speech", feature="Streaming speech-to-text transcription")
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._vad_filter = vad_filter
        self._download_root = download_root
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                download_root=self._download_root,
            )
        return self._model

    def transcribe_chunk(self, chunk: AudioChunk) -> List[TranscriptSegment]:
        """Transcribe one :class:`AudioChunk`, returning its speech segments.

        Raises:
            ValueError: If ``chunk.sample_rate`` is not 16kHz. Both bundled
                audio sources already emit 16kHz audio; a custom source must
                resample before handing chunks to this transcriber.
        """

        if chunk.sample_rate != WHISPER_SAMPLE_RATE:
            raise ValueError(
                "FasterWhisperTranscriber requires 16kHz audio, got "
                f"{chunk.sample_rate}Hz -- resample before transcribing"
            )

        model = self._ensure_model()
        samples = _pcm16_to_mono_float32(chunk.pcm16, chunk.channels)
        segments, _info = model.transcribe(
            samples,
            language=self._language,
            vad_filter=self._vad_filter,
        )

        results: List[TranscriptSegment] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            results.append(
                TranscriptSegment(
                    text=text,
                    start=chunk.timestamp + segment.start,
                    end=chunk.timestamp + segment.end,
                    no_speech_prob=getattr(segment, "no_speech_prob", 0.0),
                )
            )
        return results


def _pcm16_to_mono_float32(pcm16: bytes, channels: int) -> "Any":
    """Convert interleaved little-endian PCM16 bytes to mono float32 in [-1, 1].

    ``faster-whisper`` (via ``numpy``, a transitive dependency of the
    ``speech`` extra) expects this shape; the conversion is kept local to this
    module so :mod:`openmed.ambient.sources` stays import-cheap.
    """

    import numpy as np

    audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio
