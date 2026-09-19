"""Audio sources for the ambient redaction pipeline.

Two sources are provided out of the box: :class:`WavFileAudioSource` (no
optional dependency -- built on the stdlib ``wave`` module and the existing
dependency-free WAV header parser in :mod:`openmed.multimodal.wav_metadata`)
and :class:`MicrophoneAudioSource` (requires the ``mic`` extra). Both yield the
same :class:`AudioChunk` shape so :class:`openmed.ambient.pipeline.AmbientRedactionPipeline`
never needs to know where the audio came from.
"""

from __future__ import annotations

import queue
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from openmed.core.capabilities import require_backend
from openmed.multimodal.wav_metadata import WavMetadataError, read_wav_metadata

__all__ = [
    "AudioChunk",
    "AudioSource",
    "MicrophoneAudioSource",
    "WavFileAudioSource",
]

_DEFAULT_CHUNK_SECONDS = 3.0
_DEFAULT_SAMPLE_RATE = 16_000
_BYTES_PER_SAMPLE_16BIT = 2


@dataclass(frozen=True)
class AudioChunk:
    """One bounded window of raw PCM16 audio.

    ``pcm16`` is little-endian, interleaved-channel signed 16-bit audio -- the
    format both ``faster-whisper`` and ``sounddevice`` consume without extra
    conversion. Carrying bytes (not a NumPy array) here keeps this module
    importable without any optional dependency installed.

    ``timestamp`` is the chunk's *start* time in seconds since the stream began;
    transcript segment times are offsets from it.
    """

    pcm16: bytes
    sample_rate: int
    channels: int
    timestamp: float
    is_last: bool = False


@runtime_checkable
class AudioSource(Protocol):
    """A bounded stream of :class:`AudioChunk` at a fixed sample rate."""

    sample_rate: int
    channels: int

    def chunks(self) -> Iterator[AudioChunk]:
        """Yield audio chunks in timestamp order until the source is exhausted."""


class WavFileAudioSource:
    """Replay a local ``.wav`` file as a chunked stream.

    Useful for demos, tests, and offline batch redaction of recorded encounters
    without a microphone -- no optional dependency is required.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        chunk_seconds: float = _DEFAULT_CHUNK_SECONDS,
    ) -> None:
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")
        self._path = Path(path)
        self._chunk_seconds = chunk_seconds
        with self._path.open("rb") as handle:
            try:
                metadata = read_wav_metadata(handle)
            except WavMetadataError as exc:
                raise ValueError(f"unsupported WAV file: {exc.category}") from exc
        if metadata.bit_depth != 16:
            raise ValueError(
                "WavFileAudioSource only supports 16-bit PCM WAV files "
                f"(got {metadata.bit_depth}-bit)"
            )
        self.sample_rate = metadata.sample_rate_hz
        self.channels = metadata.channels

    def chunks(self) -> Iterator[AudioChunk]:
        frame_size = self.channels * _BYTES_PER_SAMPLE_16BIT
        frames_per_chunk = max(1, int(self._chunk_seconds * self.sample_rate))
        with wave.open(str(self._path), "rb") as handle:
            total_frames = handle.getnframes()
            frames_read = 0
            while frames_read < total_frames:
                frames = handle.readframes(frames_per_chunk)
                if not frames:
                    break
                read_count = len(frames) // frame_size
                chunk_start = frames_read / self.sample_rate
                frames_read += read_count
                yield AudioChunk(
                    pcm16=frames,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    timestamp=chunk_start,
                    is_last=frames_read >= total_frames,
                )


class MicrophoneAudioSource:
    """Capture live audio from the default input device via ``sounddevice``.

    The capture runs in ``sounddevice``'s own callback thread and hands chunks
    back through a bounded queue, so :meth:`chunks` is a plain blocking
    generator the pipeline can iterate like any other :class:`AudioSource`.
    """

    def __init__(
        self,
        *,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        channels: int = 1,
        chunk_seconds: float = _DEFAULT_CHUNK_SECONDS,
        device: int | str | None = None,
        max_duration_seconds: float | None = None,
    ) -> None:
        require_backend("mic", feature="Live microphone capture")
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")
        self.sample_rate = sample_rate
        self.channels = channels
        self._chunk_seconds = chunk_seconds
        self._device = device
        self._max_duration_seconds = max_duration_seconds
        self._frames_per_chunk = max(1, int(chunk_seconds * sample_rate))

    def chunks(self) -> Iterator[AudioChunk]:
        import sounddevice as sd

        audio_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=32)

        def _callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
            audio_queue.put(bytes(indata))

        elapsed = 0.0
        with sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self._frames_per_chunk,
            device=self._device,
            callback=_callback,
        ):
            while True:
                pcm16 = audio_queue.get()
                chunk_start = elapsed
                elapsed += self._chunk_seconds
                is_last = (
                    self._max_duration_seconds is not None
                    and elapsed >= self._max_duration_seconds
                )
                yield AudioChunk(
                    pcm16=pcm16,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    timestamp=chunk_start,
                    is_last=is_last,
                )
                if is_last:
                    return
