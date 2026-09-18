"""Real-time ambient speech redaction.

Turns a live microphone feed or a recorded encounter (``.wav``) into a stream
of de-identified transcript segments, reusing OpenMed's existing
:func:`openmed.core.pii.deidentify` engine for every redaction decision.

Requires the ``speech`` extra (``pip install "openmed[speech]"``) for
transcription, and additionally the ``mic`` extra
(``pip install "openmed[mic]"``) for live microphone capture. Replaying a
``.wav`` file via :class:`WavFileAudioSource` needs neither extra beyond
``speech``.
"""

from __future__ import annotations

from .pipeline import AmbientRedactionPipeline, RedactedSegment
from .sources import AudioChunk, AudioSource, MicrophoneAudioSource, WavFileAudioSource
from .transcriber import (
    WHISPER_SAMPLE_RATE,
    FasterWhisperTranscriber,
    TranscriptSegment,
)
from .voice_privacy import McAdamsVoiceAnonymizer, anonymize_wav_file

__all__ = [
    "AmbientRedactionPipeline",
    "AudioChunk",
    "AudioSource",
    "FasterWhisperTranscriber",
    "McAdamsVoiceAnonymizer",
    "MicrophoneAudioSource",
    "RedactedSegment",
    "TranscriptSegment",
    "WHISPER_SAMPLE_RATE",
    "WavFileAudioSource",
    "anonymize_wav_file",
]
