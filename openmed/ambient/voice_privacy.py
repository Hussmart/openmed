"""Voiceprint anonymization via McAdams-coefficient pole warping.

HIPAA Safe Harbor lists "voice prints" as one of the 18 direct identifier
categories. Transcribing and redacting spoken PHI (see
:mod:`openmed.ambient.pipeline`) removes identifiers from the *text*, but the
raw waveform still carries the speaker's voiceprint -- :mod:`openmed.risk.reid`
already tracks ``"voiceprint"`` as a quasi-identifier category for risk
scoring, but nothing in OpenMed changed the audio itself. This module closes
that gap with the McAdams-coefficient technique: the same LPC pole-warping
baseline used in the VoicePrivacy Challenge (Patino et al.), chosen because it
is a small, deterministic signal-processing method with no model checkpoint to
download and no external service call.

The technique, per analysis frame:

1. Estimate an all-pole LPC model of the frame via Levinson-Durbin recursion.
2. Compute the LPC residual (excitation) by inverse-filtering the frame.
3. Find the LPC polynomial's poles and raise each pole's phase angle to the
   ``mcadams_coefficient`` power, leaving its magnitude untouched. This warps
   the frame's formant structure -- the main correlate of perceived speaker
   identity -- while leaving pitch and timing (carried by the excitation
   signal) intact, so the anonymized audio stays intelligible.
4. Re-synthesize by filtering the original excitation through the warped
   all-pole filter, then overlap-add the frames back together.

Requires the ``voice-privacy`` extra (``pip install "openmed[voice-privacy]"``,
which is just NumPy).
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.core.capabilities import require_backend

__all__ = ["McAdamsVoiceAnonymizer", "anonymize_wav_file"]

_DEFAULT_MCADAMS_COEFFICIENT = 0.8
_DEFAULT_WINDOW_MS = 20.0
_DEFAULT_HOP_MS = 10.0
_MIN_LPC_ORDER = 8
_MAX_LPC_ORDER = 30
_SILENCE_ENERGY_FLOOR = 1e-9


@dataclass
class McAdamsVoiceAnonymizer:
    """Warp voiceprint-carrying formants in PCM16 audio, frame by frame.

    Example:
        >>> anonymizer = McAdamsVoiceAnonymizer(mcadams_coefficient=0.8)  # doctest: +SKIP
        >>> anonymized = anonymizer.anonymize_pcm16(pcm16, sample_rate=16_000, channels=1)  # doctest: +SKIP
    """

    mcadams_coefficient: float = _DEFAULT_MCADAMS_COEFFICIENT
    window_ms: float = _DEFAULT_WINDOW_MS
    hop_ms: float = _DEFAULT_HOP_MS
    lpc_order: int | None = None

    def __post_init__(self) -> None:
        require_backend("voice_privacy", feature="Voiceprint anonymization")
        if not 0.0 < self.mcadams_coefficient <= 1.5:
            raise ValueError("mcadams_coefficient must be in (0, 1.5]")
        if self.window_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("window_ms and hop_ms must be positive")
        if self.hop_ms > self.window_ms:
            raise ValueError("hop_ms must not exceed window_ms")
        if self.lpc_order is not None and not (
            _MIN_LPC_ORDER <= self.lpc_order <= _MAX_LPC_ORDER
        ):
            raise ValueError(
                f"lpc_order must be between {_MIN_LPC_ORDER} and {_MAX_LPC_ORDER}"
            )

    def anonymize_pcm16(
        self, pcm16: bytes, *, sample_rate: int, channels: int
    ) -> bytes:
        """Return anonymized PCM16 audio, same length and format as the input."""

        import numpy as np

        order = self.lpc_order or min(
            _MAX_LPC_ORDER, max(_MIN_LPC_ORDER, sample_rate // 1000 + 4)
        )
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float64)
        if channels > 1:
            frames = samples.reshape(-1, channels)
            anonymized = np.stack(
                [
                    _anonymize_mono(
                        frames[:, ch],
                        sample_rate=sample_rate,
                        order=order,
                        window_ms=self.window_ms,
                        hop_ms=self.hop_ms,
                        mcadams_coefficient=self.mcadams_coefficient,
                    )
                    for ch in range(channels)
                ],
                axis=1,
            ).reshape(-1)
        else:
            anonymized = _anonymize_mono(
                samples,
                sample_rate=sample_rate,
                order=order,
                window_ms=self.window_ms,
                hop_ms=self.hop_ms,
                mcadams_coefficient=self.mcadams_coefficient,
            )

        clipped = np.clip(anonymized, -32768, 32767)
        return clipped.astype(np.int16).tobytes()


def anonymize_wav_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    mcadams_coefficient: float = _DEFAULT_MCADAMS_COEFFICIENT,
) -> None:
    """Anonymize a 16-bit PCM ``.wav`` file's voiceprint and write the result.

    This is the standalone entry point for archived recordings -- it needs
    only the ``voice-privacy`` extra, not ``speech`` or ``mic``.
    """

    anonymizer = McAdamsVoiceAnonymizer(mcadams_coefficient=mcadams_coefficient)
    with wave.open(str(input_path), "rb") as reader:
        if reader.getsampwidth() != 2:
            raise ValueError("anonymize_wav_file only supports 16-bit PCM WAV files")
        channels = reader.getnchannels()
        sample_rate = reader.getframerate()
        pcm16 = reader.readframes(reader.getnframes())
        params = reader.getparams()

    anonymized = anonymizer.anonymize_pcm16(
        pcm16, sample_rate=sample_rate, channels=channels
    )

    with wave.open(str(output_path), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(anonymized)


def _anonymize_mono(
    samples: "Any",
    *,
    sample_rate: int,
    order: int,
    window_ms: float,
    hop_ms: float,
    mcadams_coefficient: float,
) -> "Any":
    import numpy as np

    window_len = max(order * 2, int(sample_rate * window_ms / 1000))
    hop_len = max(1, int(sample_rate * hop_ms / 1000))
    if len(samples) < window_len:
        return samples.copy()

    window = np.hanning(window_len)
    out = np.zeros(len(samples), dtype=np.float64)
    norm = np.zeros(len(samples), dtype=np.float64)

    position = 0
    while position + window_len <= len(samples):
        frame = samples[position : position + window_len] * window
        anonymized_frame = _anonymize_frame(frame, order, mcadams_coefficient)
        out[position : position + window_len] += anonymized_frame
        norm[position : position + window_len] += window * window
        position += hop_len

    norm[norm < _SILENCE_ENERGY_FLOOR] = 1.0
    return out / norm


def _anonymize_frame(frame: "Any", order: int, mcadams_coefficient: float) -> "Any":
    import numpy as np

    autocorr = _autocorrelation(frame, order)
    if autocorr[0] < _SILENCE_ENERGY_FLOOR:
        return frame

    lpc_coeffs = _levinson_durbin(autocorr, order)
    residual = np.convolve(frame, lpc_coeffs)[: len(frame)]

    poles = np.roots(lpc_coeffs)
    warped_poles = _warp_poles(poles, mcadams_coefficient)
    warped_coeffs = np.real(np.poly(warped_poles)) if len(warped_poles) else lpc_coeffs

    return _iir_synthesize(residual, warped_coeffs)


def _autocorrelation(frame: "Any", order: int) -> "Any":
    import numpy as np

    full = np.correlate(frame, frame, mode="full")
    center = len(frame) - 1
    return full[center : center + order + 1]


def _levinson_durbin(autocorr: "Any", order: int) -> "Any":
    import numpy as np

    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    err = float(autocorr[0])
    if err <= _SILENCE_ENERGY_FLOOR:
        return a

    for i in range(1, order + 1):
        acc = autocorr[i] + float(np.dot(a[1:i], autocorr[i - 1 : 0 : -1]))
        reflection = -acc / err
        new_a = a.copy()
        new_a[1:i] = a[1:i] + reflection * a[i - 1 : 0 : -1]
        new_a[i] = reflection
        a = new_a
        err *= 1.0 - reflection * reflection
        if err <= _SILENCE_ENERGY_FLOOR:
            break
    return a


def _warp_poles(poles: "Any", mcadams_coefficient: float) -> list:
    import numpy as np

    warped: list = []
    for pole in poles:
        imag = float(np.imag(pole))
        if abs(imag) < 1e-10:
            warped.append(float(np.real(pole)))
            continue
        if imag <= 0:
            continue  # reconstructed as the conjugate of its positive-imag pair
        magnitude = abs(pole)
        angle = float(np.angle(pole))
        new_angle = angle**mcadams_coefficient
        new_pole = magnitude * np.exp(1j * new_angle)
        warped.append(new_pole)
        warped.append(np.conj(new_pole))
    return warped


def _iir_synthesize(residual: "Any", denominator: "Any") -> "Any":
    """Apply the all-pole synthesis filter ``1 / denominator`` to ``residual``."""

    import numpy as np

    output = np.zeros_like(residual)
    order = len(denominator) - 1
    for n in range(len(residual)):
        acc = residual[n]
        upper = min(order, n)
        if upper:
            # history = [output[n-1], output[n-2], ..., output[n-upper]], paired
            # with denominator[1..upper]. Sliced forward then reversed to avoid
            # negative-stop slicing (output[n-1:n-1-upper:-1] misbehaves once
            # n-1-upper goes negative, since Python reinterprets a negative stop
            # as an index from the end rather than "below zero").
            history = output[n - upper : n][::-1]
            acc -= float(np.dot(denominator[1 : upper + 1], history))
        output[n] = acc
    return output
