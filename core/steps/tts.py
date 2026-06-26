"""Kokoro-ONNX TTS wrapper with lazy model loading."""
import io
import logging
import numpy as np

log = logging.getLogger(__name__)

_model = None
_voices = None


def _load_model():
    global _model, _voices
    if _model is not None:
        return _model, _voices
    log.info("Loading Kokoro-ONNX model...")
    from kokoro_onnx import Kokoro
    _model = Kokoro("kokoro-v0_19.onnx", "voices.bin")
    log.info("Kokoro model loaded.")
    return _model, _voices


def synthesize(text: str, voice: str = "af_heart") -> bytes:
    """Synthesize text to WAV audio bytes.

    Args:
        text: The text to synthesize.
        voice: Kokoro voice ID.

    Returns:
        WAV audio as bytes.
    """
    import soundfile as sf

    model, _ = _load_model()
    samples, sample_rate = model.create(text, voice=voice, speed=1.0, lang="en-us")

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()
