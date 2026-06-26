"""Replicate API fallback for MuseTalk lipsync when no GPU is available."""
import os
import logging
import replicate

log = logging.getLogger(__name__)

MUSETALK_MODEL = "lucataco/musetalk:latest"


def run_lipsync(face_image_url: str, audio_url: str) -> str:
    """Run MuseTalk lipsync via Replicate API.

    Args:
        face_image_url: Public URL to the face image (JPG/PNG).
        audio_url: Public URL to the audio file (WAV/MP3).

    Returns:
        URL of the output video.
    """
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        raise RuntimeError("REPLICATE_API_TOKEN not set")

    log.info(f"Running Replicate MuseTalk: face={face_image_url}")

    output = replicate.run(
        MUSETALK_MODEL,
        input={
            "video": face_image_url,
            "audio": audio_url,
        },
    )

    # output is typically a URL string or a list
    if isinstance(output, list):
        return str(output[0])
    return str(output)
