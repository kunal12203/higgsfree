"""FFmpeg compositor for building the final video from segments."""
import os
import json
import logging
import tempfile
import subprocess
from typing import Any

import ffmpeg

log = logging.getLogger(__name__)

MUSIC_BED_PATH = os.environ.get("MUSIC_BED_PATH", "")


def _probe_duration(path: str) -> float:
    """Return the duration of a media file in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True,
    )
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def _merge_video_audio(video_path: str, audio_path: str, output_path: str):
    """Merge a video with an audio track.

    Audio is the master: video is padded (last-frame freeze) if it is shorter
    than the audio. If video is longer, it is trimmed to audio length.
    No silent truncation — the full audio always plays.
    """
    audio_dur = _probe_duration(audio_path)
    video_dur = _probe_duration(video_path)

    if audio_dur <= 0:
        # No audio info — fall back to muxing without correction
        v = ffmpeg.input(video_path).video
        a = ffmpeg.input(audio_path).audio
        ffmpeg.output(v, a, output_path, vcodec="libx264", acodec="aac",
                      pix_fmt="yuv420p").overwrite_output().run(quiet=True)
        return

    if video_dur < audio_dur:
        # Video is shorter — pad with last-frame freeze so audio plays fully
        pad_secs = audio_dur - video_dur + 0.5  # small buffer
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-filter_complex",
            f"[0:v]tpad=stop_mode=clone:stop_duration={pad_secs:.3f}[vpad]",
            "-map", "[vpad]",
            "-map", "1:a",
            "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
            "-shortest",
            output_path,
        ], check=True, capture_output=True)
    else:
        # Video is longer or equal — trim to audio length
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
            "-t", f"{audio_dur:.3f}",
            output_path,
        ], check=True, capture_output=True)


def _create_title_card(text: str, duration: float, output_path: str, width: int = 1920, height: int = 1080):
    (
        ffmpeg
        .input(f"color=c=0x0f0f12:s={width}x{height}:d={duration}", f="lavfi")
        .drawtext(
            text=text,
            fontsize=72,
            fontcolor="white",
            x="(w-text_w)/2",
            y="(h-text_h)/2",
            fontfile="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        )
        .output(output_path, vcodec="libx264", pix_fmt="yuv420p", r=30, t=duration)
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path


def _create_broll_clip(broll_path: str, duration: float, output_path: str,
                       overlay_text: str = None, width: int = 1920, height: int = 1080):
    stream = ffmpeg.input(broll_path, t=duration)
    video = stream.video.filter("scale", width, height, force_original_aspect_ratio="decrease")
    video = video.filter("pad", width, height, "(ow-iw)/2", "(oh-ih)/2")

    if overlay_text:
        video = video.drawtext(
            text=overlay_text,
            fontsize=48,
            fontcolor="white",
            x="(w-text_w)/2",
            y="h-100",
            box=1,
            boxcolor="black@0.5",
            boxborderw=10,
        )

    (
        ffmpeg
        .output(video, output_path, vcodec="libx264", pix_fmt="yuv420p", r=30, t=duration, an=None)
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path


def _concat_videos(video_paths: list[str], output_path: str):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in video_paths:
            f.write(f"file '{p}'\n")
        list_file = f.name

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
         "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", output_path],
        check=True,
        capture_output=True,
    )
    os.unlink(list_file)


def compose_video(segments: list[dict[str, Any]], output_path: str) -> str:
    """Compose the final video from pipeline segments.

    Audio is always the master timeline. For avatar segments the lipsync video
    is padded (last-frame freeze) or trimmed so it matches the TTS audio exactly.
    This ensures the full script is always audible — no silent truncation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        clip_paths = []

        for i, seg in enumerate(segments):
            seg_type = seg.get("type", "avatar")
            duration = float(seg.get("duration", 5))
            clip_out = os.path.join(tmpdir, f"clip_{i:03d}.mp4")

            if seg_type == "title_card":
                _create_title_card(
                    text=seg.get("overlay_text") or seg.get("text", ""),
                    duration=duration,
                    output_path=clip_out,
                )
                clip_paths.append(clip_out)

            elif seg_type == "broll":
                broll_path = seg.get("broll_path")
                if broll_path and os.path.exists(broll_path):
                    _create_broll_clip(
                        broll_path=broll_path,
                        duration=duration,
                        output_path=clip_out,
                        overlay_text=seg.get("overlay_text"),
                    )
                    clip_paths.append(clip_out)
                else:
                    _create_title_card(seg.get("text", ""), duration, clip_out)
                    clip_paths.append(clip_out)

            elif seg_type == "avatar":
                video_path = seg.get("video_path")
                audio_path = seg.get("audio_path")

                if video_path and os.path.exists(video_path):
                    if audio_path and os.path.exists(audio_path):
                        # Audio is master: pad/trim video to match audio duration exactly
                        _merge_video_audio(video_path, audio_path, clip_out)
                    else:
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", video_path,
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", clip_out],
                            check=True, capture_output=True,
                        )
                    clip_paths.append(clip_out)

                elif audio_path and os.path.exists(audio_path):
                    # Audio only: solid background + full audio
                    audio_dur = _probe_duration(audio_path)
                    bg = os.path.join(tmpdir, f"bg_{i:03d}.mp4")
                    _create_title_card(seg.get("text", ""), audio_dur or duration, bg)
                    _merge_video_audio(bg, audio_path, clip_out)
                    clip_paths.append(clip_out)

                else:
                    _create_title_card(seg.get("text", ""), duration, clip_out)
                    clip_paths.append(clip_out)

        if not clip_paths:
            raise RuntimeError("No clips to compose")

        if len(clip_paths) == 1:
            import shutil
            shutil.copy(clip_paths[0], output_path)
        else:
            _concat_videos(clip_paths, output_path)

        return output_path
