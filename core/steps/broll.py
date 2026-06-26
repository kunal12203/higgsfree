"""Pexels API b-roll video fetcher with S3 cache."""
import os
import hashlib
import logging
import tempfile
import requests
import boto3

log = logging.getLogger(__name__)

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
S3_BUCKET = os.environ.get("S3_BUCKET", "avatar-graperoot-assets")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

s3 = boto3.client("s3", region_name=AWS_REGION)


def _cache_key(query: str) -> str:
    slug = hashlib.md5(query.lower().encode()).hexdigest()[:12]
    return f"broll-cache/{slug}.mp4"


def _check_s3_cache(key: str) -> str | None:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        # Download to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        s3.download_file(S3_BUCKET, key, tmp.name)
        return tmp.name
    except Exception:
        return None


def _save_to_s3(local_path: str, key: str):
    try:
        s3.upload_file(local_path, S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
    except Exception as e:
        log.warning(f"Failed to cache broll to S3: {e}")


def fetch_video(query: str, duration_hint: float = 5.0) -> str:
    """Fetch b-roll stock video for a query.

    Checks S3 cache first; falls back to Pexels API.

    Args:
        query: Search query for stock footage.
        duration_hint: Desired duration in seconds (used for filtering).

    Returns:
        Local file path to the downloaded video.
    """
    cache_key = _cache_key(query)
    cached = _check_s3_cache(cache_key)
    if cached:
        log.info(f"B-roll cache hit: {query}")
        return cached

    if not PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY not set")

    log.info(f"Fetching b-roll from Pexels: {query}")
    resp = requests.get(
        "https://api.pexels.com/videos/search",
        headers={"Authorization": PEXELS_API_KEY},
        params={"query": query, "per_page": 5, "orientation": "landscape"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    videos = data.get("videos", [])
    if not videos:
        raise RuntimeError(f"No Pexels results for: {query}")

    # Pick video closest to duration_hint
    best = min(
        videos,
        key=lambda v: abs(v.get("duration", 999) - duration_hint),
    )

    # Get HD file
    files = best.get("video_files", [])
    hd_files = [f for f in files if f.get("quality") in ("hd", "sd")]
    if not hd_files:
        hd_files = files
    video_file = hd_files[0]
    video_url = video_file["link"]

    # Download
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    with requests.get(video_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            tmp.write(chunk)
    tmp.flush()

    _save_to_s3(tmp.name, cache_key)
    return tmp.name
