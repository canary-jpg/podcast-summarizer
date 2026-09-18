"""
Fetches podcast episodes from the Listen Notes API and transcribes
audio using OpenAI Whisper. Outpurs structured JSON to data/processed/.
Usage:
    python fetch_episodes.py --query "machine learning" --max_episodes 5
    python fetch_episodes.py --podcast_id <id> --max_episodes 10
    python fetch_episodes.py --rss_url https://feeds.example.com/podcast.rss

Requirements:
    pip install requests openai-whisper pydantic python-dotenv tqdm
"""

import argparse
import json 
import os 
import time 
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path 
from typing import Optional
import requests 
import whisper 
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# -- Logging ---------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ingestion")

#--Config-------------------------------------------
LISTEN_NOTES_API_KEY = os.getenv("LISTEN_NOTES_API_KEY")
LISTEN_NOTES_BASE_URL = "https://listen-api.listennotes.com/api/v2"

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

#--Data Models----------------------------------------------
@dataclass 
class Episode:
    """Represents a single podcast episode with metadata and transcript. """
    id: str
    title: str 
    podcast_name: str
    description: str 
    audio_url: str 
    pub_date: str  #ISO 8601
    duration_seconds: int 
    transcript: Optional[str] = None 
    transcript_segments: list = field(default_factory=list)
    source: str = "listen_notes" #for provence tracking
    ingested_at: str = ""

    def to_json(self) -> dict:
        return asdict(self)


#--Listen Notes client-----------------------------------------
class ListenNotesClient:
    """Thin wrapper around the Listen Notes REST API. """

    def __init__(self, api_key:str):
        if not api_key:
            raise ValueError(
                "LISTEN_NOTES_API_KEY not set. "
                "Get a free key at https://www.listennotes.com/api/"
            )
        self.session = requests.Session()
        self.session.headers.update({"X-ListenAPI-Key": api_key})


    def _get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{LISTEN_NOTES_BASE_URL}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        #Listen Notes returns remaining quota in headers - log it
        quota = resp.headers.get("X-ListenAPI-FreeQuota")
        if quota:
            logger.debug("Listen Notes free quota remaining: %s", quota)
        return resp.json()

    def search_episodes(self, query: str, max_results: int = 10) -> list[dict]:
        """Search for episode matching a keyword query."""
        logger.info("Searching Listen Notes for: '%s'", query)
        data = self._get("search", params={
            "q": query,
            "type": "episode",
            "len_min": 10, #skip very short clips
            "language": "English",
            "page_size": min(max_results, 10), #API max per page is 10
        })
        episodes = data.get("results", [])
        logger.info("Found %d episodes", len(episodes))
        return episodes[:max_results]

    def get_podcast_episodes(self, podcast_id: str, max_results: int = 10) -> list[dict]:
        """Fetch recent episodes from a specific podcast by ID """
        logger.info("Fetching episodes for podcast ID: %s", podcast_id)
        data = self._get(f"podcasts/{podcast_id}", params={"sort": "recent_first"})
        episodes = data.get("episodes", [])
        logger.info("Found %d episodes", len(episodes))
        return episodes[:max_results]

    def parse_episode(self, raw: dict) -> Episode:
        """Normalize a Listen Notes episode dict into our Episode dataclass. """
        pub_date = raw.get("pub_date_ms", 0)
        if pub_date:
            from datetime import datetime, timezone 
            pub_date = datetime.fromtimestamp(
                pub_date / 1000, tz=timezone.utc 
            ).isoformat()

        return Episode(
            id=raw.get("id", ""),
            title=raw.get("title_original", raw.get("title", "")),
            podcast_name=raw.get("podcast", {}).get("title_original", ""),
            description=raw.get("description_original", raw.get("description", "")),
            audio_url=raw.get("audio", ""),
            pub_date=pub_date,
            duration_seconds=raw.get("audio_length_sec", 0),
            source="listen_notes",
        )


#--RSS Fallback------------------------------------------------------------------------
def fetch_rss_episodes(rss_url: str, max_results: int = 10) -> list[Episode]:
    """
    Parse a podcast RSS feed directly; useful when you have a feed URL
    but no Listen Notes ID, or when working on shows not indexed there.
    """
    try:
        import feedparser
    except ImportError:
        raise ImportError("pip install feedparser to use RSS ingestion")

    logger.info("Parsing RSS feed: %s", rss_url)
    feed = feedparser.parse(rss_url)
    podcast_name = feed.feed.get("title", "Unknown podcast")
    episodes = []

    for entry in feed.entries[:max_results]:
        #find the audio enclosure
        audio_url = ""
        for link in entry.get("links", []):
            if link.get("type", "").startswith("audio"):
                audio_url = link.get("href", "")
                break 

    pub_date = entry.get("published", "")
    duration_str = entry.get("itunes_duration", "0")
    duration_seconds = _parse_duration(duration_str)

    episodes.append(Episode(
        id=entry.get("id", entry.get("link", "")),
        title=entry.get("title", ""),
        podcast_name=podcast_name,
        description=entry.get("summary", ""),
        audio_url=audio_url,
        pub_date=pub_date,
        duration_seconds=duration_seconds,
        source="rss",
    ))

    logger.info("Parsed %d episodes from RSS", len(episodes))
    return episodes 


def _parse_duration(duration_str: str) -> int:
    """Convert 'HH:MM'SS' or 'MM:SS' or plain seconds string to int seconds """
    if not duration_str:
        return 0
    parts = duration_str.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        else:
            return int(parts[0])
    except ValueError:
        return 0

#--Whisper transcription-------------------------------------------------------------

class Transcriber:
    """
    Wraps OpenAI Whisper for local audio transcription.
    Model sizes: tiny, base, small, medium, large
    Tradeoff: larger = more accurate but slower & more RAM.
    'base' is a good starting point for experimentation.
    """

    def __init__(self, model_size: str = "base"):
        logger.info("Loading Whisper model: %s", model_size)
        self.model = whisper.load_model(model_size)
        logger.info("Whisper mode loaded")

    def transcribe_from_url(self, audio_url: str) -> dict:
        """
        Download audio to a temp file and transcribe it.
        Returns Whisper's full result dict (text + segments)
        """
        import tempfile 
        import requests

        logger.info("Downloaded audio: %s", audio_url[:80] + "...")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name 

        try:
            r = requests.get(audio_url, allow_redirects=True, stream=True)
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Transcribing with Whisper...")
            result = self.model.transcribe(tmp_path, verbose=False)
            return result
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def transcribe_from_file(self, audio_path: str) -> dict:
        """Transcribe a local audio file """
        logger.info("Transcribe local file: %s", audio_path)
        return self.model.transcribe(audio_path, verbose=False)


#--Pipeline--------------------------------------------------------------------------

def save_episode(episode: Episode, output_dir: Path) -> Path:
    """ Persist an Episode to JSON. Returns the output path"""
    from datetime import datetime, timezone 
    episode.ingested_at = datetime.now(tz=timezone.utc).isoformat()
    out_path = output_dir / f"{episode.id}.json"
    with open(out_path, "w") as f:
        json.dump(episode.to_json(), f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", out_path)
    return out_path

def run_pipeline(
    episodes: list[Episode],
    transcriber: Optional[Transcriber],
    skip_existing: bool = True,) -> list[Path]:
    """
    Main ingestion loop:
    1. Check if episode already processed (idempotency)
    2. Transcribe audio if a Transcriber is provided
    3. Save to disk
    """
    saved_paths = []

    for ep in tqdm(episodes, desc="Processing episodes"):
        out_path = PROCESSED_DIR / f"{ep.id}.json"
        
        if skip_existing and out_path.exists():
            logger.info("Skipping already-processed episode: %s", ep.title)
            saved_paths.append(out_path)
            continue
        
        if transcriber and ep.audio_url:
            try:
                result = transcriber.transcribe_from_url(ep.audio_url)
                ep.transcript = result["text"]
                ep.transcript_segments = [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"]
                    }
                    for seg in result.get("segments", [])
                ]
            except Exception as e:
                logger.warning("Transcription failed for '%s': %s", ep.title, e)
                #dont bail: save the episode with metadata even without transcript

    path = save_episode(ep, PROCESSED_DIR)
    saved_paths.append(path)

    #be polite to Listen Notes API 
    time.sleep(0.5)
    return saved_paths 


#--CLI-----------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest podcast episodes and transcribe with Whisper"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query", help="Keyword search via Listen Notes")
    source.add_argument("--podcast_id", help="Listen Notes podcast ID")
    source.add_argument("--rss_url", help="RSS feed URL (no API key needed)")

    parser.add_argument(
        "--max_episodes", type=int, default=5,
        help="Max number of episodes to fetch (default: 5)"
    )
    parser.add_argument(
        "--whisper_model", default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--no_transcribe", action="store_true",
        help="Skip transcription - fetch metadata only"
    )
    parser.add_argument(
        "--no_skip", action="store_true",
        help="Re-process episodes even if aleady on disk"
    )
    return parser.parse_args()

def main():
    args = parse_args()


#--Fetch episodes-------------------------------------------------------------
    if args.rss_url:
        episodes = fetch_rss_episodes(args.rss_url, max_results=args.max_episodes)
    else:
        client = ListenNotesClient(api_key=LISTEN_NOTES_API_KEY)
        if args.query:
            raw_episdes = client.search_episodes(args.query, max_results=args.max_episodes)
        else:
            raw_episdes = client.get_podcast_episodes(
                args.podcast_id, max_results=args.max_episodes
            )
        episodes = [client.parse_episode(r) for r in raw_episdes]

    if not episodes:
        logger.warning("No episodes found. Exiting.")
        return
        

    logger.info("Episodes to process: %d", len(episodes))


    #--Transcription--------------------------------------------------------------------------
    transcriber = None 
    if not args.no_transcribe:
        transcriber = Transcriber(model_size=args.whisper_model)

    #--Run--------------------------------------------------------------------------------------
    saved = run_pipeline(
        episodes,
        transcriber,
        skip_existing=not args.no_skip,
    )

    logger.info("Done. %d episodes saved to %s", len(saved), PROCESSED_DIR)
    for p in saved:
        print(p)

if __name__ == "__main__":
    main()
