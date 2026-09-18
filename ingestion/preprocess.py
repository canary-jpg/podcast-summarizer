"""
Cleans and chunks raw podcast transcripts produced by fetch_episodes.py
Outputs and preprocessed JSON to data/processed, ready for the summarizer.
Chunking strategy:
    - Prefer splitting on Whipser segment boundaries (preserves sentence flow)
    - Fall back to sentence-boundary splitting when segments aren't available
    - Respect a configurable token bugdet per chunk (default: 300 tokens)
    - Add overlap between chunks so context isn't lost at boundaries

Usage:
    # Process all episodes in data/processed/
    python preprocess.py

    # Process a single file
    python preprocess.py --file data/processed/<id>.json

    # Override chunk size and overlap
    python preprocess.py --max_token 2000 --overlap_tokens 100

Requirements:
    pip install tiktoken nltk tqdm
    python -m nltk.downloader punkt
"""

import argparse
import json
import logging
import re 
import unicodedata 
from dataclasses import asdict, dataclass, field 
from pathlib import Path 
from typing import Optional 

import nltk 
import tiktoken 
from tqdm import tqdm 

#dowload sentence tokenizer quietly if not already present
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

#--Logging-----------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("preprocess")

#--Config-------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
ENCODING = tiktoken.get_encoding("cl100k_base") #matches GPT-4/Claude tokenization

#filler words and disfluencies common in spoken audio
FILLER_PATTERNS = [
    r"\b(um+|uh+|hmm+| mhm+|hm+)\b",
    r"\b(you know|i mean| like,? )\b",
    r"\b(so,?)+\b",
    r"\b(basically|literally|actually),? ",
    r"(\.\.\.|u2026)+",
    r"\[inaudible\]|\[crosstalk\]|\[music\]",
    r"\s{2,}",
]

#--Data Models----------------------------------------------------------------
@dataclass
class Chunk:
    chunk_index: int
    text: str
    token_count: int
    start_time: Optional[float] = None # seconds into episode
    end_time: Optional[float] = None 
    word_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass 
class ProcessedEpisode:
    id: str 
    title: str 
    podcast_name: str
    description: str 
    pub_date: str 
    duration_seconds: int 
    source: str 
    ingested_at: str 
    chunks: list[dict] = field(default_factory=list)
    total_tokens: int = 0
    total_chunks: int = 0
    cleaning_stats: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


#--Text cleaning---------------------------------------------------------------------

def clean_transcript(text: str) -> tuple[str, dict]:
    """
    Clean raw Whisper output.
    Returns (cleaned_text, stats_dict) so callers can log what changed.
    """
    original_len = len(text)

    #normalize unicode (e.g. smart quotes, em-dashes -> ASCII equivalents)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    #strip speaker labels like "SPEAKER 1:" "Host:", "[John]:"
    text = re.sub(r"^\s*(\[?[A-Z][a-z]*\]?|SPEAKER\s*\d+)\s*:\s*", "", text, flags=re.MULTILINE)

    #remove timestamps sometimes interjected by Whisper verbose mode: "[00:01:23]"
    text = re.sub(r"\[\d{2}:\d{2}:\d{2}]", "", text)

    #apply filler patterns (order matters; whitespace collapse is last)
    for pattern in FILLER_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    #fixing spacing around punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.,!?])\s*([A-Z])", r"\1 \2", text)

    #final whitespace normalization
    text = re.sub(r"\s+", " ", text).strip()

    cleaned_len = len(text)
    stats = {
        "original_chars": original_len,
        "cleaned_chars": cleaned_len,
        "chars_removed": original_len - cleaned_len,
        "pct_removed": round((original_len - cleaned_len) / max(original_len, 1) * 100, 1)
    }
    return text, stats 


#--Token counting----------------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))

#--Chunking strategies------------------------------------------------------------------------------

def chunk_by_segments(
    segments: list[dict],
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """
    Preferred strategy: accumulate Whisper segments until we approach the 
    token budget, then start a new chunk. Overlap is achieved by including
    the last N tokens of previous chunk at the start of the next
    """
    chunks: list[Chunk] = []
    current_texts: list[str] = []
    current_tokens = 0
    current_start: Optional[float] = None 
    overlap_text = ""

    def flush(end_time: float) -> None:
        nonlocal current_texts, current_tokens, current_start, overlap_text
        if not current_texts:
            return 
        
        text = (overlap_text + " " + " ".join(current_texts)).strip()
        token_count = count_tokens(text)

        chunks.append(Chunk(
            chunk_index=len(chunks),
            text=text,
            token_count=token_count,
            start_time=current_start,
            end_time=end_time,
            word_count=len(text.split())
        ))

        #compute overlap: take the tail of current_texts to fill overlap budget
        overlap_tokens_remaining = overlap_tokens
        overlap_parts: list[str] = []
        for seg_text in reversed(current_texts):
            seg_tokens = count_tokens(seg_text)
            if overlap_tokens_remaining <= 0:
                break
            overlap_parts.insert(0, seg_text)
            overlap_tokens_remaining-= seg_tokens 

        overlap_text = " ".join(overlap_parts)
        current_text = []
        current_tokens = 0
        current_start = None 
    
    for seg in segments:
        seg_text = seg.get("text", "").strip()
        if not seg_text:
            continue 

        seg_tokens = count_tokens(seg_text)
        seg_start = seg.get("start")
        seg_end = seg.get("end")

        if current_start is None:
            current_start = seg_start 

        #if adding this segment would exceed budget, flush first
        if current_tokens + seg_tokens > max_tokens and current_texts:
            flush(end_time=seg_start)

        current_texts.append(seg_text)
        current_tokens += seg_tokens 

    #flush remaining
    if current_texts:
        last_end = segments[-1].get("end") if segments else None 
        flush(end_time=last_end)
    
    return chunks 

def chunk_by_sentences(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """
    Fallback strategy when Whisper segment aren't available.
    Splits on sentence boundaries using NLTK, then acculumates sentences
    into chunks respecting the token budget.
    """
    sentences = nltk.sent_tokenize(text)
    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    overlap_text=""

    def flush() -> None:
        nonlocal current_sentences, current_tokens, overlap_text 
        if not current_sentences:
            return 

        text_block = (overlap_text + " " + " ".join(current_sentences)).strip()
        token_count = count_tokens(text_block)

        chunks.append(Chunk(
            chunk_index=len(chunks),
            text=text_block,
            token_count=token_count,
            word_count=len(text_block.split()),
        ))

        #compute overlap from tail of current_sentences
        overlap_tokens_remaining = overlap_tokens
        overlap_parts: list[str] = []
        for sent in reversed(current_sentences):
            if overlap_tokens_remaining <= 0:
                break
            overlap_parts.insert(0, sent)
            overlap_tokens_remaining -= count_tokens(sent)

        overlap_text = " ".join(overlap_parts)
        current_sentences = []
        current_tokens = 0
    
    for sent in sentences:
        sent_tokens = count_tokens(sent)

        #edge case: single sentence exceeds budget - split it hard
        if sent_tokens > max_tokens:
            if current_sentences:
                flush()
            words = sent.split()
            mini_chunk_words: list[str] = []
            mini_tokens = 0
            for word in words:
                word_tokens = count_tokens(word)
                if mini_tokens + word_tokens > max_tokens and mini_chunk_words:
                    mini_text = " ".join(mini_chunk_words)
                    chunks.append(Chunk(
                        chunk_index=len(chunks),
                        text=mini_text,
                        token_count=count_tokens(mini_text),
                        word_count=len(mini_chunk_words),
                    ))
                mini_chunk_words = []
                mini_tokens = 0
                mini_chunk_words.append(word)
            if mini_chunk_words:
                mini_text = " ".join(mini_chunk_words)
                chunks.append(Chunk(
                    chunk_index=len(chunks),
                    text=mini_text,
                    token_count=count_tokens(mini_text),
                    word_count=len(mini_chunk_words),
                ))
            continue

        if current_tokens + sent_tokens > max_tokens and current_sentences:
            flush()

        current_sentences.append(sent)
        current_tokens += sent_tokens 
    
    if current_sentences:
        flush()

    return chunks


#--Main preprocessing---------------------------------------------------------------------------

def preprocess_episode(
    episode_data: dict,
    max_tokens: int,
    overlap_tokens: int,
) -> Optional[ProcessedEpisode]:
    """
    Full preprocessing pipeline for a single episode:
    1. extract and clean transcript
    2. choose chunking strategy (segments > sentences > skip)
    3. return a ProcessedEpisode ready for the summarizer
    """
    episode_id = episode_data.get("id", "unknown")
    title = episode_data.get("title", "")
    transcript = episode_data.get("transcript", "")
    segments = episode_data.get("transcript_segments", [])

    if not transcript and not segments:
        logger.warning("Episode '%s' has no transcript - skipping", title)
        return None 
    
    #use segment text if full transcript is missing (shouldn't happen but defensive)
    if not transcript and segments:
        transcript = " ".join(s.get("text", "") for s in segments)

    #clean
    cleaned_text, cleaning_stats = transcript, {"original_chars": len(transcript), "cleaned_chars": len(transcript), "chars_removed": 0, "pct_removed": 0.0}

    if not cleaned_text:
        logger.warning("Episode '%s' is empty after cleaning - skipping", title)
        return None 
    
    logger.info(
        "Cleaning '%s': removed %s%% of characters (%d -> %d chars)",
        title,
        cleaning_stats["pct_removed"],
        cleaning_stats["original_chars"],
        cleaning_stats["cleaned_chars"],
    )

    #also clean segment texts so they match cleaned transcript
    cleaned_segments = []
    for seg in segments:
        seg_text, _ = clean_transcript(seg.get("text", ""))
        if seg_text:
            cleaned_segments.append({**seg, "text": seg_text})
    
    #chunk
    if cleaned_segments:
        logger.info("Chunking by Whisper segments")
        chunks = chunk_by_segments(cleaned_segments, max_tokens, overlap_tokens)
    else:
        logger.info("No segments available - chunking by sentences")
        chunks = chunk_by_sentences(cleaned_text, max_tokens, overlap_tokens)
    
    total_tokens = sum(c.token_count for c in chunks)
    logger.info(
        "Episode '%s': %d chunks, %d total chunks",
        title, len(chunks), total_tokens,
    )

    return ProcessedEpisode(
        id=episode_id,
        title=title,
        podcast_name=episode_data.get("podcast_name", ""),
        description=episode_data.get("description", ""),
        pub_date=episode_data.get("pub_date", ""),
        duration_seconds=episode_data.get("duration_seconds", 0),
        source=episode_data.get("source", ""),
        ingested_at=episode_data.get("ingested_at", ""),
        chunks=[c.to_dict() for c in chunks],
        total_tokens=total_tokens,
        total_chunks=len(chunks),
        cleaning_stats=cleaning_stats,
    )


def process_file(
    input_path: Path,
    output_dir: Path,
    max_tokens: int,
    overlap_tokens: int,
    skip_existing: bool,
) -> Optional[Path]:
    """Load a raw episode JSON, preprocess it, and write the result """
    out_path = output_dir / input_path.name.replace(".json", "_preprocessed.json")

    if skip_existing and out_path.exists():
        logger.info("Skipping already-preprocessed: %s", input_path.name)
        return out_path
    
    with open(input_path) as f:
        episode_data = json.load(f)

    #skips files that were already preprocessed
    if "chunks" in episode_data and "total_chunks" in episode_data:
        logger.info("File already contains chunks, skipping: %s", input_path.name)
        return out_path 

    result = preprocess_episode(episode_data, max_tokens, overlap_tokens)
    if result is None:
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result.to_json(), f, indent=2, ensure_ascii=False)

    logger.info("Written: %s", out_path)
    return out_path

#--CLI--------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean and chunk podcast transcripts for summarization"
    )
    parser.add_argument(
        "--file", type=Path, default=None,
        help="Process a single episode JSON (default: all files in data/processed/)"
    )
    parser.add_argument(
        "--input_dir", type=Path, default=PROCESSED_DIR,
        help="Directory of raw episode JSON (default: data/processed/)"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=PROCESSED_DIR,
        help="Output directory for preprocessed JSONs (default: data/processed/)"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=3000,
        help=(
            "Max tokens per chunk (default: 3000). "
            "Rule of thumb: keep well under your model's context window. "
            "3000 leaves room for a system prompt + summary in a 4096-token model."
        ),
    )
    parser.add_argument(
        "--overlap_tokens", type=int, default=150,
        help="Overlap between consecutive chunks in tokens (default: 150)"
    )
    parser.add_argument(
        "--no_skip", action="store_true",
        help="Reprocess files even if preprocessed output already exists"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.file:
        files = [args.file]
    else:
        files = sorted(args.input_dir.glob("*.json"))
        #exclude files that are already preprocessed outputs
        files = [f for f in files if "_preprocessed" not in f.name]

    if not files: 
        logger.warning("No episode JSON files found in %s", args.input_dir)
        return 

    logger.info(
        "Preprocessing %d episodes(s) | max_tokens=%d | overlap=%d",
        len(files), args.max_tokens, args.overlap_tokens,
    )

    saved = []
    for path in tqdm(files, desc="Preprocessing"):
        out = process_file(
            input_path=path,
            output_dir=args.output_dir,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            skip_existing=not args.no_skip,
        )

        if out:
            saved.append(out)

    logger.info("Done. %d episode(s) preprocessed.", len(saved))
    for p in saved:
        print(p)

if __name__ == "__main__":
    main()