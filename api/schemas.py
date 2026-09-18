"""
Pydantic request and response models for the podcast summarizer API.
Keeping schemas separate from route logic makes them easy to version
and reuse across routes, tests, and client SDKs.
"""

from enum import Enum 
from typing import Optional 
from pydantic import BaseModel, Field, httpUrl

#--Enums-------------------------------------------------------------

class WhisperModel(str, Enum):
    tiny = "tiny"
    base = "base"
    small = "small"
    medium = "medium"
    large = "large"

class LLMProvider(str, Enum):
    anthropic   =    "anthropic"
    openai      =     "openai"
    local       =     "local"

class IngestionSource(str, Enum):
    search     = "search"
    podcast_id = "podcast_id"
    rss        = "rss"

#--Requests----------------------------------------------------------

class SummarizedRequest(BaseModel):
    """
    Request body for POST /summarize.
    The caller provider either a search query, a podcast ID, or an RSS URL.
    The API handles ingestion -> preprocessing -> summarization end-to-end
    """
    #source: exactly one of these should be set
    query: Optional[str] = Field(None, description="Keyword search via Listen Notes")
    podcast_id: Optional[str] = Field(None, description="Listen Notes podcast ID")
    rss_url: Optional[httpUrl] = Field(None, description="Direct RSS feed URL")

    #ingestion options
    max_episodes: int = Field(1, ge=1, le=10, description="Episodes to fetch and summarize")
    whisper_model: WhisperModel = Field(WhisperModel.base, description="Whisper ASR model size")
    skip_transcribe: bool = Field(False, description="Use episode description only, skip audio transcription")

    #summarization options
    provider: LLMProvider = Field(LLMProvider.anthropic, description="LLM provider")
    model: Optional[str] = Field(None, description="Model name override")
    prompt_version: str = Field("v1", description="Prompt version from configs/prompts.yaml")

    #preprocessing options
    max_tokens_per_chunk: int = Field(3000, ge=500, le=8000)
    overlap_tokens: int = Field(150, ge=0, le=500)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "machine_learning",
                "max_episodes": 2, 
                "whisper_model": "base",
                "provider": "anthropic",
                "prompt_version": "v1",
            }
        }


class EvalRequest(BaseModel):
    """Request body for POST /eval: score an existing summary. """
    episode_id: str = Field(..., description="Episode ID from a previous /summarize call")
    reference_text: Optional[str] = Field(None, description="Human-written reference summary for ROUGE/BERTScore")
    run_llm_judge: bool = Field(True, description="Run LLM-as-judge scorer")


#--Responses-------------------------------------------------------------------------------------------------------

class ChunkSummaryOut(BaseModel):
    chunk_index:        int
    summary:            str
    input_tokens:       int
    output_tokens:      int 
    latency_ms:         float

class EpisodeSummaryOut(BaseModel):
    episode_id:      str 
    title:           str 
    podcast_name:    str 
    pub_date:        str 
    final_sumary:    str
    chunk_summaries: list[ChunkSummaryOut]
    model:           str 
    provider:        str 
    prompt_version:  str 
    total_input_tokens: int 
    total_output_tokens: int 
    total_latency_ms: float 
    num_chunks: int 
    summarized_at: str 


class RougeOut(BaseModel):
    rouge1_f: float 
    rouge2_f: float 
    rougeL_f: float 

class BertScoreOut(BaseModel):
    f1:         float
    precision:  float 
    recall:     float 
    model_used: str 

class LLMJudgeOut(BaseModel):
    faithfulness:   float 
    coverage:       float 
    fluency:        float 
    conciseness:    float 
    overall:        float 
    reasons:        dict[str, str]

class EvalOut(BaseModel):
    episode_id:         str 
    title:              str 
    rouge:              Optional[RougeOut] = None 
    bertscore:          Optional[BertScoreOut] = None 
    llm_judge:          Optional[LLMJudgeOut] = None 
    passed_thresholds:  Optional[bool] = None 
    threshold_failures: list[str] = []
    evaluated_at:       str 


class SummarizeResponse(BaseModel):
    """Full response from POST /summarize: one entry per episode processed """
    summaries:              list[EpisodeSummaryOut]
    total_episodes:         int 
    total_input_tokens:     int 
    total_output_tokens:    int 

class HealthResponse(BaseModel):
    status:     str 
    version:    str 
    model:      str 
    provider:   str 

class ErrorResponse(BaseModel):
    detail:     str 
    code:       str 