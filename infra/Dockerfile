# Multi-stage build:
#   Stage 1 (builder) - install dependencies into a venv
#   Stage 2 (runtime) - copy venv + app code, run as non-root user

# Whisper and bert-score are large; they're included here but you can
# remove them if you're only using the API with skip_transcribe=true
# and no_llm_judge=true (metadata-only mode)

# Stage 1: Builder
FROM python:3.11-slim AS builder 

WORKDIR /build 

# System deps for audio processing (ffmpeg required by Whisper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/ap/lists/*

COPY infra/requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install -r requirements.txt 

# prep-download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"


# Stage 2: runtime
FROM python:3.11-slim as runtime 

# ffmpeg needed at runtime for Whisper audio decoding
RUN apt-get update && update apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# copy installed packages from builder
COPY --from=builder /install /usr/local
COPY --from=builder /root/nltk_data /root/nltk_data 

WORKDIR /app 

# copy application code
COPY ingestion/     ingestion/
COPY summarizer/    summarizer/
COPY evaluation/    evaluation/
COPY monitoring/    monitoring/
COPY api/           api/ 
COPY configs/      configs/

#create data directories (will be volume-mounted in production)
RUN mkdir -p data/raw data/processed data/outputs data/eval_results 

#run as non-root for security 
RUN useradd -m appuser && chown -R appuser /app 
USER appuser 

EXPOSE 8000

#uvicorn with 2 workers - increase for production load 
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]