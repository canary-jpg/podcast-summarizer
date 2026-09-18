"""
Middleware for the podcast summarizer FastAPI app:
    - Request/response logging with timing
    - Unique request ID injection (useful for tracing in logs)
    - Simple in-memory rate limiting per IP
    - Correlation of slow requests for performance monitoring
"""

import logging 
import time 
import uuid 
from collections import defaultdict
from typing import Callable 

from fastapi import Request, Response 
from starlette.middleware.base import BaseHTTPMiddleware 

logger = logging.getLogger("api.middleware")

#--Request logging + timing---------------------------------------

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Attaches a unique request ID to every request and logs:
        - Incoming method + path
        - Response status code an wall-clock latency
        - A WARNING for any request over SLOW_THRESHOLD_MS
    """
    SLOW_THRESHOLD_MS = 30_000 #30s: summarization can be slow

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id 

        start = time.monotonic()
        logger.info(
            "[%s] -> %s %s",
            request_id,
            request.method,
            request.url.path,
        )

        response = await calle_next(request)

        latency_ms = (time.monotonic() - start) * 1000
        response.headers["X-Request-ID"] = request_id 
        response.headers["X-Latency-MS"] = str(round(latency_ms))

        log_fn = logger.warning if latency_ms > self.SLOW_THRESHOLD_MS else logger.info 
        log_fn = (
            "[%s] <- %d %.0fms %s %s",
            request_id,
            response.status_code,
            latency_ms,
            request.method,
            request.url.path,
        )

        return response 

#--Rate limiting-------------------------------------------------------------------------

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple token-bucket rate limiter keyed by client IP.
    Limits:
        - Default: 10 requests per 60s window (enough for a dev workflow)
        - /health is excluded (monitoring probes shouldn't count)
    For production you'd replace this with Redis + a proper sliding window,
    but in-memory is fine for a single-process deployment
    """

    EXCLUDED_PATHS = {"/health", "/docs", "/openai.json", "/redoc"}

    def __init__(self, app, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        #{ip: [(timestamp, count), ...]}
        self._windows: dict[str, list[float]] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispath(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)

        ip = self._get_client_ip(request)
        now = time.monotonic()
        window_start = now - self.window_seconds 

        #prune old timestamps
        self._windows[ip] = [t for t in self._windows[ip] if t > window_start]

        if len(self._windows[ip]) >= self.max_requests:
            logger.warning("Rate limit exceeded for IP: %s", ip)
            return Response(
                content='{"detail": "Rate limit exceeded", "code": "rate_limited"}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(self.window_seconds)},
            )
        
        self._windows[ip].append(now)
        response = await call_next(request)

        #tell the client how many requests they have left
        remaining = self.max_requests - len(self._windows[ip])
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(self.window_seconds)

        return response 

        