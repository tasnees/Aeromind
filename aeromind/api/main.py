from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import time

from aeromind.api.routers import router

try:
    from prometheus_client import (  # type: ignore
        CONTENT_TYPE_LATEST,
        Counter,
        Histogram,
        generate_latest,
    )
    PROMETHEUS_AVAILABLE = True

    REQUEST_COUNT = Counter(
        "aeromind_http_requests_total",
        "Total HTTP requests processed by AeroMind API",
        ["method", "endpoint", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "aeromind_http_request_duration_seconds",
        "HTTP request duration in seconds",
        ["endpoint"],
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False


app = FastAPI(
    title="AeroMind Ops — Predictive Maintenance & PHM API",
    description=(
        "Enterprise API providing Remaining Useful Life (RUL) deep temporal inference, "
        "quantile uncertainty quantification, sensor attribution, and an LLM maintenance copilot."
    ),
    version="1.0.0",
)

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_and_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time-Sec"] = f"{process_time:.4f}"

    if PROMETHEUS_AVAILABLE:
        endpoint = request.url.path
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(process_time)

    return response


app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "AeroMind Ops",
        "docs_url": "/docs",
        "health_check": "/api/v1/monitoring/health",
        "metrics": "/metrics",
        "version": "1.0.0",
    }


@app.get("/metrics")
def get_metrics():
    """Prometheus metrics scrape endpoint."""
    if PROMETHEUS_AVAILABLE:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    return Response(content="# Prometheus client not installed\n", media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("aeromind.api.main:app", host="0.0.0.0", port=8000, reload=False)
