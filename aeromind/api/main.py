from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import time

from aeromind.api.routers import router

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
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time-Sec"] = f"{process_time:.4f}"
    return response


app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "AeroMind Ops",
        "docs_url": "/docs",
        "health_check": "/api/v1/monitoring/health",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("aeromind.api.main:app", host="0.0.0.0", port=8000, reload=False)

