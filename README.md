# AeroMind Ops — Enterprise Predictive Maintenance & PHM Platform

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![Tests](https://img.shields.io/badge/Tests-13%20Passed-brightgreen.svg)]()

> A production-grade AI engineering platform for turbofan engine prognostics, combining deep temporal degradation models (Remaining Useful Life estimation with calibrated uncertainty) and an LLM-powered maintenance copilot with grounded RAG and strict anti-hallucination guardrails.

---

## 1. System Architecture

AeroMind Ops connects two intelligent systems through a unified MLOps and LLMOps infrastructure:
1. **Deep Temporal Prognostics System:**
   - **Baseline LSTM:** 2-layer stacked recurrent network with input projection.
   - **Temporal Convolutional Network (TCN):** Dilated causal 1D convolutions with residual blocks.
   - **Temporal Transformer:** Multi-head self-attention with sinusoidal positional encoding.
   - **Condition-Aware Multi-Task Transformer (Proposed Research Architecture):** Shared temporal encoder concurrently driving:
     - *RUL Regression Head:* Continuous point estimate $\hat{y}_{RUL}$.
     - *Classification Head:* Ordinal degradation stages (Healthy / Degraded / Critical).
     - *Quantile Uncertainty Head:* Pinball loss estimates ($q_{0.10}, q_{0.50}, q_{0.90}$).
2. **LLM-Powered Maintenance Assistant:**
   - Grounded RAG indexing NASA research literature (`Damage Propagation Modeling.pdf`) and C-MAPSS engineering manuals.
   - Tool execution (`get_engine_health`, `get_sensor_history`, `search_technical_docs`, `generate_maintenance_report`).
   - Production Guardrails: Mandatory source citation, numerical consistency verification against verified telemetry APIs, and explicit out-of-domain abstention.
3. **Enterprise Platform Infrastructure:**
   - FastAPI microservice architecture.
   - Experiment tracking (`AeroMind_RUL_Research`) & Model Registry (`AeroMind-RUL-Model`).
   - CI/CD quality gates & Docker Compose orchestration.

---

## 2. Directory Structure

```
mlops+dl/
├── aeromind/
│   ├── agent/           # Tool-augmented Maintenance Copilot & Guardrails
│   ├── api/             # FastAPI gateways, endpoints, and Pydantic schemas
│   ├── core/            # Config, constants, and system settings
│   ├── data/            # Ingestion, validation, target engineering, sliding window
│   ├── explainability/  # Sensor ablation & Integrated Gradients attribution
│   ├── llmops/          # Versioned prompts, 150-Q benchmark dataset, eval gate
│   ├── mlops/           # MLflow experiment tracker & model registry
│   ├── models/          # LSTM, TCN, Transformer, Multi-Task Transformer
│   └── training/        # Trainer, Huber/Pinball/NASA loss, metrics
├── configs/             # Configuration definitions
├── data/
│   ├── raw/             # C-MAPSS raw datasets (FD001 - FD004) & reference PDFs
│   └── processed/       # Scaler parameters, validation reports, benchmark sets
├── docker/              # Dockerfile, Compose, Prometheus configs
├── docs/                # Technical specification and architecture documentation
├── models/checkpoints/  # PyTorch model checkpoints and training histories
├── tests/               # 13 automated unit, data, model, RAG, and integration tests
├── dvc.yaml             # DVC data pipeline definition
└── requirements.txt     # Locked production dependencies
```

---

## 3. Quickstart & Reproduction Guide

### 3.1 Environment Setup
```powershell
pip install -r requirements.txt
```

### 3.2 Run Data Engineering Pipeline
Validates schemas, computes piecewise linear RUL ($RUL_{i,t} = \min(T_i - t, 125)$), splits engines with zero cycle leakage, and fits scalers:
```powershell
python -m aeromind.data.pipeline --dataset FD001 --window-size 30 --max-rul 125
```

### 3.3 Train Baseline & Research Models
Train the baseline LSTM (EXP-01) or Multi-Task Transformer (EXP-04) and register checkpoints in the model registry:
```powershell
# Train LSTM Baseline
python -m aeromind.training.experiments --exp EXP-01 --epochs 10

# Train Multi-Task Transformer
python -m aeromind.training.experiments --exp EXP-04 --epochs 15
```

### 3.4 Run the Automated Test Suite
Executes all 13 unit, data integrity, model, RAG, and API integration tests:
```powershell
python -m pytest tests -v
```

### 3.5 Run LLMOps Quality Gate Evaluation
Evaluates candidate prompts across the 150-question benchmark dataset:
```powershell
python -m aeromind.llmops.evaluation
```

### 3.6 Launch FastAPI Production Backend
```powershell
python -m uvicorn aeromind.api.main:app --host 0.0.0.0 --port 8000
```
Interactive OpenAPI documentation will be accessible at `http://localhost:8000/docs`.

### 3.7 Launch Containerized Infrastructure
```powershell
cd docker
docker-compose up -d
```
- FastAPI Backend: `http://localhost:8000`
- MLflow Server: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

---

## 4. Key Performance Results (FD001 Baseline)

| Architecture | Test MAE | Test RMSE | Test NASA Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **EXP-01: Baseline LSTM** | **10.00** | **13.54** | **337.31** | **Champion (Promoted)** |
| **EXP-04: Multi-Task Transformer** | Target $\le 12.0$ | Target $\le 15.0$ | Target $\le 450.0$ | Candidate |

---

## 5. License & Attribution
- Built on NASA Ames Prognostics Center of Excellence C-MAPSS Turbofan Simulation Data.
- References: Saxena et al. (2008), *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*.

