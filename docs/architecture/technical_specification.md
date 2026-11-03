# AeroMind Ops — Technical Specification and Architecture Document

**System Version:** `1.0.0-PROD-SPEC`  
**Domain:** Predictive Maintenance & Prognostics Health Management (PHM) for Turbofan Engines  
**Data Foundation:** NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)  
**Author:** AeroMind Ops Lead AI Systems Engineer  

---

## 1. Executive Summary & Problem Formulation

Modern aerospace operations mandate strict zero-unplanned-downtime tolerance. Turbofan engines operate under extreme thermal, mechanical, and aerodynamic stresses, causing continuous, non-linear degradation of sub-assemblies (e.g., High-Pressure Compressor (HPC), Fan, Low-Pressure Turbine (LPT)). 

**AeroMind Ops** is a dual-engine intelligent PHM platform designed to:
1. **Accurately Predict Remaining Useful Life (RUL)** and degradation phases before catastrophic failure using deep temporal neural architectures under varying operating conditions and sensor noise.
2. **Quantify Prediction Uncertainty** using quantile regression to establish confidence bounds critical for risk-averse aviation safety thresholds.
3. **Provide Explainable Attributions** identifying which sensor trends and physical subsystems drove the degradation assessment.
4. **Deliver an LLM-Powered Maintenance Copilot** with Retrieval-Augmented Generation (RAG) and tool execution capabilities, enabling maintenance engineers to query engine status, compare fleet trends, diagnose root causes, and retrieve grounded operating manuals with strict anti-hallucination guardrails.
5. **Enforce Enterprise MLOps and LLMOps Lifecycles**, ensuring end-to-end data versioning (DVC), experiment tracking and model registry (MLflow), prompt versioning, automated evaluation gates, and containerized deployment.

---

## 2. Functional Requirements

### 2.1 Deep Learning System (RUL & Degradation Engine)
- **FR-DL-01 (Multi-Dataset Ingestion):** Support NASA C-MAPSS sub-datasets FD001 (single condition, single fault), FD002 (six conditions, single fault), FD003 (single condition, dual fault), and FD004 (six conditions, dual fault).
- **FR-DL-02 (Leakage-Free Preprocessing):** Execute engine-level dataset splitting. Window generation and scaling parameters must strictly derive from training engines without temporal leakage.
- **FR-DL-03 (Piecewise Linear Target Engineering):** Clip RUL targets at $RUL_{max} = 125$ cycles to reflect physical plateauing during healthy run-in periods.
- **FR-DL-04 (Multi-Architecture Suite):** Implement and evaluate four architectures:
  1. *Baseline LSTM:* 2-layer recurrent network with input projection and dropout.
  2. *Temporal Convolutional Network (TCN):* Dilated causal 1D convolutions with residual blocks.
  3. *Temporal Transformer:* Multi-head self-attention with sinusoidal positional encoding.
  4. *Condition-Aware Multi-Task Transformer (Proposed Research Contribution):* Shared encoder with condition embedding driving three heads simultaneously:
     - Regression Head: Point estimate $\hat{y}_{RUL}$.
     - Classification Head: Degradation stage (Healthy / Warning / Critical).
     - Quantile Head: Uncertainty bounds ($q_{0.10}, q_{0.50}, q_{0.90}$).
- **FR-DL-05 (Sensor Explainability):** Provide sensor ablation impact metrics and integrated gradients attributing RUL predictions to physical sensors over time windows.

### 2.2 LLM-Powered Maintenance Assistant (Copilot & RAG)
- **FR-LLM-01 (Modular Inference):** Decoupled inference client supporting local models (Ollama Qwen2.5-3B-Instruct) and enterprise LLM APIs.
- **FR-LLM-02 (Domain Technical Grounding):** Ingestion and vector indexing of NASA prognostic papers, C-MAPSS documentation, and maintenance guidelines into vector storage (Qdrant).
- **FR-LLM-03 (Tool-Augmented Orchestration):** Copilot agent capable of autonomous tool execution:
  - `get_engine_health(engine_id)`
  - `get_sensor_history(engine_id, window)`
  - `search_technical_docs(query)`
  - `generate_maintenance_report(engine_id)`
- **FR-LLM-04 (Safety & Guardrails):** Mandatory citation of retrieved documents, strict verification against verified numerical prediction APIs, rejection of prompt injection, and explicit abstention when technical documentation is absent.

### 2.3 Backend & Integration Platform
- **FR-BE-01 (FastAPI Service Tier):** RESTful endpoints for real-time inference, fleet inventory, historical analytics, copilot conversations, and report downloads.
- **FR-BE-02 (Data Persistence):** Relational storage (PostgreSQL/SQLite) for fleet metadata, sensor history, prediction audits, and user interactions; Vector storage (Qdrant) for technical documentation chunks.
- **FR-BE-03 (Security & Isolation):** Zero execution privilege from LLM to write/alter model weights or database schemas.

---

## 3. High-Level System Architecture

See `aeromind_technical_specification.md` in the system artifacts for full interactive Mermaid diagrams and data flow topologies.

---

## 4. Formal Data Schemas

### 4.1 Raw C-MAPSS Stream Schema
Space-delimited text files consisting of 26 numerical fields:
```
[unit_nr] [time_cycles] [setting_1] [setting_2] [setting_3] [s_1] [s_2] ... [s_21]
```

| Field Name | Type | Unit / Range | Description |
| :--- | :--- | :--- | :--- |
| `unit_nr` | Integer | $1 \dots N$ | Engine serial / trajectory identifier |
| `time_cycles` | Integer | $1 \dots T_i$ | Discrete operating cycle timestamp |
| `setting_1` | Float | Altitude (0 to 42,000 ft) | Flight operating condition parameter 1 |
| `setting_2` | Float | Mach number (0.0 to 0.84) | Flight operating condition parameter 2 |
| `setting_3` | Float | Throttle resolver (60 to 100) | Flight operating condition parameter 3 |
| `sensor_1` to `sensor_21` | Float | Physical sensors | Temperatures, pressures, fan/core speeds, fuel flows |

### 4.2 Constant vs Degraded Sensor Analysis (FD001 Baseline)
In single-condition environments (FD001), 7 sensors remain invariant across all operational cycles:
- **Invariant (Dropped in Preprocessing):** `sensor_1` (Total temp at fan inlet), `sensor_5` (Pressure at fan inlet), `sensor_10` (Engine pressure ratio), `sensor_16` (Burner fuel-air ratio), `sensor_18` (Demanded fan speed), `sensor_19` (Demanded corrected fan speed).
- **Informative (Selected Degradation Indicators):** 
  - `sensor_2` (LPC outlet temp)
  - `sensor_3` (HPC outlet temp)
  - `sensor_4` (LPT outlet temp)
  - `sensor_7` (HPC outlet pressure)
  - `sensor_8` (Physical fan speed)
  - `sensor_9` (Physical core speed)
  - `sensor_11` (Static pressure at HPC outlet)
  - `sensor_12` (Ratio of fuel flow to static pressure)
  - `sensor_13` (Corrected fan speed)
  - `sensor_14` (Corrected core speed)
  - `sensor_15` (Bypass ratio)
  - `sensor_17` (Bleed enthalpy)
  - `sensor_20` (HPT coolant bleed)
  - `sensor_21` (LPT coolant bleed)

### 4.3 Processed Sliding Window Tensor Contract
Each training observation is a 3-dimensional tensor tuple:
$$X_k \in \mathbb{R}^{W \times D_{features}}, \quad y_{RUL} \in \mathbb{R}, \quad y_{stage} \in \{0, 1, 2\}$$
Where:
- Window length $W = 30$ historical consecutive cycles.
- Feature dimension $D_{features} = 14$ informative sensors (plus optional 3 operational settings in multi-condition sets).
- $y_{RUL} = \min(T_{engine} - t_{window\_end}, 125)$.
- $y_{stage} \in \{0: \text{Healthy } (RUL > 80), 1: \text{Degraded } (30 < RUL \le 80), 2: \text{Critical } (RUL \le 30)\}$.

---

## 5. Model Evaluation Metrics

### 5.1 Regression Metrics
1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} | \hat{RUL}_i - RUL_i |$$
2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{RUL}_i - RUL_i)^2}$$

### 5.2 NASA Prognostics Asymmetric Scoring Function
$$d_i = \hat{RUL}_i - RUL_i$$
$$S = \sum_{i=1}^{N} s_i, \quad s_i = \begin{cases} e^{-\frac{d_i}{13}} - 1 & \text{for } d_i < 0 \text{ (Early prediction)} \\ e^{\frac{d_i}{10}} - 1 & \text{for } d_i \ge 0 \text{ (Late prediction / Severe penalty)} \end{cases}$$

### 5.3 Quantile Calibration & Uncertainty Metrics
1. **Pinball Loss (Quantile Loss) for target quantile $\tau \in \{0.10, 0.50, 0.90\}$:**
   $$\mathcal{L}_{\tau}(y, \hat{y}^{(\tau)}) = \max\left(\tau(y - \hat{y}^{(\tau)}), (\tau - 1)(y - \hat{y}^{(\tau)})\right)$$
2. **Prediction Interval Coverage Probability (PICP):**
   $$\text{PICP} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(\hat{q}_{0.10, i} \le RUL_i \le \hat{q}_{0.90, i}\right)$$

---

## 6. LLM & RAG Evaluation Metrics

| Metric | Measurement Technique | Target Threshold |
| :--- | :--- | :--- |
| **Faithfulness** | Ratio of statements in response directly supported by retrieved chunks | $\ge 0.92$ |
| **Answer Relevance** | Semantic embedding similarity between engineer prompt and generated answer | $\ge 0.88$ |
| **Context Precision** | Mean Reciprocal Rank (MRR) of relevant ground-truth chunks in retrieved top-$k$ | $\ge 0.85$ |
| **Context Recall** | Proportion of ground-truth evidence successfully retrieved in context | $\ge 0.85$ |
| **Citation Correctness** | Verification that every cited page/document explicitly contains the claimed claim | $100\%$ |
| **Abstention Rate (Out-of-Domain)** | Frequency of polite abstention when queried on ungrounded or out-of-scope topics | $\ge 0.95$ |

---

## 7. Reproducibility & Deployment Requirements

1. **Deterministic Seeding:** Master seed ($42$) pinned across PyTorch, NumPy, Python random.
2. **Data & Artifact Versioning:** Raw data referenced via DVC; Scaler parameters stored as versioned JSON artifacts alongside model weights.
3. **MLflow Tracking & Model Registry:** Experiment name `AeroMind_RUL_Research`, Model `AeroMind-RUL-Model`.
4. **LLM Observability:** Langfuse integration logging token usage, prompt version ID, latency, tool calls, and user ratings.
5. **Containerization & CI/CD:** Multi-container Docker Compose with isolated networks: API gateway, vector store, relational DB, monitoring stack.

