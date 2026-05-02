<div align="center">

  <h1>Enhancing Translational Abilities of Longitudinal Mental Health Applications: An Adaptive Approach to Idiographic Modelling by Leveraging Ontology-based Agentic AI</h1>

  <h2>🐦‍🔥 PHOENIX Engine</h2>

  <p>The PHOENIX engine conceptualises mental health support as a closed-loop workflow that iteratively optimizes the digital intervention proposal based on multi-modal data from previous collection cycles.</p>

  <p>(Personalized Hierarchical Optimization Engine for Navigating Insightful eXplorations)</p>

  <p>
    <a href="#"><img src="https://img.shields.io/badge/Type-Software_Tool-4f46e5.svg?style=flat-square" alt="Software Tool" /></a>
    <a href="#"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg?style=flat-square" alt="License GPL v3" /></a>
    <a href="#"><img src="https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=flat-square&logo=docker&logoColor=white" alt="Docker Ready" /></a>
    <a href="#"><img src="https://img.shields.io/badge/Ghent_University-Master's_Thesis-007a64.svg?style=flat-square" alt="Ghent University Master's Thesis" /></a>
  </p>

</div>

---

## 📋 Table of Contents

- [🏛️ Academic Context](#-academic-context)
- [🧭 PHOENIX Scope](#-phoenix-scope)
- [🔁 End-to-End Stage Map](#-end-to-end-stage-map)
- [🐦‍🔥 PHOENIX Ontology](#-phoenix-ontology-with-llm-based-mappings)
- [🏗️ Technical Architecture](#-technical-architecture)
- [🚀 Quick Setup](#-quick-setup-of-phoenix-engine)
- [🗂️ Repository Structure](#-repository-structure)
- [🐳 Docker](#-docker)
- [💻 Run from CLI](#-run-phoenix-from-cli)
- [🖥️ Run from Frontend](#-run-phoenix-from-frontend)
- [📦 Outputs and Validation](#-outputs-and-validation-targets)
- [📊 PHOENIX Engine Evaluation Framework](#-phoenix-engine-evaluation-framework)
- [✅ Quality Assurance and CI/CD](#-quality-assurance-and-cicd)
- [📜 License](#-license)

---

## 🏛️ Academic Context

This research-grade software is being created for a Ghent University **master's thesis** that aims to enhance the clinical translation abilities of longitudinal **mental health applications**: toward an adaptive approach for **idiographic modelling** by using **ontology-based multi-agentic workflows**.

| **Field** | **Value** |
|---|---|
| **Institution** | Ghent University |
| **Author** | Stijn Van Severen |
| **Supervisors** | Geert Crombez, Annick De Paepe |

---

## 🧭 PHOENIX Scope

PHOENIX separates two concerns:

- **Core engine flow**: clinical/analytic decision flow from intake to iterative model carry-over.
- **Research support flow**: visualization, QA, and research reporting for validation and communication.

This separation keeps scientific validation transparent without mixing support tasks into core decision logic.

---

## 🔁 End-to-End Stage Map

PHOENIX is a modular, multi-agent system that starts from free-text complaints, builds an initial observation model, analyzes time-series dynamics, proposes targets/interventions, and packages iterative updates for the next cycle.

![PHOENIX engine — Multi Agent System Architecture](src/backend/overview/create_flowchart.png)

---

## 🐦‍🔥 PHOENIX Ontology with LLM-based Mappings

The following ontology was developed to support the PHOENIX engine's reasoning and decision-making processes.

![PHOENIX Aggregated Ontology](src/backend/overview/PHOENIX_ontology.png)

---

## 🏗️ Technical Architecture

### Five PHOENIX Ontologies

All stages are constrained by five stable ontologies that enforce structural guarantees across the full pipeline:

| Ontology | Role | Source |
|---|---|---|
| **CRITERION** | Operationalized mental health variables (DSM-5-TR, RDoC) | `src/backend/SystemComponents/PHOENIX_ontology/separate/CRITERION/` |
| **PREDICTOR** | Hierarchically structured treatment-solution entities used to model actionable intervention pathways and candidate refinement space | `src/backend/SystemComponents/PHOENIX_ontology/separate/PREDICTOR/` |
| **PERSON** | Individual characteristics (demographics, comorbidity, history) | `src/backend/SystemComponents/PHOENIX_ontology/separate/PERSON/` |
| **CONTEXT** | Situational and environmental factors | `src/backend/SystemComponents/PHOENIX_ontology/separate/CONTEXT/` |
| **HAPA** | Health Action Process Approach (barriers, coping, phases) | `src/backend/SystemComponents/PHOENIX_ontology/separate/HAPA/` |

### Runtime Multi-Agent Design

In the real integrated pipeline, all decision-making stages use live LLM reasoning in normal operation, but they do so with different scaffolds. Step 01 combines always-on complaint decomposition, a local LLM critic loop, hybrid ontology retrieval, and optional final leaf adjudication, while later stages use explicit actor-critic loops with bounded iterations and heuristic fallback only for deterministic or degraded runs.

| Integrated Step | Runtime Component | Live LLM Use | Critic Dimensions | Core Runtime Method |
|---|---|---|---|---|
| 01 | Complaint Operationalization Agent | Yes, always-on for free-text decomposition and local critic review; optional again for final leaf adjudication | schema_validity, coverage_grounding, atomicity_nonoverlap, granularity_fit, current_actionability | LLM decomposition + local LLM critic refinement + HTSSF retrieval (dense + BM25 + token overlap + fuzzy) |
| 02 | Initial Observation Model Constructor | Yes, real-time HyDE generation and structured model construction | predictor_grounding, criterion_continuity, ontology_strictness, evidence_quality | HyDE-based predictor RAG + actor-critic refinement |
| 03 | Treatment Target Identifier | Yes, real-time structured actor output when enabled | safety, domain_boundary, lineage_consistency | BFS candidate selector + idiographic-nomothetic fusion |
| 04 | Updated Observation Model Constructor | Yes, real-time structured actor output when enabled | safety, domain_boundary, lineage_consistency | BFS-guided hierarchical model update with ontology-constrained refinement |
| 05 | HAPA Intervention Mapper | Yes, real-time structured intervention generation when enabled | reasoning_quality, evidence_grounding, hapa_consistency, medical_safety | Barrier scoring: 0.60·predictor + 0.20·profile + 0.15·context + 0.05·complaint |

Runtime interpretation:

- **Step 01** is not just retrieval: it always begins with LLM-based complaint decomposition, with no non-LLM fallback for that decomposition phase, and then runs a local structured LLM critic loop to review complaint coverage, granularity, overlap, and present-state actionability before mapping to ontology leaves.
- **Step 01** still uses hybrid retrieval for ontology grounding, and can optionally run an additional LLM adjudication pass to choose the final leaf or return `UNMAPPED`.
- **Steps 02, 03, 04, and 05** use live LLM calls in normal operation, with component-specific actor-critic prompts, bounded iterations, and heuristic fallback paths for deterministic or degraded runs.
- The intervention module is **Step 05** in the actual integrated pipeline, even if some earlier summaries compressed it into a four-stage abstraction.

**Optional DAG orchestrator** (`src/backend/orchestrator.py`): for complex tasks, a flexible orchestrator creates DAG-based parallel/sequential execution plans — otherwise the pipeline runs sequentially (primary evaluation path).

### Hierarchical Updating Algorithm (HUA)

Quantitative backbone bridging EMA data to adaptive model weighting:

1. **Readiness classifier** — stationarity (ADF/KPSS), collinearity, effective sample size → tier selection (tv-gVAR / gVAR / GGM / correlation / descriptives)
2. **Network time-series analyst** — kernel-smoothed VAR(1), L1-penalized stationary gVAR, partial correlations (Ledoit-Wolf shrinkage), time-varying GIF animations
3. **Momentary impact quantifier** — leave-one-predictor-out MSE delta + coefficient magnitude composite
4. **BFS candidate selector** — `score = 0.45·mapping + 0.25·HyDE + 0.20·idiographic_anchor + 0.10·domain_bonus`

**Adaptive idiographic-nomothetic weighting** per cycle:
```
idiographic_weight = clamp(0.30 + 0.50 × readiness_score / 100)
nomothetic_weight  = 1.0 - idiographic_weight
```

### Iterative Cycle Design

PHOENIX implements a breadth-first iterative algorithm across cycles:

1. **Cycle N** produces: criterion leaf, initial model, pseudodata, HUA results, treatment targets, HAPA intervention
2. **Cycle N+1** seeds from Cycle N via a history ledger: impact scores → `idiographic_anchor` in BFS; prior cycle scores modulate `domain_bonus`; `composite_score = 0.35·similarity + 0.25·impact[N] + 0.15·target_scores + 0.10·priority_scores + 0.15·quality_scores`

---

## 🚀 Quick Setup of PHOENIX engine

### 1. Clone repository

```bash
git clone https://github.com/stvsever/ThesisMaster.git
cd MASTERPROEF
```

### 2. Create Python environment (3.11+)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure `.env` for LLM-enabled runs

Create or update `.env` in repository root:

```bash
OPENROUTER_API_KEY=<your_openrouter_key>
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

Runtime behavior:
- `OPENROUTER_API_KEY` is primary.
- Runtime mirrors it to `OPENAI_API_KEY` for backward-compatible scripts.
- Default model is `gpt-5-nano` (resolved as `openai/gpt-5-nano` when routed via OpenRouter).

### 4. Optional smoke validation

If you want to quickly validate the integrated pipeline on a single profile with minimal iterations, you can run the smoke test:

```bash
make pipeline-smoke
```

---

## 🗂️ Repository Structure

A client-side graph creator (GitNexus) was used to generate a comprehensive knowledge graph of the entire codebase; its component interactions are provided below:

<div align="center">
  <img src="src/backend/overview/gitnexus_overview.png" alt="PHOENIX GitNexus Codebase Graph" width="800" />
</div>

The main codebase is organized around `src/` and `evaluation/`. Inside `src/`, the canonical runtime split is now `src/frontend/` for the Flask application and `src/backend/` for the engine, ontologies, shared runtime utilities, and architecture assets.

```text
MASTERPROEF/
├── src/                            # Canonical application source tree
│   ├── backend/                      # Engine runtime, SystemComponents, utils, orchestrator, overview assets
│   ├── frontend/                     # Flask app, UI routes, runtime workspace integration
│   ├── __init__.py                   # Package root for `src.frontend` and `src.backend`
│   └── README.md                     # Architecture overview for the `src/` tree
├── evaluation/                     # Sequential scripts + integrated pipeline + QA/research
│   ├── sequential/                    # Stage-wise run_step.py scripts (00..08)
│   ├── integrated_pipeline/           # run_pipeline.py and run_engine_pipeline.py
│   ├── survey_analysis/               # 6-study evaluation framework with analysis scripts
│   └── quality_and_research/          # pytest suites, schema contracts, research reporting
├── docker/                         # Dockerfile + docker-compose for reproducible deployment
├── .github/                        # CI/CD workflows
├── pyproject.toml                  # Python package metadata and constraints
├── requirements.txt                # Dependency baseline
└── README.md                       # Root documentation
```

---

## 🐳 Docker

PHOENIX ships with a ready-to-use Docker configuration for reproducible execution:

```bash
git clone https://github.com/stvsever/ThesisMaster.git
cd MASTERPROEF

# Optional for LLM-enabled runs; deterministic mode can skip this.
cat > .env <<'EOF'
OPENROUTER_API_KEY=<your_openrouter_key>
OPENAI_BASE_URL=https://openrouter.ai/api/v1
EOF

cd docker
docker compose up --build
```

This starts the Flask frontend on [http://127.0.0.1:5050](http://127.0.0.1:5050). The Docker setup bundles the project dependencies, mounts integrated-pipeline outputs back to the host, and also supports CLI runs through the `phoenix-cli` service. See [docker/README.md](./docker/README.md) for the full workflow.

---

## 💻 Run PHOENIX from CLI

### A. Standard integrated run

The following command executes the full PHOENIX pipeline with default settings, processing the synthetic_v1 dataset through all stages and generating comprehensive outputs:

```bash
python evaluation/integrated_pipeline/run_pipeline.py --mode synthetic_v1
```

### B. Single profile selection

The following command runs the pipeline on the `synthetic_v1` dataset but limits the execution to a single profile matching the pattern `pseudoprofile_FTC_ID001`. This allows for focused testing and debugging on a specific case:

```bash
python evaluation/integrated_pipeline/run_pipeline.py --mode synthetic_v1 \
  --pattern pseudoprofile_FTC_ID001 \
  --max-profiles 1
```

### C. Iterative run (2 cycles)

The following command executes the PHOENIX pipeline for 2 complete cycles, allowing you to observe how the system iteratively refines its outputs based on previous cycle data. The `--profile-memory-window 3` flag enables the system to retain information from the last 3 profiles for informed decision-making in subsequent cycles:

```bash
python evaluation/integrated_pipeline/run_pipeline.py --mode synthetic_v1 \
  --cycles 2 \
  --profile-memory-window 3
```

### D. Deterministic mode (no LLM)

```bash
python evaluation/integrated_pipeline/run_pipeline.py --mode synthetic_v1 --disable-llm
```

Runtime note:
- If a cycle is `readiness_aligned` and only contemporaneous correlation analysis is feasible, PHOENIX now applies a correlation-baseline impact fallback so downstream Step-03/04/05 and communication stages still execute and persist outputs.
- If Step-02 model generation fails (for example provider/network failure), PHOENIX now builds complaint-grounded fallback Step-02 artifacts directly from Step-01 operationalization output, instead of copying unrelated historical profile artifacts.
- For iterative cycles started via `--start-from-pseudodata`, PHOENIX now resolves `initial_model_runs_root` from the active run lineage (same run id) so Step-03/04 stay anchored to the current cycle history.

---

## 🖥️ Run PHOENIX from Frontend

Use the following command to start the Flask frontend:

```bash
python src/frontend/app.py
# or
python evaluation/integrated_pipeline/run_pipeline.py --ui
```

Open [http://127.0.0.1:5050](http://127.0.0.1:5050).

Frontend provides:
- Intake for complaint/person/environment context
- Live component status and streaming logs
- One-click full end-to-end run from free-text complaint (with iterative cycle controls)
- Step-level run controls and advanced configuration toggles
- Wizard-style iterative execution: INTAKE → MODEL → DATA → ANALYSIS → INTERVENTION → MODEL (cycle N+1)
- Interactive Chart.js dashboard — all visualizations are dynamic (no static PNGs in UI)
- Canvas-based animated network visualization with per-frame scrubbing
- Session persistence and cohort batch execution

---

## 📦 Outputs and Validation Targets

Integrated outputs are saved under:

```text
evaluation/integrated_pipeline/runs/<run_id>/
```

Key artifacts to inspect:
- `00_operationalization/` through `10_research_reports/`
- `pipeline_summary.json`
- `llm_startup_health_check.json`
- Stage logs (`stage.log`, `stage_events.jsonl`, `stage_trace.json`)
- Profile-specific JSON/CSV outputs per step
- Profile-specific human-readable summaries:
  - `07_hapa_digital_intervention/<profile_id>/step05_hapa_intervention.md`
  - `08_treatment_translation_communication/<profile_id>/treatment_translation_communication.md`
- Time-varying network animation: `04_time_series_analysis/<profile_id>/tv_network_animation.gif`
- Publication-ready PNGs: `09_impact_visualizations/<profile_id>/` (for human healthcare expert comparison)

---

## 📊 PHOENIX Engine Evaluation Framework

The evaluation framework assesses PHOENIX output quality across five clinical tasks using a double-blind LLM-as-judge design. For each of 10 clinical cases, PHOENIX and HCP outputs are independently rated on a bipolar −10 to +10 absolute quality scale across 38 dimensions by a `google/gemini-3.1-flash-lite-preview` judge in three separate runs — without knowledge of source identity — producing 2,340 quality ratings. Effects are quantified via linear mixed-effects models (`quality_score ~ entity + (1|case) + (1|judge_run)`), standardized as Cohen's dz, and tested for equivalence against a ±1.5-point margin.

![PHOENIX Evaluation Workflow Overview](evaluation/survey_analysis/llm_as_judge/overview/evaluation_overview.png)

The `evaluation/survey_analysis/` directory implements a five-part double-blind evaluation of PHOENIX against 10 licensed HCPs (one per case). Both sources receive identical Qualtrics-derived inputs and complete the same five clinical tasks; outputs are rated anonymously by an LLM judge across 38 dimensions on a bipolar −10 to +10 scale.

| Part | Task | Dimensions |
|---|---|---:|
| 1 | Symptom labelling | 7 |
| 2 | Modifiable treatment options | 8 |
| 3 | Treatment target ranking (network + EMA) | 7 |
| 4 | EMA item selection (6 items, 2 per goal) | 8 |
| 5 | Personalised mobile coaching message | 9 |

Per-part effects are estimated with `quality_score ~ entity_ec + (1|case_id) + (1|judge_run)` (PHOENIX = +0.5, HCP = −0.5), standardized as Cohen's dz, and tested for equivalence against a ±1.5-point margin (TOST). A cross-part holistic synthesis and supplementary ICC, calibration, and sensitivity diagnostics are run automatically.

Run the full pipeline:

```bash
export OPENROUTER_API_KEY=...
python evaluation/survey_analysis/pipeline.py --mode real --judge openrouter --n-runs 3
```

Results are saved under `evaluation/survey_analysis/results/` with publication-ready figures and statistical reports.

---

## ✅ Quality Assurance and CI/CD

Run locally:

```bash
make qa-unit
make qa-integration
make qa-smoke
make qa-all
```

Automated workflows:
- `.github/workflows/ci.yml`
- `.github/workflows/smoke_pipeline.yml`

Schema/contract validation entrypoint:
- `evaluation/quality_and_research/quality_assurance/validate_contract_schemas.py`

**Contract validation**: 7 JSON schemas enforce structural guarantees on every stage output: `readiness_report`, `network_comparison_summary`, `momentary_impact`, `step03_target_selection`, `step04_updated_model`, `step05_hapa_intervention`, `pipeline_summary`.

---

## 📜️ License

This project is licensed under **GNU General Public License v3.0**. See [`LICENSE`](./LICENSE).

What this means in practice:
- You may **use, study, modify, and redistribute** this code.
- If you distribute modified versions (or software that includes GPL-covered parts), you must:
  - keep it under GPL-compatible terms,
  - provide corresponding source code,
  - preserve copyright and license notices,
  - document meaningful changes.
- The software is provided **without warranty**.

For academic reuse, cite the thesis context appropriately and keep provenance of methodological changes explicit.

> [!CAUTION]
> **EU MDR / PRE-CLINICAL DISCLAIMER**
> PHOENIX is a **Clinical Decision Support System (CDSS) prototype** designed for research purposes. It is **NOT** a certified medical device under the EU Medical Device Regulation (MDR 2017/745) or FDA guidelines. Do not use for primary diagnostic decisions. All outputs must be verified by a qualified clinician.
