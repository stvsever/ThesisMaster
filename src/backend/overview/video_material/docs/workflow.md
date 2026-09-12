# Workflow fidelity

The demonstration follows the root README, its architecture image, the integrated pipeline stage registry and the current shared candidate-selection code. The architecture image is read together with the newer runtime descriptions: for example, complaint decomposition and its local critic precede hybrid ontology retrieval, rather than being reduced to retrieval alone.

## Source map

Paths below are repository-relative. Links open their corresponding files from this document.

| Source                                                                                 | What it establishes                                                              |
| -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| [Root README](../../../../../README.md)                                                | Full stage map, ontology roles, runtime nuances and cycle carry-over             |
| [Root architecture image](../../create_flowchart.png)                                  | Actor-critic stages, HUA, EMA and iterative memory                               |
| [Agentic framework](../../../SystemComponents/Agentic_Framework/README.md)             | The five generative stages and their critics                                     |
| [Integrated pipeline](../../../../../evaluation/integrated_pipeline/README.md)         | Core stages, communication, next-cycle input and separate research support       |
| [HUA](../../../SystemComponents/Hierarchical_Updating_Algorithm/README.md)             | Readiness, time-series analysis, momentary impact and breadth-first selection    |
| [Candidate refinement source](../../../utils/agentic_core/shared/target_refinement.py) | Exact domain grouping, three selection phases, weighted scoring and model fusion |
| [PHOENIX ontology](../../../SystemComponents/PHOENIX_ontology/README.md)               | CRITERION, PREDICTOR, PERSON, CONTEXT and HAPA roles                             |
| [Portable snapshot](../data/snapshot.json)                                             | The actual tree, selected leaf paths and source fingerprints used in these films |

## Engine sequence

1. Start from the free-text complaint with separate person and context information.
2. Decompose the complaint into atomic, grounded concepts. Review coverage, overlap, granularity and actionability. Retrieve CRITERION paths through hybrid matching. Live adjudication can leave uncertain items `UNMAPPED`.
3. Construct an initial predictor-criterion observation model with HyDE-informed retrieval. Critic review checks grounding, continuity, ontology constraints and evidence quality.
4. Use the approved model to define EMA measurements. Collect repeated within-person observations. The demonstration uses fictional, deterministic traces.
5. Classify data readiness. Actual selection considers stationarity, collinearity, effective sample size and an analysis execution plan. Low readiness does not justify drawing an estimated temporal network.
6. Apply the appropriate analysis. Methods can range from descriptives and correlations to reduced baselines, stationary gVAR or time-varying gVAR, conditional on the actual diagnostic plan.
7. Quantify predictor impact. The code describes predictive error change and coefficient magnitude components. Association and predictive impact do not establish causality.
8. Explore eligible predictor candidates across domains before refining them by score.
9. Review candidate targets against evidence, lineage, suitability, safety, feasibility and personal preferences.
10. Update the observation model using bounded readiness-dependent fusion. Step 03 and Step 04 are co-located in parts of the implementation, while remaining distinct reasoning stages.
11. Translate selected targets through HAPA motivation/volition, barrier identification, coping strategies and phase-appropriate delivery. Review the proposed intervention with a critic.
12. Communicate the proposal and retain output lineage, selections and cycle history. Seed the next collection and analysis cycle from the updated model. Research reporting and impact visualisation remain support activities.

## Three-phase selection

The relevant runtime unit is a secondary domain key, formed from the first two path segments, such as `BIO / Sleep_Circadian_and_Restoration`. It is more specific than merely alternating three BIO, PSYCHO and SOCIAL roots.

The visual implementation follows these allocation rules:

1. Select the best-scoring candidate from each eligible secondary domain, subject to the candidate budget.
2. Rotate across domains until the selected set reaches `min(budget, 3 × number_of_domains)` or no candidates remain.
3. Fill remaining slots by descending global score, without repeating paths.

The score shown is:

```text
0.45 × mapping
+ 0.25 × HyDE relevance
+ 0.20 × idiographic anchor
+ 0.10 × domain bonus
```

The demonstration computes this weighted combination from authored fixture values. It does not execute the runtime embedding, mapping, HyDE generation or impact-estimation steps. The pool includes the eight featured case candidates and deterministic background samples from their actual secondary domains. The same pool and upper budget apply to both comparison panels.

The constrained comparators retain a particular PSYCHO secondary branch. They may use fewer slots if that branch contains fewer candidates in the shared pool. The integrative comparator takes the top candidates globally and can achieve the same domain or criterion coverage as PHOENIX. None of these counts measure treatment benefit.

## Readiness and fusion

The engine formula is:

```text
personal_weight = clamp(0.30 + 0.50 × readiness / 100, 0.30, 0.80)
population_weight = 1 - personal_weight
```

The studio's tier breakpoints at 35, 55 and 72 are explicitly illustrative interface thresholds. They are not a reproduction of the runtime readiness classifier. The preference for time-varying gVAR is case-specific and still presented as conditional on diagnostic support.

For the sparse Maya and Elias defaults, candidate scoring uses no simulated idiographic impact anchor. The Elias scene displays no estimated network and marks network impact as unavailable. Fusion still visualises the bounded source formula, which can allocate personal weight to non-network within-person information. Its rank-change bars are authored illustrations, not fitted output.

Changing cycle number perturbs synthetic anchors in either direction. It demonstrates retained evidence and re-ranking without scripting an improvement trajectory. It does not simulate treatment response.

## HAPA and human-facing output

The displayed barrier coefficients follow the documented formula:

```text
0.60 × predictor
+ 0.20 × person profile
+ 0.15 × context
+ 0.05 × complaint
```

The case plans are authored communication examples. The on-stage phone is a delivery preview. Its visible button is part of the film image, while the studio's actual controls remain outside the canvas. Each case retains its own obstacle, feasible cue, action, coping alternative, support route and monitoring question.

## Ontology extraction

`scripts/build_data.py` reads the raw aggregated CRITERION and PREDICTOR trees and the PERSON, CONTEXT and HAPA sources. It requires every authored mapping query to identify exactly one leaf. It also verifies each highlighted span against the exact complaint text. The resulting snapshot contains all PREDICTOR branches and source hashes; only selected CRITERION paths are copied into case records.

The raw aggregated CRITERION ontology contains 24,280 leaves. PREDICTOR contains 7,265 leaves, with path lengths from 4 to 10 segments. Root BIO/PSYCHO/SOCIAL segments count as level 1. The central PREDICTOR logo is a visual root and does not add a level to those counts.
