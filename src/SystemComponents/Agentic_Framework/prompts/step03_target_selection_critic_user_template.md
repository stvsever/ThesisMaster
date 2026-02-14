Review this Step-03 output candidate and return a guardrail review JSON.

Stage: $STAGE
Pass threshold (0-1): $PASS_THRESHOLD

Evidence bundle:
$EVIDENCE_BUNDLE_JSON

Requirements:
- Provide weighted_subscores_0_1 using these keys exactly:
  - reasoning_quality
  - evidence_grounding
  - readiness_feasibility_alignment
  - bfs_policy_adherence
  - ontology_alignment
- Provide feedback_for_revision if pass_decision is REVISE.
- Keep all scores in [0,1].
