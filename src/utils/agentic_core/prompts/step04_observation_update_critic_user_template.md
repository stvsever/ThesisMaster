Review this Step-04 output candidate and return a guardrail review JSON.

Stage: $STAGE
Pass threshold (0-1): $PASS_THRESHOLD

Evidence bundle:
$EVIDENCE_BUNDLE_JSON

Requirements:
- Provide weighted_subscores_0_1 using these keys exactly:
  - predictor_grounding
  - criterion_continuity
  - bfs_depth_balance
  - fusion_consistency
  - feasibility_alignment
- Provide feedback_for_revision if pass_decision is REVISE.
- Keep all scores in [0,1].
