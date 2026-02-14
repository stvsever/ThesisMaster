Review this Step-05 intervention candidate and return a guardrail review JSON.

Pass threshold (0-1): $PASS_THRESHOLD

Evidence bundle:
$EVIDENCE_BUNDLE_JSON

Requirements:
- Provide weighted_subscores_0_1 with these keys exactly:
  - reasoning_quality
  - evidence_grounding
  - hapa_consistency
  - medical_safety
  - personalization_context_fit
  - regulatory_ethical_alignment
  - intervention_feasibility
- Provide concrete feedback_for_revision when pass_decision is REVISE.
- Keep all scores in [0,1].
