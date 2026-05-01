"""
Per-part dimension specifications for the LLM-as-judge.

Each :class:`Dimension` carries a stable ``key`` (used as the column value
in ``judgments_long.csv`` and as the JSON key in the judge response), a
human-readable ``display_label``, a one-paragraph ``goal_description``, a
``rationale`` for why the dimension matters at this pipeline step, and
short ``anchor_examples`` describing low / mid / high anchors on the 1..7
Likert scale.

PROMPT_VERSION is bumped whenever the prompt template, the dimension set,
or the rating scale changes; downstream judgments record this constant
verbatim so analyses can stratify by prompt version if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

PROMPT_VERSION: str = "2026-05-01-v1"

LIKERT_MIN: int = 1
LIKERT_MAX: int = 7
LIKERT_NEUTRAL: int = 4

LIKERT_ANCHOR_NAMES: Tuple[str, ...] = (
    "1 = very poor",
    "2 = poor",
    "3 = below acceptable",
    "4 = neutral / acceptable",
    "5 = good",
    "6 = very good",
    "7 = excellent",
)

PART_TITLES: Dict[str, str] = {
    "part1": "Operationalisation of mental state",
    "part2": "Initial observational model",
    "part3": "Treatment-target prioritisation",
    "part4": "Updated observational model",
    "part5": "Tailored intervention message",
}


@dataclass(frozen=True)
class Dimension:
    """One rating dimension within a part."""

    key: str
    display_label: str
    goal_description: str
    rationale: str
    anchor_examples: Dict[str, str] = field(default_factory=dict)

    def anchor_block(self) -> str:
        """Return a compact text block of anchors for use in the prompt."""
        if not self.anchor_examples:
            return "  (no extra anchor examples)"
        order = ["low", "mid", "high"]
        chunks = []
        for tier in order:
            if tier in self.anchor_examples:
                chunks.append(f"  - {tier.upper()}: {self.anchor_examples[tier]}")
        return "\n".join(chunks) if chunks else "  (no extra anchor examples)"


# ──────────────────────────────────────────────────────────────────────────────
# PART 1 — OPERATIONALISATION
# ──────────────────────────────────────────────────────────────────────────────

PART1_DIMENSIONS: List[Dimension] = [
    Dimension(
        key="clinical_accuracy",
        display_label="Clinical accuracy",
        goal_description=(
            "Each label and description names a real, recognisable clinical "
            "phenomenon and accurately reflects the symptoms reported in the "
            "vignette."
        ),
        rationale=(
            "Operationalisation is the foundation of the rest of the pipeline; "
            "wrong or hallucinated symptoms propagate into the network and "
            "monitoring plan."
        ),
        anchor_examples={
            "low": "Several invented or clinically implausible labels.",
            "mid": "Most labels are recognisable but one or two are vague.",
            "high": "Every label maps cleanly onto a textbook construct.",
        },
    ),
    Dimension(
        key="construct_interoperability",
        display_label="Construct interoperability",
        goal_description=(
            "Labels use standard clinical vocabulary (DSM-5/ICD-11 or the "
            "transdiagnostic literature) so that any other clinician would "
            "interpret them the same way."
        ),
        rationale=(
            "Common vocabulary lets the network and intervention modules be "
            "compared across patients and across clinicians."
        ),
        anchor_examples={
            "low": "Idiosyncratic phrasing only the author would use.",
            "mid": "Mixed: some standard terms, some informal labels.",
            "high": "Crisp clinical terminology throughout.",
        },
    ),
    Dimension(
        key="resolution_preservation",
        display_label="Resolution preservation",
        goal_description=(
            "The level of detail in the labels preserves nuance from the "
            "vignette without being overly broad or overly specific."
        ),
        rationale=(
            "Too coarse and signal is lost; too fine and downstream modules "
            "over-fit to one phrasing."
        ),
        anchor_examples={
            "low": "All-or-nothing categorical labels with no detail.",
            "mid": "Adequate granularity but a few details are missed.",
            "high": "Each label captures the right grain of nuance.",
        },
    ),
    Dimension(
        key="behavioural_specificity",
        display_label="Behavioural specificity",
        goal_description=(
            "Each item is operationalisable as something observable or "
            "measurable, not a purely subjective trait."
        ),
        rationale=(
            "The next pipeline step turns these items into EMA predictors; "
            "they must be observable to be measurable."
        ),
        anchor_examples={
            "low": "Items are pure trait labels with no anchor in behaviour.",
            "mid": "Mix of behavioural and trait-style items.",
            "high": "Every item is grounded in observable behaviour.",
        },
    ),
    Dimension(
        key="internal_consistency",
        display_label="Internal consistency",
        goal_description=(
            "Labels and descriptions match each other and do not contradict "
            "themselves or the vignette."
        ),
        rationale=(
            "Contradictions surface as compounded errors downstream."
        ),
        anchor_examples={
            "low": "Multiple internal contradictions.",
            "mid": "Largely consistent with one minor mismatch.",
            "high": "Fully consistent throughout.",
        },
    ),
    Dimension(
        key="completeness",
        display_label="Completeness",
        goal_description=(
            "The set of items covers all symptom domains that are present in "
            "the vignette."
        ),
        rationale=(
            "Missing domains lead to blind spots in the personalised plan."
        ),
        anchor_examples={
            "low": "Major symptom domains are missing.",
            "mid": "Covers most domains; one minor area missing.",
            "high": "Comprehensive coverage of all relevant domains.",
        },
    ),
    Dimension(
        key="conciseness_redundancy",
        display_label="Conciseness / non-redundancy",
        goal_description=(
            "No item duplicates another; each adds unique information."
        ),
        rationale=(
            "Redundant items inflate the model without improving fidelity."
        ),
        anchor_examples={
            "low": "Several near-duplicate labels.",
            "mid": "One pair of overlapping items.",
            "high": "Every item adds a unique facet.",
        },
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# PART 2 — INITIAL OBSERVATIONAL MODEL
# ──────────────────────────────────────────────────────────────────────────────

PART2_DIMENSIONS: List[Dimension] = [
    Dimension(
        key="clinical_appropriateness",
        display_label="Clinical appropriateness",
        goal_description=(
            "Each predictor is clinically meaningful for this symptom "
            "profile, not generic boilerplate."
        ),
        rationale=(
            "Predictor selection determines whether monitoring will surface "
            "actionable information."
        ),
        anchor_examples={
            "low": "Generic predictors unrelated to the case.",
            "mid": "Mostly relevant but a few generic items.",
            "high": "All predictors closely match the patient's profile.",
        },
    ),
    Dimension(
        key="network_validity",
        display_label="Network validity",
        goal_description=(
            "The predictor-criterion mapping is theoretically sound and "
            "consistent with the network-of-symptoms literature."
        ),
        rationale=(
            "The bipartite network derived from this step drives Part 3."
        ),
        anchor_examples={
            "low": "Mappings violate basic network theory.",
            "mid": "Plausible mappings with one or two doubtful edges.",
            "high": "Theoretically defensible mappings throughout.",
        },
    ),
    Dimension(
        key="ema_feasibility",
        display_label="EMA feasibility",
        goal_description=(
            "Each predictor can realistically be sampled momentarily on a "
            "phone in fewer than ~30 seconds per item."
        ),
        rationale=(
            "Predictors that cannot be measured momentarily break the "
            "monitoring assumption of the pipeline."
        ),
        anchor_examples={
            "low": "Most predictors require lab equipment or interviews.",
            "mid": "Mixed feasibility; one or two impractical predictors.",
            "high": "All predictors fit a 1-minute phone prompt budget.",
        },
    ),
    Dimension(
        key="predictor_diversity",
        display_label="Predictor diversity",
        goal_description=(
            "The predictors span cognitive, affective, behavioural, and "
            "contextual domains rather than clustering in one bucket."
        ),
        rationale=(
            "Diverse predictors maximise the chance of finding a leverage "
            "point downstream."
        ),
        anchor_examples={
            "low": "All predictors live in one domain.",
            "mid": "Two domains covered.",
            "high": "All four domains represented.",
        },
    ),
    Dimension(
        key="measurement_specificity",
        display_label="Measurement specificity",
        goal_description=(
            "The measurement schedule and criteria are concrete (units, "
            "anchors, response options) and unambiguous."
        ),
        rationale=(
            "Vague measurements propagate noise into all downstream stats."
        ),
        anchor_examples={
            "low": "Criteria are abstract or missing units.",
            "mid": "Mostly concrete with a few ambiguous fields.",
            "high": "Every criterion is fully specified.",
        },
    ),
    Dimension(
        key="intervention_potential",
        display_label="Intervention potential",
        goal_description=(
            "Predictors are modifiable behaviours or states rather than "
            "stable traits."
        ),
        rationale=(
            "Non-modifiable predictors offer no actionable handle for "
            "intervention."
        ),
        anchor_examples={
            "low": "Mostly trait-like items (e.g. personality).",
            "mid": "Mix of modifiable and trait-like predictors.",
            "high": "All predictors are modifiable in the short term.",
        },
    ),
    Dimension(
        key="construct_coverage",
        display_label="Construct coverage",
        goal_description=(
            "Breadth across the symptom space identified in Part 1, without "
            "two predictors covering the same construct."
        ),
        rationale=(
            "Tight coverage maximises information per measurement."
        ),
        anchor_examples={
            "low": "Several constructs left uncovered.",
            "mid": "Most constructs covered with one overlap.",
            "high": "Every construct uniquely represented.",
        },
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# PART 3 — TREATMENT-TARGET PRIORITISATION
# ──────────────────────────────────────────────────────────────────────────────

PART3_DIMENSIONS: List[Dimension] = [
    Dimension(
        key="top_target_appropriateness",
        display_label="Top-target appropriateness",
        goal_description=(
            "The rank-1 choice is clinically defensible given the network "
            "and the EMA monitoring summary."
        ),
        rationale=(
            "The top target receives the most resource and matters most."
        ),
        anchor_examples={
            "low": "The top choice contradicts the evidence.",
            "mid": "Defensible top choice with a plausible alternative.",
            "high": "The top choice is the obvious best fit.",
        },
    ),
    Dimension(
        key="evidence_alignment",
        display_label="Evidence alignment",
        goal_description=(
            "The full ordering aligns with the strength of network edges and "
            "the current state levels in the monitoring summary."
        ),
        rationale=(
            "An ordering uncoupled from evidence undermines the entire "
            "personalisation pipeline."
        ),
        anchor_examples={
            "low": "Ordering ignores the provided evidence.",
            "mid": "Ordering tracks evidence with one or two reversals.",
            "high": "Ordering closely tracks the evidence.",
        },
    ),
    Dimension(
        key="rank_coherence",
        display_label="Rank coherence",
        goal_description=(
            "The ordering is internally logical: no jumps that contradict "
            "the implicit reasoning."
        ),
        rationale=(
            "Coherent rankings are easier for clinicians to audit and accept."
        ),
        anchor_examples={
            "low": "Rankings appear arbitrary.",
            "mid": "Mostly coherent with one questionable swap.",
            "high": "Fully coherent ordering.",
        },
    ),
    Dimension(
        key="network_impact_awareness",
        display_label="Network impact awareness",
        goal_description=(
            "The rationale considers downstream and cascading effects across "
            "the bipartite network."
        ),
        rationale=(
            "Network-aware reasoning is the value-add over picking the "
            "simplest target."
        ),
        anchor_examples={
            "low": "No mention of downstream effects.",
            "mid": "Some awareness of cascades.",
            "high": "Cascades explicitly drive the ordering.",
        },
    ),
    Dimension(
        key="monitoring_integration",
        display_label="Monitoring integration",
        goal_description=(
            "The rank reflects the current burden levels seen in the EMA "
            "summary, not just static edge weights."
        ),
        rationale=(
            "Personalisation requires using the patient-specific monitoring "
            "evidence, not only the group-level network."
        ),
        anchor_examples={
            "low": "Monitoring data is ignored.",
            "mid": "Monitoring data influences a few ranks.",
            "high": "Monitoring data drives the ordering.",
        },
    ),
    Dimension(
        key="modifiability_weighting",
        display_label="Modifiability weighting",
        goal_description=(
            "High-rank targets are realistically actionable for this patient "
            "in the short term."
        ),
        rationale=(
            "An optimal but unactionable target is clinically useless."
        ),
        anchor_examples={
            "low": "High-rank items are unactionable.",
            "mid": "Mixed actionability.",
            "high": "Top picks are highly actionable.",
        },
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# PART 4 — UPDATED OBSERVATIONAL MODEL
# ──────────────────────────────────────────────────────────────────────────────

PART4_DIMENSIONS: List[Dimension] = [
    Dimension(
        key="adaptive_reasoning",
        display_label="Adaptive reasoning",
        goal_description=(
            "Revisions reflect what the 21-day EMA actually showed (drops "
            "saturated variables, tightens noisy ones, adds gaps)."
        ),
        rationale=(
            "If the model does not change in response to data, monitoring is "
            "pointless."
        ),
        anchor_examples={
            "low": "Model is essentially unchanged.",
            "mid": "Some sensible revisions, some inertia.",
            "high": "Every revision is justified by the EMA evidence.",
        },
    ),
    Dimension(
        key="target_alignment",
        display_label="Target alignment",
        goal_description=(
            "The revised plan keeps measurement focused on Part 3's top "
            "treatment targets."
        ),
        rationale=(
            "Coherence between the prioritised targets and what is measured."
        ),
        anchor_examples={
            "low": "Revisions ignore the prioritised targets.",
            "mid": "Targets are partially aligned.",
            "high": "Plan tightly serves the prioritised targets.",
        },
    ),
    Dimension(
        key="personalisation",
        display_label="Personalisation",
        goal_description=(
            "Variables fit the specific patient pattern (e.g. weekend-only "
            "spikes, evening-only dips)."
        ),
        rationale=(
            "Personalisation is the central claim of the PHOENIX pipeline."
        ),
        anchor_examples={
            "low": "Generic plan that ignores the pattern.",
            "mid": "Some patient-specific adjustments.",
            "high": "Plan is tailored to the observed pattern.",
        },
    ),
    Dimension(
        key="measurement_quality",
        display_label="Measurement quality",
        goal_description=(
            "Chosen variables remain EMA-feasible and well-defined post-revision."
        ),
        rationale=(
            "Revisions sometimes introduce poorly-defined new variables."
        ),
        anchor_examples={
            "low": "New variables are vague or infeasible.",
            "mid": "Mostly well-defined; one weak addition.",
            "high": "All revisions remain EMA-feasible and well-defined.",
        },
    ),
    Dimension(
        key="parsimony",
        display_label="Parsimony",
        goal_description=(
            "Avoids bloating the model; drops redundant or saturated "
            "variables when warranted."
        ),
        rationale=(
            "EMA fatigue grows non-linearly with item count."
        ),
        anchor_examples={
            "low": "Model grows without justification.",
            "mid": "Modest growth; some pruning.",
            "high": "Model size shrinks where warranted.",
        },
    ),
    Dimension(
        key="theoretical_coherence",
        display_label="Theoretical coherence",
        goal_description=(
            "Revisions preserve a coherent network model rather than a bag "
            "of unrelated items."
        ),
        rationale=(
            "A coherent revised network is interpretable for the clinician."
        ),
        anchor_examples={
            "low": "Revisions break network structure.",
            "mid": "Mostly coherent with one or two ad-hoc additions.",
            "high": "Revised network is fully coherent.",
        },
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# PART 5 — TAILORED INTERVENTION MESSAGE
# ──────────────────────────────────────────────────────────────────────────────

PART5_DIMENSIONS: List[Dimension] = [
    Dimension(
        key="hapa_phase_appropriateness",
        display_label="HAPA-phase appropriateness",
        goal_description=(
            "Tone, content, and call-to-action match the assigned HAPA phase "
            "(pre-intentional / intentional / action / maintenance). The "
            "judge also classifies the message's HAPA phase in the ``extra`` "
            "field."
        ),
        rationale=(
            "Mis-staged messages either alienate patients (e.g. action-style "
            "calls in pre-intentional phase) or under-serve them."
        ),
        anchor_examples={
            "low": "Tone clearly mis-matched to the phase.",
            "mid": "Mostly appropriate with a tonal slip.",
            "high": "Tone perfectly matches the phase.",
        },
    ),
    Dimension(
        key="behavioural_change_potential",
        display_label="Behavioural-change potential",
        goal_description=(
            "Message is plausibly likely to nudge the patient toward action."
        ),
        rationale=(
            "The intervention only has value if it changes behaviour."
        ),
        anchor_examples={
            "low": "Generic encouragement unlikely to move behaviour.",
            "mid": "Plausible nudge.",
            "high": "Strong, evidence-aligned nudge.",
        },
    ),
    Dimension(
        key="personalisation_specificity",
        display_label="Personalisation specificity",
        goal_description=(
            "Incorporates patient-specific cues from the case (their "
            "predictors, their schedule, their goals)."
        ),
        rationale=(
            "Generic messages defeat the purpose of the PHOENIX pipeline."
        ),
        anchor_examples={
            "low": "Could apply to anyone.",
            "mid": "One concrete patient-specific cue.",
            "high": "Multiple patient-specific cues woven in.",
        },
    ),
    Dimension(
        key="professional_tone",
        display_label="Professional tone",
        goal_description=(
            "Clinically appropriate register: neither preachy, infantilising, "
            "nor cold."
        ),
        rationale=(
            "Tone is the single most predictive factor of message uptake."
        ),
        anchor_examples={
            "low": "Tone is preachy or dismissive.",
            "mid": "Mostly professional with one slip.",
            "high": "Warm, professional register throughout.",
        },
    ),
    Dimension(
        key="empathy_warmth",
        display_label="Empathy / warmth",
        goal_description=(
            "Validates the patient's perspective and effort."
        ),
        rationale=(
            "Validation precedes change in motivational interviewing."
        ),
        anchor_examples={
            "low": "No validation or acknowledgement.",
            "mid": "Brief acknowledgement.",
            "high": "Genuine, specific validation.",
        },
    ),
    Dimension(
        key="clarity_actionability",
        display_label="Clarity / actionability",
        goal_description=(
            "The action requested is concrete, specific, and feasible."
        ),
        rationale=(
            "Vague calls-to-action do not produce behaviour change."
        ),
        anchor_examples={
            "low": "Vague or no clear action.",
            "mid": "One concrete action.",
            "high": "Concrete, time-bound, feasible action.",
        },
    ),
    Dimension(
        key="message_appropriateness_length",
        display_label="Length appropriateness",
        goal_description=(
            "Respects the 2..3 sentence constraint, neither truncated nor "
            "bloated."
        ),
        rationale=(
            "Mobile delivery requires tight messages."
        ),
        anchor_examples={
            "low": "Far too short or far too long.",
            "mid": "Slightly off the 2..3 sentence target.",
            "high": "Hits the 2..3 sentence target cleanly.",
        },
    ),
]


DIMENSIONS_BY_PART: Dict[str, List[Dimension]] = {
    "part1": PART1_DIMENSIONS,
    "part2": PART2_DIMENSIONS,
    "part3": PART3_DIMENSIONS,
    "part4": PART4_DIMENSIONS,
    "part5": PART5_DIMENSIONS,
}


def dimensions_for(part: str) -> List[Dimension]:
    """Return the list of :class:`Dimension` for a given part."""
    if part not in DIMENSIONS_BY_PART:
        raise ValueError(f"Unknown part {part!r}")
    return DIMENSIONS_BY_PART[part]


__all__ = [
    "PROMPT_VERSION",
    "LIKERT_MIN", "LIKERT_MAX", "LIKERT_NEUTRAL",
    "LIKERT_ANCHOR_NAMES",
    "PART_TITLES",
    "Dimension",
    "DIMENSIONS_BY_PART",
    "dimensions_for",
]
