"""
Per-part dimension specifications for the double-blind LLM-as-judge design.

The judge does not assign independent absolute ratings to Output A and
Output B. It makes one signed comparative judgment per dimension:

    +9  = Output A is decisively better than Output B
     0  = no meaningful quality difference
    -9  = Output B is decisively better than Output A

After unblinding, the runner converts that A-vs-B value into a PHOENIX-vs-HCP
score, where positive values always favour PHOENIX and negative values always
favour the HCP output. This is the outcome used by the mixed models.

PROMPT_VERSION must be bumped whenever the prompt template, dimension set,
or scale definition changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

PROMPT_VERSION: str = "2026-05-01-v2-signed-comparison"

SCALE_MIN: int = -9
SCALE_MAX: int = 9
SCALE_NEUTRAL: int = 0

SIGNED_SCALE_ANCHORS: Tuple[str, ...] = (
    "-9 = Output B is decisively better",
    "-6 = Output B is strongly better",
    "-3 = Output B is modestly better",
    "0 = no meaningful difference / tie",
    "+3 = Output A is modestly better",
    "+6 = Output A is strongly better",
    "+9 = Output A is decisively better",
)

PART_TITLES: Dict[str, str] = {
    "part1": "01_Identifying_Symptoms",
    "part2": "02_Identifying_Modifiable_Treatment_Options",
    "part3": "03_Prioritising_Treatment_Targets",
    "part4": "04_Selecting_EMA_Measurement_Items",
    "part5": "05_Mobile_Coaching_Message",
}


@dataclass(frozen=True)
class Dimension:
    """One comparative evaluation dimension within a survey part."""

    key: str
    display_label: str
    goal_description: str
    rationale: str
    anchor_examples: Dict[str, str] = field(default_factory=dict)

    def anchor_block(self) -> str:
        """Return compact dimension-specific examples for the prompt."""
        if not self.anchor_examples:
            return "  (use the global -9..+9 comparative anchors)"
        order = ["negative", "tie", "positive"]
        labels = {
            "negative": "NEGATIVE SCORE",
            "tie": "ZERO / TIE",
            "positive": "POSITIVE SCORE",
        }
        chunks = []
        for tier in order:
            if tier in self.anchor_examples:
                chunks.append(f"  - {labels[tier]}: {self.anchor_examples[tier]}")
        return "\n".join(chunks) if chunks else "  (use the global -9..+9 comparative anchors)"


# Part 1: clinicians identify 3..6 symptom labels from the free complaint text.
PART1_DIMENSIONS: List[Dimension] = [
    Dimension(
        key="complaint_coverage",
        display_label="Complaint coverage",
        goal_description=(
            "Captures the major current complaint and state dimensions that are "
            "explicitly or strongly implied by the vignette."
        ),
        rationale=(
            "Missed symptoms create blind spots for all later treatment-option, "
            "network, EMA, and coaching steps."
        ),
        anchor_examples={
            "negative": "Output B covers substantially more of the clinically relevant complaint domains.",
            "tie": "Both outputs cover essentially the same clinically relevant domains.",
            "positive": "Output A covers substantially more of the clinically relevant complaint domains.",
        },
    ),
    Dimension(
        key="symptom_boundary_validity",
        display_label="Symptom boundary validity",
        goal_description=(
            "Labels are symptoms or state dimensions, not treatments, external "
            "causes, broad diagnoses, fixed traits, or contextual stressors."
        ),
        rationale=(
            "The survey explicitly distinguishes symptoms in Part 1 from "
            "modifiable treatment options in Part 2."
        ),
        anchor_examples={
            "negative": "Output B more cleanly avoids behaviours, contexts, diagnoses, and treatments.",
            "tie": "Both outputs respect the symptom boundary to a similar degree.",
            "positive": "Output A more cleanly avoids behaviours, contexts, diagnoses, and treatments.",
        },
    ),
    Dimension(
        key="granularity_resolution",
        display_label="Granularity / resolution",
        goal_description=(
            "Labels are concrete enough for daily self-report and network "
            "analysis, without being too broad or too atomised."
        ),
        rationale=(
            "Overly broad labels hide mechanisms; overly narrow labels overfit "
            "one wording in the vignette."
        ),
        anchor_examples={
            "negative": "Output B has a more clinically useful level of specificity.",
            "tie": "Both outputs preserve about the same level of useful resolution.",
            "positive": "Output A has a more clinically useful level of specificity.",
        },
    ),
    Dimension(
        key="nonredundancy_discriminability",
        display_label="Non-redundancy / discriminability",
        goal_description=(
            "The symptom labels are mutually distinct and do not duplicate the "
            "same construct under different words."
        ),
        rationale=(
            "Redundant symptom nodes distort a later symptom-behaviour network."
        ),
        anchor_examples={
            "negative": "Output B has fewer overlapping or duplicate labels.",
            "tie": "Both outputs have comparable overlap or distinctness.",
            "positive": "Output A has fewer overlapping or duplicate labels.",
        },
    ),
    Dimension(
        key="clinical_interoperability",
        display_label="Clinical interoperability",
        goal_description=(
            "Uses concise, recognisable clinical vocabulary that another HCP "
            "or the PHOENIX ontology could interpret consistently."
        ),
        rationale=(
            "Interoperable labels are needed for later mapping to standardised "
            "criterion constructs."
        ),
        anchor_examples={
            "negative": "Output B uses clearer and more standard clinical terms.",
            "tie": "Both outputs are similarly interpretable.",
            "positive": "Output A uses clearer and more standard clinical terms.",
        },
    ),
    Dimension(
        key="ema_measurability",
        display_label="EMA measurability",
        goal_description=(
            "The selected symptoms could plausibly be measured repeatedly via "
            "brief daily app self-report."
        ),
        rationale=(
            "The full PHOENIX workflow depends on EMA-compatible constructs."
        ),
        anchor_examples={
            "negative": "Output B is more readily translatable into daily EMA items.",
            "tie": "Both outputs are similarly measurable via EMA.",
            "positive": "Output A is more readily translatable into daily EMA items.",
        },
    ),
]


# Part 2: clinicians generate 3..5 modifiable treatment-option labels.
PART2_DIMENSIONS: List[Dimension] = [
    Dimension(
        key="modifiability_actionability",
        display_label="Modifiability / actionability",
        goal_description=(
            "Treatment options describe behaviours, routines, strategies, or "
            "processes the patient or therapist can realistically change."
        ),
        rationale=(
            "Part 2 is not about naming symptoms; it is about identifying "
            "handles for intervention."
        ),
        anchor_examples={
            "negative": "Output B contains more directly changeable options.",
            "tie": "Both outputs are similarly actionable.",
            "positive": "Output A contains more directly changeable options.",
        },
    ),
    Dimension(
        key="symptom_relevance",
        display_label="Symptom relevance",
        goal_description=(
            "Options plausibly target the specific symptoms and complaint "
            "pattern supplied for the case."
        ),
        rationale=(
            "Generic wellness behaviours are weak unless they connect to the "
            "case-specific symptom pattern."
        ),
        anchor_examples={
            "negative": "Output B is more tightly connected to the case symptoms.",
            "tie": "Both outputs are similarly connected to the case symptoms.",
            "positive": "Output A is more tightly connected to the case symptoms.",
        },
    ),
    Dimension(
        key="causal_plausibility",
        display_label="Causal plausibility",
        goal_description=(
            "It is clinically plausible that changing the option could reduce "
            "one or more target symptoms."
        ),
        rationale=(
            "The treatment-option network assumes the option can influence "
            "symptoms, not merely correlate with them."
        ),
        anchor_examples={
            "negative": "Output B offers more plausible mechanisms of change.",
            "tie": "Both outputs have similarly plausible mechanisms.",
            "positive": "Output A offers more plausible mechanisms of change.",
        },
    ),
    Dimension(
        key="daily_ema_feasibility",
        display_label="Daily EMA feasibility",
        goal_description=(
            "Options can be tracked with short daily app questions using "
            "yes/no, counts, minutes, frequency, or 0..10 ratings."
        ),
        rationale=(
            "The survey asks for treatment options that can enter daily "
            "mobile monitoring."
        ),
        anchor_examples={
            "negative": "Output B is easier to measure in daily EMA.",
            "tie": "Both outputs are similarly EMA-feasible.",
            "positive": "Output A is easier to measure in daily EMA.",
        },
    ),
    Dimension(
        key="symptom_option_separation",
        display_label="Symptom-option separation",
        goal_description=(
            "Avoids re-listing symptoms, diagnostic states, or impairment as "
            "if they were treatment options."
        ),
        rationale=(
            "Conflating symptom nodes with behaviour nodes breaks the bipartite "
            "network logic."
        ),
        anchor_examples={
            "negative": "Output B more cleanly separates options from symptoms.",
            "tie": "Both outputs show similar boundary discipline.",
            "positive": "Output A more cleanly separates options from symptoms.",
        },
    ),
    Dimension(
        key="option_diversity_complementarity",
        display_label="Diversity / complementarity",
        goal_description=(
            "The set spans complementary mechanisms rather than several near "
            "duplicates of the same behaviour."
        ),
        rationale=(
            "Diverse options give the later network and ranking stages more "
            "useful intervention candidates."
        ),
        anchor_examples={
            "negative": "Output B offers a broader and less redundant option set.",
            "tie": "Both outputs have comparable diversity.",
            "positive": "Output A offers a broader and less redundant option set.",
        },
    ),
    Dimension(
        key="label_precision",
        display_label="Label precision",
        goal_description=(
            "Labels are compact but specific enough to be understood without "
            "extra explanation."
        ),
        rationale=(
            "The survey asks for short labels, so precision has to come from "
            "the wording itself."
        ),
        anchor_examples={
            "negative": "Output B uses clearer short labels.",
            "tie": "Both outputs have similarly clear labels.",
            "positive": "Output A uses clearer short labels.",
        },
    ),
]


# Part 3: clinicians rank five standardised treatment options using the network.
PART3_DIMENSIONS: List[Dimension] = [
    Dimension(
        key="network_weight_alignment",
        display_label="Network-weight alignment",
        goal_description=(
            "Ranking reflects the strength of symptom-treatment associations "
            "in the bipartite network."
        ),
        rationale=(
            "The survey's Part 3 task is explicitly network-informed."
        ),
        anchor_examples={
            "negative": "Output B follows the strongest network evidence more closely.",
            "tie": "Both rankings use the network evidence similarly.",
            "positive": "Output A follows the strongest network evidence more closely.",
        },
    ),
    Dimension(
        key="current_state_integration",
        display_label="Current-state integration",
        goal_description=(
            "Ranking uses the 21-day EMA burden, frequency, trends, and "
            "current symptom levels, not only edge strength."
        ),
        rationale=(
            "The survey warns that a strong edge is necessary but not sufficient "
            "when the current state is already favourable."
        ),
        anchor_examples={
            "negative": "Output B better integrates current burden and frequency.",
            "tie": "Both rankings integrate current state similarly.",
            "positive": "Output A better integrates current burden and frequency.",
        },
    ),
    Dimension(
        key="edge_direction_interpretation",
        display_label="Edge-direction interpretation",
        goal_description=(
            "Correctly interprets positive and negative network relations: "
            "increasing a harmful behaviour differs from increasing a protective one."
        ),
        rationale=(
            "The network legend distinguishes positive from negative edges, and "
            "the ranking should not invert that meaning."
        ),
        anchor_examples={
            "negative": "Output B handles positive and negative edge direction more accurately.",
            "tie": "Both outputs handle edge direction similarly.",
            "positive": "Output A handles positive and negative edge direction more accurately.",
        },
    ),
    Dimension(
        key="top_target_defensibility",
        display_label="Top-target defensibility",
        goal_description=(
            "The rank-1 and top-3 choices are clinically and empirically "
            "defensible as highest priorities."
        ),
        rationale=(
            "Part 4 and Part 5 depend mainly on the highest-priority targets."
        ),
        anchor_examples={
            "negative": "Output B has a more defensible highest-priority set.",
            "tie": "Both outputs have similarly defensible top priorities.",
            "positive": "Output A has a more defensible highest-priority set.",
        },
    ),
    Dimension(
        key="modifiability_feasibility_weighting",
        display_label="Modifiability / feasibility weighting",
        goal_description=(
            "Higher-ranked targets are realistic to modify for this patient in "
            "the near term."
        ),
        rationale=(
            "A statistically central but practically unreachable target is a "
            "poor digital intervention target."
        ),
        anchor_examples={
            "negative": "Output B gives more weight to feasible change targets.",
            "tie": "Both outputs weight feasibility similarly.",
            "positive": "Output A gives more weight to feasible change targets.",
        },
    ),
    Dimension(
        key="rank_order_coherence",
        display_label="Rank-order coherence",
        goal_description=(
            "The full 1..5 order is internally consistent and free of arbitrary "
            "swaps."
        ),
        rationale=(
            "A coherent ranking is easier to audit and justify clinically."
        ),
        anchor_examples={
            "negative": "Output B has a more coherent full ordering.",
            "tie": "Both outputs are similarly coherent.",
            "positive": "Output A has a more coherent full ordering.",
        },
    ),
]


# Part 4: clinicians select exactly six concrete EMA items, two per target.
PART4_DIMENSIONS: List[Dimension] = [
    Dimension(
        key="target_item_mapping_accuracy",
        display_label="Target-item mapping accuracy",
        goal_description=(
            "Selected EMA items directly operationalise the three abstract "
            "treatment targets supplied for the case."
        ),
        rationale=(
            "Part 4 translates abstract targets into concrete app-measurable "
            "sub-treatment options."
        ),
        anchor_examples={
            "negative": "Output B maps items to the abstract targets more accurately.",
            "tie": "Both outputs map items to targets similarly.",
            "positive": "Output A maps items to the abstract targets more accurately.",
        },
    ),
    Dimension(
        key="coverage_balance",
        display_label="Coverage balance",
        goal_description=(
            "Selects exactly six items with two well-chosen items per "
            "treatment target."
        ),
        rationale=(
            "The survey's design fixes the measurement budget at three targets "
            "times two items."
        ),
        anchor_examples={
            "negative": "Output B better satisfies the 2-items-per-target constraint.",
            "tie": "Both outputs satisfy the constraint similarly.",
            "positive": "Output A better satisfies the 2-items-per-target constraint.",
        },
    ),
    Dimension(
        key="measurement_concreteness",
        display_label="Measurement concreteness",
        goal_description=(
            "Chosen items are specific daily behaviours or strategies that can "
            "be measured by yes/no, counts, minutes, or 0..10 ratings."
        ),
        rationale=(
            "A selected EMA item has to become an app prompt without additional "
            "clinical interpretation."
        ),
        anchor_examples={
            "negative": "Output B chooses more concretely measurable items.",
            "tie": "Both outputs are similarly concrete.",
            "positive": "Output A chooses more concretely measurable items.",
        },
    ),
    Dimension(
        key="directness_specificity",
        display_label="Directness / specificity",
        goal_description=(
            "Avoids items that only tangentially relate to the target when more "
            "direct operationalisations are available."
        ),
        rationale=(
            "The instructions explicitly tell participants to avoid side-path "
            "items even when they seem related."
        ),
        anchor_examples={
            "negative": "Output B selects more direct indicators for the targets.",
            "tie": "Both outputs are similarly direct.",
            "positive": "Output A selects more direct indicators for the targets.",
        },
    ),
    Dimension(
        key="dynamic_informativeness",
        display_label="Dynamic informativeness",
        goal_description=(
            "Items are likely to vary day to day and reveal actionable patterns "
            "during monitoring."
        ),
        rationale=(
            "A static or rarely changing item contributes little to EMA-based "
            "adaptation."
        ),
        anchor_examples={
            "negative": "Output B would produce more informative daily time series.",
            "tie": "Both outputs are similarly informative.",
            "positive": "Output A would produce more informative daily time series.",
        },
    ),
    Dimension(
        key="monitoring_burden_parsimony",
        display_label="Monitoring burden / parsimony",
        goal_description=(
            "The selected set gives useful coverage without unnecessary burden, "
            "ambiguity, or redundancy."
        ),
        rationale=(
            "Mobile adherence depends on a compact, non-redundant item set."
        ),
        anchor_examples={
            "negative": "Output B is more parsimonious and less redundant.",
            "tie": "Both outputs have similar burden and redundancy.",
            "positive": "Output A is more parsimonious and less redundant.",
        },
    ),
    Dimension(
        key="feedback_value_for_coaching",
        display_label="Feedback value for coaching",
        goal_description=(
            "Selected items would provide data that can meaningfully adapt the "
            "tone or content of later mobile coaching messages."
        ),
        rationale=(
            "The digital intervention is only useful if monitoring can inform "
            "what the app says next."
        ),
        anchor_examples={
            "negative": "Output B would better inform adaptive coaching.",
            "tie": "Both outputs have similar coaching feedback value.",
            "positive": "Output A would better inform adaptive coaching.",
        },
    ),
]


# Part 5: clinicians write a 2..4 sentence mobile coaching message.
PART5_DIMENSIONS: List[Dimension] = [
    Dimension(
        key="treatment_goal_alignment",
        display_label="Treatment-goal alignment",
        goal_description=(
            "Message addresses the primary treatment goal and stays focused "
            "on the intended behaviour or coping shift."
        ),
        rationale=(
            "A warm message still fails if it targets the wrong behavioural "
            "lever."
        ),
        anchor_examples={
            "negative": "Output B is more tightly aligned with the treatment goal.",
            "tie": "Both messages align with the treatment goal similarly.",
            "positive": "Output A is more tightly aligned with the treatment goal.",
        },
    ),
    Dimension(
        key="barrier_responsiveness",
        display_label="Barrier responsiveness",
        goal_description=(
            "Message explicitly acknowledges or works around the main barrier "
            "given in the case context."
        ),
        rationale=(
            "Behaviour change depends on addressing the reason the patient is "
            "not already doing the target behaviour."
        ),
        anchor_examples={
            "negative": "Output B responds more directly to the stated barrier.",
            "tie": "Both messages respond to the barrier similarly.",
            "positive": "Output A responds more directly to the stated barrier.",
        },
    ),
    Dimension(
        key="action_specificity_feasibility",
        display_label="Action specificity / feasibility",
        goal_description=(
            "Includes a concrete next action that is small, feasible, and "
            "time-anchored enough for a mobile prompt."
        ),
        rationale=(
            "Specific implementation intentions are more likely to translate "
            "into behaviour than generic encouragement."
        ),
        anchor_examples={
            "negative": "Output B gives a clearer and more feasible next action.",
            "tie": "Both messages are similarly actionable.",
            "positive": "Output A gives a clearer and more feasible next action.",
        },
    ),
    Dimension(
        key="behaviour_change_potential",
        display_label="Behaviour-change potential",
        goal_description=(
            "Overall likelihood that the message would increase self-efficacy, "
            "intent, or concrete action in the next day."
        ),
        rationale=(
            "The message is evaluated as an intervention component, not as "
            "general psychoeducation."
        ),
        anchor_examples={
            "negative": "Output B is more likely to move behaviour.",
            "tie": "Both messages have similar behaviour-change potential.",
            "positive": "Output A is more likely to move behaviour.",
        },
    ),
    Dimension(
        key="tone_empathy_professionalism",
        display_label="Tone / empathy / professionalism",
        goal_description=(
            "Tone is warm, direct, respectful, non-infantilising, and free of "
            "clinical jargon or diagnostic labelling."
        ),
        rationale=(
            "Mobile coaching needs enough warmth to be received and enough "
            "professional restraint to remain clinically appropriate."
        ),
        anchor_examples={
            "negative": "Output B has a more empathic and professional tone.",
            "tie": "Both messages have similar tone quality.",
            "positive": "Output A has a more empathic and professional tone.",
        },
    ),
    Dimension(
        key="mobile_concision_readability",
        display_label="Mobile concision / readability",
        goal_description=(
            "Fits the 2..4 sentence mobile-screen constraint and is readable "
            "without dense phrasing."
        ),
        rationale=(
            "A message that is too long or cognitively heavy will not function "
            "well in a mobile app."
        ),
        anchor_examples={
            "negative": "Output B is more concise and readable for a mobile screen.",
            "tie": "Both messages are similarly concise and readable.",
            "positive": "Output A is more concise and readable for a mobile screen.",
        },
    ),
    Dimension(
        key="personalisation_specificity",
        display_label="Personalisation specificity",
        goal_description=(
            "Uses concrete case-specific cues rather than a generic supportive "
            "message that could fit any patient."
        ),
        rationale=(
            "The evaluation concerns personalised digital intervention."
        ),
        anchor_examples={
            "negative": "Output B uses more relevant case-specific cues.",
            "tie": "Both messages are similarly personalised.",
            "positive": "Output A uses more relevant case-specific cues.",
        },
    ),
    Dimension(
        key="clinical_safety_nonjudgment",
        display_label="Clinical safety / non-judgment",
        goal_description=(
            "Avoids shame, coercion, blame, unsafe advice, overpromising, or "
            "language that could worsen distress."
        ),
        rationale=(
            "Low-risk, non-judgmental wording is essential in automated mental "
            "health messaging."
        ),
        anchor_examples={
            "negative": "Output B is safer and less judgmental.",
            "tie": "Both messages have similar safety and non-judgment.",
            "positive": "Output A is safer and less judgmental.",
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
    """Return the dimensions for a given part key."""
    if part not in DIMENSIONS_BY_PART:
        raise ValueError(f"Unknown part {part!r}")
    return DIMENSIONS_BY_PART[part]


__all__ = [
    "PROMPT_VERSION",
    "SCALE_MIN",
    "SCALE_MAX",
    "SCALE_NEUTRAL",
    "SIGNED_SCALE_ANCHORS",
    "PART_TITLES",
    "Dimension",
    "DIMENSIONS_BY_PART",
    "dimensions_for",
]
