"""
Per-part dimension specifications for the double-blind LLM-as-judge design.

Design
------
The judge assigns one **absolute quality rating** (1–5 Likert scale) to each
dimension for a single anonymous output.  No comparative judgement is made.

After judging, the long-format CSV records the entity source (phoenix / hcp)
alongside each quality score so that downstream mixed models can estimate the
PHOENIX-vs-HCP quality gap as a predictor coefficient.

Scale conventions
-----------------
    1 = Poor       — fails the criterion; serious clinical problem
    2 = Below avg  — notable gaps; would require significant revision
    3 = Acceptable — meets criterion adequately; only minor issues
    4 = Good       — clearly meets criterion; no meaningful gaps
    5 = Excellent  — exceeds criterion; exemplary response

PROMPT_VERSION must be bumped whenever the prompt template, dimension set,
scale definition, or anchor text changes, so that CSV rows from different
versions can be distinguished in analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

PROMPT_VERSION: str = "2026-05-02-absolute-quality-research-grade"

# Absolute 1..5 quality scale.
QUALITY_MIN: int = 1
QUALITY_MAX: int = 5
QUALITY_NEUTRAL: int = 3

# Legacy aliases kept for backward compatibility with calling code.
SCALE_MIN: int = QUALITY_MIN
SCALE_MAX: int = QUALITY_MAX
SCALE_NEUTRAL: int = QUALITY_NEUTRAL

QUALITY_SCALE_ANCHORS: Tuple[str, ...] = (
    "1 = Poor       — fails the criterion significantly; serious clinical problem",
    "2 = Below avg  — notable gaps; would require significant revision",
    "3 = Acceptable — meets the criterion adequately; minor issues only",
    "4 = Good       — clearly meets the criterion well; no meaningful gaps",
    "5 = Excellent  — exceeds the criterion; could serve as an exemplar",
)

# Legacy alias
SIGNED_SCALE_ANCHORS: Tuple[str, ...] = QUALITY_SCALE_ANCHORS

PART_TITLES: Dict[str, str] = {
    "part1": "01_Identifying_Symptoms",
    "part2": "02_Identifying_Modifiable_Treatment_Options",
    "part3": "03_Prioritising_Treatment_Targets",
    "part4": "04_Selecting_EMA_Measurement_Items",
    "part5": "05_Mobile_Coaching_Message",
}


@dataclass(frozen=True)
class Dimension:
    """
    One quality-evaluation dimension within a survey part.

    anchor_examples keys should use the five-score format:
        "score_1", "score_2", "score_3", "score_4", "score_5"
    Legacy 3-tier keys ("poor", "acceptable", "excellent") are also
    supported for backward compatibility.
    """

    key: str
    display_label: str
    goal_description: str
    rationale: str
    anchor_examples: Dict[str, str] = field(default_factory=dict)

    def anchor_block(self) -> str:
        """Return compact dimension-specific quality anchors for the prompt.

        Prefers the five-score format; falls back to legacy 3-tier if
        only those keys are present.
        """
        score_keys = ["score_1", "score_2", "score_3", "score_4", "score_5"]
        score_labels = {
            "score_1": "SCORE 1 (poor)",
            "score_2": "SCORE 2 (below avg)",
            "score_3": "SCORE 3 (acceptable)",
            "score_4": "SCORE 4 (good)",
            "score_5": "SCORE 5 (excellent)",
        }
        chunks = []
        for k in score_keys:
            if k in self.anchor_examples:
                chunks.append(f"  - {score_labels[k]}: {self.anchor_examples[k]}")
        if chunks:
            return "\n".join(chunks)

        # Fallback: legacy 3-tier format
        legacy_order = ["poor", "acceptable", "excellent"]
        legacy_labels = {
            "poor": "SCORE 1–2 (poor/below avg)",
            "acceptable": "SCORE 3 (acceptable)",
            "excellent": "SCORE 4–5 (good/excellent)",
        }
        for tier in legacy_order:
            if tier in self.anchor_examples:
                chunks.append(f"  - {legacy_labels[tier]}: {self.anchor_examples[tier]}")
        return "\n".join(chunks) if chunks else "  (use the global 1..5 quality anchors)"


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1: Identify 3–6 symptom labels from the patient complaint vignette
# ═══════════════════════════════════════════════════════════════════════════════

PART1_DIMENSIONS: List[Dimension] = [
    Dimension(
        key="task_adherence_label_format",
        display_label="Task adherence / label format",
        goal_description=(
            "Provides 3–6 short symptom labels and does not add diagnoses, "
            "lengthy explanations, treatment suggestions, or unsolicited context."
        ),
        rationale=(
            "The survey task requests compact labels only; extra structure "
            "makes outputs less comparable and less useful as downstream nodes."
        ),
        anchor_examples={
            "score_1": "Ignores the label format entirely: provides prose descriptions, diagnoses, or treatment suggestions instead of labels.",
            "score_2": "Attempts labels but with significant format violations: long sentences, embedded rationales, or only 1–2 labels.",
            "score_3": "Mostly label-only format with 3–6 items; minor formatting deviation (e.g., one slightly verbose label).",
            "score_4": "Clean, compact labels within the 3–6 range; no meaningful format violations.",
            "score_5": "Perfectly formatted 3–6 short labels; exemplary adherence to the task format.",
        },
    ),
    Dimension(
        key="complaint_coverage",
        display_label="Complaint coverage",
        goal_description=(
            "Captures all major current complaint and state dimensions "
            "explicitly or strongly implied by the vignette."
        ),
        rationale=(
            "Missed symptom domains create blind spots for treatment-option "
            "selection, network construction, EMA design, and coaching."
        ),
        anchor_examples={
            "score_1": "Misses two or more major complaint domains that are explicitly present in the vignette.",
            "score_2": "Covers some domains but omits one prominent symptom cluster or state dimension.",
            "score_3": "Covers most major domains; one minor omission or one marginally tangential label.",
            "score_4": "Comprehensively captures all major complaint domains; no meaningful omission.",
            "score_5": "Exhaustively covers all complaint domains without over-inclusion; a model of complaint coverage.",
        },
    ),
    Dimension(
        key="symptom_boundary_validity",
        display_label="Symptom boundary validity",
        goal_description=(
            "Labels represent symptoms or internal state dimensions — not "
            "treatments, external causes, broad diagnoses, fixed traits, or "
            "contextual stressors."
        ),
        rationale=(
            "Part 1 captures symptom nodes; Part 2 captures modifiable "
            "treatment options.  Boundary violations corrupt the bipartite "
            "network logic downstream."
        ),
        anchor_examples={
            "score_1": "Multiple labels are treatments, diagnoses, or external causes rather than internal symptom states.",
            "score_2": "One label clearly crosses the symptom/treatment or symptom/diagnosis boundary.",
            "score_3": "Mostly valid symptoms; one label blurs the boundary (e.g., is both a symptom and a behaviour).",
            "score_4": "All labels are clearly symptoms or internal state dimensions; no boundary violations.",
            "score_5": "All labels are precisely scoped as internal symptom nodes; exemplary boundary adherence.",
        },
    ),
    Dimension(
        key="granularity_resolution",
        display_label="Granularity / resolution",
        goal_description=(
            "Labels are concrete enough for daily EMA self-report and network "
            "analysis, without being too broad to be informative or too "
            "atomised to be analytically useful."
        ),
        rationale=(
            "Overly broad labels (e.g., 'stress') hide symptom mechanisms; "
            "overly narrow labels over-fit a single sentence in the vignette."
        ),
        anchor_examples={
            "score_1": "Labels are uniformly too vague (e.g., 'mental health problems') or trivially atomistic to be analytically useful.",
            "score_2": "Several labels are poorly calibrated: either too broad or too narrow for network or EMA use.",
            "score_3": "Most labels at appropriate resolution; one or two are slightly over- or under-specified.",
            "score_4": "All labels appropriately calibrated for symptom-network nodes and daily EMA operationalisation.",
            "score_5": "Precisely calibrated granularity throughout; optimal specificity for both network analysis and EMA translation.",
        },
    ),
    Dimension(
        key="nonredundancy_discriminability",
        display_label="Non-redundancy / discriminability",
        goal_description=(
            "Labels are mutually distinct and do not duplicate the same "
            "construct under different surface wordings."
        ),
        rationale=(
            "Redundant symptom nodes inflate one construct's centrality and "
            "distort the symptom-behaviour network."
        ),
        anchor_examples={
            "score_1": "Two or more label pairs clearly represent the same construct (e.g., 'fatigue' and 'low energy' as separate items).",
            "score_2": "One pair of labels overlaps substantially — distinguishable in wording but not in construct.",
            "score_3": "Labels mostly distinct; one pair has minor thematic overlap but measures different facets.",
            "score_4": "All labels clearly discriminable; each targets a distinct symptom construct.",
            "score_5": "Zero redundancy; every label is conceptually independent and immediately distinguishable.",
        },
    ),
    Dimension(
        key="clinical_interoperability",
        display_label="Clinical interoperability",
        goal_description=(
            "Uses concise, recognisable clinical or psychological vocabulary "
            "that another HCP or the PHOENIX ontology could interpret "
            "consistently."
        ),
        rationale=(
            "Interoperable labels support reliable mapping to standardised "
            "criterion constructs across clinicians and systems."
        ),
        anchor_examples={
            "score_1": "Idiosyncratic, colloquial, or invented terms that could not be consistently mapped by another clinician.",
            "score_2": "Some standard terms but one or two are so ambiguous or colloquial as to hinder interoperability.",
            "score_3": "Mostly standard clinical vocabulary; one term is slightly informal but recognisable.",
            "score_4": "All terms are precise, standard clinical or psychological vocabulary.",
            "score_5": "All labels use canonical clinical terminology that is immediately mappable to established constructs.",
        },
    ),
    Dimension(
        key="ema_measurability",
        display_label="EMA measurability",
        goal_description=(
            "The selected symptoms could plausibly be measured repeatedly "
            "via brief daily mobile self-report."
        ),
        rationale=(
            "The full PHOENIX workflow depends on EMA-compatible symptom "
            "nodes that translate directly into daily app prompts."
        ),
        anchor_examples={
            "score_1": "Multiple labels describe stable traits or unmeasurable internal states incompatible with daily app prompts.",
            "score_2": "One or two labels are marginal for EMA (e.g., describe long-term states rather than daily fluctuations).",
            "score_3": "Most labels EMA-compatible; one is marginal but arguable.",
            "score_4": "All labels translate to plausible daily self-report items (yes/no, 0–10 ratings, counts).",
            "score_5": "Every label is optimally operationalisable as a brief daily mobile prompt; a textbook EMA item set.",
        },
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2: Generate 3–5 modifiable treatment-option labels for the symptom pattern
# ═══════════════════════════════════════════════════════════════════════════════

PART2_DIMENSIONS: List[Dimension] = [
    Dimension(
        key="task_adherence_label_format",
        display_label="Task adherence / label format",
        goal_description=(
            "Provides 3–5 short treatment-option labels; avoids embedded "
            "measurement definitions, rationales, or unrequested prose."
        ),
        rationale=(
            "Both sources complete the same label-generation task; extra "
            "explanatory content must not influence the quality score."
        ),
        anchor_examples={
            "score_1": "Provides prose paragraphs, measurement specs, or fewer than 3 / more than 5 items instead of labels.",
            "score_2": "Attempts labels but with significant violations: embedded rationales, long explanations, or wrong count.",
            "score_3": "3–5 labels mostly in format; one minor prose addition or count boundary issue.",
            "score_4": "3–5 clean, compact treatment-option labels; no meaningful format violation.",
            "score_5": "Perfectly formatted 3–5 labels; exemplary task adherence.",
        },
    ),
    Dimension(
        key="modifiability_actionability",
        display_label="Modifiability / actionability",
        goal_description=(
            "Treatment options describe behaviours, routines, strategies, or "
            "processes the patient or therapist can realistically change."
        ),
        rationale=(
            "Part 2 identifies modifiable handles for intervention — not "
            "symptoms, states, or fixed traits."
        ),
        anchor_examples={
            "score_1": "All or most options describe states or traits the patient cannot directly modify (e.g., 'sleep quality' as a fixed state).",
            "score_2": "Multiple options are borderline — phrased as states rather than changeable behaviours.",
            "score_3": "Most options are actionable; one is borderline between a state and a modifiable behaviour.",
            "score_4": "All options describe patient- or therapist-modifiable behaviours or routines.",
            "score_5": "All options are concretely actionable with a clear change mechanism; exemplary modifiability.",
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
            "Generic wellness behaviours (e.g., 'exercise') are weak unless "
            "they connect to the case-specific symptom pattern."
        ),
        anchor_examples={
            "score_1": "All options are generic wellness behaviours with no clear connection to the case symptoms.",
            "score_2": "Most options are generic; only one or two map to the case-specific symptom pattern.",
            "score_3": "Most options connect to case symptoms; one is tangential.",
            "score_4": "Every option directly addresses one or more of the case-specific symptom domains.",
            "score_5": "Each option precisely targets a specific symptom mechanism in the case; no generic padding.",
        },
    ),
    Dimension(
        key="causal_plausibility",
        display_label="Causal plausibility",
        goal_description=(
            "It is clinically plausible that changing the option would reduce "
            "one or more target symptoms via a recognisable mechanism."
        ),
        rationale=(
            "The bipartite network assumes options causally influence symptoms; "
            "spurious or implausible links distort network-based prioritisation."
        ),
        anchor_examples={
            "score_1": "Multiple options lack any recognisable causal mechanism linking them to the target symptoms.",
            "score_2": "One option has a weak or speculative causal pathway; mechanism is not clinically recognised.",
            "score_3": "Most options have plausible mechanisms; one has limited empirical support.",
            "score_4": "All options have clear, clinically recognised causal pathways to symptom reduction.",
            "score_5": "All options have strong evidence-based causal mechanisms; would satisfy peer review.",
        },
    ),
    Dimension(
        key="daily_ema_feasibility",
        display_label="Daily EMA feasibility",
        goal_description=(
            "Options can be tracked with short daily app questions using "
            "yes/no, counts, minutes, frequency, or 0–10 ratings."
        ),
        rationale=(
            "The survey task selects treatment options that can enter daily "
            "mobile monitoring for personalised feedback loops."
        ),
        anchor_examples={
            "score_1": "Options are too abstract or dispositional to operationalise as daily app questions.",
            "score_2": "Multiple options need significant creative operationalisation to become app prompts.",
            "score_3": "Most options translate to EMA questions; one requires moderate operationalisation effort.",
            "score_4": "All options map to simple daily app prompts (yes/no, minutes, count, 0–10 rating).",
            "score_5": "Every option is immediately deployable as a brief daily app item; optimal EMA feasibility.",
        },
    ),
    Dimension(
        key="symptom_option_separation",
        display_label="Symptom-option separation",
        goal_description=(
            "Avoids re-listing symptoms, diagnostic states, or impairment as "
            "if they were modifiable treatment options."
        ),
        rationale=(
            "Conflating symptom nodes with behaviour nodes breaks the bipartite "
            "network logic and makes Part 3 prioritisation meaningless."
        ),
        anchor_examples={
            "score_1": "Multiple options are clearly symptom labels or diagnostic states, not behavioural targets.",
            "score_2": "One option is clearly a symptom re-labelled as a treatment option.",
            "score_3": "Options mostly well-separated; one label blurs the symptom/behaviour boundary but is arguable.",
            "score_4": "All options are clearly behavioural/routine; no symptom re-labelling.",
            "score_5": "Perfectly clean separation: all items are unambiguously modifiable treatment options.",
        },
    ),
    Dimension(
        key="option_diversity_complementarity",
        display_label="Diversity / complementarity",
        goal_description=(
            "The set spans complementary mechanisms or domains rather than "
            "clustering around near-duplicates of the same behaviour."
        ),
        rationale=(
            "Diverse options give the network and ranking stages more useful "
            "intervention candidates across different causal pathways."
        ),
        anchor_examples={
            "score_1": "Multiple options target the same behavioural mechanism; little complementary coverage.",
            "score_2": "Moderate redundancy between two or three options; limited mechanism diversity.",
            "score_3": "Moderate diversity; slight overlap between one option pair but covers different aspects.",
            "score_4": "Options span clearly complementary mechanisms with minimal redundancy.",
            "score_5": "Maximally diverse option set covering all major modifiable pathways for the case.",
        },
    ),
    Dimension(
        key="label_precision",
        display_label="Label precision",
        goal_description=(
            "Labels are compact but specific enough to be understood without "
            "additional explanation."
        ),
        rationale=(
            "Because the survey asks for short labels, precision must come "
            "from wording alone, without embedded definitions."
        ),
        anchor_examples={
            "score_1": "Labels are so vague or ambiguous they could refer to many different behaviours.",
            "score_2": "One or two labels are ambiguous and would require clarification before clinical use.",
            "score_3": "Labels mostly clear; one is slightly ambiguous but interpretable in context.",
            "score_4": "All labels are concise and immediately interpretable without clarification.",
            "score_5": "All labels achieve high precision in minimal words; ideal for network node labelling.",
        },
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3: Rank 5 treatment options from highest to lowest priority
#         using the bipartite symptom-treatment network + 21-day EMA data
# ═══════════════════════════════════════════════════════════════════════════════

PART3_DIMENSIONS: List[Dimension] = [
    Dimension(
        key="ranking_validity_completeness",
        display_label="Ranking validity / completeness",
        goal_description=(
            "Ranks all five available treatment options exactly once, with a "
            "clear 1–5 priority order and no missing, duplicated, or invalid "
            "option identifiers."
        ),
        rationale=(
            "A structurally invalid ranking cannot be clinically interpreted; "
            "validity is a prerequisite for all other dimensions."
        ),
        anchor_examples={
            "score_1": "Missing options, duplicate ranks, or identifiers not in the provided option list.",
            "score_2": "Structurally mostly valid but with one missing option or one unclear identifier.",
            "score_3": "All five options ranked once with correct IDs; minor presentation inconsistency.",
            "score_4": "Complete, valid 1–5 ranking using all correct option identifiers; no structural issue.",
            "score_5": "Perfectly formed ranking with unambiguous 1–5 order and exact option identifiers.",
        },
    ),
    Dimension(
        key="network_weight_alignment",
        display_label="Network-weight alignment",
        goal_description=(
            "Ranking reflects the relative strength of symptom-treatment "
            "associations in the bipartite network (edge weight and degree)."
        ),
        rationale=(
            "The survey task is explicitly network-informed; network-weight "
            "alignment is the primary ranking criterion."
        ),
        anchor_examples={
            "score_1": "Top-ranked options have the weakest network connections; highest-degree options ranked last.",
            "score_2": "Broadly misaligned with network weights; several strong-link options ranked low.",
            "score_3": "Broadly aligned with network weights; one or two unexpected rank swaps.",
            "score_4": "Ranking closely mirrors edge strength and connectivity degree across all five options.",
            "score_5": "Near-optimal alignment with network weights; would be defensible in a peer-reviewed methods paper.",
        },
    ),
    Dimension(
        key="current_state_integration",
        display_label="Current-state integration",
        goal_description=(
            "Ranking incorporates 21-day EMA burden, frequency, trend, and "
            "current symptom level alongside network edge strength."
        ),
        rationale=(
            "A strong network edge is necessary but not sufficient: a currently "
            "favourable target may be less urgent than a weaker link to a "
            "highly active, burdening symptom."
        ),
        anchor_examples={
            "score_1": "Ranking appears based solely on static network edges; ignores all EMA monitoring data.",
            "score_2": "Partially accounts for EMA data; misses one important current-state signal (e.g., high burden, worsening trend).",
            "score_3": "Integrates most current-state information; one EMA signal underweighted.",
            "score_4": "Top priorities clearly reflect both network strength and current EMA burden and trend.",
            "score_5": "Optimally integrates network structure with all EMA signals; could be a methodology reference.",
        },
    ),
    Dimension(
        key="edge_direction_interpretation",
        display_label="Edge-direction interpretation",
        goal_description=(
            "Correctly distinguishes positive-edge options (increasing a "
            "harmful behaviour) from negative-edge options (increasing a "
            "protective factor), and ranks accordingly."
        ),
        rationale=(
            "The network legend distinguishes sign directions; misinterpreting "
            "a negative protective edge as a positive risk factor inverts the "
            "clinical logic of the ranking."
        ),
        anchor_examples={
            "score_1": "Treats protective edges as risk factors or vice versa; direction logic systematically reversed.",
            "score_2": "One edge-direction misinterpretation that materially affects a rank position.",
            "score_3": "Edge directions mostly correct; one ambiguous or implicit interpretation.",
            "score_4": "All positive (risk) and negative (protective) edges interpreted correctly.",
            "score_5": "Impeccable edge-direction interpretation; positive and protective relationships explicitly distinguished.",
        },
    ),
    Dimension(
        key="top_target_defensibility",
        display_label="Top-target defensibility",
        goal_description=(
            "The rank-1 and top-3 choices are clinically and empirically "
            "defensible as the highest intervention priorities for this case."
        ),
        rationale=(
            "Parts 4 and 5 depend directly on the highest-priority targets; "
            "an indefensible rank-1 choice cascades errors downstream."
        ),
        anchor_examples={
            "score_1": "Rank-1 choice is clinically implausible or clearly a low-priority option given the case data.",
            "score_2": "Rank-1 is arguable but one of the top-3 is a clearly poor choice.",
            "score_3": "Top 3 are generally defensible; one choice within top-3 is debatable.",
            "score_4": "Rank-1 and top-3 are clearly the strongest, most defensible priorities for this case.",
            "score_5": "Top-target selection is optimal and would be endorsed by an expert panel without reservation.",
        },
    ),
    Dimension(
        key="modifiability_feasibility_weighting",
        display_label="Modifiability / feasibility weighting",
        goal_description=(
            "Higher-ranked targets are realistic to modify for this patient "
            "given their current state, resources, and context."
        ),
        rationale=(
            "A statistically central but practically unreachable target is a "
            "poor digital intervention priority."
        ),
        anchor_examples={
            "score_1": "Top-ranked target is theoretically central but practically unmodifiable for this patient's situation.",
            "score_2": "Top-ranked targets have questionable feasibility given the patient's current state or context.",
            "score_3": "Most top-ranked targets are feasible; one is marginally practical for this patient.",
            "score_4": "All high-priority targets are realistic and actionable for this patient's specific situation.",
            "score_5": "Feasibility perfectly calibrated to patient context; top targets are both central and immediately actionable.",
        },
    ),
    Dimension(
        key="rank_order_coherence",
        display_label="Rank-order coherence",
        goal_description=(
            "The full 1–5 order is internally consistent: higher-ranked "
            "options are systematically more justified than lower-ranked ones."
        ),
        rationale=(
            "A coherent ranking is auditable and shows a clear clinical "
            "priority logic rather than arbitrary ordering."
        ),
        anchor_examples={
            "score_1": "Adjacent ranks appear interchangeable; no discernible priority logic across the full order.",
            "score_2": "The ranking contains one or two rank swaps that violate the stated or implied priority logic.",
            "score_3": "Mostly coherent; one adjacent-rank pair seems arbitrarily ordered.",
            "score_4": "Full 1–5 ordering is internally consistent and clinically logical throughout.",
            "score_5": "The rank-order logic is transparent, coherent, and fully defensible from rank 1 to rank 5.",
        },
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Part 4: Select exactly 6 EMA items from a 20-item candidate list (2 per target)
# ═══════════════════════════════════════════════════════════════════════════════

PART4_DIMENSIONS: List[Dimension] = [
    Dimension(
        key="valid_candidate_selection",
        display_label="Valid candidate selection",
        goal_description=(
            "Selects only items present in the 20-item candidate list; avoids "
            "invented, paraphrased, duplicated, or unidentifiable items."
        ),
        rationale=(
            "Part 4 is a constrained selection task; inventing better-sounding "
            "items is not a valid answer and corrupts the controlled comparison."
        ),
        anchor_examples={
            "score_1": "One or more selected items are not in the candidate list (invented or significantly paraphrased).",
            "score_2": "All items are from the list but one is a close paraphrase that could not be uniquely identified.",
            "score_3": "All items from the candidate list; minor transcription variation but unambiguously identifiable.",
            "score_4": "All 6 items match the candidate list exactly or with trivial phrasing differences.",
            "score_5": "All 6 items are exact matches from the candidate list; perfect selection validity.",
        },
    ),
    Dimension(
        key="target_item_mapping_accuracy",
        display_label="Target-item mapping accuracy",
        goal_description=(
            "Selected EMA items directly operationalise the three abstract "
            "treatment targets supplied for the case."
        ),
        rationale=(
            "Part 4 translates abstract targets into concrete measurable "
            "sub-behaviours; poor mapping breaks the intervention logic."
        ),
        anchor_examples={
            "score_1": "Items are generic or map to different targets; most do not operationalise the specified targets.",
            "score_2": "Some items map to targets; one or two clearly miss the target they are assigned to.",
            "score_3": "Most items map to their targets; one is only tangentially related.",
            "score_4": "Every item directly operationalises one of the three treatment targets.",
            "score_5": "Each item is the most direct available operationalisation of its target; optimal mapping.",
        },
    ),
    Dimension(
        key="coverage_balance",
        display_label="Coverage balance",
        goal_description=(
            "Selects exactly 6 items with 2 well-chosen items per treatment "
            "target."
        ),
        rationale=(
            "The survey's design fixes the measurement budget at three targets "
            "× two items; imbalanced coverage under-measures one target."
        ),
        anchor_examples={
            "score_1": "Fewer or more than 6 items, or strong imbalance (e.g., 4 items for one target, 0 for another).",
            "score_2": "Exactly 6 items but clear imbalance: one target has 3+ items, another has only 1.",
            "score_3": "Exactly 6 items with approximately 2 per target; one target slightly over- or under-represented.",
            "score_4": "Exactly 6 items with 2 well-matched items per treatment target.",
            "score_5": "Exactly 6 items with 2 optimal items per target; perfectly balanced coverage.",
        },
    ),
    Dimension(
        key="measurement_concreteness",
        display_label="Measurement concreteness",
        goal_description=(
            "Chosen items are specific daily behaviours or strategies that "
            "can be measured via yes/no, counts, minutes, or 0–10 ratings."
        ),
        rationale=(
            "A selected EMA item must become an app prompt without additional "
            "clinical interpretation at the point of deployment."
        ),
        anchor_examples={
            "score_1": "Items are too abstract or dispositional to become app prompts without extensive operationalisation.",
            "score_2": "Multiple items require significant operationalisation effort before they can be app-deployed.",
            "score_3": "Most items are concrete; one needs minor reframing to become a deployable app question.",
            "score_4": "All items are immediately deployable as mobile app questions without further work.",
            "score_5": "All items are optimally concrete: specific, brief, and directly deployable as daily app prompts.",
        },
    ),
    Dimension(
        key="directness_specificity",
        display_label="Directness / specificity",
        goal_description=(
            "Avoids items that only tangentially relate to the target when "
            "more direct operationalisations are available in the candidate "
            "list."
        ),
        rationale=(
            "The task instructions explicitly require selecting the most "
            "direct available items rather than side-path proxies."
        ),
        anchor_examples={
            "score_1": "Selected items are indirect proxies when clearly more direct operationalisations were available in the list.",
            "score_2": "One item is a side-path proxy when a clearly better direct item was available.",
            "score_3": "Mostly direct; one item is slightly indirect but defensible.",
            "score_4": "All items are the most direct available operationalisation of each target.",
            "score_5": "Every item represents the single most direct operationalisation; could not be improved by swapping.",
        },
    ),
    Dimension(
        key="dynamic_informativeness",
        display_label="Dynamic informativeness",
        goal_description=(
            "Items are expected to vary day-to-day and reveal actionable "
            "patterns during the monitoring period."
        ),
        rationale=(
            "A static or rarely varying item contributes little to EMA-based "
            "adaptive digital intervention."
        ),
        anchor_examples={
            "score_1": "Items measure stable traits or almost never-varying states; no sensitivity to daily change.",
            "score_2": "One or two items are relatively static and unlikely to vary meaningfully day-to-day.",
            "score_3": "Most items capture daily variation; one is relatively static but arguable.",
            "score_4": "All items are expected to vary daily and reveal actionable monitoring patterns.",
            "score_5": "All items are maximally sensitive to daily variation; ideal for adaptive EMA-based intervention.",
        },
    ),
    Dimension(
        key="monitoring_burden_parsimony",
        display_label="Monitoring burden / parsimony",
        goal_description=(
            "The selected 6-item set provides useful coverage without "
            "unnecessary redundancy or cognitive burden."
        ),
        rationale=(
            "Mobile adherence depends on a compact, non-redundant item set; "
            "redundancy without coverage gain wastes monitoring budget."
        ),
        anchor_examples={
            "score_1": "Multiple redundant item pairs cover the same behavioural micro-facet; set is unnecessarily burdensome.",
            "score_2": "One redundant pair reduces effective coverage without adding new information.",
            "score_3": "Reasonable parsimony; minor redundancy between one pair of items.",
            "score_4": "6 items provide good coverage with minimal overlap; acceptable burden.",
            "score_5": "6 items achieve maximal coverage across all three targets with zero redundancy; optimal parsimony.",
        },
    ),
    Dimension(
        key="feedback_value_for_coaching",
        display_label="Feedback value for coaching",
        goal_description=(
            "Selected items would provide day-to-day data that can meaningfully "
            "adapt the tone or content of later mobile coaching messages."
        ),
        rationale=(
            "The digital intervention is only useful if monitoring data can "
            "inform what the app communicates next."
        ),
        anchor_examples={
            "score_1": "Items would not differentiate patient states relevant to coaching (e.g., all stable traits).",
            "score_2": "Most items have limited coaching feedback value; only one or two would usefully adapt a message.",
            "score_3": "Most items provide coaching-relevant signal; one has limited feedback discriminability.",
            "score_4": "All items would enable meaningful adaptation of coaching content based on daily variability.",
            "score_5": "Every item provides high-value coaching signal; the set is optimally designed for adaptive messaging.",
        },
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Part 5: Write a 2–4 sentence personalised mobile coaching message
# ═══════════════════════════════════════════════════════════════════════════════

PART5_DIMENSIONS: List[Dimension] = [
    Dimension(
        key="message_format_direct_address",
        display_label="Message format / direct address",
        goal_description=(
            "Uses 2–4 sentences, addresses the patient in second person "
            "(you/your), and reads as a phone-ready coaching message rather "
            "than a clinical note, treatment plan, or advice letter."
        ),
        rationale=(
            "The task evaluates a deployable mobile intervention message; "
            "a clinical note or third-person formulation is the wrong format "
            "regardless of its content quality."
        ),
        anchor_examples={
            "score_1": "More than 4 sentences, third-person phrasing, or structured as a clinical note/recommendation.",
            "score_2": "Broadly in the right format but with a significant deviation: e.g., 5 sentences or 1 sentence only.",
            "score_3": "2–4 sentences in second person; minor format deviation (e.g., one slightly clinical phrasing).",
            "score_4": "2–4 concise sentences in direct second-person voice; immediately readable as a phone message.",
            "score_5": "Perfectly formatted phone message: 2–4 punchy sentences, second-person voice, zero clinical-note language.",
        },
    ),
    Dimension(
        key="treatment_goal_alignment",
        display_label="Treatment-goal alignment",
        goal_description=(
            "Message targets the primary treatment goal and stays focused "
            "on the intended behaviour or coping shift."
        ),
        rationale=(
            "A warm, well-written message still fails if it targets a different "
            "behavioural domain than the specified treatment goal."
        ),
        anchor_examples={
            "score_1": "Message targets a different behavioural domain than the treatment goal entirely.",
            "score_2": "Message partially relates to the goal; one or more sentences target a different domain.",
            "score_3": "Message relates to the treatment goal; one sentence drifts slightly off-target.",
            "score_4": "Every sentence is focused on the treatment goal behaviour; no off-target content.",
            "score_5": "Tightly aligned to the goal throughout; each sentence advances the specific treatment target.",
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
            "Behaviour change requires addressing the reason the patient is "
            "not already doing the target behaviour; generic encouragement "
            "ignores that barrier."
        ),
        anchor_examples={
            "score_1": "Message ignores the stated barrier entirely; offers only generic encouragement.",
            "score_2": "Message vaguely alludes to difficulty but does not address the specific stated barrier.",
            "score_3": "Message implicitly acknowledges the barrier but does not directly name or work around it.",
            "score_4": "Message explicitly acknowledges the specific barrier and offers a concrete way around it.",
            "score_5": "Barrier is named, validated, and addressed with a specific, practical work-around strategy.",
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
            "Specific implementation intentions (what, when, how) are more "
            "likely to translate into behaviour than generic encouragement."
        ),
        anchor_examples={
            "score_1": "No concrete action proposed; only general motivation or awareness raising.",
            "score_2": "Proposes an action but it is too vague or lacks any time anchor or feasibility consideration.",
            "score_3": "Action is present but either slightly vague or without a clear time anchor.",
            "score_4": "Specific, small, feasible action step with a time anchor; immediately actionable.",
            "score_5": "Implementation intention is precise (what + when + how), small, and calibrated to the patient's context.",
        },
    ),
    Dimension(
        key="behaviour_change_potential",
        display_label="Behaviour-change potential",
        goal_description=(
            "Overall likelihood that the message would increase self-efficacy, "
            "intent, or concrete action in the next day or two."
        ),
        rationale=(
            "The message is evaluated as an intervention component; high "
            "behaviour-change potential requires combining barrier work, "
            "specific action, and self-efficacy support."
        ),
        anchor_examples={
            "score_1": "Message is unlikely to shift intent or self-efficacy; no recognisable behaviour-change mechanism.",
            "score_2": "Message has limited behaviour-change potential: one mechanism is present but insufficient.",
            "score_3": "Moderately likely to motivate; limited by vague action or generic framing on one dimension.",
            "score_4": "Strong behaviour-change potential: barrier addressed, specific action provided, self-efficacy supported.",
            "score_5": "Optimally designed for behaviour change: integrates all key mechanisms (barrier, action, efficacy, personalisation).",
        },
    ),
    Dimension(
        key="tone_empathy_professionalism",
        display_label="Tone / empathy / professionalism",
        goal_description=(
            "Tone is warm, direct, respectful, non-infantilising, and free "
            "of clinical jargon or diagnostic labelling."
        ),
        rationale=(
            "Mobile coaching messages need warmth to be received and "
            "professional restraint to remain clinically appropriate."
        ),
        anchor_examples={
            "score_1": "Tone is cold, patronising, shame-inducing, or heavy with diagnostic/clinical jargon.",
            "score_2": "Tone is generally acceptable but with one phrase that is dismissive, overly clinical, or slightly patronising.",
            "score_3": "Appropriate tone overall; one minor jargon word or slightly detached sentence.",
            "score_4": "Warm, empathic, respectful, and jargon-free throughout; immediately readable.",
            "score_5": "Exceptional tone: genuinely empathic, professionally precise, and free of any problematic language.",
        },
    ),
    Dimension(
        key="mobile_concision_readability",
        display_label="Mobile concision / readability",
        goal_description=(
            "Message fits the 2–4 sentence mobile constraint and is readable "
            "at a glance without dense or complex phrasing."
        ),
        rationale=(
            "A message that is too long or cognitively heavy will not function "
            "well as a mobile push notification or in-app message."
        ),
        anchor_examples={
            "score_1": "Too long, or sentences are so dense and complex they would not function as mobile text.",
            "score_2": "Within length limit but one sentence is quite long or uses complex vocabulary for the medium.",
            "score_3": "Generally concise and readable; one sentence is slightly longer than ideal for a phone screen.",
            "score_4": "Perfectly concise; each sentence is short and immediately readable on a phone screen.",
            "score_5": "Exemplary mobile readability: short, punchy sentences, zero cognitive overhead, ideal for push notifications.",
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
            "The entire PHOENIX evaluation concerns personalised digital "
            "intervention; generic messages defeat the personalisation purpose."
        ),
        anchor_examples={
            "score_1": "Could have been written for any patient; no case-specific details at all.",
            "score_2": "Only one generic reference to the broad complaint area; no case-specific cues.",
            "score_3": "Partially personalised; uses one case-specific detail (symptom, goal, or barrier).",
            "score_4": "Uses two or more concrete case-specific cues (e.g., specific symptom + named barrier).",
            "score_5": "Highly personalised throughout; three or more specific case cues woven naturally into the message.",
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
            "Non-judgmental, low-risk wording is non-negotiable in automated "
            "mental-health messaging where safeguards are limited."
        ),
        anchor_examples={
            "score_1": "Contains shaming language, unrealistic promises, coercive pressure, or advice that could increase distress.",
            "score_2": "One phrase is slightly judgmental, mildly coercive, or makes a questionable promise.",
            "score_3": "Generally safe and non-judgmental; one minor phrase slightly overpromising or implicitly pressuring.",
            "score_4": "Completely safe, non-judgmental, no coercive language, no unrealistic promises.",
            "score_5": "Exemplary safety profile: unconditionally non-judgmental, validating, zero risk of iatrogenic effect.",
        },
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════

DIMENSIONS_BY_PART: Dict[str, List[Dimension]] = {
    "part1": PART1_DIMENSIONS,
    "part2": PART2_DIMENSIONS,
    "part3": PART3_DIMENSIONS,
    "part4": PART4_DIMENSIONS,
    "part5": PART5_DIMENSIONS,
}


def dimensions_for(part: str) -> List[Dimension]:
    """Return the evaluation dimensions for a given part key."""
    if part not in DIMENSIONS_BY_PART:
        raise ValueError(f"Unknown part {part!r}; expected one of {list(DIMENSIONS_BY_PART)}")
    return DIMENSIONS_BY_PART[part]


__all__ = [
    "PROMPT_VERSION",
    "QUALITY_MIN",
    "QUALITY_MAX",
    "QUALITY_NEUTRAL",
    "QUALITY_SCALE_ANCHORS",
    # Legacy aliases
    "SCALE_MIN",
    "SCALE_MAX",
    "SCALE_NEUTRAL",
    "SIGNED_SCALE_ANCHORS",
    "PART_TITLES",
    "Dimension",
    "DIMENSIONS_BY_PART",
    "dimensions_for",
    "PART1_DIMENSIONS",
    "PART2_DIMENSIONS",
    "PART3_DIMENSIONS",
    "PART4_DIMENSIONS",
    "PART5_DIMENSIONS",
]
