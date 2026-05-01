"""
Per-part dimension specifications for the double-blind LLM-as-judge design.

Design
------
The judge assigns one **absolute quality rating** on a bipolar −10 to +10
semantic differential scale to each dimension for a single anonymous output.
No comparative judgement is made during judging.

Scale conventions
-----------------
    −10 = Catastrophic failure  — clinically unusable; may cause harm
     −5 = Notably deficient     — major gaps requiring extensive revision
      0 = Acceptable            — meets criterion adequately; clinical baseline
     +5 = Clearly good          — above acceptable; no meaningful gaps
    +10 = Outstanding           — gold-standard exemplar

After judging, the long-format CSV records the entity source (phoenix / hcp)
alongside each quality score so that downstream mixed models can estimate the
PHOENIX-vs-HCP quality gap as a predictor coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

PROMPT_VERSION: str = "2026-05-02-bipolar-research-grade"

# Bipolar semantic differential scale.
QUALITY_MIN: int = -10
QUALITY_MAX: int = +10
QUALITY_NEUTRAL: int = 0

# Legacy aliases kept for backward compatibility.
SCALE_MIN: int = QUALITY_MIN
SCALE_MAX: int = QUALITY_MAX
SCALE_NEUTRAL: int = QUALITY_NEUTRAL

QUALITY_SCALE_ANCHORS: Tuple[str, ...] = (
    "−10 = Catastrophic failure  — clinically unusable or harmful",
    " −5 = Notably deficient     — major gaps requiring extensive revision",
    "  0 = Acceptable            — meets criterion adequately; clinical baseline",
    " +5 = Clearly good          — above acceptable; no meaningful gaps",
    "+10 = Outstanding           — gold-standard exemplar; definitively exceeds criterion",
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

    anchor_examples keys use five calibrated anchor points on the −10..+10 scale:
        "n10"  → −10 (catastrophic failure)
        "n5"   → −5  (notably deficient)
        "z0"   →  0  (acceptable baseline)
        "p5"   → +5  (clearly good)
        "p10"  → +10 (outstanding)
    """

    key: str
    display_label: str
    goal_description: str
    rationale: str
    anchor_examples: Dict[str, str] = field(default_factory=dict)

    def anchor_block(self) -> str:
        """Return dimension-specific quality anchors at five calibrated points."""
        anchors = [
            ("n10", "SCORE −10 (catastrophic failure)"),
            ("n5",  "SCORE  −5 (notably deficient)"),
            ("z0",  "SCORE   0 (acceptable)"),
            ("p5",  "SCORE  +5 (clearly good)"),
            ("p10", "SCORE +10 (outstanding)"),
        ]
        chunks = []
        for key, label in anchors:
            if key in self.anchor_examples:
                chunks.append(f"  - {label}: {self.anchor_examples[key]}")
        return "\n".join(chunks) if chunks else "  (use the global −10..+10 quality anchors)"


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
            "n10": "Completely ignores the label format — provides prose paragraphs, clinical notes, diagnoses, or a treatment plan; no individual labels identifiable.",
            "n5":  "Attempts labels but with major violations: most items are multi-sentence explanations or diagnoses; only 1–2 usable labels.",
            "z0":  "3–6 short labels in the correct format; one minor deviation (e.g., one slightly verbose label) but recognisably label-only output.",
            "p5":  "3–6 compact, cleanly formatted labels; all items are single-phrase symptom labels with no extraneous content.",
            "p10": "Perfectly formatted 3–6 labels; exemplary adherence to the label-only task format; could serve as a canonical reference response.",
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
            "n10": "Misses almost all major complaint domains; only one superficial aspect addressed; vignette content largely ignored.",
            "n5":  "Misses two or more prominent complaint domains explicitly stated in the vignette; significant clinical blind spots.",
            "z0":  "Covers the main complaint domains; one minor omission or one tangential label; functional for downstream steps.",
            "p5":  "Comprehensively covers all major complaint domains from the vignette; no meaningful omission; thematically complete.",
            "p10": "Exhaustive coverage of every major and secondary complaint domain without over-inclusion; optimal signal-to-noise for network construction.",
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
            "treatment options. Boundary violations corrupt the bipartite "
            "network logic downstream."
        ),
        anchor_examples={
            "n10": "Multiple labels are diagnoses, treatments, or external causes; the output confuses the symptom layer with treatment/cause layers entirely.",
            "n5":  "One or two labels clearly cross into treatment, diagnosis, or external-cause territory; meaningful boundary violations.",
            "z0":  "Mostly valid symptom labels; one label blurs the boundary (e.g., is both a symptom descriptor and a behaviour) but the set is usable.",
            "p5":  "All labels are unambiguously internal symptom or state dimensions; no boundary violations.",
            "p10": "Impeccable symptom boundary adherence; every label is a precisely scoped internal state that maps cleanly to the network ontology.",
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
            "Overly broad labels hide mechanisms; overly narrow labels "
            "over-fit a single sentence in the vignette."
        ),
        anchor_examples={
            "n10": "Labels are entirely non-specific (e.g., 'mental health issues') or absurdly atomistic; not usable as network nodes.",
            "n5":  "Several labels are poorly calibrated — too broad or too narrow for meaningful network analysis or EMA operationalisation.",
            "z0":  "Most labels at appropriate resolution; one or two slightly over- or under-specified but the set is usable.",
            "p5":  "All labels precisely calibrated for symptom-network nodes; optimal specificity for EMA translation.",
            "p10": "Perfect granularity throughout; each label hits the exact resolution that maximises both network discriminability and EMA operationalisability.",
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
            "n10": "Three or more label pairs clearly represent the same construct under different words; the set is severely informationally deflated.",
            "n5":  "One label pair clearly overlaps or represents the same construct; the set conflates two distinct symptom nodes.",
            "z0":  "Labels mostly distinct; minor thematic overlap between one pair but they measure different facets.",
            "p5":  "All labels clearly discriminable; each targets a distinct symptom construct with no overlap.",
            "p10": "Zero redundancy; every label is conceptually orthogonal; the set is maximally informative per label.",
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
            "n10": "All or most terms are idiosyncratic, colloquial, or invented; no other clinician or ontology could map these reliably.",
            "n5":  "Several terms are ambiguous or non-standard; significant mapping uncertainty across clinicians.",
            "z0":  "Mostly standard clinical vocabulary; one term is informal but recognisable in context.",
            "p5":  "All terms are precise, standard clinical vocabulary immediately interpretable by any trained HCP.",
            "p10": "Canonical clinical terminology throughout; every label maps directly to an established criterion construct; optimal ontology interoperability.",
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
            "n10": "Multiple labels describe fixed traits, past events, or chronic states that cannot vary day-to-day via app self-report.",
            "n5":  "One or two labels are fundamentally incompatible with daily EMA (e.g., describe personality features or historical events).",
            "z0":  "Most labels EMA-compatible; one is marginal but arguable as a daily self-report item.",
            "p5":  "All labels translate directly to plausible daily self-report items (0–10 rating, yes/no, count).",
            "p10": "Every label is optimally EMA-compatible — sensitive to daily fluctuation, unambiguously measurable, immediately deployable as a mobile prompt.",
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
            "n10": "Completely ignores the format — provides a full treatment plan, clinical summary, or free text; no treatment-option labels identifiable.",
            "n5":  "Attempts labels but with major violations: embedded rationales, measurement definitions, or fewer than 2 / more than 6 items.",
            "z0":  "3–5 treatment-option labels in the correct format; one minor format deviation.",
            "p5":  "3–5 compact, clean treatment-option labels with no extraneous content; exactly the format requested.",
            "p10": "Exemplary format adherence — exactly 3–5 short, precise labels; could serve as a reference answer for this task type.",
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
            "n10": "All options describe fixed states or traits the patient cannot change (e.g., 'childhood trauma', 'personality type'); entirely non-actionable.",
            "n5":  "Most options are borderline states rather than concrete changeable behaviours; intervention handles are unclear.",
            "z0":  "Most options are actionable; one is borderline between a fixed state and a modifiable behaviour.",
            "p5":  "All options describe clearly modifiable behaviours, routines, or strategies with a direct change mechanism.",
            "p10": "Maximally actionable set; every option specifies an immediately modifiable behaviour with an obvious clinical change lever.",
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
            "n10": "All options are generic wellness behaviours with no discernible connection to the case-specific symptom profile.",
            "n5":  "Most options are generic; only one or two show any case-specific relevance.",
            "z0":  "Most options connect to case symptoms; one is tangential or only loosely related.",
            "p5":  "Every option directly targets one or more of the case-specific symptom domains identified in Part 1.",
            "p10": "Each option precisely targets a specific symptom mechanism in the case; the set represents a maximally case-specific intervention menu.",
        },
    ),
    Dimension(
        key="causal_plausibility",
        display_label="Causal plausibility",
        goal_description=(
            "It is clinically plausible that changing the option could reduce "
            "one or more target symptoms via a recognisable mechanism."
        ),
        rationale=(
            "The bipartite network assumes options causally influence symptoms; "
            "spurious links distort network-based prioritisation."
        ),
        anchor_examples={
            "n10": "Multiple options lack any recognisable mechanism linking behaviour change to symptom improvement; causal logic is entirely absent.",
            "n5":  "One option has a speculative or clinically implausible causal pathway; mechanism is not evidence-based.",
            "z0":  "Most options have plausible mechanisms; one has limited but arguable empirical support.",
            "p5":  "All options have clear, clinically recognised causal pathways to symptom reduction.",
            "p10": "All options have strong, evidence-based causal mechanisms that would satisfy peer review; the causal logic is explicit and compelling.",
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
            "n10": "Options are too dispositional or abstract to operationalise as daily app questions; no feasible monitoring approach.",
            "n5":  "Multiple options need major creative reinterpretation before they could become daily app prompts.",
            "z0":  "Most options map to EMA questions; one requires moderate operationalisation effort.",
            "p5":  "All options map directly to simple daily app prompts (yes/no, minutes, count, 0–10 rating).",
            "p10": "Every option is immediately deployable as a daily mobile EMA item; optimal feasibility throughout.",
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
            "n10": "Multiple options are clearly symptom labels or diagnostic states re-listed as treatment options; the bipartite network logic is entirely violated.",
            "n5":  "One option is clearly a symptom masquerading as a treatment option; a meaningful boundary violation.",
            "z0":  "Options mostly well-separated; one label blurs the boundary but is arguable as an intervention target.",
            "p5":  "All options are clearly behavioural or routine-based; no symptom re-labelling.",
            "p10": "Perfect boundary adherence; every item is unambiguously a modifiable treatment behaviour distinct from any symptom node.",
        },
    ),
    Dimension(
        key="option_diversity_complementarity",
        display_label="Diversity / complementarity",
        goal_description=(
            "The set spans complementary mechanisms or domains rather than "
            "clustering near-duplicates of the same behaviour."
        ),
        rationale=(
            "Diverse options give the network and ranking stages more useful "
            "intervention candidates across different causal pathways."
        ),
        anchor_examples={
            "n10": "All options target the same mechanism or behavioural domain; the set provides no complementary coverage.",
            "n5":  "Significant clustering — multiple options target similar mechanisms with little diversity.",
            "z0":  "Moderate diversity; slight overlap between one option pair but covers different aspects.",
            "p5":  "Options span clearly complementary mechanisms with minimal redundancy.",
            "p10": "Maximally diverse and complementary option set; covers all major modifiable pathways for the case with zero redundancy.",
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
            "from the wording alone."
        ),
        anchor_examples={
            "n10": "Labels are so vague or ambiguous they could refer to many entirely different behaviours or interventions.",
            "n5":  "One or two labels are ambiguous and would require clarification before any clinical or network use.",
            "z0":  "Labels mostly clear; one is slightly ambiguous but interpretable in context.",
            "p5":  "All labels are concise and immediately interpretable without clarification.",
            "p10": "Every label achieves maximum precision in minimal words; ideal for network node labelling with no ambiguity possible.",
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
            "n10": "Ranking is completely invalid — multiple options missing, severe duplicate ranks, or identifiers not from the provided list; unusable.",
            "n5":  "Structurally deficient — one option missing or one duplicate rank; structural repairs needed before interpretation.",
            "z0":  "All five options ranked once with correct IDs; one minor presentation inconsistency.",
            "p5":  "Complete, valid 1–5 ranking using all correct option identifiers; no structural issue.",
            "p10": "Perfectly formed ranking — unambiguous 1–5 order, exact identifiers, no edge cases; structurally exemplary.",
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
            "n10": "Top-ranked options are the weakest-connected network nodes; the highest-degree options are systematically ranked last.",
            "n5":  "Broadly misaligned — several strong-link options ranked low while weak options are prioritised.",
            "z0":  "Broadly aligned with network weights; one or two unexpected rank swaps but the overall pattern is correct.",
            "p5":  "Ranking closely mirrors edge strength and connectivity degree across all five options.",
            "p10": "Near-optimal network alignment; the ranking would be endorsed by a network analysis expert as the most parsimonious interpretation of the edge data.",
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
            "n10": "Ranking appears based entirely on static network edges; all EMA monitoring data completely ignored.",
            "n5":  "Partially accounts for EMA data; misses one important current-state signal (e.g., high burden or worsening trend).",
            "z0":  "Integrates most current-state information; one EMA signal underweighted.",
            "p5":  "Top priorities clearly reflect both network strength and current EMA burden and trend.",
            "p10": "Optimally integrates all EMA signals with network structure; could serve as a reference for adaptive intervention prioritisation.",
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
            "a protective edge as a risk factor inverts the clinical logic."
        ),
        anchor_examples={
            "n10": "Edge directions systematically misinterpreted — protective edges treated as risk factors or vice versa throughout.",
            "n5":  "One substantial edge-direction error that materially inverts the priority logic for a key option.",
            "z0":  "Edge directions mostly correct; one ambiguous or marginally incorrect interpretation.",
            "p5":  "All positive (risk) and negative (protective) edges correctly interpreted throughout.",
            "p10": "Impeccable edge-direction interpretation; positive and protective relationships explicitly and correctly distinguished; logic is transparent and verifiable.",
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
            "n10": "Rank-1 choice is clinically implausible or clearly contraindicated for this patient based on the provided data.",
            "n5":  "Rank-1 is arguable but one of the top-3 is clearly a poor choice given the case data.",
            "z0":  "Top 3 are generally defensible; one choice within top-3 is debatable but not indefensible.",
            "p5":  "Rank-1 and top-3 are clearly the strongest, most defensible priorities for this case.",
            "p10": "Top-target selection is optimal and would be unanimously endorsed by an expert panel; the defensibility case is airtight.",
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
            "n10": "Top-ranked targets are theoretically central but practically impossible to modify for this patient's specific context.",
            "n5":  "Top-ranked targets have questionable feasibility given the patient's current state or resources.",
            "z0":  "Most top-ranked targets are feasible; one is marginally practical for this specific patient.",
            "p5":  "All high-priority targets are realistic and immediately actionable for this patient's situation.",
            "p10": "Feasibility perfectly calibrated to the patient's context; top targets are simultaneously maximally central and immediately achievable.",
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
            "n10": "No discernible priority logic; adjacent ranks appear arbitrarily assigned; the ordering cannot be explained by any coherent clinical principle.",
            "n5":  "The ranking contains one or two rank swaps that directly contradict the stated or implied priority logic.",
            "z0":  "Mostly coherent; one adjacent-rank pair seems arbitrarily ordered.",
            "p5":  "Full 1–5 ordering is internally consistent and clinically logical throughout.",
            "p10": "Rank-order logic is transparent, fully defensible, and internally coherent from rank 1 to rank 5; every step in the priority cascade is justified.",
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
            "n10": "Multiple selected items are invented, significantly paraphrased, or not traceable to any item in the candidate list; structurally invalid.",
            "n5":  "One item is clearly not in the candidate list or is such a significant paraphrase it cannot be uniquely matched.",
            "z0":  "All items from the candidate list; one is a close paraphrase but unambiguously identifiable.",
            "p5":  "All 6 items match the candidate list exactly or with trivial phrasing differences.",
            "p10": "All 6 items are exact verbatim matches from the candidate list; perfect selection validity.",
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
            "n10": "Items are generic or map to entirely different targets; the selection completely fails to operationalise the specified treatment targets.",
            "n5":  "Some items map to targets; one or two clearly miss the target they are assigned to.",
            "z0":  "Most items map to their targets; one is only tangentially related to its assigned target.",
            "p5":  "Every item directly operationalises one of the three treatment targets.",
            "p10": "Each item is the single most direct available operationalisation of its target; mapping is optimal throughout.",
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
            "times two items; imbalanced coverage under-measures one target."
        ),
        anchor_examples={
            "n10": "Fewer or more than 6 items, or grossly imbalanced (e.g., 5 items for one target, 0 for another).",
            "n5":  "Exactly 6 items but clear imbalance — one target has 3+ items, another has only 1.",
            "z0":  "Exactly 6 items with approximately 2 per target; one target slightly over- or under-represented.",
            "p5":  "Exactly 6 items with 2 well-matched items per treatment target; balanced coverage.",
            "p10": "Exactly 6 items with the 2 optimal items per target; perfectly balanced and theoretically grounded.",
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
            "clinical interpretation at deployment."
        ),
        anchor_examples={
            "n10": "Items are entirely dispositional or abstract; none could become an app prompt without extensive clinical reinterpretation.",
            "n5":  "Multiple items require significant operationalisation effort before they could be app-deployed.",
            "z0":  "Most items are concrete; one needs minor reframing to become a deployable app question.",
            "p5":  "All items are immediately deployable as mobile app questions without further work.",
            "p10": "All items are optimally concrete — specific, brief, unambiguous, immediately deployable as daily mobile prompts.",
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
            "n10": "All selected items are indirect proxies when clearly more direct operationalisations were available in the candidate list.",
            "n5":  "One item is a side-path proxy when a clearly superior direct item was available.",
            "z0":  "Mostly direct; one item is slightly indirect but defensible.",
            "p5":  "All items are the most direct available operationalisation of each target.",
            "p10": "Every item represents the single most direct available operationalisation; no closer match exists in the candidate list.",
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
            "n10": "Items measure stable traits or chronic states; no sensitivity to daily variation; unsuitable for adaptive EMA.",
            "n5":  "One or two items are relatively static and unlikely to vary meaningfully day-to-day.",
            "z0":  "Most items capture daily variation; one is relatively static but arguable.",
            "p5":  "All items are expected to vary daily and reveal actionable monitoring patterns.",
            "p10": "Every item is maximally sensitive to daily variation; the set is optimally designed for adaptive EMA-based digital intervention.",
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
            "n10": "Multiple redundant item pairs cover the same behavioural micro-facet; set is unnecessarily burdensome without coverage gain.",
            "n5":  "One redundant pair substantially reduces effective coverage without adding new information.",
            "z0":  "Reasonable parsimony; minor redundancy between one pair of items.",
            "p5":  "6 items provide good coverage with minimal overlap; acceptable burden for daily monitoring.",
            "p10": "6 items achieve maximal coverage across all three targets with zero redundancy; the theoretically optimal parsimonious set.",
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
            "n10": "Items would not differentiate patient states relevant to coaching at all (e.g., all stable traits; no day-to-day signal).",
            "n5":  "Most items have limited coaching feedback value; only one or two would usefully adapt a coaching message.",
            "z0":  "Most items provide coaching-relevant signal; one has limited feedback discriminability.",
            "p5":  "All items would enable meaningful adaptation of coaching content based on daily variability.",
            "p10": "Every item provides maximum coaching signal; the set is optimally designed for personalised adaptive digital coaching.",
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
            "regardless of content quality."
        ),
        anchor_examples={
            "n10": "Completely wrong format — a clinical note, third-person formulation, treatment plan, or single-sentence response; entirely unsuitable for mobile deployment.",
            "n5":  "Broadly in the right direction but with major deviation: 5+ sentences, third-person voice, or structured as a recommendation letter.",
            "z0":  "2–4 sentences in second person; one minor format deviation (e.g., one slightly clinical phrase).",
            "p5":  "2–4 concise sentences in direct second-person voice; immediately readable as a phone coaching message.",
            "p10": "Perfect mobile message format — 2–4 punchy second-person sentences; completely phone-ready; zero clinical-note language; exemplary.",
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
            "n10": "Message targets an entirely different behavioural domain from the treatment goal; the goal content is absent.",
            "n5":  "Message partially relates to the goal; most sentences target a different domain.",
            "z0":  "Message relates to the treatment goal; one sentence drifts slightly off-target.",
            "p5":  "Every sentence focused on the treatment goal behaviour; no off-target content.",
            "p10": "Tightly aligned to the goal throughout; each sentence advances the specific treatment target in a precise and intentional way.",
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
            "n10": "Message completely ignores the stated barrier; provides generic encouragement with no acknowledgement that a barrier exists.",
            "n5":  "Message vaguely alludes to difficulty without addressing the specific stated barrier.",
            "z0":  "Message implicitly acknowledges the barrier but does not directly name or provide a concrete work-around.",
            "p5":  "Message explicitly acknowledges the specific barrier and offers a concrete work-around strategy.",
            "p10": "Barrier is named, validated, and addressed with a specific, practical strategy; the work-around is both clinically sound and immediately actionable.",
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
            "n10": "No action whatsoever; only general motivation or awareness-raising with no behavioural suggestion.",
            "n5":  "Proposes an action but it is extremely vague or entirely disconnected from the patient's feasible context.",
            "z0":  "Action is present but either slightly vague or lacks a clear time anchor.",
            "p5":  "Specific, small, feasible action step with a time anchor; immediately actionable.",
            "p10": "Implementation intention is precise (what, when, how), calibrated to the patient's specific context, and optimally designed to translate into same-day behaviour.",
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
            "n10": "Message has no recognisable behaviour-change mechanism; highly unlikely to produce any shift in intent, self-efficacy, or action.",
            "n5":  "Message has limited behaviour-change potential; only one weak mechanism present (e.g., vague validation without specificity).",
            "z0":  "Moderately likely to motivate; limited by vague action framing or one missing mechanism.",
            "p5":  "Strong behaviour-change potential: barrier addressed, specific action provided, self-efficacy supported.",
            "p10": "Optimally designed for behaviour change; integrates all key mechanisms (barrier, specific action, efficacy, personalisation) in a coherent and compelling message.",
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
            "n10": "Tone is actively harmful — shaming, patronising, clinically cold, or riddled with diagnostic jargon that distances the patient.",
            "n5":  "Tone is generally acceptable but contains one phrase that is dismissive, condescending, or overly clinical.",
            "z0":  "Appropriate tone overall; one minor jargon word or slightly detached sentence.",
            "p5":  "Warm, empathic, respectful, and jargon-free throughout; immediately readable by the patient.",
            "p10": "Exceptional tone — genuinely empathic, professionally precise, and completely free of problematic language; an exemplar of digital therapeutic communication.",
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
            "n10": "Message is far too long or so cognitively dense it cannot function as a mobile notification.",
            "n5":  "Within length limit but one sentence is very long or uses complex nested syntax unsuitable for the mobile medium.",
            "z0":  "Generally concise and readable; one sentence is slightly longer than ideal for a phone screen.",
            "p5":  "Perfectly concise; each sentence is short and immediately readable on a phone screen.",
            "p10": "Exemplary mobile readability — every sentence is short, punchy, and immediately comprehensible; optimal for push-notification delivery.",
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
            "n10": "Could have been written for any patient in any therapeutic context; no case-specific details whatsoever.",
            "n5":  "Only one generic reference to the broad complaint area; no meaningful case-specific cues.",
            "z0":  "Partially personalised; uses one concrete case-specific detail (symptom, goal, or barrier).",
            "p5":  "Uses two or more concrete case-specific cues woven naturally into the message.",
            "p10": "Highly personalised throughout; three or more specific case cues integrated seamlessly; reads as written for exactly this patient.",
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
            "n10": "Contains shaming language, coercive pressure, unrealistic promises, or advice that could directly worsen the patient's distress.",
            "n5":  "One phrase is notably judgmental, mildly coercive, or makes a questionable promise.",
            "z0":  "Generally safe and non-judgmental; one minor phrase slightly overpromising or implicitly pressuring.",
            "p5":  "Completely safe, non-judgmental, no coercive language, no unrealistic promises.",
            "p10": "Exemplary safety profile — unconditionally validating, zero risk of iatrogenic effect; a model of clinical safety in automated mental-health messaging.",
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
        raise ValueError(
            f"Unknown part {part!r}; expected one of {list(DIMENSIONS_BY_PART)}"
        )
    return DIMENSIONS_BY_PART[part]


__all__ = [
    "PROMPT_VERSION",
    "QUALITY_MIN",
    "QUALITY_MAX",
    "QUALITY_NEUTRAL",
    "QUALITY_SCALE_ANCHORS",
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
