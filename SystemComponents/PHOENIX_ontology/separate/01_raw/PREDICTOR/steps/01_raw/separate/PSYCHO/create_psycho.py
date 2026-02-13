#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
create_psycho.py — PHOENIX_ontology PSYCHO sub-ontology generator (Solution-Variable-Oriented)

Design principles (hard constraints):
- Leaf nodes represent actionable PSYCHO "solution variables" (interventions, skills, choices, delivery formats).
- No disorder-labeled branches (avoid disorder names as category labels; avoid *_Disorder / *_Syndrome / DSM / ICD).
- No frequency/duration/intensity parameters as nodes (no schedules, doses, minutes, sessions, Hz, etc.).
- PSYCHO only (exclude BIO predictors such as meds/nutrition/physiology; exclude SOCIAL predictors such as finances/housing/policy).
- High-resolution psychotherapy coverage (breadth + technique-level depth).

Outputs:
  1) PSYCHO.json
  2) metadata.txt (same folder)

Override output path:
  PSYCHO_OUT_PATH="/path/to/PSYCHO.json" python create_psycho.py

Notes:
- This file intentionally uses verbose lists to keep ontology explicit and inspectable.
- Keys use underscore naming and are meant to be stable identifiers.
"""

import os
import json
import re
import hashlib
import datetime
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple


# ------------------------ Ontology builder ------------------------

@dataclass
class OntologyBuilder:
    root: Dict[str, Any] = field(default_factory=dict)

    def add_path(self, path: List[str]) -> None:
        node = self.root
        for p in path:
            if not isinstance(node, dict):
                raise TypeError(f"Ontology corruption: expected dict at '{p}', got {type(node)}")
            node = node.setdefault(p, {})

    def add_leaves(self, base_path: List[str], leaves: Iterable[str]) -> None:
        for leaf in leaves:
            self.add_path(base_path + [leaf])

    def merge_subtree(self, base_path: List[str], subtree: Dict[str, Any]) -> None:
        """
        Merge a dict subtree at base_path. Existing keys are recursively merged.
        """
        self.add_path(base_path)
        node = self._get_node(base_path)
        self._deep_merge(node, subtree)

    def _get_node(self, path: List[str]) -> Dict[str, Any]:
        node = self.root
        for p in path:
            node = node[p]
        return node

    @staticmethod
    def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        for k, v in src.items():
            if k not in dst:
                dst[k] = v
            else:
                if isinstance(dst[k], dict) and isinstance(v, dict):
                    OntologyBuilder._deep_merge(dst[k], v)
                else:
                    raise TypeError(f"Merge conflict at key '{k}': {type(dst[k])} vs {type(v)}")


# ------------------------ Tree helpers ------------------------

def count_leaves(node: Any) -> int:
    # Leaves are empty dicts
    if isinstance(node, dict):
        if not node:
            return 1
        return sum(count_leaves(v) for v in node.values())
    return 1

def count_nodes(node: Any) -> int:
    if isinstance(node, dict):
        return 1 + sum(count_nodes(v) for v in node.values())
    return 1

def max_depth(node: Any, depth: int = 0) -> int:
    if isinstance(node, dict) and node:
        return max(max_depth(v, depth + 1) for v in node.values())
    return depth

def iter_leaf_paths(node: Any, prefix: Tuple[str, ...] = ()) -> Iterable[Tuple[str, ...]]:
    if isinstance(node, dict):
        if not node:
            yield prefix
        else:
            for k, v in node.items():
                yield from iter_leaf_paths(v, prefix + (k,))
    else:
        yield prefix

def subtree_leaf_counts(root: Dict[str, Any]) -> Dict[str, int]:
    return {k: count_leaves(v) for k, v in root.items()}

def scan_forbidden_tokens(paths: List[Tuple[str, ...]], forbidden_patterns: List[str], limit: int = 50):
    bad = []
    for p in paths:
        s = " ".join(p)
        for pat in forbidden_patterns:
            if re.search(pat, s, flags=re.IGNORECASE):
                bad.append((p, pat))
                break
        if len(bad) >= limit:
            break
    return bad


# ------------------------ PSYCHO ontology content ------------------------

def build_psycho_ontology() -> Dict[str, Any]:
    """
    Returns dict shaped as {"PSYCHO": {...}}.
    """
    b = OntologyBuilder(root={})
    PSYCHO: Dict[str, Any] = {}

    # ============================================================
    # 0) Foundations / Common Factors / Therapy Architecture
    # ============================================================
    foundations = OntologyBuilder().root

    alliance = [
        # Alliance + stance
        "Therapeutic_Alliance_Explicit_Contracting",
        "Collaborative_Goal_Setting",
        "Shared_Case_Formulation_Creation",
        "Empathic_Reflection_Skills",
        "Validation_and_Normalization",
        "Rupture_Recognition_and_Repair",
        "Expectation_Alignment",
        "Therapy_Rationale_Clarity",
        "Session_Agenda_Collaboration",
        "Feedback_Informed_Therapy_Checkins",
        "Cultural_Humility_Practices",
        "Values_Sensitive_Language_Choices",
        "Therapist_Transparency_Appropriate",
        "Strengths_Based_Framing",
        "Hope_Induction_Strategies",
        "Motivation_Enhancement_Dialogue",
        "Collaborative_Experiment_Mindset",
        "Boundary_Setting_and_Containment",
        "Therapeutic_Consistency_and_FollowThrough",
        "Therapeutic_Stance_Flexibility",
        "Shared_Decision_Making_Practice",
        "Therapeutic_Pacing_Adjustment",
        "Alliance_Monitoring_And_Repair_Routine",
        # Engagement + adherence (psychological levers; schedule-free)
        "Barrier_Elicitation_And_Problem_Solving_For_Engagement",
        "Treatment_Expectancy_Shaping",
        "Motivation_Clarification_Values_Link",
        "Dropout_Risk_Conversation",
        "Therapy_Process_Psychoeducation",
        "Normalize_Setbacks_And_Learning_Frame",
        "Skill_Generalization_Conversation",
        "Therapy_Toolbox_Consolidation",
        # Safety in a non-crisis framing (still psycho-level)
        "Early_Warning_Signs_Recognition_Framework",
        "Coping_Plan_Structure_NonCrisis",
        "Support_Seeking_Scripts_NonCrisis",
        "Means_Safety_Conversation_Framework",
    ]
    add_leaves(foundations, ["Therapeutic_Relationship_and_Context"], alliance)

    formulation = [
        "CBT_5P_Formulation_Framework",
        "Functional_Analysis_ABC_Model",
        "Behavior_Chain_Analysis",
        "ACT_Matrix_Formulation",
        "Schema_Mode_Map",
        "Psychodynamic_Conflict_Formulation",
        "Interpersonal_Pattern_Map",
        "Attachment_Informed_Formulation",
        "Narrative_Reauthoring_Map",
        "Systemic_Relational_Map",
        "Strengths_and_Resources_Map",
        "Trigger_Response_Consequence_Map",
        "Emotion_Process_Map",
        "Values_Goals_Barriers_Map",
        "Cognitive_Model_Personalized_Map",
        "Learning_Model_Inhibitory_Learning_Map",
        "Maintenance_Cycle_Diagramming",
        "Risk_And_Protective_Factors_Map",
        "Self_Regulation_Cycle_Map",
        "Avoidance_Safety_Behavior_Map",
        # Enriched conceptual tools
        "Metacognitive_Model_Formulation",
        "Threat_Monitoring_And_Attention_Cycle_Map",
        "Self_Criticism_Shame_Cycle_Map",
        "Interoceptive_Appraisal_Cycle_Map",
        "Perfectionism_Standards_Cycle_Map",
        "Emotion_Driven_Behavior_Map",
        "Values_Conflict_Map",
        "Coping_Strategy_Function_Map",
        "Strengths_Deployment_Map",
    ]
    add_leaves(foundations, ["Case_Formulation_and_Treatment_Planning"], formulation)

    goal_work = [
        "SMART_Goal_Specification",
        "Values_Clarification_Exercise",
        "Life_Domain_Prioritization",
        "Approach_Goal_Refinement",
        "Avoidance_Goal_Reframe_to_Approach",
        "Implementation_Intention_IfThen_Plans",
        "Barrier_Anticipation_and_Contingencies",
        "Skill_Generalization_Plan",
        "Relapse_Prevention_Framework",
        "Maintenance_Plan_Creation",
        "Homework_Planning_and_Review_System",
        "Between_Session_Support_Structure",
        "Progress_Milestone_Mapping",
        "Choice_Point_Identification",
        "Personal_Resource_Activation_Plan",
        "Treatment_Prioritization_Triage_Map",
        "Therapy_Toolbox_Personalization",
        # Evidence-based self-regulation planning (schedule-free)
        "Mental_Contrasting_Future_Obstacle_Link",
        "WOOP_Framework_Practice",
        "Identity_Based_Goal_Linking",
        "Commitment_Device_Design_Psychological",
        "Obstacle_Forecasting_And_Alternative_Paths",
        "Setback_Recovery_Scripts",
        "Self_Efficacy_Building_Milestones",
        "Values_Aligned_Boundary_Setting_Plan",
    ]
    add_leaves(foundations, ["Goals_Values_and_Maintenance_Planning"], goal_work)

    # ============================================================
    # Psychoeducation Modules (expanded; broad + deep; psycho-only)
    # Structure: Domain -> Subdomain -> Leaf modules
    # Leaf nodes = teachable psychoeducation units (not dosage; not disorders)
    # ============================================================

    psychoed = OntologyBuilder().root

    # ------------------------------------------------------------
    # A) Stress, Arousal, and Self-Regulation
    # ------------------------------------------------------------
    add_leaves(psychoed, ["Stress_and_Arousal", "Stress_Response_Systems"], [
        "Psychoeducation_Stress_Response_System",
        "Psychoeducation_Sympathetic_Parasympathetic_Balance",
        "Psychoeducation_Arousal_Curves_And_Performance",
        "Psychoeducation_Stress_Appraisal_And_Coping",
        "Psychoeducation_Threat_Safety_Signals",
        "Psychoeducation_Allostatic_Load_Concept_Psychological",
    ])

    add_leaves(psychoed, ["Stress_and_Arousal", "Interoception_and_Bodily_Sensations"], [
        "Psychoeducation_Interoception_And_Arousal",
        "Psychoeducation_Body_Sensations_and_Appraisal",
        "Psychoeducation_Panic_Sensations_As_Safe_But_Intense",
        "Psychoeducation_Somatic_Attention_Amplification",
        "Psychoeducation_Breathing_Arousal_Link",
        "Psychoeducation_Temperature_Tension_Tremor_Normalization",
    ])

    add_leaves(psychoed, ["Stress_and_Arousal", "Self_Regulation_Cycles"], [
        "Psychoeducation_Self_Regulation_Model",
        "Psychoeducation_Emotion_Regulation_Model",
        "Psychoeducation_Window_Of_Tolerance_Concept",
        "Psychoeducation_Recovery_And_Depletion",
        "Psychoeducation_Decision_Fatigue_And_Self_Control",
        "Psychoeducation_TopDown_BottomUp_Regulation_Language",
    ])

    # ------------------------------------------------------------
    # B) Emotion Science (functions, differentiation, action)
    # ------------------------------------------------------------
    add_leaves(psychoed, ["Emotion_Science", "Emotion_Functions_and_Signals"], [
        "Psychoeducation_Emotion_Functions",
        "Psychoeducation_Emotions_As_Information_Not_Instructions",
        "Psychoeducation_Action_Tendencies_And_Choice",
        "Psychoeducation_Primary_vs_Secondary_Emotions",
        "Psychoeducation_Emotion_Granularity_Why_It_Matters",
        "Psychoeducation_Moods_vs_Emotions_Distinction",
    ])

    add_leaves(psychoed, ["Emotion_Science", "Emotion_Driven_Behavior_and_Learning"], [
        "Psychoeducation_Emotion_Driven_Behavior",
        "Psychoeducation_Emotion_Reinforcement_Loops",
        "Psychoeducation_Avoidance_Maintains_Distress",
        "Psychoeducation_Safety_Behaviors_Maintain_Threat_Beliefs",
        "Psychoeducation_Opposite_Action_Rationale",
        "Psychoeducation_Exposure_Learning_Model",
    ])

    add_leaves(psychoed, ["Emotion_Science", "Complex_Emotions"], [
        "Psychoeducation_Shame_Guilt_Differences",
        "Psychoeducation_Anger_As_Boundary_Signal",
        "Psychoeducation_Jealousy_And_Threat_Perception",
        "Psychoeducation_Compassion_and_Self_Criticism",
        "Psychoeducation_Self_Compassion_Systems_Model",
        "Psychoeducation_Grief_and_Adaptation",
    ])

    # ------------------------------------------------------------
    # C) Cognition, Attention, and Metacognition
    # ------------------------------------------------------------
    add_leaves(psychoed, ["Cognition_and_Thinking", "Thoughts_Beliefs_and_Appraisals"], [
        "Psychoeducation_Thoughts_Are_Not_Facts",
        "Psychoeducation_Appraisal_Emotion_Link",
        "Psychoeducation_Core_Beliefs_And_Rules",
        "Psychoeducation_Cognitive_Distortions_As_Patterns",
        "Psychoeducation_Meaning_Making_And_Interpretation",
        "Psychoeducation_Beliefs_As_Testable_Hypotheses",
    ])

    add_leaves(psychoed, ["Cognition_and_Thinking", "Attention_and_Control"], [
        "Psychoeducation_Attention_and_Worry",
        "Psychoeducation_Attention_Is_Trainable",
        "Psychoeducation_Attentional_Bias_To_Threat",
        "Psychoeducation_Internal_vs_External_Attention",
        "Psychoeducation_Attention_Fatigue_And_Reset",
        "Psychoeducation_Distraction_vs_Avoidance_Distinction",
    ])

    add_leaves(psychoed, ["Cognition_and_Thinking", "Metacognition_and_Beliefs_About_Thinking"], [
        "Psychoeducation_Metacognitive_Beliefs_About_Thinking",
        "Psychoeducation_Thought_Suppression_Rebound_Effect",
        "Psychoeducation_Worry_As_Cognitive_Avoidance",
        "Psychoeducation_Rumination_vs_Reflection",
        "Psychoeducation_Meta_Worry_And_Safety_Seeking",
        "Psychoeducation_Cognitive_Flexibility_Concept",
    ])

    add_leaves(psychoed, ["Cognition_and_Thinking", "Cognitive_Biases_and_Heuristics"], [
        "Psychoeducation_Cognitive_Biases",
        "Psychoeducation_Confirmation_Bias",
        "Psychoeducation_Negativity_Bias",
        "Psychoeducation_Availability_Heuristic",
        "Psychoeducation_Intolerance_Of_Uncertainty_Concept",
        "Psychoeducation_Perfectionism_Standards_And_Costs",
    ])

    # ------------------------------------------------------------
    # D) Learning, Habits, and Motivation
    # ------------------------------------------------------------
    add_leaves(psychoed, ["Learning_and_Habit_Change", "Learning_Principles"], [
        "Psychoeducation_Learning_and_Habit_Loops",
        "Psychoeducation_Classical_Conditioning_Basics",
        "Psychoeducation_Operant_Learning_Basics",
        "Psychoeducation_Generalization_And_Discrimination",
        "Psychoeducation_Extinction_And_New_Learning",
        "Psychoeducation_Inhibitory_Learning_Principles",
    ])

    add_leaves(psychoed, ["Learning_and_Habit_Change", "Habit_Systems_and_Automaticity"], [
        "Psychoeducation_Habit_Automaticity_And_Cues",
        "Psychoeducation_Trigger_Response_Reward_Loop",
        "Psychoeducation_Safety_Behavior_Habit_Loops",
        "Psychoeducation_Implementation_Intentions_Rationale",
        "Psychoeducation_Identity_And_Habit_Stability",
        "Psychoeducation_Context_Dependence_Of_Habits",
    ])

    add_leaves(psychoed, ["Learning_and_Habit_Change", "Motivation_and_Reward"], [
        "Psychoeducation_Reward_and_Motivation",
        "Psychoeducation_Approach_Avoidance_Motivation",
        "Psychoeducation_Dopamine_As_Learning_Signal_Lay_Explanation",
        "Psychoeducation_Small_Wins_And_Momentum",
        "Psychoeducation_Self_Efficacy_Concept",
        "Psychoeducation_Values_As_Motivation_Source",
    ])

    add_leaves(psychoed, ["Learning_and_Habit_Change", "Cravings_Urges_and_Compulsions_PsychoOnly"], [
        "Psychoeducation_Cravings_And_Urges_As_Transient",
        "Psychoeducation_Urge_Surfing_Rationale",
        "Psychoeducation_Compulsions_As_Negative_Reinforcement",
        "Psychoeducation_Checking_Reassurance_Trap",
        "Psychoeducation_Delay_Distraction_And_Choice",
        "Psychoeducation_Lapse_vs_Relapse_Distinction",
    ])

    # ------------------------------------------------------------
    # E) Interpersonal and Attachment Processes (psychoeducation, not “social interventions”)
    # ------------------------------------------------------------
    add_leaves(psychoed, ["Interpersonal_Processes", "Interpersonal_Cycles_and_Patterns"], [
        "Psychoeducation_Interpersonal_Cycles",
        "Psychoeducation_Conflict_Escalation_Cycle",
        "Psychoeducation_Pursue_Withdraw_Dance",
        "Psychoeducation_Rupture_Repair_Normalization",
        "Psychoeducation_Boundaries_And_Relational_Safety",
        "Psychoeducation_Validation_Why_It_Works",
    ])

    add_leaves(psychoed, ["Interpersonal_Processes", "Attachment_Lens_Psychoeducation"], [
        "Psychoeducation_Attachment_Needs_And_Security",
        "Psychoeducation_Protest_Despair_Repair_Map",
        "Psychoeducation_Internal_Working_Models_Concept",
        "Psychoeducation_Mentalizing_And_Misattunement",
        "Psychoeducation_Shame_And_Connection_Dynamics",
        "Psychoeducation_Trust_As_Predictable_Repair",
    ])

    # ------------------------------------------------------------
    # F) Mindfulness and Acceptance (principles + mechanisms)
    # ------------------------------------------------------------
    add_leaves(psychoed, ["Mindfulness_and_Acceptance", "Core_Principles"], [
        "Psychoeducation_Mindfulness_Principles",
        "Psychoeducation_Nonjudgment_vs_Evaluation",
        "Psychoeducation_Decentering_And_Reperceiving",
        "Psychoeducation_Acceptance_vs_Resignation",
        "Psychoeducation_Experiential_Avoidance_Concept",
        "Psychoeducation_Observer_Stance_Concept",
    ])

    add_leaves(psychoed, ["Mindfulness_and_Acceptance", "Mechanisms_and_Training"], [
        "Psychoeducation_Attentional_Training_In_Mindfulness",
        "Psychoeducation_Relationship_To_Thoughts_Shift",
        "Psychoeducation_Urges_As_Bodily_Events",
        "Psychoeducation_Self_Compassion_And_Safety_System",
        "Psychoeducation_Pain_vs_Suffering_Distinction_PsychoOnly",
    ])

    # ------------------------------------------------------------
    # G) Trauma, Memory, and Meaning (psycho-only)
    # ------------------------------------------------------------
    add_leaves(psychoed, ["Trauma_Memory_and_Adaptation", "Memory_Systems_and_Trauma"], [
        "Psychoeducation_Trauma_Memory_Basics",
        "Psychoeducation_Trauma_Triggers_And_Cue_Linkages",
        "Psychoeducation_Intrusions_As_Memory_Reactivation",
        "Psychoeducation_Avoidance_And_NonIntegration",
        "Psychoeducation_Reconsolidation_Lay_Explanation",
        "Psychoeducation_PostEvent_Meaning_Making_Process",
    ])

    add_leaves(psychoed, ["Trauma_Memory_and_Adaptation", "Safety_and_Stabilization_Concepts"], [
        "Psychoeducation_Stabilization_And_Resourcing_Rationale",
        "Psychoeducation_Grounding_Why_It_Works",
        "Psychoeducation_Window_Of_Tolerance_In_Trauma_Lens",
        "Psychoeducation_Shame_Guilt_In_Trauma_Lens",
    ])

    # ------------------------------------------------------------
    # H) Sleep, Energy, and Cognitive-Affective Spirals (psycho-only framing)
    # ------------------------------------------------------------
    add_leaves(psychoed, ["Sleep_and_Restoration_Psychoeducation", "Sleep_Cognition_Loops"], [
        "Psychoeducation_Sleep_Worry_Cycle_Psychological",
        "Psychoeducation_PreSleep_Hyperarousal_Loop",
        "Psychoeducation_Insomnia_As_Learned_Arousal_Lay_Model",
        "Psychoeducation_Safety_Behaviors_Around_Sleep",
        "Psychoeducation_Paradoxical_Intention_Rationale",
    ])

    # ------------------------------------------------------------
    # I) Values, Goals, and Identity (meaning-based psychoeducation)
    # ------------------------------------------------------------
    add_leaves(psychoed, ["Values_Goals_and_Identity", "Values_and_Choice_Points"], [
        "Psychoeducation_Values_And_Choice_Points",
        "Psychoeducation_Approach_vs_Avoidance_Goals",
        "Psychoeducation_Identity_Narratives_And_Behavior",
        "Psychoeducation_Self_Concept_Flexibility",
        "Psychoeducation_Goal_Conflict_And_Ambivalence",
        "Psychoeducation_Hope_As_Goal_Pathways_Thinking",
    ])

    # Attach subtree
    foundations["Psychoeducation_Modules"] = psychoed

    ethics = [
        "Informed_Consent_Clarity",
        "Confidentiality_And_Limits_Psychoeducation",
        "Collaborative_Risk_Discussion_Framework",
        "Goal_Ownership_And_Autonomy_Support",
        "Shared_Decision_Making_Risks_Benefits",
        "Privacy_And_Data_Use_Explanation_For_Digital_Tools",
        "Scope_Of_Practice_Transparency",
        "Documentation_Transparency_Conversation",
    ]
    add_leaves(foundations, ["Ethics_Consent_and_Safety"], ethics)

    PSYCHO["Foundations_and_Common_Factors"] = foundations

    # ============================================================
    # 1) Cognitive-Behavioral Therapies (WAY expanded, CBT-first, technique-level)
    # Design rules honored:
    # - Leaves are actionable PSYCHO solution variables
    # - No disorder-labeled branches
    # - No dose/frequency/time parameters as nodes
    # - Rich categorization (multiple layers) instead of long flat lists
    # ============================================================

    cbt = OntologyBuilder().root

    # ------------------------------------------------------------
    # 1.0 CBT Foundations (micro-processes that organize CBT work)
    # ------------------------------------------------------------
    add_leaves(cbt, ["CBT_Foundations", "Case_Conceptualization_and_Targeting", "Models_and_Maps"], [
        "CBT_Cycle_Thought_Emotion_Behavior_Map",
        "Situation_Thought_Emotion_Behavior_Link_Map",
        "Trigger_Response_Consequence_Map_CBT",
        "Safety_Behavior_Map",
        "Avoidance_Maintenance_Map",
        "Cognitive_Model_Personalized_Map",
        "Functional_Analysis_ABC_Model_CBT",
        "Behavior_Chain_Analysis_CBT",
        "Problem_List_And_Target_Prioritization",
        "Mechanism_Target_Linkage_Map",
        "Feedback_Loop_Mapping_For_Change",
    ])

    add_leaves(cbt, ["CBT_Foundations", "Collaborative_Methods", "Therapy_Workstyle"], [
        "Guided_Discovery_Stance",
        "Collaborative_Empiricism_Practice",
        "Agenda_Setting_CBT_Style",
        "Homework_Collaboration_Framework",
        "Experiment_Mindset_Orientation",
        "Skills_Generalization_Planning_CBT",
        "Relapse_Prevention_Framework_CBT",
        "Maintenance_Plan_CBT",
    ])

    # ------------------------------------------------------------
    # 1.1 Cognitive Restructuring (decomposed into skill families)
    # ------------------------------------------------------------
    # 1.1.A Identify & capture cognition
    add_leaves(cbt, ["Cognitive_Restructuring", "Elicit_and_Capture", "Monitoring_and_Noticing"], [
        "Thought_Monitoring_Log",
        "Automatic_Thought_Identification",
        "Hot_Thought_Identification",
        "Image_Or_Memory_As_Cognition_Identification",
        "Prediction_Identification",
        "Interpretation_Identification",
        "Assumption_Rule_Identification",
        "Self_Talk_Transcription_Practice",
        "Situation_Specific_Belief_Elicitation",
    ])

    add_leaves(cbt, ["Cognitive_Restructuring", "Elicit_and_Capture", "Clarify_Meaning"], [
        "Clarify_Thought_To_Single_Sentence",
        "Specify_Trigger_Context_Details",
        "Specify_Emotion_Link_To_Thought",
        "Specify_Behavior_Urge_Link_To_Thought",
        "Distinguish_Observation_vs_Interpretation",
    ])

    # 1.1.B Evaluate cognition (evidence, logic, probability)
    add_leaves(cbt, ["Cognitive_Restructuring", "Evaluate", "Socratic_and_Guided_Discovery"], [
        "Socratic_Questioning",
        "Guided_Discovery",
        "Downward_Arrow_Technique",
        "Upward_Arrow_Values_Link",
        "Perspective_Taking_Reframe",
        "Self_Distancing_Reappraisal",
        "Perspective_Broadening_View_From_Above",
        "Temporal_Distancing_Reappraisal",
    ])

    add_leaves(cbt, ["Cognitive_Restructuring", "Evaluate", "Evidence_and_Data"], [
        "Evidence_For_Against_Evaluation",
        "Reality_Testing_With_Data_Collection",
        "Survey_Method_Belief_Testing",
        "Behavioral_Experiment_Design",
        "Prediction_Checking_With_Data_Log",
        "Alternative_Attribution_Generation",
        "Alternative_Explanation_Generation",
        "Base_Rate_Check_Practice",
    ])

    add_leaves(cbt, ["Cognitive_Restructuring", "Evaluate", "Probability_and_Catastrophe_Work"], [
        "Probability_Reestimation",
        "Decatastrophizing_Sequence",
        "Best_Worst_MostLikely_Scenarios",
        "Time_Projection_Reappraisal",
        "Reappraisal_Of_Uncertainty_As_Survivable",
        "Ambiguity_Interpretation_Rebalance",
    ])

    # 1.1.C Restructure cognition (generate and install alternatives)
    add_leaves(cbt, ["Cognitive_Restructuring", "Restructure_and_Install", "Alternative_Thinking"], [
        "Cognitive_Reappraisal_Practice",
        "Balanced_Thought_Generation",
        "Helpful_Unhelpful_Thought_Sorting",
        "Meaning_Reappraisal_Practice",
        "Values_Consistent_Cognition_Filter",
        "Compassionate_Reframe",
        "Double_Standard_Technique",
        "Imagery_Cognitive_Reframe",
    ])

    add_leaves(cbt, ["Cognitive_Restructuring", "Restructure_and_Install", "Belief_Strength_Modulation"], [
        "Belief_Rating_and_ReRating",
        "Belief_Conditionality_Softening",
        "Continuum_Technique",
        "Perfectionism_Rule_Softening",
        "Reframe_Control_As_Influence",
        "Cognitive_Reframing_Of_Intentionality",
    ])

    # 1.1.D Cognitive distortion “checks” (organized by bias families)
    add_leaves(cbt, ["Cognitive_Restructuring", "Cognitive_Bias_Checks", "Inference_Biases"], [
        "Mind_Reading_Check",
        "Fortune_Telling_Check",
        "Jumping_To_Conclusions_Check",
        "Confirmation_Bias_Check",
        "Selective_Abstraction_Check",
    ])

    add_leaves(cbt, ["Cognitive_Restructuring", "Cognitive_Bias_Checks", "Appraisal_Biases"], [
        "All_Or_Nothing_Shade_Finding",
        "Overgeneralization_Check",
        "Personalization_Check",
        "Labeling_Reframe",
        "Should_Statements_Reframe",
        "Emotional_Reasoning_Check",
    ])

    # 1.1.E Emotion-linked reappraisal modules (content-specific but disorder-free)
    add_leaves(cbt, ["Cognitive_Restructuring", "Emotion_Linked_Reappraisal", "Threat_and_Anxiety"], [
        "Threat_Appraisal_Reframe",
        "Reattribute_Threat_To_Arousal_Sensitivity",
        "Safety_Probability_Reframe",
        "Tolerate_Uncertainty_Cognitive_Frame",
    ])

    add_leaves(cbt, ["Cognitive_Restructuring", "Emotion_Linked_Reappraisal", "Anger_and_Moral_Appraisal"], [
        "Anger_Appraisal_Reframe",
        "Intent_Assignment_Check_Practice",
        "Fairness_Standard_Flexibilization",
        "Responsibility_Boundary_Reframe",
    ])

    add_leaves(cbt, ["Cognitive_Restructuring", "Emotion_Linked_Reappraisal", "Shame_Guilt_Self_Evaluation"], [
        "Shame_Appraisal_Reframe",
        "Responsibility_Reappraisal",
        "Self_Critical_Thought_Restructure",
        "Compassionate_Self_Evaluation_Reframe",
        "Pie_Chart_Responsibility_Share",
    ])

    # 1.1.F Tools and worksheets (structured)
    add_leaves(cbt, ["Cognitive_Restructuring", "Tools_and_Worksheets", "Core_Forms"], [
        "Thought_Record_Short_Form",
        "Thought_Record_Long_Form",
        "Evidence_Log_Practice",
        "Cognitive_Distortion_Checklist",
        "Reframe_Prompt_Cards",
    ])

    add_leaves(cbt, ["Cognitive_Restructuring", "Tools_and_Worksheets", "Belief_Level_Work"], [
        "Belief_Ladder_Worksheet",
        "Core_Belief_Worksheet",
        "Responsibility_Distribution_Worksheet",
        "Uncertainty_Appraisal_Worksheet",
        "Values_Belief_Consistency_Checksheet",
        "Self_Compassion_Reframe_Worksheet",
        "Perspective_Shift_Prompt_Cards",
    ])

    add_leaves(cbt, ["Cognitive_Restructuring", "Tools_and_Worksheets", "Rehearsal_and_Consolidation"], [
        "Cognitive_Rehearsal_Script",
        "Coping_Statement_Cards",
        "Learning_Summary_From_Reappraisal",
    ])

    # ------------------------------------------------------------
    # 1.1.1 REBT / Rational belief work (explicit; more layered)
    # ------------------------------------------------------------
    add_leaves(cbt, ["Rational_Belief_Work_REBT_Family", "ABCDE_and_Disputation", "ABCDE_Core"], [
        "ABCDE_Model_Application",
        "Belief_Disputation_Empirical",
        "Belief_Disputation_Logical",
        "Belief_Disputation_Pragmatic",
    ])

    add_leaves(cbt, ["Rational_Belief_Work_REBT_Family", "Philosophical_Shifts", "Demandingness_and_Awfulizing"], [
        "Preference_Over_Demandingness_Shift",
        "Catastrophe_To_Bad_Not_Terrible_Reframe",
        "High_Frustration_Tolerance_Building",
    ])

    add_leaves(cbt, ["Rational_Belief_Work_REBT_Family", "Acceptance_Work", "Unconditional_Acceptance"], [
        "Unconditional_Self_Acceptance_Practice",
        "Unconditional_Other_Acceptance_Practice",
        "Unconditional_Life_Acceptance_Practice",
    ])

    add_leaves(cbt, ["Rational_Belief_Work_REBT_Family", "Imagery_and_Emotive_Methods", "Emotive_Exercises"], [
        "Rational_Emotive_Imagery_Practice",
        "Shame_Attack_REBT_Variant",
    ])

    add_leaves(cbt, ["Rational_Belief_Work_REBT_Family", "Rational_Self_Talk", "Coping_Language"], [
        "Rational_Belief_Formulation",
        "Coping_Self_Statements_Rational",
        "Philosophical_Change_Dialogue",
    ])

    # ------------------------------------------------------------
    # 1.2 Behavioral Activation & Behavior Change (more structured)
    # ------------------------------------------------------------
    add_leaves(cbt, ["Behavior_Change_and_Activation", "Assessment_and_Functional_Analysis", "Monitoring"], [
        "Activity_Monitoring",
        "Procrastination_Functional_Analysis",
        "Avoidance_Reduction_Plan",
        "Approach_Behavior_Planning",
        "Behavioral_Review_And_Adjust_Cycle",
    ])

    add_leaves(cbt, ["Behavior_Change_and_Activation", "Planning_and_Grading", "Activation_Design"], [
        "Values_Based_Activity_Selection",
        "Pleasure_Mastery_Balance_Planning",
        "Graded_Task_Assignment",
        "Behavioral_Activity_Planning_System",
        "Task_Initiation_Cues",
        "Action_Labeling_And_Start_Cue",
        "Obstacle_IfThen_Planning",
        "Implementation_Intention_Activation",
    ])

    add_leaves(cbt, ["Behavior_Change_and_Activation", "Contingency_and_Environment", "Reinforcement_and_Design"], [
        "Reward_Engineering",
        "Contingency_Management_Self_Administered",
        "Response_Cost_Self_Administered",
        "Stimulus_Control_For_Habits",
        "Environmental_Restructuring_For_Action",
        "Reduce_Friction_For_Target_Behavior",
        "Increase_Friction_For_Undesired_Behavior",
        "Precommitment_To_Remove_Temptations",
        "Commitment_Device_Self_Administered",
        "Temptation_Bundling_Design",
    ])

    add_leaves(cbt, ["Behavior_Change_and_Activation", "Skill_Shaping_and_Routine", "Building_Momentum"], [
        "Behavioral_Momentum_Strategy",
        "Small_Wins_Protocol",
        "Shaping_And_Successive_Approximation",
        "Behavioral_Substitution_Planning",
        "Behavioral_Routine_Design",
        "Behavioral_Contracting_Self",
        "Identity_Statement_For_Behavior",
        "Opposite_Action_Behavioral",
        "Activation_With_Accountability_Partner",
    ])

    add_leaves(cbt, ["Behavior_Change_and_Activation", "Experiment_Linked_Action", "Behavioral_Tests"], [
        "Behavioral_Experiment_Activity_Test",
        "Behavioral_Tracking_With_IfThen_Adjustment",
    ])

    # ------------------------------------------------------------
    # 1.3 Behavioral Experiments & Learning (richer taxonomy)
    # ------------------------------------------------------------
    add_leaves(cbt, ["Behavioral_Experiments_and_Learning", "Design_Principles", "Core_Design"], [
        "Prediction_Experiment_Setup",
        "Alternative_Prediction_Setup",
        "Experiment_Single_Variable_Change",
        "Experiment_Minimize_Confounds_Planning",
        "Experiment_Ethical_Safety_Check",
        "Experiment_Outcome_Measures_Definition",
    ])

    add_leaves(cbt, ["Behavioral_Experiments_and_Learning", "Safety_Behavior_and_Attention", "Disconfirm_Maintainers"],
               [
                   "Safety_Behavior_Elimination_Experiment",
                   "Experiment_Manipulate_Attention_Allocation",
                   "Attention_Focus_Shift_Experiment",
                   "Experiment_Manipulate_Interpretation_Style",
               ])

    add_leaves(cbt, ["Behavioral_Experiments_and_Learning", "Interpersonal_Experiments", "Social_Learning"], [
        "Interpersonal_Signaling_Experiment",
        "Experiment_Test_Other_Minds_Assumptions",
        "Experiment_Test_Assertiveness_Outcome",
        "Experiment_Test_Boundary_Setting_Outcome",
    ])

    add_leaves(cbt, ["Behavioral_Experiments_and_Learning", "Tolerance_and_Emotion", "Affective_Learning"], [
        "Experiment_Test_Emotional_Tolerance",
        "Tolerance_Of_Uncertainty_Experiment",
        "Perfectionism_Rule_Bending_Experiment",
        "Self_Compassion_Behavior_Experiment",
    ])

    add_leaves(cbt, ["Behavioral_Experiments_and_Learning", "Values_and_Approach", "Approach_Learning"], [
        "Values_Aligned_Action_Experiment",
        "Experiment_Test_Approach_Behavior",
        "Approach_Action_In_Hard_Context_Practice",
    ])

    add_leaves(cbt, ["Behavioral_Experiments_and_Learning", "Measurement_and_Consolidation", "Logging_and_Learning"], [
        "Behavioral_Experiment_Data_Log",
        "Post_Experiment_Learning_Summary",
        "Generalization_Experiment_Planning",
        "Context_Variation_Experiment",
        "Experiment_Learning_Summary_Scripting",
    ])

    # ------------------------------------------------------------
    # 1.4 Exposure & Inhibitory Learning (high-resolution subfamilies)
    # ------------------------------------------------------------
    # 1.4.A Design & rationale
    add_leaves(cbt, ["Exposure_and_Inhibitory_Learning", "Design_and_Rationale", "Learning_Model"], [
        "Inhibitory_Learning_Rationale",
        "Design_Exposure_For_Learning_Goals",
        "Expectancy_Violation_Planning",
        "Expectation_Laddering",
        "Violation_Salience_Enhancement",
        "Inhibitory_Learning_Summary_Scripting",
    ])

    add_leaves(cbt, ["Exposure_and_Inhibitory_Learning", "Design_and_Rationale", "Hierarchy_and_Targeting"], [
        "Exposure_Hierarchy_Construction",
        "Graduated_Exposure_Principles",
        "Stimulus_Discrimination_Training",
        "Approach_Action_During_Exposure",
        "Multiple_Context_Exposure",
        "Context_Variation_Exposure",
        "Variability_Exposure_Planning",
        "Retrieval_Cue_Planning",
    ])

    # 1.4.B Safety behaviors & subtle avoidance
    add_leaves(cbt, ["Exposure_and_Inhibitory_Learning", "Safety_Behavior_Work", "Identify_and_Remove"], [
        "Safety_Behavior_Drop_Plan",
        "Identify_And_Remove_Subtle_Safety_Behaviors",
        "Reduce_Reassurance_Seeking_And_Checking",
        "Competing_Response_Prevention",
    ])

    # 1.4.C Modalities
    add_leaves(cbt, ["Exposure_and_Inhibitory_Learning", "Exposure_Modalities", "In_Vivo"], [
        "InVivo_Exposure_Framework",
        "Social_Evaluation_Exposure",
        "Shame_Attack_Exercises",
        "Perfectionism_Exposure_Practice",
        "Uncertainty_Exposure_Practice",
    ])

    add_leaves(cbt, ["Exposure_and_Inhibitory_Learning", "Exposure_Modalities", "Imaginal"], [
        "Imaginal_Exposure_Framework",
        "Imagery_Exposure_Script_Construction",
        "Hotspot_Imaginal_Focus_Practice",
    ])

    add_leaves(cbt, ["Exposure_and_Inhibitory_Learning", "Exposure_Modalities", "Interoceptive"], [
        "Interoceptive_Exposure_Framework",
        "Arousal_Sensation_Reinterpretation_Practice",
        "Interoceptive_Safety_Behavior_Drop",
    ])

    add_leaves(cbt, ["Exposure_and_Inhibitory_Learning", "Exposure_Modalities", "Cue_and_Trigger"], [
        "Cue_Exposure_Framework",
        "Trigger_Cue_Mapping",
        "Response_Prevention_Principles",
    ])

    add_leaves(cbt, ["Exposure_and_Inhibitory_Learning", "Exposure_Modalities", "Mindfulness_Integrated"], [
        "Mindful_Exposure_Practice",
        "Allowing_Anxiety_During_Exposure_Practice",
    ])

    # 1.4.D Advanced inhibitory learning enrichments
    add_leaves(cbt, ["Exposure_and_Inhibitory_Learning", "Advanced_Learning_Enhancers", "Extinction_Enhancement"], [
        "Deepened_Extinction_Planning",
        "Occasional_Reinforcement_Exposure",
        "Competing_Expectancy_Training",
        "Cognitive_Load_Exposure_Variation",
    ])

    add_leaves(cbt, ["Exposure_and_Inhibitory_Learning", "Advanced_Learning_Enhancers", "Relapse_Learning_Prevention"],
               [
                   "Reinstatement_Prevention_Planning",
                   "Self_Compassion_After_Exposure",
                   "Post_Exposure_Processing_Guidelines",
                   "Exposure_Learning_Log",
               ])

    # 1.4.E Response prevention & urge work (expanded)
    add_leaves(cbt,
               ["Exposure_and_Inhibitory_Learning", "Response_Prevention_and_Urge_Work", "Response_Prevention_Core"], [
                   "Response_Prevention_Planning",
                   "Ritual_Prevention_Support",
                   "Relapse_Prevention_For_Urges",
               ])

    add_leaves(cbt,
               ["Exposure_and_Inhibitory_Learning", "Response_Prevention_and_Urge_Work", "Urge_Tolerance_and_Surfing"],
               [
                   "Urge_Tolerance_Practice",
                   "Urge_Surfing_Expanded_Practice",
                   "Surf_The_Impulse_With_Breath_Anchor",
                   "Cognitive_Reframe_Urge_As_Wave",
                   "Values_Linked_Urge_Endurance",
                   "Compassionate_Self_Talk_During_Urge",
               ])

    add_leaves(cbt, ["Exposure_and_Inhibitory_Learning", "Response_Prevention_and_Urge_Work", "Delay_and_Alternatives"],
               [
                   "Delay_And_Distract_Urge_Skills",
                   "Urge_Triggered_IfThen_Plan",
                   "Behavioral_Substitution_For_Urge",
               ])

    add_leaves(cbt, ["Exposure_and_Inhibitory_Learning", "Response_Prevention_and_Urge_Work", "Habit_Reversal_Family"],
               [
                   "Habit_Reversal_Competing_Response",
                   "Awareness_Training_For_Urges",
               ])

    # 1.4.F Classical conditioning & counterconditioning subtree (as you had; keep merge)
    sd_subtree: Dict[str, Any] = {}
    add_leaves(sd_subtree, ["Systematic_Desensitization", "Core_Procedure"], [
        "Reciprocal_Inhibition_Rationale",
        "Counterconditioning_Rationale",
        "Fear_Hierarchy_Construction_Classical",
        "Stimulus_Anxiety_Ranking",
        "Relaxation_Response_Training_For_Counterconditioning",
        "Pairing_Relaxation_With_Threat_Cues",
        "Gradual_Stimulus_Progression_Counterconditioning",
        "Conditioned_Response_Replacement_Framework",
        "Physiological_Calm_Association_Building",
        "Anxiety_Response_Inhibition_Practice",
        "Subjective_Anxiety_Rating_Framework",
        "Desensitization_Learning_Log",
        "Generalization_Planning_For_Counterconditioning",
    ])
    add_leaves(sd_subtree, ["Systematic_Desensitization", "Modality_Variants"], [
        "Imaginal_Systematic_Desensitization",
        "InVivo_Systematic_Desensitization",
        "Interoceptive_Systematic_Desensitization",
        "Symbolic_Cue_Desensitization",
        "Script_Based_Desensitization",
        "Audio_Guided_Desensitization",
        "VR_Assisted_Desensitization_Framework",
    ])
    add_leaves(sd_subtree, ["Systematic_Desensitization", "Competing_Response_Skills"], [
        "Progressive_Muscle_Relaxation_Skill",
        "Diaphragmatic_Breathing_Skill",
        "Paced_Breathing_Skill",
        "Autogenic_Training_Skill",
        "Guided_Imagery_Relaxation_Skill",
        "Applied_Relaxation_Skill",
        "Cue_Controlled_Relaxation_Skill",
        "Mindful_Body_Relaxation_Skill",
        "Soothing_Rhythm_Breathing_Skill",
        "Grounding_Five_Senses_Skill",
    ])
    add_leaves(sd_subtree, ["Systematic_Desensitization", "Pairing_Formats"], [
        "Relaxation_Then_Exposure_Pairing",
        "Simultaneous_Relaxation_Exposure_Pairing",
        "Alternating_Relaxation_Exposure_Pairing",
        "Mastery_Imagery_Exposure_Pairing",
        "Safe_Place_To_Threat_Cue_Transition_Pairing",
        "Counterconditioning_With_Safety_Cue_Installation",
        "Counterconditioning_With_Positive_Affect_Induction",
    ])
    add_leaves(sd_subtree, ["Counterconditioning", "Counterconditioning_Types"], [
        "Counterconditioning_With_Approach_Behavior",
        "Counterconditioning_With_Compassionate_Response",
        "Counterconditioning_With_Playfulness_Response",
        "Counterconditioning_With_Self_Efficacy_Priming",
        "Counterconditioning_With_Reward_Association",
        "Counterconditioning_With_Neutralization_Through_Naming",
    ])
    add_leaves(sd_subtree, ["Desensitization_Hierarchy_Design", "Stimulus_Representation_Types"], [
        "Hierarchy_By_Situational_Cues",
        "Hierarchy_By_Sensory_Cues",
        "Hierarchy_By_Internal_Sensations",
        "Hierarchy_By_Social_Evaluation_Cues",
        "Hierarchy_By_Images_And_Memories",
        "Hierarchy_By_Uncertainty_Cues",
        "Hierarchy_By_Performance_Demands",
    ])
    add_leaves(sd_subtree, ["Desensitization_Implementation_Variants"], [
        "Brief_Relaxation_Pairing_Protocol",
        "Extended_Relaxation_Training_First_Protocol",
        "Graduated_Chunked_Exposure_Steps",
        "Micro_Step_Exposure_Progression",
        "Context_Variation_Within_Desensitization",
        "Stimulus_Generalization_Training",
        "Retrieval_Cue_Use_For_Calm_Association",
    ])
    add_leaves(sd_subtree, ["Counterconditioning_and_Inhibitory_Learning_Integration"], [
        "Blend_Counterconditioning_With_Expectancy_Violation",
        "Counterconditioning_As_Aftercare_Post_Exposure",
        "Relaxation_As_Safety_Behavior_Risk_Check",
        "Flexibly_Drop_Relaxation_When_Appropriate",
        "Shift_From_Habituation_Goals_To_Learning_Goals",
    ])

    b2 = OntologyBuilder(root=cbt)
    b2.merge_subtree(
        ["Exposure_and_Inhibitory_Learning", "Classical_Conditioning_and_Counterconditioning"],
        sd_subtree
    )

    # ------------------------------------------------------------
    # 1.5 Perseverative Cognition (worry/rumination) – structured
    # ------------------------------------------------------------
    add_leaves(cbt, ["Perseverative_Cognition_Interventions", "Worry_Management", "Contain_and_Reorient"], [
        "Worry_Postponement_Skill",
        "Worry_Container_Practice",
        "Worry_Triggers_Map",
        "Attention_Refocusing_Routine",
        "Worry_To_Planning_Discrimination",
        "Problem_Solving_vs_Worry_Discrimination",
    ])

    add_leaves(cbt, ["Perseverative_Cognition_Interventions", "Rumination_Interventions", "Disengage_and_Shift"], [
        "Set_Shifting_From_Rumination",
        "Rumination_Cue_Interruption",
        "Rumination_Functional_Analysis",
        "Detach_From_Inner_Debate_Practice",
        "Replace_Rumination_With_Values_Action",
    ])

    add_leaves(cbt, ["Perseverative_Cognition_Interventions", "Metacognitive_Beliefs", "Challenge_Meta_Beliefs"], [
        "Metacognitive_Worry_Belief_Challenge",
        "Meta_Worry_Challenge",
        "Belief_About_Rumination_Challenge",
        "Threat_Monitoring_Reduction_Practice",
        "Reduce_Reassurance_Seeking_And_Checking",
    ])

    add_leaves(cbt, ["Perseverative_Cognition_Interventions", "Tolerance_of_Uncertainty", "Cognitive_and_Behavioral"], [
        "Uncertainty_Tolerance_Practice",
        "Letting_Go_Practice",
        "Cognitive_Defusion_Worry",
        "Worry_As_Avoidance_Recognition",
    ])

    add_leaves(cbt, ["Perseverative_Cognition_Interventions", "Tools_and_Worksheets", "Structured_Forms"], [
        "Constructive_Worry_Worksheet",
        "Unhelpful_Worry_Benefit_Cost_Check",
        "Worry_And_Rumination_Learning_Log",
        "Self_Compassion_For_Rumination",
    ])

    # ------------------------------------------------------------
    # 1.6 Problem Solving Training (micro-skills added; layered)
    # ------------------------------------------------------------
    add_leaves(cbt, ["Problem_Solving_Training", "Orientation_and_Definition", "Set_Up"], [
        "Problem_Orientation_Reframe",
        "Problem_Definition_Clarification",
        "Define_Success_Criteria_For_Solution",
        "Emotion_Informed_Problem_Solving",
    ])

    add_leaves(cbt, ["Problem_Solving_Training", "Generate_and_Select", "Options_and_Decisions"], [
        "Solution_Brainstorming",
        "Decision_Matrix_Evaluation",
        "Values_Aligned_Solution_Filter",
        "Risk_Assessment_and_Mitigation",
    ])

    add_leaves(cbt, ["Problem_Solving_Training", "Implement_and_Learn", "Action_and_Review"], [
        "Stepwise_Action_Plan",
        "Implementation_Barrier_Check",
        "Obstacle_Identification_and_Workaround",
        "Test_Smallest_Safe_Action_Step",
        "Contingency_Planning_If_Blockers",
        "Plan_Do_Review_Cycle",
        "Post_Action_Learning_Review",
        "Problem_Solving_With_Compassion",
    ])

    add_leaves(cbt, ["Problem_Solving_Training", "Applied_Problem_Solving", "Domains"], [
        "Communication_Problem_Solving",
        "Conflict_Resolution_Problem_Solving",
        "Time_Management_Problem_Solving",
    ])

    # ------------------------------------------------------------
    # 1.7 CBT Skills Training Libraries (organized; still CBT-first)
    # ------------------------------------------------------------
    add_leaves(cbt, ["CBT_Skills_Training_Libraries", "Assertiveness_and_Boundaries", "Core_Skills"], [
        "Assertiveness_Training",
        "Request_Making_Skills",
        "Refusal_Skills",
        "Boundary_Setting_Scripts",
        "Self_Advocacy_Scripts",
        "Boundary_Assertion_With_Warmth",
    ])

    add_leaves(cbt, ["CBT_Skills_Training_Libraries", "Communication_and_Repair", "Conversation_Skills"], [
        "Communication_Skills_I_Statements",
        "Active_Listening_Skills",
        "Nondefensive_Listening_Practice",
        "Validation_Then_Request_Sequence",
        "Emotion_Labeling_In_Conversation",
        "Repair_Intent_Statement_Practice",
        "Conflict_Repair_Sequence",
        "Deescalation_By_Slowing_And_Summarizing",
    ])

    add_leaves(cbt, ["CBT_Skills_Training_Libraries", "Conflict_and_Negotiation", "Resolution_Skills"], [
        "Negotiation_Skills",
        "Anger_Deescalation_Scripts",
        "Perspective_Taking_Training",
    ])

    add_leaves(cbt, ["CBT_Skills_Training_Libraries", "Social_Confidence_Practice", "Rehearsal_and_Feedback"], [
        "Social_Skills_Rehearsal",
        "Role_Playing_Scenarios",
        "Behavioral_Rehearsal_With_Feedback",
        "Exposure_Aligned_Social_Skills_Practice",
        "Interpersonal_Feedback_Practice",
    ])

    # ------------------------------------------------------------
    # 1.8 Schema-Focused CBT Derivatives (more detailed layering)
    # ------------------------------------------------------------
    schema = OntologyBuilder().root

    add_leaves(schema, ["Assessment_and_Conceptualization", "Identify_and_Map", "Schema_and_Modes"], [
        "Schema_Theme_Identification",
        "Mode_Identification",
        "Mode_Switching_Awareness",
        "Schema_Coping_Style_Identification",
        "Inner_Critic_Mode_Map",
    ])

    add_leaves(schema, ["Assessment_and_Conceptualization", "Triggers_and_Maintenance", "Maps"], [
        "Schema_Trigger_Mapping",
        "Vulnerability_Triggers_Map",
        "Needs_Assessment_Framework",
        "Schema_Maintenance_Cycle_Map",
    ])

    add_leaves(schema, ["Change_Techniques", "Experiential", "Imagery_and_Chairwork"], [
        "Imagery_Rescripting_For_Needs",
        "Chairwork_Mode_Dialogue",
        "Corrective_Experience_Scripting",
        "Mode_Specific_Coping_Cards",
    ])

    add_leaves(schema, ["Change_Techniques", "Relational_Stance", "Therapeutic_Method"], [
        "Limited_Reparenting_Stance",
        "Empathic_Confrontation_Practice",
    ])

    add_leaves(schema, ["Change_Techniques", "Behavioral_Pattern_Change", "Practice_and_Planning"], [
        "Behavior_Pattern_Breaking_Plans",
        "Behavioral_Practice_Healthy_Adult",
        "Healthy_Adult_Mode_Strengthening",
        "Needs_Fulfillment_Planning",
        "Boundary_Setting_With_Inner_Critic",
        "Self_Compassionate_Inner_Dialogue",
        "Values_Aligned_Mode_Coaching",
    ])

    cbt["Schema_Focused_CBT_Derivatives"] = schema

    # ------------------------------------------------------------
    # 1.9 Transdiagnostic / Process-Based CBT (structured)
    # ------------------------------------------------------------
    add_leaves(cbt, ["Transdiagnostic_CBT_Frameworks", "Unified_Protocol_Style_Processes", "Core_Process_Targets"], [
        "Unified_Protocol_Emotion_Awareness",
        "Unified_Protocol_Cognitive_Flexibility",
        "Unified_Protocol_Emotion_Exposure",
        "Unified_Protocol_Emotion_Driven_Behavior_Change",
        "Unified_Protocol_Interoceptive_Awareness",
    ])

    add_leaves(cbt, ["Transdiagnostic_CBT_Frameworks", "Avoidance_and_Safety_Behaviors", "Reduce_Maintainers"], [
        "Transdiagnostic_Avoidance_Reduction",
        "Transdiagnostic_Safety_Behavior_Reduction",
    ])

    add_leaves(cbt, ["Transdiagnostic_CBT_Frameworks", "Reward_and_Engagement", "Rebuild_Approach"], [
        "Transdiagnostic_Reward_Reconnection",
    ])

    add_leaves(cbt, ["Transdiagnostic_CBT_Frameworks", "Process_Based_CBT", "Mechanism_Linked_Selection"], [
        "Process_Based_CBT_Target_Process_Selection",
        "Process_Based_CBT_Mechanism_Linked_Intervention",
        "Process_Based_CBT_Values_And_Goals_Alignment",
        "Process_Based_CBT_Context_Sensitivity_Training",
    ])

    # ------------------------------------------------------------
    # 1.10 Cognitive Bias / Attention / Interpretation Modification (more layered)
    # ------------------------------------------------------------
    add_leaves(cbt, ["Cognitive_Bias_and_Attention_Modification", "Interpretation_Training", "Bias_Modification_Tasks"],
               [
                   "Interpretation_Bias_Modification_Task",
                   "Positive_Interpretation_Training",
                   "Ambiguity_Tolerance_Training",
                   "Interpretation_Flexibility_Drills",
                   "Compassionate_Interpretation_Training",
               ])

    add_leaves(cbt, ["Cognitive_Bias_and_Attention_Modification", "Attention_Training", "Disengagement_and_Shift"], [
        "Attention_Bias_Retraining_Task",
        "Threat_Signal_Disengagement_Practice",
        "Attention_Switching_Practice_CBT",
    ])

    add_leaves(cbt, ["Cognitive_Bias_and_Attention_Modification", "Bias_Awareness_and_Correction", "Cognitive_Drills"],
               [
                   "Cognitive_Bias_Awareness_Training",
                   "Alternative_Hypotheses_Drill",
                   "Evidence_Weighting_Practice",
                   "Confirmation_Bias_Check_Practice",
                   "Negative_Filtering_Reversal_Practice",
                   "Discounting_Positive_Reversal_Practice",
               ])

    # ------------------------------------------------------------
    # 1.11 Sleep-Related CBT Components (psychological only; expanded & categorized)
    # ------------------------------------------------------------
    add_leaves(cbt, ["Sleep_Related_Cognitive_Behavioral_Interventions", "Conditioning_and_Associations", "Relearning"],
               [
                   "Stimulus_Control_Bed_Association_Relearning",
                   "Conditioned_Arousal_Deconditioning_Framework",
                   "Reduce_Time_Checking_Behavior",
               ])

    add_leaves(cbt, ["Sleep_Related_Cognitive_Behavioral_Interventions", "Cognitive_Work", "Sleep_Appraisals"], [
        "Cognitive_Reframe_Sleep_Threat_Appraisals",
        "Sleep_Related_Worry_Defusion",
        "Paradoxical_Intention_Sleep_Principle",
    ])

    add_leaves(cbt, ["Sleep_Related_Cognitive_Behavioral_Interventions", "PreSleep_Mental_Skills",
                     "Detachment_and_Unloading"], [
                   "PreSleep_Detachment_Routine_Design_Psychological",
                   "Constructive_Worry_Before_Bed_Worksheet",
               ])

    # Attach to PSYCHO (outside this snippet you’ll do: PSYCHO["Cognitive_Behavioral_Therapies"] = cbt)
    PSYCHO["Cognitive_Behavioral_Therapies"] = cbt

    # ============================================================
    # 2) Third-Wave / Contextual / Skills-Based Therapies
    # ============================================================

    # NOTE: Third-wave therapies are an evolution of Cognitive Behavioral Therapy (CBT) that focus less on changing the content of thoughts
    # and more on changing one's relationship with thoughts and feelings, emphasizing mindfulness, acceptance, values, and psychological flexibility
    # to build a richer, more meaningful life, rather than just symptom reduction.

    third = OntologyBuilder().root

    # ACT (hexaflex)
    act_processes = {
        "Values_and_Direction": [
            "Values_Interview",
            "Values_Card_Sort",
            "Life_Compass_Exercise",
            "Values_Behavior_Linkage",
            "Committed_Action_Planning",
            "Choice_Point_Mapping",
            "Values_Based_Goal_Setting",
            "Barriers_and_Willingness_Map",
            "Values_Clarification_Narrative",
            "Values_Consistency_Check",
            "Values_Conflict_Resolution_Dialogue",
            "Values_As_Compass_Recenter_Practice",
        ],
        "Acceptance_and_Willingness": [
            "Expansion_Exercise",
            "Urge_Surfing",
            "Willingness_Dial",
            "Allowing_Sensations_Practice",
            "Acceptance_of_Emotions_Practice",
            "Self_Compassionate_Willingness",
            "Drop_The_Rope_TugOfWar_Metaphor",
            "Passengers_On_The_Bus_Metaphor",
            "Acceptance_With_Breath_Anchor",
            "Allowing_Thoughts_Practice",
            "Make_Room_For_Affect_Practice",
            "Allow_Discomfort_While_Acting",
        ],
        "Cognitive_Defusion": [
            "Labeling_Thoughts_As_Thoughts",
            "Silly_Voice_Defusion",
            "Leaves_On_A_Stream_Exercise",
            "Thank_Your_Mind_Exercise",
            "Word_Repetition_Defusion",
            "Observer_Stance_To_Thoughts",
            "Noticing_Mental_Images_Defusion",
            "Defusion_With_Writing",
            "Name_The_Story_Defusion",
            "Physicalize_Thought_Defusion",
            "Defusion_From_Self_Criticism",
            "Defusion_From_Rules_And_Shoulds",
        ],
        "Present_Moment_Awareness": [
            "Mindfulness_of_Breath",
            "Mindfulness_of_Body_Scan",
            "Mindfulness_of_Sounds",
            "Mindfulness_of_Thoughts",
            "Mindfulness_of_Emotions",
            "Five_Senses_Grounding",
            "Open_Monitoring_Practice",
            "Brief_CheckIn_Practice",
            "Noticing_And_Naming_Practice",
            "Recenter_On_Values_Practice",
            "Mindful_Pause_Before_Action",
            "Mindful_Savoring_Practice",
        ],
        "Self_As_Context": [
            "Observing_Self_Exercise",
            "Perspective_Taking_Self_As_Context",
            "Chessboard_Metaphor",
            "Sky_And_Weather_Metaphor",
            "Noticer_Exercise",
            "Story_Observer_Exercise",
            "Perspective_Shift_Practice",
            "Observer_Self_In_Emotion_Storm",
        ],
        "Psychological_Flexibility_Integration": [
            "ACT_Matrix_Practice",
            "Hexaflex_Process_Mapping",
            "Values_Exposure_Integration",
            "Flexible_Attention_Shifting",
            "Flexible_Rule_Use_Practice",
            "Compassionate_Choice_Point",
            "Willingness_Action_Link",
            "Committed_Action_Review_And_Adjust",
        ],
    }
    for proc, leaves in act_processes.items():
        add_leaves(third, ["ACT_Contextual_Behavioral_Therapy", proc], leaves)

    # DBT
    dbt = OntologyBuilder().root
    add_leaves(dbt, ["Mindfulness_Skills"], [
        "Wise_Mind_Practice",
        "Observe_Skill",
        "Describe_Skill",
        "Participate_Skill",
        "Nonjudgmental_Stance",
        "One_Mindfully_Practice",
        "Effectiveness_Focus",
        "Mindfulness_Of_Current_Emotion",
        "Mindfulness_Of_Thoughts",
        "Lovingkindness_Practice",
        "Urge_Observation_Practice",
        "Grounding_Practice",
        "Mindful_Walking_Practice",
        "Mindful_Eating_Practice",
        "Mindfulness_Of_Interpersonal_Moments",
        "Mindfulness_Of_Body_Cues",
    ])
    add_leaves(dbt, ["Distress_Tolerance_Skills"], [
        "TIP_Skills_Framework",
        "Self_Soothing_Five_Senses",
        "Distraction_Wise_List",
        "IMPROVE_Moment_Framework",
        "Pros_Cons_Decision_Aid",
        "Radical_Acceptance_Practice",
        "Turning_The_Mind_Practice",
        "Willingness_Practice",
        "Reality_Acceptance_Scripts",
        "Crisis_Survival_Kit_Preparation",
        "Grounding_With_Temperature_Stimulus",
        "Breathing_To_Downshift_Arousal",
        "Urge_Surfing_Distress",
        "Cognitive_Redefinition_Of_Crisis",
        "Self_Compassionate_Distress_Holding",
    ])
    # DBT – Emotion Regulation Skills (expanded; 2 extra layers:
    # 1) subcategory, 2) high-resolution technique leaves)
    dbt_er: dict = {}

    # ------------------------------------------------------------
    # A) Emotion literacy & awareness (identify/describe)
    # ------------------------------------------------------------
    add_leaves(dbt_er, ["Emotion_Literacy_and_Awareness", "Identify_and_Name"], [
        "Emotion_Labeling_Practice",
        "Name_The_Emotion_To_Tame_Practice",
        "Arousal_Labeling_Practice",
        "Differentiate_Emotion_vs_Mood",
        "Differentiate_Emotion_vs_Thought",
        "Differentiate_Primary_vs_Secondary_Emotion",
        "Differentiate_Mixed_Emotions_Practice",
        "Emotion_Intensity_Rating_Framework",
        "Emotion_Granularity_Building_Practice",
        "Emotion_Wave_Tracking_Practice",
    ])

    add_leaves(dbt_er, ["Emotion_Literacy_and_Awareness", "Map_Emotion_Patterns"], [
        "Function_Of_Emotion_Analysis",
        "Emotion_Action_Tendency_Map",
        "Emotion_Trigger_Cue_Mapping",
        "Body_Sensation_Emotion_Link_Map",
        "Vulnerability_Factors_Identification",
        "Emotion_Context_Consequence_Mapping",
        "Emotion_Driven_Behavior_Recognition",
        "Emotion_Myths_Check_Practice",
    ])

    # ------------------------------------------------------------
    # B) Reality-based appraisal (reduce misinterpretation)
    # ------------------------------------------------------------
    add_leaves(dbt_er, ["Reality_Based_Appraisal_and_Reappraisal", "Check_The_Facts"], [
        "Check_The_Facts_Skill",
        "Interpretation_vs_Fact_Discrimination",
        "Assumption_Testing_Practice",
        "Perspective_Taking_Reappraisal",
        "Intent_Assignment_Check_Practice",
        "Mind_Reading_Check_Practice",
        "Catastrophe_Check_Practice",
        "Threat_Appraisal_Reframe_Practice",
        "Self_Blaming_Reappraisal_Practice",
    ])

    add_leaves(dbt_er, ["Reality_Based_Appraisal_and_Reappraisal", "Meaning_Making_and_Learning"], [
        "Meaning_Reappraisal_Practice",
        "Emotion_Learning_Summary_Practice",
        "Update_Appraisal_After_New_Data",
        "Compassionate_Reframe_For_Emotion",
    ])

    # ------------------------------------------------------------
    # C) Change emotions directly (action + exposure)
    # ------------------------------------------------------------
    add_leaves(dbt_er, ["Change_Based_Skills", "Opposite_Action_and_Action_Shaping"], [
        "Opposite_Action_Skill",
        "Opposite_Action_Planning_Worksheet",
        "Approach_Action_For_Avoidance_Urges",
        "Gentle_Approach_For_Fear_Responses",
        "Behavioral_Activation_For_Low_Mood",
        "Anger_To_Assertiveness_Planning",
        "Shame_To_Pride_Action_Planning",
        "Self_Respect_Action_Planning",
        "Values_Aligned_Action_For_Emotion",
    ])

    add_leaves(dbt_er, ["Change_Based_Skills", "Emotion_Exposure_and_Willingness"], [
        "Emotion_Exposure_Practice",
        "Willingness_To_Feel_Practice",
        "Allowing_Emotion_Wave_Practice",
        "Drop_Safety_Behaviors_During_Emotion",
        "Stay_With_Sensation_Practice",
        "Approach_Triggers_Gradually_Framework",
    ])

    # ------------------------------------------------------------
    # D) Solve problems when change is possible (reduce drivers)
    # ------------------------------------------------------------
    add_leaves(dbt_er, ["Problem_Solving_for_Emotion_Drivers", "Problem_Definition_and_Targeting"], [
        "Problem_Solving_Emotion_Driven",
        "Define_Problem_Behavior_Emotion_Link",
        "Prioritize_Change_Targets_Practice",
        "Barrier_Analysis_For_Solutions",
    ])

    add_leaves(dbt_er, ["Problem_Solving_for_Emotion_Drivers", "Generate_and_Test_Solutions"], [
        "Solution_Brainstorming_Emotion_Drivers",
        "Decision_Matrix_For_Emotion_Solutions",
        "Plan_Do_Review_For_Emotion_Problems",
        "Interpersonal_Repair_As_Solution",
        "Boundary_Setting_As_Solution",
    ])

    # ------------------------------------------------------------
    # E) Build positive emotions (short-term + long-term)
    # ------------------------------------------------------------
    add_leaves(dbt_er, ["Build_Positive_Emotions", "Short_Term_Positive_Emotion_Skills"], [
        "Build_Positive_Experiences",
        "Pleasant_Events_List",
        "Savoring_Practice",
        "Gratitude_Practice",
        "Positive_Event_Noticing_Practice",
        "Values_Based_MicroJoy_Planning",
    ])

    add_leaves(dbt_er, ["Build_Positive_Emotions", "Long_Term_Positive_Emotion_Skills"], [
        "Accumulate_Positive_Experiences_LongTerm",
        "Values_Aligned_Goal_Pursuit_Planning",
        "Build_Mastery_Framework",
        "Self_Efficacy_Building_Practice",
        "Meaning_And_Purpose_Integration_Practice",
    ])

    # ------------------------------------------------------------
    # F) Reduce vulnerability to emotion mind (PLEASE-like but schedule-free)
    # (Kept psychological; no BIO dosing; framed as self-regulation routines/choices)
    # ------------------------------------------------------------
    add_leaves(dbt_er, ["Reduce_Vulnerability_and_Increase_Resilience", "Vulnerability_Reduction_Frameworks"], [
        "Reduce_Emotional_Vulnerability_Framework",
        "Check_Vulnerability_Factors_Practice",
        "Balance_Demands_And_Resources_Practice",
        "Stress_Load_Awareness_Practice",
        "Recovery_Routine_Design_Practice",
    ])

    add_leaves(dbt_er, ["Reduce_Vulnerability_and_Increase_Resilience", "Build_Mastery_and_Efficacy"], [
        "Build_Mastery_Framework",
        "Graded_Task_Mastery_Planning",
        "Competence_Tracking_Practice",
        "Strengths_Use_Planning",
    ])

    # ------------------------------------------------------------
    # G) Self-validation, self-compassion, and repair
    # ------------------------------------------------------------
    add_leaves(dbt_er, ["Self_Relating_and_Repair", "Validate_and_Soothe"], [
        "Self_Validation_Practice",
        "Validation_Scripts_Self",
        "Self_Compassion_Practice",
        "Compassionate_Self_Talk_Practice",
        "Common_Humanity_Reframe_Practice",
        "Nonjudgment_Toward_Emotion_Practice",
    ])

    add_leaves(dbt_er, ["Self_Relating_and_Repair", "Aftercare_and_Repair_Plans"], [
        "Aftercare_For_Intense_Emotion",
        "Repair_After_Dysregulation_Plan",
        "Repair_Conversation_Rehearsal",
        "Self_Forgiveness_Practice",
        "Relapse_Prevention_For_Dysregulation",
    ])

    # ------------------------------------------------------------
    # H) Chain analysis & contingency learning (behavior-emotion loops)
    # ------------------------------------------------------------
    add_leaves(dbt_er, ["Chain_Analysis_and_Learning_Loops", "Chain_Analysis_Core"], [
        "Chain_Analysis_Skill",
        "Vulnerability_Link_Identification",
        "Prompting_Event_Identification",
        "Links_In_Chain_Identification",
        "Consequences_Map_Practice",
        "Skill_Alternative_Link_Planning",
    ])

    add_leaves(dbt_er, ["Chain_Analysis_and_Learning_Loops", "Solution_Planning_From_Chain"], [
        "Prevention_Plan_From_Chain_Analysis",
        "Missing_Skill_Identification_From_Chain",
        "Trigger_Plan_Creation_From_Chain",
        "Contingency_IfThen_From_Chain",
    ])

    # Attach into DBT subtree
    dbt["Emotion_Regulation_Skills"] = dbt_er

    add_leaves(dbt, ["Interpersonal_Effectiveness_Skills"], [
        "DEAR_MAN_Skill",
        "GIVE_Skill",
        "FAST_Skill",
        "Boundary_Setting_Practice",
        "Saying_No_Practice",
        "Asking_For_Needs_Practice",
        "Repair_After_Conflict_Practice",
        "Validation_Others_Practice",
        "Self_Respect_Effectiveness_Practice",
        "Dialectical_Thinking_Practice",
        "Middle_Path_Practice",
        "Negotiation_Practice",
        "Interpersonal_Values_Clarification",
        "Balancing_Objectives_And_Relationship",
        "Nondefensive_Receiving_Feedback",
    ])
    third["DBT_Skills_Based_Therapy"] = dbt

    # Mindfulness-based interventions (MBIs + MBCT-like skills)
    # High-resolution, multi-layer taxonomy (schedule-free; technique-level leaves)
    mindfulness: dict = {}

    # ------------------------------------------------------------
    # 1) Core Attentional Skills (samatha-style foundations)
    # ------------------------------------------------------------
    attentional = [
        "Posture_And_Embodiment_Setup",
        "Intention_Setting_Practice",
        "Attentional_Stability_Training",
        "Attention_Anchor_Selection",
        "Breath_Following_Practice",
        "Breath_Counting_Practice",
        "Single_Point_Focus_Practice",
        "Attention_Returning_Skill",
        "Attention_Wandering_Noticing",
        "Distraction_Labeling_And_Release",
        "Balance_Effort_And_Ease_Practice",
        "Sustained_Attention_Practice",
        "Attentional_Switching_Practice",
        "Concentration_Refinement_Practice",
    ]
    add_leaves(mindfulness, ["Core_Attentional_Skills"], attentional)

    # ------------------------------------------------------------
    # 2) Open Monitoring & Meta-Awareness (vipassana-style)
    # ------------------------------------------------------------
    open_monitoring = [
        "Open_Awareness_Practice",
        "Choiceless_Awareness_Practice",
        "Noting_Practice_Basic",
        "Noting_Practice_Granular",
        "Monitoring_Of_Mind_States",
        "Meta_Awareness_Of_Attention",
        "Witness_Consciousness_Practice",
        "Decentering_Practice",
        "Reactivity_Noticing_Practice",
        "Allowing_And_Letting_Be_Practice",
        "Nonattachment_To_Experience_Practice",
        "Equanimity_Cultivation_Practice",
        "Nonjudgment_Training",
        "Nonreactivity_Training",
    ]
    add_leaves(mindfulness, ["Open_Monitoring_and_Meta_Awareness"], open_monitoring)

    # ------------------------------------------------------------
    # 3) Body-Based Mindfulness (interoception + embodiment)
    # ------------------------------------------------------------
    body_based = [
        "Body_Scan_Practice",
        "Segmented_Body_Scan_Practice",
        "Body_Scan_With_Breath_Integration",
        "Mindfulness_Of_Body_Sensations_Practice",
        "Interoceptive_Awareness_Practice",
        "Somatic_Labeling_Practice",
        "Embodied_Grounding_Practice",
        "Mindful_Posture_CheckIn",
        "Mindful_Relaxation_Release_Practice",
        "Pain_Mindfulness_Practice",
        "Pleasant_Sensation_Savoring_Practice",
    ]
    add_leaves(mindfulness, ["Body_Based_Mindfulness"], body_based)

    # ------------------------------------------------------------
    # 4) Affective Mindfulness (emotions, urges, craving, compassion)
    # ------------------------------------------------------------
    affective = [
        "Mindfulness_Of_Emotions_Practice",
        "Emotion_Labeling_Mindfulness",
        "Riding_Emotion_Waves_Practice",
        "Turning_Toward_Experience_Practice",
        "Allowing_Difficult_Emotion_Practice",
        "Self_Validation_Mindfulness_Practice",
        "Urge_Surfing_Practice",
        "Craving_Mindfulness_Practice",
        "RAIN_Framework_Practice",
        "Compassionate_Presence_With_Emotion",
        "Self_Compassion_Mindfulness_Practice",
    ]
    add_leaves(mindfulness, ["Affective_Mindfulness_and_Urge_Work"], affective)

    # ------------------------------------------------------------
    # 5) Cognitive Mindfulness (thoughts, rumination, decentering)
    # ------------------------------------------------------------
    cognitive = [
        "Mindfulness_Of_Thoughts_Practice",
        "Thought_Labeling_As_Mental_Event",
        "Thoughts_As_Mental_Events_Practice",
        "Cognitive_Defusion_Mindfulness",
        "Rumination_Noticing_And_Disengaging",
        "Worry_Noticing_And_Returning",
        "Mind_Stories_Naming_Practice",
        "Judgment_Noticing_Practice",
        "Letting_Go_Of_Analysis_Practice",
        "Beginner_Mind_Training",
    ]
    add_leaves(mindfulness, ["Cognitive_Mindfulness_and_Decentering"], cognitive)

    # ------------------------------------------------------------
    # 6) Formal Meditation Formats
    # ------------------------------------------------------------
    formal_formats = [
        "Sitting_Meditation_Practice",
        "Standing_Meditation_Practice",
        "Walking_Meditation_Practice",
        "Lying_Down_Meditation_Practice",
        "Seated_Breath_Meditation",
        "Sound_As_Anchor_Meditation",
        "Object_Focus_Meditation",
        "Open_Monitoring_Sitting_Meditation",
    ]
    add_leaves(mindfulness, ["Formal_Meditation_Formats"], formal_formats)

    # ------------------------------------------------------------
    # 7) Informal / Daily-Life Mindfulness
    # ------------------------------------------------------------
    informal = [
        "Mindful_Eating_Practice",
        "Mindful_Drinking_Practice",
        "Mindful_Showering_Practice",
        "Mindful_Hands_Washing_Practice",
        "Mindful_Commute_Practice",
        "Mindful_Pausing_Practice",
        "Mindful_One_Tasking_Practice",
        "Mindful_Transitions_Practice",
        "Mindful_Technology_Use_Practice",
        "Mindful_Routine_Embedding_Practice",
    ]
    add_leaves(mindfulness, ["Informal_Everyday_Mindfulness"], informal)

    # ------------------------------------------------------------
    # 8) Mindful Movement (MBSR-style + contemplative movement)
    # ------------------------------------------------------------
    movement = [
        "Mindful_Movement_Practice",
        "Mindful_Stretching_Practice",
        "Yoga_Based_Mindful_Movement_Practice",
        "TaiChi_Informed_Mindful_Movement_Practice",
        "Qigong_Informed_Mindful_Movement_Practice",
        "Walking_Meditation_With_Sensation_Focus",
        "Mindful_Strength_Training_Practice",
        "Mindful_Dance_Movement_Practice",
    ]
    add_leaves(mindfulness, ["Mindful_Movement_and_Embodied_Practices"], movement)

    # ------------------------------------------------------------
    # 9) Compassion / Lovingkindness / Prosocial Meditation (brahmavihara-family)
    # ------------------------------------------------------------
    compassion = [
        "Lovingkindness_Practice",
        "Self_Directed_Lovingkindness_Practice",
        "Other_Directed_Lovingkindness_Practice",
        "Compassion_Meditation_Practice",
        "Self_Compassion_Meditation_Practice",
        "Compassion_For_Others_Meditation_Practice",
        "Sympathetic_Joy_Practice",
        "Equanimity_Meditation_Practice",
        "Tonglen_Informed_Compassion_Practice",
        "Forgiveness_Meditation_Practice",
        "Receiving_Compassion_Practice",
    ]
    add_leaves(mindfulness, ["Compassion_and_Prosocial_Meditations"], compassion)

    # ------------------------------------------------------------
    # 10) MBCT / MBSR Structured Micro-Practices (cognitive relapse-prevention skills)
    # ------------------------------------------------------------
    mbct_mbsr = [
        "Brief_Breathing_Space_Practice",
        "Three_Minute_Breathing_Space_Concept",
        "Recognize_Autopilot_Mode",
        "Turning_Toward_Difficulty_Practice",
        "Shifting_From_Doing_To_Being_Mode",
        "Approach_And_Allow_Practice",
        "Decentering_From_Negative_Thinking",
        "Relating_Differently_To_Thoughts_Practice",
        "Mood_Thought_Link_Recognition",
        "Early_Warning_Sign_Recognition",
    ]
    add_leaves(mindfulness, ["MBCT_MBSR_Core_Practices"], mbct_mbsr)

    # ------------------------------------------------------------
    # 11) Interpersonal Mindfulness (communication + relational presence)
    # ------------------------------------------------------------
    interpersonal = [
        "Mindful_Communication_Practice",
        "Mindful_Listening_Practice",
        "Mindful_Speaking_Practice",
        "Pause_Breathe_Respond_Practice",
        "Nonviolent_Communication_Mindful_Stance",
        "Interpersonal_Trigger_Noticing_Practice",
        "Relational_Attunement_Practice",
        "Mindful_Boundary_Awareness_Practice",
        "Compassionate_Communication_Practice",
    ]
    add_leaves(mindfulness, ["Interpersonal_Mindfulness"], interpersonal)

    # ------------------------------------------------------------
    # 12) Stress & Reactivity Applications (applied mindfulness)
    # ------------------------------------------------------------
    stress_apps = [
        "Mindful_Response_To_Stress_Practice",
        "Stress_Reactivity_Cycle_Noticing",
        "Grounding_Five_Senses_Practice",
        "Anchoring_In_Breath_During_Stress",
        "Mindful_Coping_Choice_Point",
        "After_Stress_Recovery_Mindfulness",
        "Pre_Event_Centering_Practice",
        "Post_Event_Decompression_Mindfulness",
    ]
    add_leaves(mindfulness, ["Applied_Mindfulness_for_Stress_and_Reactivity"], stress_apps)

    # Attach into your ontology
    third["Mindfulness_Based_Interventions"] = mindfulness

    # Compassion-focused interventions (CFT and related)
    compassion_focused = [
        "Soothing_Rhythm_Breathing",
        "Compassionate_Image_Creation",
        "Compassionate_Self_Identity_Work",
        "Self_Criticism_To_Compassion_Reframe",
        "Compassionate_Letter_Writing",
        "Compassionate_Behavior_Planning",
        "Warmth_Tone_Posture_Practice",
        "Compassion_For_Others_Practice",
        "Receiving_Compassion_Practice",
        "Shame_Compassion_Work",
        "Threat_Drive_Soothe_Balance_Map",
        "Safe_Place_Imagery",
        "Compassionate_Inner_Ally_Practice",
        "Compassionate_Motivation_Training",
        # Added: compassion skills detail
        "Compassionate_Attention_Training",
        "Compassionate_Reasoning_Practice",
        "Compassionate_Behavioral_Rehearsal",
        "Compassion_For_Inner_Parts_Practice",
    ]
    add_leaves(third, ["Compassion_Focused_Interventions"], compassion_focused)

    # RO-DBT (overcontrol skills)
    ro_dbt = [
        "Radical_Openness_Practice",
        "Self_Enquiry_Practice",
        "Flexible_Control_Practice",
        "Social_Signaling_Skills",
        "Authentic_Self_Disclosure_Practice",
        "Repair_Of_Overcontrol_Patterns",
        "Novelty_Seeking_Behavior_Experiments",
        "Playfulness_Practice",
        "Open_Posture_Facial_Softening_Practice",
        "Receptivity_To_Feedback_Practice",
        "Soothe_Threat_System_Through_Connection",
    ]
    add_leaves(third, ["Radically_Open_Skills_Based_Therapy"], ro_dbt)

    # FAP (in-session interpersonal learning)
    functional_analytic = [
        "In_Session_Behavior_Tracking",
        "Reinforcement_In_Session_Skill",
        "Evocative_Responding_Skill",
        "Natural_Reinforcement_Practice",
        "Therapeutic_Honesty_Practice",
        "Interpersonal_Feedback_Loops",
        "Awareness_Courage_Love_Framework",
        "In_Session_Skills_Shaping",
        "Functional_Equivalence_Discussion",
        "Identify_Clinically_Relevant_Behavior_InSession",
    ]
    add_leaves(third, ["Functional_Analytic_Behavior_Therapy"], functional_analytic)

    # Stress inoculation (psychological components only)
    stress_inoculation = [
        "Coping_Skills_Training_Framework",
        "Cognitive_Preparation_For_Stress",
        "Arousal_Management_Skills",
        "Coping_Self_Statements_Practice",
        "Stress_Appraisal_Reframe",
        "Stress_Exposure_Simulation_Practice",
        "Generalization_Of_Coping_Skills",
        "Coping_Imagery_Rehearsal",
    ]
    add_leaves(third, ["Stress_Inoculation_and_Coping_Skills"], stress_inoculation)

    PSYCHO["Third_Wave_and_Contextual_Therapies"] = third

    # ============================================================
    # 3) Psychodynamic / Insight-Oriented / Mentalization
    # ============================================================
    insight = OntologyBuilder().root

    psychodynamic_core = [
        "Clarification_Technique",
        "Confrontation_Technique",
        "Interpretation_Technique",
        "Defense_Identification",
        "Pattern_Recognition_Relational",
        "Core_Conflict_Theme_Work",
        "Affect_Focus_And_Containment",
        "Transference_Exploration",
        "Countertransference_Use_As_Data",
        "Corrective_Emotional_Experience_Facilitation",
        "Exploration_of_Ambivalence",
        "Free_Association_Invitation",
        "Dream_Material_Exploration",
        "Meaning_Making_From_Symptoms",
        "Attachment_History_Exploration",
        "Reflective_Function_Prompts",
        "Supportive_Interventions_Stabilization",
        "Ego_Strengthening_Interventions",
        "Therapeutic_Frame_Stability",
        "Working_Through_Patterns",
        # Added: modern integrative psychodynamic specifics
        "Identify_Repetition_Compulsion_Patterns",
        "Affect_Clarification_InTheMoment",
        "Interpretation_Link_Affect_And_Defense",
        "Mentalization_Breakdown_Repair",
        "Relational_Needs_Exploration",
    ]
    add_leaves(insight, ["Psychodynamic_Therapy_Family", "Core_Techniques"], psychodynamic_core)

    brief_psychodynamic = [
        "Focus_Selection_For_Insight_Work",
        "Pressure_To_Experience_Affect",
        "Challenge_Avoidance_In_Session",
        "Mobilize_Emotional_Experience",
        "Unlocking_Defensive_Patterns",
        "Integrate_New_Emotional_Learning",
        "Clarify_Therapeutic_Task_And_Focus",
    ]
    add_leaves(insight, ["Psychodynamic_Therapy_Family", "Brief_Dynamic_Techniques"], brief_psychodynamic)

    mentalization = [
        "Mental_State_Language_Cultivation",
        "Not_Knowing_Stance_Practice",
        "Clarify_Misunderstandings_Practice",
        "Affect_Mentalization_Practice",
        "Interpersonal_Mentalization_Practice",
        "Stop_Rewind_Explore_Method",
        "Marking_And_Mirroring_Skill",
        "Epistemic_Trust_Building",
        "Rupture_Repair_Mentalization",
        "Attachment_Focused_Mentalization",
        "Mentalizing_The_Moment_Practice",
        # Added: MBT micro-skills
        "Recognize_Mentalizing_Vs_NonMentalizing_Modes",
        "Switch_From_Certainty_To_Curiosity_Practice",
        "Affect_Label_Then_Perspective_Take",
        "Check_Alternative_Minds_Hypotheses",
    ]
    add_leaves(insight, ["Mentalization_Based_Interventions"], mentalization)

    PSYCHO["Insight_Oriented_Therapies"] = insight

    # ============================================================
    # 4) Humanistic / Experiential / Existential / Emotion-Focused
    # ============================================================
    human = OntologyBuilder().root

    person_centered = [
        "Unconditional_Positive_Regard_Practice",
        "Accurate_Empathic_Understanding_Practice",
        "Congruence_and_Genuineness_Practice",
        "Reflective_Listening_Practice",
        "Emotion_Following_Practice",
        "Client_Led_Goal_Elicitation",
        "Presence_And_Attunement_Practice",
        "Therapeutic_Presence_Grounding",
    ]
    add_leaves(human, ["Person_Centered_Therapy"], person_centered)

    gestalt = [
        "Here_And_Now_Awareness_Practice",
        "Two_Chair_Dialogue",
        "Empty_Chair_Exercise",
        "Enactment_Practice",
        "Body_Awareness_Focusing",
        "Experiment_Design_Gestalt",
        "Polarities_Integration_Work",
        "Unfinished_Business_Processing",
        "Dialogical_Encounter_Practice",
        "Boundary_And_Contact_Cycle_Awareness",
    ]
    add_leaves(human, ["Gestalt_and_Experiential_Therapy"], gestalt)

    eft = [
        "Emotion_Tracking",
        "Primary_Emotion_Accessing",
        "Secondary_Emotion_Differentiation",
        "Emotion_Schematic_Transformation",
        "Two_Chair_Self_Criticism_Work",
        "Compassionate_Chair_Work",
        "Vulnerability_Expression_Practice",
        "Needs_Identification_From_Emotion",
        "Adaptive_Emotion_Cultivation",
        "Self_Interruption_Awareness",
        "Emotion_Focused_Rupture_Repair",
        "Emotion_Co_Regulation_Practice",
        # Added: EFT micro-processes
        "Transform_Shame_With_Compassion",
        "Transform_Fear_With_Security",
        "Transform_Anger_With_Boundaries",
        "Access_Self_Compassionate_Sadness",
    ]
    add_leaves(human, ["Emotion_Focused_Therapy"], eft)

    existential = [
        "Meaning_Exploration",
        "Values_And_Purpose_Work",
        "Freedom_Responsibility_Dialogue",
        "Authenticity_Work",
        "Mortality_Awareness_Integration",
        "Isolation_Connection_Exploration",
        "Existential_Choice_Clarification",
        "Life_Narrative_Integration",
        "Existential_Anxiety_Normalization",
        # Added: existential skills
        "Acceptance_Of_Uncertainty_Existential",
        "Values_As_Response_Ability_Practice",
        "Suffering_And_Meaning_Dual_Awareness",
    ]
    add_leaves(human, ["Existential_and_Meaning_Centered_Interventions"], existential)

    PSYCHO["Humanistic_and_Experiential_Therapies"] = human

    # ============================================================
    # 5) Interpersonal / Attachment / Relational Therapies
    # ============================================================
    rel = OntologyBuilder().root

    ipt = [
        "Interpersonal_Inventory",
        "Communication_Analysis",
        "Role_Transition_Support",
        "Role_Dispute_Negotiation",
        "Grief_Processing_Framework",
        "Interpersonal_Skills_Practice",
        "Social_Support_Mobilization",
        "Interpersonal_Pattern_Identification",
        "Affect_And_Communication_Link",
        "Assertive_Communication_Practice",
        "Boundary_Negotiation_Practice",
        "Repair_Attempts_Practice",
        "Interpersonal_Problem_Area_Focus",
        # Added: interpersonal mechanisms
        "Reduce_Interpersonal_Avoidance_Planning",
        "Increase_Support_Seeking_Behavior",
        "Interpersonal_Expectations_Reframe",
    ]
    add_leaves(rel, ["Interpersonal_Therapy_Family"], ipt)

    attachment = [
        "Attachment_Needs_Identification",
        "Secure_Base_Building_Practice",
        "Emotionally_Attuned_Responding",
        "Corrective_Relational_Experience_Planning",
        "Inner_Working_Model_Revision_Work",
        "Protest_Despair_Repair_Map",
        "Safety_And_Trust_Exercises",
        "Care_Seeking_And_Care_Giving_Balance",
        "Repair_After_Misalignment_Practice",
        "Relational_Safety_Cues_Identification",
        # Added: attachment micro-skills
        "Name_And_Share_Attachment_Needs",
        "Ask_For_Reassurance_With_Self_Respect",
        "Co_Regulation_Request_Script",
        "Differentiate_Threat_From_Need",
    ]
    add_leaves(rel, ["Attachment_Informed_Interventions"], attachment)

    PSYCHO["Interpersonal_and_Attachment_Therapies"] = rel

    # ============================================================
    # 6) Systemic / Couple / Family / Parenting Interventions
    # ============================================================
    systemic = OntologyBuilder().root

    couple_formats = [
        "Couple_Communication_Skills_Training",
        "Conflict_Deescalation_Routines",
        "Repair_Attempt_Practice",
        "Shared_Meaning_Goals_Work",
        "Trust_Rebuilding_Steps",
        "Attachment_Bonding_Conversations",
        "Emotion_Co_Regulation_Practice",
        "Problem_Solving_As_A_Team",
        "Fair_Fighting_Rules",
        "Boundary_With_External_Stressors",
        "Shared_Rituals_Design",
        "Responsibility_Sharing_Negotiation",
        "Couple_Strengths_Amplification",
        # Added: couple micro-processes
        "Soft_Startup_Conversation_Practice",
        "Accept_Influence_Practice",
        "Needs_And_Values_Clarification_As_Couple",
    ]
    add_leaves(systemic, ["Couple_Based_Interventions"], couple_formats)

    family_therapy = [
        "Family_Rules_And_Roles_Mapping",
        "Genogram_Construction",
        "Triangles_And_Boundaries_Work",
        "Structural_Boundary_Realignment",
        "Communication_Pattern_Reframe",
        "Circular_Questioning",
        "Reframing_Symptom_As_Signal",
        "Family_Strengths_Activation",
        "Family_Meeting_Structure",
        "Relational_Repair_Conversations",
        "Reducing_Expressed_Emotion_Scripts",
        "Collaborative_Problem_Solving_Family",
        "Alliance_Building_Across_Family_Members",
        # Added: systemic micro-skills
        "Identify_Interactional_Cycles",
        "Deescalate_Blame_And_Shift_To_Cycle",
        "Increase_Perspective_Taking_In_Family",
    ]
    add_leaves(systemic, ["Family_Systems_Interventions"], family_therapy)

    parenting = [
        "Positive_Attention_Skills",
        "Differential_Reinforcement_Skills",
        "Clear_Instruction_Skills",
        "Consistent_Consequences_Framework",
        "Emotion_Coaching_Parenting",
        "Limit_Setting_Scripts",
        "Routine_Building_Parenting",
        "Collaborative_Problem_Solving_Parenting",
        "Repair_After_Conflict_Parenting",
        "Parental_Self_Regulation_Skills",
        "Parent_Stress_Management_Skills",
        "Parent_Alignment_Communication",
        "Values_Based_Parenting_Goals",
        "Media_And_Device_Boundaries",
        "School_Home_Collaboration_Structure",
        "Positive_Parenting_Scripts",
        # Added: caregiver skills
        "Reflective_Parenting_Practice",
        "Validate_Then_Limit_Sequence",
        "Model_Emotion_Regulation_Explicitly",
    ]
    add_leaves(systemic, ["Parenting_And_Caregiver_Interventions"], parenting)

    PSYCHO["Systemic_Couple_Family_Parenting"] = systemic

    # ============================================================
    # 7) Trauma / Memory / Imagery / Narrative Interventions
    # ============================================================
    trauma = OntologyBuilder().root

    trauma_focused_components = [
        "Psychoeducation_Trauma_Responses",
        "Affect_Regulation_Skills_Trauma_Focused",
        "Cognitive_Processing_Of_Meaning",
        "Narrative_Processing_Structure",
        "InVivo_Masteries_Planning",
        "Enhancing_Safety_And_Trust",
        # Added: stabilization sequencing elements (no dosing)
        "Stabilization_Readiness_Check_Framework",
        "Grounding_And_Orientation_First_Principle",
        "Dual_Attention_Stimulus_Principle",
        "Titration_And_Pacing_Principle",
    ]
    add_leaves(trauma, ["Trauma_Focused_CBT_Components"], trauma_focused_components)

    emdr = [
        "Target_Memory_Selection",
        "Negative_Cognition_Identification",
        "Positive_Cognition_Installation",
        "SUD_VOC_Rating_Framework",
        "Bilateral_Stimulation_Framework",
        "Resource_Development_Installation",
        "Safe_Calm_Place_Imagery",
        "Affect_Tolerance_Preparation",
        "Cognitive_Interweaves",
        "Body_Scan_Processing",
        "Closure_Stabilization_Procedure",
        "Future_Template_Installation",
        # Added: EMDR-adjacent protocol variables
        "Floatback_To_Earlier_Memory_Method",
        "Link_Present_Trigger_To_Target_Memory",
        "Install_Adaptive_Belief_Network",
    ]
    add_leaves(trauma, ["EMDR_Informed_Interventions"], emdr)

    exposure_based_processing = [
        "Imaginal_Exposure_Script_Structure",
        "InVivo_Approach_Planning",
        "Hotspot_Processing_Guidance",
        "Between_Session_Approach_Assignments_Framework",
        "Processing_Learning_Summary",
        # Added
        "Identify_Avoided_Memory_Parts",
        "Approach_Then_Ground_Sequence",
    ]
    add_leaves(trauma, ["Exposure_Based_Trauma_Processing"], exposure_based_processing)

    cognitive_processing = [
        "Stuck_Point_Identification",
        "Socratic_Challenge_Of_Stuck_Points",
        "Alternative_Belief_Installation",
        "Meaning_Making_Integration",
        "Trauma_Related_Guilt_Shame_Reappraisal",
        # Added: belief domains without disorder labels
        "Responsibility_Appraisal_Rebalance_Trauma",
        "Safety_Trust_Power_Esteem_Intimacy_Belief_Check",
    ]
    add_leaves(trauma, ["Cognitive_Trauma_Processing"], cognitive_processing)

    imagery = [
        "Imagery_Rescripting_Framework",
        "Compassionate_Imagery_Rescripting",
        "Mastery_Imagery_Practice",
        "Scene_Revision_Imagery",
        "Somatic_Anchoring_In_Imagery",
        "Imagery_For_Safety_Cues",
        "Imagery_For_Self_Support",
        "Nightmare_Imagery_Rehearsal",
        "Memory_Reconsolidation_Informed_Imagery",
        "Observer_To_Field_Perspective_Shift",
        # Added: imagery techniques
        "Imagery_Distancing_Control",
        "Imagery_Containment_Box_Method",
        "Imagery_Rescripting_With_Protector_Figure",
    ]
    add_leaves(trauma, ["Imagery_and_Rescripting_Interventions"], imagery)

    narrative = [
        "Narrative_Exposure_Structure",
        "Life_Timeline_Construction",
        "Meaning_Making_From_Adversity",
        "Written_Emotional_Processing",
        "Letter_Writing_Unsent",
        "Story_Reauthoring_Practice",
        "Witnessing_And_Validation_Ritual",
        "Values_Integration_In_Narrative",
        "Identity_Reconstruction_Work",
        # Added: narrative micro-structures
        "Identify_Dominant_Problem_Story",
        "Thicken_Preferred_Story_With_Evidence",
        "Name_Values_And_Agency_In_Story",
    ]
    add_leaves(trauma, ["Narrative_and_Writing_Based_Interventions"], narrative)

    stabilization = [
        "Grounding_Skills_Training",
        "Window_Of_Tolerance_Education",
        "Affect_Modulation_Skills",
        "Containment_Imagery",
        "Safe_Place_Practice",
        "Orientation_To_Present_Cues",
        "Trigger_Plan_Creation",
        "Resource_Building_Framework",
        "Interpersonal_Safety_Planning",
        "Coping_Cards_Preparation",
        "Compassionate_Aftercare_Planning",
        # Added: dissociation-agnostic stabilization skills
        "Dual_Attention_Grounding_Practice",
        "Name_Date_Location_Orientation_Practice",
        "Parts_Awareness_And_Self_Leadership_Practice",
    ]
    add_leaves(trauma, ["Stabilization_and_Resourcing"], stabilization)

    PSYCHO["Trauma_Memory_Imagery_and_Narrative_Work"] = trauma

    # ============================================================
    # 8) Brief / Solution-Focused / Motivational Approaches
    # ============================================================
    brief = OntologyBuilder().root

    sfb = [
        "Miracle_Question",
        "Scaling_Questions",
        "Exception_Finding",
        "Coping_Questions",
        "Future_Oriented_Description",
        "Strengths_Amplification",
        "Small_Steps_Planning",
        "Compliments_Strategic",
        "Preferred_Future_Story",
        "Resource_Talk_Elicitation",
        # Added: solution-focused micro-structures
        "Identify_Preferred_Signs_Of_Change",
        "Amplify_Existing_Successful_Strategies",
        "Future_Self_Letter_SFBT_Variant",
    ]
    add_leaves(brief, ["Solution_Focused_Brief_Therapy"], sfb)

    mi = [
        "OARS_Skills",
        "Open_Questioning",
        "Affirmations_Practice",
        "Reflective_Listening_Simple",
        "Reflective_Listening_Complex",
        "Summarizing_Skills",
        "Change_Talk_Elicitation",
        "Sustain_Talk_Softening",
        "Decisional_Balance_Exploration",
        "Confidence_Ruler",
        "Importance_Ruler",
        "Exploring_Ambivalence",
        "Elicit_Provide_Elicit_Information_Share",
        "Autonomy_Support_Language",
        "Values_Behavior_Discrepancy",
        "Commitment_Language_Strengthening",
        "Plan_Development_MI",
        "Rolling_With_Resistance",
        "Strengthen_Self_Efficacy_Language",
        # Added: MI micro-skills
        "Complex_Reflection_Double_Sided",
        "Reflection_Amplified",
        "Reflection_Shift_Focus",
        "Ask_Permission_Before_Advice",
        "Summarize_Change_Talk_Bundle",
    ]
    add_leaves(brief, ["Motivational_Interviewing_Approach"], mi)

    coaching = [
        "Goal_Clarification_Coaching",
        "Accountability_Structure_Coaching",
        "Barrier_Removal_Coaching",
        "Skill_Building_Coaching",
        "Decision_Clarity_Coaching",
        "Time_And_Energy_Prioritization",
        "Habit_Design_Coaching",
        "Identity_Based_Habit_Work",
        "Environmental_Design_For_Habits",
        "Self_Efficacy_Building",
        "Strengths_Utilization_Planning",
        "Performance_Routine_Design",
        "Choice_Architecture_For_Behavior",
        # Added: coaching tools
        "Reflect_On_Learned_Lessons_Practice",
        "Values_Aligned_Prioritization_Coaching",
        "Cognitive_Reframe_For_Performance_Pressure",
    ]
    add_leaves(brief, ["Coaching_and_Behavior_Change_Support"], coaching)

    PSYCHO["Brief_and_Motivational_Approaches"] = brief

    # ============================================================
    # 9) Cognitive / Neuropsychological / Metacognitive Interventions
    # ============================================================
    cog = OntologyBuilder().root

    cognitive_remediation = [
        "Attention_Control_Training",
        "Sustained_Attention_Practice",
        "Selective_Attention_Practice",
        "Divided_Attention_Practice",
        "Working_Memory_Strategy_Training",
        "Chunking_Strategy",
        "Spaced_Retrieval_Practice",
        "Elaboration_Strategy",
        "Mnemonic_Strategy_Training",
        "Prospective_Memory_Aids_Training",
        "Executive_Function_Planning_Skills",
        "Task_Switching_Strategy_Training",
        "Inhibition_Strategy_Training",
        "Cognitive_Flexibility_Practice",
        "Error_Awareness_Training",
        "Metacognitive_Awareness_Training",
        "Goal_Management_Training",
        "Time_Estimation_Strategy_Training",
        "Processing_Speed_Strategy_Training",
        "Cognitive_Load_Management",
        "External_Aids_Use_Training",
        "Study_Skills_Training",
        "Learning_Strategy_Training",
        # Added: practical cognitive scaffolds (psycho-level)
        "Prioritize_Then_Sequence_Task_Steps",
        "Reduce_Multitasking_Practice",
        "Single_Task_Focus_Practice",
        "Distractor_Management_Strategy",
    ]
    add_leaves(cog, ["Cognitive_Remediation_and_Rehabilitation"], cognitive_remediation)

    metacognitive = [
        "Detached_Mindfulness_Practice",
        "Attention_Training_Technique_Framework",
        "Worry_Rumination_Postponement",
        "Reduce_Threat_Monitoring_Practice",
        "Metacognitive_Belief_Modification",
        "Cognitive_Confidence_Recalibration",
        "Uncertainty_Tolerance_Practice",
        "Rumination_Interrupt_Routines",
        "Refocusing_Strategy_Training",
        "Perspective_Shifting_Metacognitive",
        "Cognitive_Bias_Awareness_Training",
        "Attributional_Reframe_Practice",
        "Jumping_To_Conclusions_Check",
        "Alternative_Hypotheses_Drill",
        "Evidence_Weighting_Practice",
        "Attention_Switching_Practice",
        "Meta_Awareness_Of_Mental_Events",
        # Added: metacognitive therapy style elements
        "Modify_Beliefs_About_Thought_Danger",
        "Stop_Threat_Monitoring_And_Checking",
        "Postpone_Analysis_And_Ruminate_Less",
        "Practice_Attention_Flexibility_Shifts",
        "Metacognitive_Detachment_From_Inner_Dialogue",
    ]
    add_leaves(cog, ["Metacognitive_and_Attention_Based_Interventions"], metacognitive)

    PSYCHO["Cognitive_and_Metacognitive_Interventions"] = cog

    # ============================================================
    # 10) Core Psychological Skills Toolkits (cross-model, trans-theoretical)
    # Purpose:
    # - Portable “skills primitives” usable across CBT/ACT/DBT/MBI/psychodynamic/humanistic work
    # - Organized by capacity domains with 2-layer structure: Domain -> Subdomain -> Leaves
    # ============================================================

    core_skills = OntologyBuilder().root

    # ------------------------------------------------------------
    # 10.1 Emotion Regulation & Affective Mastery
    # 10.1 Emotion Regulation & Affective Mastery
    # ------------------------------------------------------------
    add_leaves(core_skills,
               ["Emotion_Regulation_and_Affective_Mastery", "Awareness_and_Clarity", "Labeling_and_Differentiation"], [
                   "Emotion_Labeling_Granularity",
                   "Emotion_Differentiation_Practice",
                   "Primary_vs_Secondary_Emotion_Distinction",
                   "Body_Emotion_Link_Identification",
                   "Emotion_Action_Tendency_Map",
                   "Affect_Label_And_Breathe_Practice",
                   "Name_Need_Request_Practice",
                   "Needs_Identification_From_Emotion",
               ])

    add_leaves(core_skills, ["Emotion_Regulation_and_Affective_Mastery", "Tolerance_and_Surfing", "Affect_Tolerance"], [
        "Affect_Tolerance_Skills",
        "Distress_Tolerance_Toolkit",
        "Grounding_Toolkit",
        "Urge_Surfing_Skill",
        "Impulse_Pause_Practice",
        "Arousal_Sensations_Reframe",
        "Self_Soothing_Sensory_Toolkit",
    ])

    add_leaves(core_skills, ["Emotion_Regulation_and_Affective_Mastery", "Change_Strategies", "Emotion_Modulation"], [
        "Cognitive_Reappraisal_Emotion",
        "Opposite_Action_Emotion",
        "Emotion_Exposure_Planning",
        "Emotional_Approach_Coping",
        "Emotion_Action_Tendency_Choice",
    ])

    add_leaves(core_skills,
               ["Emotion_Regulation_and_Affective_Mastery", "Self_and_Other_Validation", "Validation_Skills"], [
                   "Self_Validation_Practice",
                   "Other_Validation_Practice",
                   "Validation_Specificity_Practice",
                   "Co_Regulation_Request_Scripts",
               ])

    add_leaves(core_skills, ["Emotion_Regulation_and_Affective_Mastery", "Difficult_Emotion_Modules",
                             "Shame_Guilt_Anger_Jealousy_Grief"], [
                   "Reduce_Self_Criticism_Scripts",
                   "Compassionate_Self_Talk",
                   "Self_Compassion_In_Shame_Moment",
                   "Shame_Resilience_Practices",
                   "Anger_Regulation_Scripts",
                   "Jealousy_Appraisal_Reframe",
                   "Grief_Rituals_And_Meaning",
               ])

    add_leaves(core_skills,
               ["Emotion_Regulation_and_Affective_Mastery", "Repair_and_Aftercare", "Post_Dysregulation_Repair"], [
                   "Repair_After_Emotional_Outburst",
                   "Repair_After_Dysregulation_Plan",
                   "Aftercare_For_Intense_Emotion",
               ])

    # ------------------------------------------------------------
    # 10.2 Stress Resilience & Downregulation
    # ------------------------------------------------------------
    add_leaves(core_skills,
               ["Stress_Resilience_and_Downregulation", "Physiological_Downshift", "Breath_and_Relaxation_Skills"], [
                   "Diaphragmatic_Breathing_Practice",
                   "Paced_Breathing_Practice",
                   "Progressive_Muscle_Relaxation_Practice",
                   "Guided_Imagery_Relaxation",
                   "Soothing_Rhythm_Breathing",
                   "Arousal_Downshift_Scripts",
               ])

    add_leaves(core_skills, ["Stress_Resilience_and_Downregulation", "Mindfulness_For_Stress", "Orient_and_Ground"], [
        "Mindfulness_CheckIn_Practice",
        "Body_Scan_Grounding",
        "Five_Senses_Grounding_Practice",
        "Present_Moment_Orientation_Practice",
    ])

    add_leaves(core_skills,
               ["Stress_Resilience_and_Downregulation", "Cognitive_Stress_Skills", "Appraisal_and_Self_Efficacy"], [
                   "Stress_Signals_Awareness_Training",
                   "Stress_Appraisal_Reframe_Practice",
                   "Self_Efficacy_Priming_Under_Stress",
                   "Cognitive_Unloading_Journaling",
               ])

    add_leaves(core_skills, ["Stress_Resilience_and_Downregulation", "Recovery_Design", "Restoration_and_Self_Care"], [
        "Micro_Restoration_Routines",
        "Recovery_Rituals_Planning",
        "Self_Care_Plan_Design",
        "Boundary_Setting_For_Stress_Load",
    ])

    add_leaves(core_skills,
               ["Stress_Resilience_and_Downregulation", "Stress_Problem_Solving", "Action_Oriented_Coping"], [
                   "Problem_Solving_Stressors",
                   "Stressor_Controlability_Discrimination",
                   "Coping_Cards_For_Stress",
               ])

    # ------------------------------------------------------------
    # 10.3 Interpersonal Effectiveness & Relational Repair
    # ------------------------------------------------------------
    add_leaves(core_skills,
               ["Interpersonal_Effectiveness_and_Relational_Repair", "Core_Communication", "Listen_Express_Clarify"], [
                   "Active_Listening_Practice",
                   "Empathic_Communication_Practice",
                   "I_Statement_Communication_Practice",
                   "Clarifying_Questions_Practice",
                   "Summarize_Then_Check_Understanding",
               ])

    add_leaves(core_skills, ["Interpersonal_Effectiveness_and_Relational_Repair", "Assertiveness_and_Boundaries",
                             "Ask_Say_No_Negotiate"], [
                   "Assertiveness_Scripts",
                   "Boundary_Setting_Scripts",
                   "Request_Making_Skills",
                   "Refusal_Skills",
                   "Negotiation_And_Compromise_Skills",
                   "Compassionate_Boundary_Setting",
                   "Assertive_No_For_Overcommitment",
               ])

    add_leaves(core_skills,
               ["Interpersonal_Effectiveness_and_Relational_Repair", "Conflict_and_Deescalation", "Reduce_Escalation"],
               [
                   "Conflict_Deescalation_Scripts",
                   "Repair_Attempt_Practice",
                   "Rupture_Repair_Scripts",
                   "Boundary_Repair_After_Overstep",
                   "Nondefensive_Listening_Practice",
               ])

    add_leaves(core_skills,
               ["Interpersonal_Effectiveness_and_Relational_Repair", "Feedback_Skills", "Give_Receive_Integrate"], [
                   "Feedback_Giving_Skills",
                   "Feedback_Receiving_Skills",
                   "Receptivity_To_Feedback_Practice",
               ])

    add_leaves(core_skills, ["Interpersonal_Effectiveness_and_Relational_Repair", "Connection_and_Social_Fluency",
                             "Initiate_and_Deepen"], [
                   "Social_Initiation_Practice",
                   "Small_Talk_Practice",
                   "Vulnerability_Sharing_Practice",
                   "Trust_Building_MicroBehaviors",
                   "Nonviolent_Communication_Framework",
               ])

    # ------------------------------------------------------------
    # 10.4 Compassionate Self-Relating & Inner Support (CFT-adjacent, trans-model)
    # ------------------------------------------------------------
    add_leaves(core_skills, ["Compassionate_Self_Relating_and_Inner_Support", "Self_Compassion_Practices",
                             "Warmth_and_Common_Humanity"], [
                   "Self_Compassion_Break_Practice",
                   "Common_Humanity_Reframe",
                   "Permission_To_Be_Imperfect_Practice",
                   "Compassion_For_Mistakes_Learning_Frame",
               ])

    add_leaves(core_skills,
               ["Compassionate_Self_Relating_and_Inner_Support", "Compassionate_Cognition", "Reframes_and_Voice"], [
                   "Self_Criticism_To_Care_Reframe",
                   "Supportive_Inner_Voice_Practice",
                   "Compassionate_Letter_Writing",
               ])

    add_leaves(core_skills, ["Compassionate_Self_Relating_and_Inner_Support", "Compassionate_Imagery_and_Affect",
                             "Imagery_and_Breath"], [
                   "Compassionate_Imagery",
                   "Lovingkindness_Practice",
                   "Compassionate_Reparenting_Self_Practice",
                   "Soothing_Rhythm_Breathing",
               ])

    add_leaves(core_skills,
               ["Compassionate_Self_Relating_and_Inner_Support", "Self_Respect_and_Values", "Dignity_Boundaries"], [
                   "Values_Based_Self_Respect_Practice",
                   "Compassionate_Boundary_Setting",
               ])

    # ------------------------------------------------------------
    # 10.5 Executive Function & Self-Management (psychological, not BIO/SOCIAL)
    # ------------------------------------------------------------
    add_leaves(core_skills,
               ["Executive_Function_and_Self_Management", "Planning_and_Prioritization", "Goal_Structuring"], [
                   "Prioritization_Skill",
                   "Task_Breakdown_Skill",
                   "Decision_Clarity_ProsCons_Skill",
                   "Plan_Do_Review_Self_Management",
               ])

    add_leaves(core_skills,
               ["Executive_Function_and_Self_Management", "Initiation_and_Followthrough", "Start_and_Persist"], [
                   "Start_Cue_Design_Skill",
                   "Implementation_Intention_Self_Management",
                   "Self_Monitor_Then_Adjust_Skill",
               ])

    add_leaves(core_skills, ["Executive_Function_and_Self_Management", "Attention_and_Distraction", "Focus_Protection"],
               [
                   "Distractor_Management_Skill",
                   "Cognitive_Offloading_System_Design",
                   "Boundaries_For_Deep_Work_Practice",
               ])

    # Attach to PSYCHO with clearer top-level name
    PSYCHO["Core_Psychological_Skills_Toolkits"] = core_skills

    # ============================================================
    # 11) Integrative / Creative / Body-Oriented Psychotherapies (psychological techniques)
    # ============================================================
    integrative = OntologyBuilder().root

    hypnosis = [
        "Hypnotic_Induction_Framework",
        "Therapeutic_Suggestion_Design",
        "Imagery_Deepening_Procedure",
        "Ego_Strengthening_Suggestions",
        "Post_Hypnotic_Cue_Planning",
        "Hypnosis_For_Arousal_Reduction_Skills",
        "Self_Hypnosis_Skill_Training",
        # Added
        "Hypnotic_Safe_Place_Installation",
        "Hypnotic_Reframe_Of_Somatic_Sensations",
    ]
    add_leaves(integrative, ["Hypnosis_and_Suggestion_Based_Interventions"], hypnosis)

    expressive = [
        "Expressive_Writing_Prompting",
        "Metaphor_Development_Practice",
        "Values_Based_Narrative_Creation",
        "Therapeutic_Storytelling_Practice",
        "Art_Based_Emotion_Expression_Practice",
        "Music_Based_Emotion_Regulation_Practice",
        "Movement_Based_Emotion_Expression_Practice",
        "Psychodrama_Role_Reversal",
        "Psychodrama_Future_Rehearsal",
        "Psychodrama_Empty_Chair_Enactment",
        # Added: creative variations
        "Poetry_Based_Emotion_Labeling_Practice",
        "Collage_Based_Values_Clarification",
        "Imagery_And_Symbol_Work_Integration",
    ]
    add_leaves(integrative, ["Creative_and_Expressive_Interventions"], expressive)

    body_oriented = [
        "Interoceptive_Awareness_Training",
        "Somatic_Tracking_Practice",
        "Pendulation_Practice",
        "Titration_Of_Activation_Practice",
        "Grounding_Through_Posture_Practice",
        "Orienting_Response_Practice",
        "Completion_Of_Defensive_Response_Imagery",
        "Body_Map_Of_Emotion_Practice",
        "Somatic_Resourcing_Practice",
        # Added: somatic psychotherapy variables (non-BIO)
        "Somatic_Boundary_Sensing_Practice",
        "Tracking_Impulse_And_Choice_Point_Somatic",
        "Resourcing_With_Supportive_Movement",
    ]
    add_leaves(integrative, ["Somatic_and_Body_Oriented_Psychotherapies"], body_oriented)

    # Parts/inner-systems work (psychological; brand-agnostic)
    parts_work = [
        "Parts_Mapping_Practice",
        "Self_Leadership_Practice",
        "Inner_Critic_Externalization",
        "Protector_Part_Appreciation_Practice",
        "Vulnerable_Part_Compassion_Practice",
        "Internal_Dialogue_Chairwork_Parts",
        "Unblending_From_Intense_Part_Practice",
        "Negotiate_New_Roles_For_Parts",
        "Integrate_Learned_Protective_Strategies",
    ]
    add_leaves(integrative, ["Parts_And_Internal_Systems_Work"], parts_work)

    PSYCHO["Integrative_Creative_and_Body_Oriented_Interventions"] = integrative

    # ============================================================
    # 12) Digital / Guided Self-Help / Tech-Enabled Psychotherapy
    # ============================================================
    digital = OntologyBuilder().root

    # ------------------------------------------------------------
    # 12.1 Care delivery models, pathways, and session structures
    # (leaf nodes = actionable delivery “solution variables”)
    # ------------------------------------------------------------

    # A) Setting / channel (where care happens)
    add_leaves(digital, ["Care_Delivery_Models_and_Pathways", "Setting_and_Channel"], [
        "Individual_Therapy_InPerson",
        "Individual_Therapy_Telehealth_Video",
        "Individual_Therapy_Telehealth_Audio",
        "Individual_Therapy_Telehealth_TextBased",

        "Group_Therapy_InPerson",
        "Group_Therapy_Telehealth_Video",
        "Group_Therapy_Telehealth_TextBased",

        "Couple_Therapy_Format",
        "Family_Therapy_Format",

        "Blended_Care_Format_InPerson_Plus_Digital",
        "Blended_Care_Format_Telehealth_Plus_Digital",
    ])

    # B) Guidance / support intensity (who supports and how)
    add_leaves(digital, ["Care_Delivery_Models_and_Pathways", "Guidance_and_Support_Intensity"], [
        "Unguided_Self_Help_Format",
        "Guided_Self_Help_Format",
        "Clinician_Supported_Digital_Format",
        "Coach_Supported_Digital_Format",
        "Peer_Supported_Format",
        "Asynchronous_Therapy_Format",
        "Synchronous_Therapy_Format",
        "Hybrid_Synchronous_Asynchronous_Format",
        "Stepped_Care_Format",
        "Collaborative_Care_Digital_Adjunct_Format",
    ])

    # C) Session structure variants (still schedule-free; structure not dosage)
    add_leaves(digital, ["Care_Delivery_Models_and_Pathways", "Session_and_Contact_Structures"], [
        "Brief_CheckIn_Format",
        "Skills_Group_Format",
        "Workshop_Format",
        "OnDemand_Message_Coaching_Format",
        "Between_Session_Support_Channel_Format",
        "Measurement_Based_CheckIn_Format",
        "Assessment_Feedback_Only_Format",
        "Digital_Intake_And_Onboarding_Format",
    ])

    # D) Entry, triage, and routing (non-disorder-labeled; still psycho-level)
    add_leaves(digital, ["Care_Delivery_Models_and_Pathways", "Entry_Triage_and_Routing"], [
        "Digital_Self_Assessment_Then_Recommendation_Routing",
        "Preference_Based_Modality_Selection_Flow",
        "Readiness_And_Barrier_Screen_Then_Adapt_Flow",
        "Skill_Gap_Identification_Then_Module_Selection",
        "Stepped_Care_Escalation_Rules_Framework",
        "Care_Navigation_Assisted_Selection_Flow",
    ])

    # ------------------------------------------------------------
    # 12.2 Human support components (roles + workflows)
    # ------------------------------------------------------------
    add_leaves(digital, ["Human_Support_Components", "Support_Roles"], [
        "Clinician_Guidance_Role",
        "Coach_Guidance_Role",
        "Care_Manager_Role",
        "Peer_Moderator_Role",
        "Group_Facilitator_Role",
        "Technical_Onboarding_Support_Role",
    ])

    add_leaves(digital, ["Human_Support_Components", "Support_Workflows"], [
        "Asynchronous_Message_Review_Workflow",
        "Homework_Review_And_Feedback_Workflow",
        "Skills_Coaching_Feedback_Workflow",
        "Goal_Setting_And_Revision_Workflow",
        "Alliance_Feedback_CheckIn_Workflow",
        "Rupture_Repair_Conversation_Workflow_Digital",
        "Between_Session_Support_Boundaries_Workflow",
        "Referral_And_Handoff_Workflow_Digital",
    ])

    # ------------------------------------------------------------
    # 12.3 Interaction modalities and communication primitives
    # ------------------------------------------------------------
    add_leaves(digital, ["Interaction_Modalities_and_Channels", "Communication_Channels"], [
        "Secure_Messaging_Between_Sessions",
        "InApp_Voice_Note_Coaching_Channel",
        "InApp_Video_Session_Channel",
        "InApp_Audio_Session_Channel",
        "Chat_Based_Session_Channel",
        "Email_Like_Asynchronous_Therapy_Channel",
        "SMS_Based_Skills_Prompt_Channel",
    ])

    add_leaves(digital, ["Interaction_Modalities_and_Channels", "Interfaces_and_Interaction_Styles"], [
        "Conversational_Coaching_Interface",
        "Structured_Form_Based_Interface",
        "Guided_Wizard_StepByStep_Interface",
        "Microlearning_Card_Interface",
        "Interactive_Exercise_Interface",
        "Audio_Guided_Practice_Interface",
        "Video_Guided_Practice_Interface",
        "VR_Therapeutic_Experience_Framework",
        "AR_Exposure_Or_Skills_Practice_Framework",
    ])

    add_leaves(digital, ["Interaction_Modalities_and_Channels", "Sync_And_Response_Patterns"], [
        "RealTime_Coaching_Interaction",
        "NearRealTime_Response_Window_Expectation_Setting",
        "Batch_Response_Coaching_Mode",
        "User_Initiated_OnDemand_Support_Mode",
        "System_Initiated_CheckIn_Mode",
    ])

    # ------------------------------------------------------------
    # 12.4 Digital program templates (structured intervention “containers”)
    # (leaves represent implementable program archetypes)
    # ------------------------------------------------------------
    add_leaves(digital, ["Digital_Intervention_Program_Templates", "CBT_Family_Programs"], [
        "Digital_CBT_Skills_Course_Program",
        "Digital_Behavioral_Activation_Program",
        "Digital_Cognitive_Restructuring_Program",
        "Digital_Exposure_And_Inhibitory_Learning_Program",
        "Digital_Problem_Solving_Program",
        "Digital_Worry_And_Rumination_Management_Program",
        "Digital_Sleep_Skills_CBTI_Informed_Program_PsychoOnly",
    ])

    add_leaves(digital, ["Digital_Intervention_Program_Templates", "Third_Wave_Programs"], [
        "Digital_ACT_Psychological_Flexibility_Program",
        "Digital_DBT_Skills_Program",
        "Digital_Mindfulness_Based_Program",
        "Digital_Self_Compassion_Program",
        "Digital_Radical_Openness_Skills_Program",
    ])

    add_leaves(digital, ["Digital_Intervention_Program_Templates", "Interpersonal_And_Emotion_Programs"], [
        "Digital_Interpersonal_Skills_Program",
        "Digital_Communication_And_Repair_Program",
        "Digital_Emotion_Regulation_Program",
        "Digital_Stress_Resilience_Program",
        "Digital_Values_And_Identity_Program",
    ])

    add_leaves(digital, ["Digital_Intervention_Program_Templates", "Wellbeing_And_Optimization_Programs"], [
        "Digital_Gratitude_And_Savoring_Program",
        "Digital_Strengths_And_Resource_Activation_Program",
        "Digital_Hope_And_Goal_Pathways_Program",
        "Digital_Flow_And_Engagement_Design_Program",
        "Digital_Resilience_And_Setback_Recovery_Program",
    ])

    # ------------------------------------------------------------
    # 12.5 Digital tools and supports (atomic components)
    # (leaf nodes represent concrete candidate tools/features)
    # ------------------------------------------------------------
    add_leaves(digital, ["Digital_Tools_and_Supports", "Core_Exercise_Tools"], [
        "Digital_Workbook_CBT",
        "Digital_Workbook_ACT",
        "Digital_Workbook_DBT_Skills",
        "Digital_Workbook_Mindfulness",
        "Digital_Workbook_Self_Compassion",

        "Guided_Journaling_App",
        "Cognitive_Unloading_Journal_Tool",
        "Thought_Record_App",
        "Values_Clarification_App",
        "Behavior_Activation_Planner_App",
        "Goal_Setting_And_Tracking_Tool_Psycho",
        "Implementation_Intention_IfThen_Planner_Tool",
        "Problem_Solving_Worksheet_Tool",
        "Decision_Balance_ProsCons_Tool",

        "Exposure_Planning_Tool",
        "Digital_Exposure_Hierarchy_Builder",
        "Response_Prevention_Support_Tool",
        "Safety_Behavior_Audit_Checklist_Tool",

        "Emotion_Labeling_And_Granularity_Tool",
        "Urge_Surfing_Guide_Tool",
        "Grounding_Five_Senses_Guide_Tool",
        "Self_Soothing_Toolkit_Builder",
        "Aftercare_Plan_Builder_Tool",
    ])

    add_leaves(digital, ["Digital_Tools_and_Supports", "Skills_Practice_and_Rehearsal"], [
        "Interactive_Skills_Practice_Module",
        "Roleplay_Simulation_Conversation_Practice_Module",
        "Behavioral_Rehearsal_With_Feedback_Module",
        "Guided_Imagery_Practice_Player",
        "Exposure_Simulation_Scenario_Module",
        "Cognitive_Rehearsal_Script_Generator_Tool",
        "Coping_Cards_And_Prompt_Cards_Builder",
    ])

    add_leaves(digital, ["Digital_Tools_and_Supports", "Mindfulness_And_Compassion_Media_Libraries"], [
        "Mindfulness_Timer_And_Prompt_Framework",
        "Mindfulness_Guided_Audio_Library",
        "Self_Compassion_Guided_Audio_Library",
        "Breath_Pacing_Guide_Framework",
        "Progressive_Muscle_Relaxation_Audio_Library",
        "Lovingkindness_Practice_Library",
    ])

    add_leaves(digital, ["Digital_Tools_and_Supports", "Tracking_And_Self_Monitoring_PsychoTools"], [
        "Mood_And_Context_Tracking_Tool",
        "EMA_Prompting_Tool",
        "Trigger_Response_Log_Tool",
        "Urge_Log_Tool",
        "Skill_Use_Tracking_Tool",
        "Values_Aligned_Action_Tracking_Tool",
        "Therapy_Takeaway_Summary_Log_Tool",
        "Personal_Progress_Dashboard_Psycho",
    ])

    add_leaves(digital, ["Digital_Tools_and_Supports", "Homework_And_Between_Session_Support"], [
        "Therapy_Homework_Manager",
        "Homework_Assignment_Library_And_Templates",
        "Homework_Submission_And_Feedback_Channel",
        "Between_Session_CheckIn_Form",
        "Between_Session_Reflection_Prompts_Tool",
        "Between_Session_Support_Request_Button_NonCrisis",
    ])

    add_leaves(digital, ["Digital_Tools_and_Supports", "Content_Library_and_Psychoeducation"], [
        "Therapeutic_Content_Library",
        "Microlearning_Psychoeducation_Cards",
        "Interactive_Psychoeducation_Quizzes",
        "Case_Example_Story_Library",
        "Skills_Demonstration_Video_Library",
    ])

    # ------------------------------------------------------------
    # 12.6 Personalization, adaptation, and just-in-time support
    # ------------------------------------------------------------
    add_leaves(digital, ["Personalization_and_Adaptive_Systems", "Personalization_Inputs_PsychoOnly"], [
        "Preference_Based_Content_Customization",
        "Values_Based_Module_Recommendation",
        "Goal_Based_Module_Recommendation",
        "Barrier_Based_Adaptation_Logic",
        "Skill_Gap_Based_Recommendation",
        "Context_Tag_Based_Recommendation_PsychoOnly",
    ])

    add_leaves(digital, ["Personalization_and_Adaptive_Systems", "Adaptive_Delivery_and_JITAI"], [
        "JustInTime_Adaptive_Skills_Cueing_Framework",
        "Trigger_Detected_Skills_Suggestion_RuleSet",
        "Emotion_Intensity_Based_Skill_Suggestion",
        "Rumination_Worry_Detected_Interrupt_And_Shift_Prompt",
        "PreEvent_Centering_Prompt_Framework",
        "PostEvent_Debrief_And_Learning_Prompt_Framework",
        "Adaptive_Practice_Difficulty_Scaffolding_Framework",
    ])

    add_leaves(digital, ["Personalization_and_Adaptive_Systems", "Digital_Coach_Reasoning_Styles_Psycho"], [
        "Socratic_Questioning_Digital_Coach_Mode",
        "Motivational_Interviewing_Digital_Coach_Mode",
        "Compassionate_Coach_Voice_Mode",
        "Behavioral_Coach_Action_First_Mode",
        "Values_Compass_Recenter_Mode",
    ])

    # ------------------------------------------------------------
    # 12.7 Engagement, adherence, and behavior design (schedule-free)
    # ------------------------------------------------------------
    add_leaves(digital, ["Engagement_and_Adherence_Design", "Onboarding_and_Commitment"], [
        "Digital_Onboarding_Guided_Setup_Wizard",
        "Expectation_Setting_And_Rationale_Explainer",
        "Values_Linked_Commitment_Contract_Digital",
        "Barrier_Elicitation_And_Problem_Solving_Module",
        "Early_Wins_Planning_Module",
        "Practice_Friction_Audit_Module",
    ])

    add_leaves(digital, ["Engagement_and_Adherence_Design", "Prompting_And_Reminders"], [
        "Skills_Cueing_Notifications",
        "Contextual_Reminders_For_Practice",
        "Gentle_Nudge_After_Dropoff_Framework",
        "Streaks_And_Consistency_Feedback_Framework",
        "Reflective_CheckIn_Prompt_Framework",
    ])

    add_leaves(digital, ["Engagement_and_Adherence_Design", "Motivation_And_Reward_Design_Psycho"], [
        "Gamified_Skills_Progression_Framework",
        "Micro_Rewards_For_Completion_Framework",
        "Self_Efficacy_Building_Feedback_Messages",
        "Strengths_Based_Progress_Feedback",
        "Identity_Based_Encouragement_Messaging",
    ])

    add_leaves(digital, ["Engagement_and_Adherence_Design", "Accountability_And_Social_Commitment_Psycho"], [
        "Accountability_Partner_Link_Feature",
        "Coach_CheckIn_Accountability_Feature",
        "Peer_Group_Accountability_Feature",
        "Share_Progress_Summary_With_Chosen_Support",
    ])

    # ------------------------------------------------------------
    # 12.8 Safety, escalation, and scope boundaries (non-crisis framing)
    # (psycho-level: recognition + routing; not emergency operations)
    # ------------------------------------------------------------
    add_leaves(digital, ["Safety_and_Escalation_NonCrisis", "Early_Warning_And_Safety_Checks"], [
        "Early_Warning_Signs_CheckIn_Module",
        "Coping_Plan_Structure_NonCrisis_Digital",
        "Support_Seeking_Scripts_NonCrisis_Digital",
        "Means_Safety_Conversation_Framework_Digital",
    ])

    add_leaves(digital, ["Safety_and_Escalation_NonCrisis", "Escalation_And_Handoff_Routing"], [
        "Escalate_To_Clinician_Review_Rule",
        "Escalate_To_Care_Manager_Contact_Rule",
        "Warm_Handoff_To_Higher_Intensity_Care_Pathway",
        "Scope_Limits_And_Referral_Explanation_Module",
    ])

    # ------------------------------------------------------------
    # 12.9 Clinician-facing tools for tech-enabled psychotherapy
    # ------------------------------------------------------------
    add_leaves(digital, ["Clinician_Facing_Tools", "Session_Support_and_Documentation_Psycho"], [
        "Shared_Session_Agenda_Tool",
        "Shared_Case_Formulation_Whiteboard_Tool",
        "Shared_Goals_And_Values_Map_Tool",
        "Session_Notes_Sharing_Summary_Tool",
        "Between_Session_Message_Triage_Dashboard",
    ])

    add_leaves(digital, ["Clinician_Facing_Tools", "Measurement_And_Feedback_Views_Psycho"], [
        "Progress_Timeline_Dashboard_For_Clinician",
        "Skill_Use_And_Adherence_View_For_Clinician",
        "Client_Feedback_InSession_Snapshot",
        "Alliance_CheckIn_View_For_Clinician",
    ])

    add_leaves(digital, ["Clinician_Facing_Tools", "Homework_And_Exposure_Support"], [
        "Homework_Assignment_And_Review_Dashboard",
        "Digital_Behavioral_Experiment_Logger",
        "Exposure_Hierarchy_Shared_Edit_Tool",
        "Between_Session_Practice_Review_Summary",
    ])

    # ------------------------------------------------------------
    # 12.10 Peer and community features (psycho-only; supportive formats)
    # ------------------------------------------------------------
    add_leaves(digital, ["Peer_and_Community_Features", "Community_Design_Elements"], [
        "Peer_Support_Group_Chat_Moderated",
        "Peer_Led_Skills_Practice_Group_Format",
        "Community_Norms_And_Safety_Guidelines_Module",
        "Structured_Peer_CheckIn_Prompts",
        "Anonymous_Sharing_With_Boundaries_Feature",
    ])

    add_leaves(digital, ["Peer_and_Community_Features", "Prosocial_And_Belonging_Levers_Psycho"], [
        "Encouragement_And_Recognition_Feature",
        "Share_Success_Strategies_Library_Community",
        "Buddy_System_Pairing_Feature",
    ])

    # ------------------------------------------------------------
    # 12.11 Accessibility, inclusion, and usability supports
    # ------------------------------------------------------------
    add_leaves(digital, ["Accessibility_and_Inclusion_Features", "Accessibility_Supports"], [
        "Low_Literacy_Mode_Simplified_Language",
        "Multilingual_Content_Support",
        "Audio_First_Mode_For_Low_Reading_Load",
        "Captioned_Video_Content_Support",
        "Screen_Reader_Compatibility_Feature",
        "Cognitive_Load_Reduction_UI_Mode",
    ])

    add_leaves(digital, ["Accessibility_and_Inclusion_Features", "Personalization_For_Identity_And_Context"], [
        "Culturally_Sensitive_Examples_Library",
        "Values_Sensitive_Language_Options_Digital",
        "Pronoun_And_Name_Preferences_Support",
        "Content_Trigger_Warnings_And_OptOut_Controls",
    ])

    # ------------------------------------------------------------
    # 12.12 Privacy, consent, and data-control (psycho-level solution variables)
    # (kept here because digital delivery materially changes these choices)
    # ------------------------------------------------------------
    add_leaves(digital, ["Privacy_Consent_and_Data_Control", "Consent_And_Transparency_Flows"], [
        "Privacy_And_Data_Use_Explanation_For_Digital_Tools",
        "InApp_Consent_Checkpoint_For_Sensitive_Exercises",
        "Explain_AI_Assistance_Role_And_Limits_Module",
        "Explain_Coach_Or_Clinician_Response_Boundaries_Module",
    ])

    add_leaves(digital, ["Privacy_Consent_and_Data_Control", "User_Data_Control_Features"], [
        "User_Control_Data_Sharing_With_Clinician_Feature",
        "Selective_Sharing_Per_Module_Feature",
        "Data_Download_And_Deletion_Request_Flow",
        "Anonymized_Mode_For_Self_Reflection_Notes",
    ])

    # Attach subtree
    PSYCHO["Digital_and_Tech_Enabled_Solutions"] = digital

    # ============================================================
    # 13) Measurement-Based Care / Feedback / Self-Monitoring
    # ============================================================
    mbc = OntologyBuilder().root

    monitoring = [
        "Symptom_Self_Report_Tracking",
        "Functioning_Tracking",
        "Values_Aligned_Action_Tracking",
        "Behavior_Chain_Logs",
        "Thought_Record_Tracking",
        "Exposure_Progress_Tracking",
        "Activity_Engagement_Tracking",
        "Interpersonal_Event_Logs",
        "Stressors_And_Coping_Log",
        "Trigger_Response_Log",
        "Urge_Log",
        "Compassion_Practice_Log",
        "Goal_Progress_Mapping",
        "Session_Impact_Feedback_Form",
        "Alliance_Feedback_Form",
        "Therapist_Feedback_Review_Routine",
        "Personalized_Outcome_Monitoring",
        "Progress_Barrier_CheckIn",
        # Added: measurement enrichments
        "Goal_Attainment_Scaling_Framework",
        "Idiographic_Item_Tracking_Framework",
        "Daily_Context_Tagging_Framework_NoFrequency",
        "Skill_Use_Tracking",
        "Therapy_Takeaway_Summary_Log",
    ]
    add_leaves(mbc, ["Measurement_and_Feedback_Systems"], monitoring)

    PSYCHO["Measurement_Based_Care_and_Feedback"] = mbc

    # ============================================================
    # 14) Positive Psychology / Wellbeing / Resilience (NEW primary node)
    # ============================================================
    positive = OntologyBuilder().root

    gratitude = [
        "Gratitude_Journaling_Practice",
        "Gratitude_Letter_Writing",
        "Gratitude_Visit_Planning",
        "Three_Good_Things_Practice",
        "Savoring_Positive_Moments_Practice",
        "Positive_Event_Capitalization_Sharing",
        "Appreciation_Expression_Practice",
    ]
    add_leaves(positive, ["Gratitude_and_Savoring"], gratitude)

    strengths = [
        "Strengths_Identification_Practice",
        "Strengths_Use_Planning",
        "Signature_Strengths_Activation",
        "Strengths_Based_Reframe",
        "Strengths_In_New_Context_Experiment",
    ]
    add_leaves(positive, ["Strengths_and_Resource_Activation"], strengths)

    meaning = [
        "Meaning_Making_Interview",
        "Values_To_Purpose_Link_Practice",
        "Best_Possible_Self_Exercise",
        "Life_Story_Integration_For_Meaning",
        "Acts_Of_Kindness_Planning_Psychological",
        "Self_Transcendence_Practice",
    ]
    add_leaves(positive, ["Meaning_Purpose_and_Future_Orientation"], meaning)

    resilience = [
        "Optimism_Balanced_Realism_Practice",
        "Hope_Pathways_And_Agency_Practice",
        "Self_Efficacy_Mastery_Recall_Practice",
        "Coping_Flexibility_Training",
        "Resilience_Lesson_Learning_Review",
        "Post_Adversity_Growth_Narrative_Practice",
    ]
    add_leaves(positive, ["Resilience_and_Optimism_Skills"], resilience)

    PSYCHO["Positive_Psychology_and_Wellbeing"] = positive

    # ============================================================
    # 15) Self, Identity, Shame, Perfectionism, Values Conflicts (NEW primary node)
    # ============================================================
    self_identity = OntologyBuilder().root

    self_esteem = [
        "Self_Esteem_Evidence_Log",
        "Self_Worth_From_Values_Practice",
        "Reduce_Self_Comparison_Practice",
        "Self_Compassion_For_Failure_Practice",
        "Identify_And_Challenge_Self_Labeling",
        "Balanced_Self_Appraisal_Practice",
    ]
    add_leaves(self_identity, ["Self_Esteem_and_Self_Worth_Work"], self_esteem)

    perfectionism = [
        "Set_Good_Enough_Standard_Practice",
        "Behavioral_Experiment_Imperfect_Action",
        "Reduce_Reassurance_And_Checking_Perfectionism",
        "Reframe_Mistakes_As_Learning",
        "Values_Over_Standards_Choice_Point",
        "Compassionate_Response_To_Inner_Critic",
    ]
    add_leaves(self_identity, ["Perfectionism_and_Standards_Work"], perfectionism)

    shame_guilt = [
        "Shame_Externalization_Practice",
        "Name_Shame_And_Return_To_Values",
        "Guilt_Repair_Planning",
        "Self_Forgiveness_Practice",
        "Compassionate_Witnessing_Practice",
        "Disclosure_With_Boundaries_Practice",
    ]
    add_leaves(self_identity, ["Shame_Guilt_and_Self_Forgiveness"], shame_guilt)

    values_conflict = [
        "Values_Conflict_Clarification",
        "Moral_Dilemma_Processing_Practice",
        "Integrate_Competing_Values_Practice",
        "Commitment_To_Chosen_Value_Path",
        "Self_Respect_Decision_Practice",
    ]
    add_leaves(self_identity, ["Values_Conflict_and_Moral_Repair_Work"], values_conflict)

    PSYCHO["Self_and_Identity_Work"] = self_identity

    # ============================================================
    # 13) Change Process and Therapeutic Phases
    # ============================================================
    change_process = OntologyBuilder().root

    # ------------------------------------------------------------
    # Crosscutting: Phase Management, Sequencing, and Optimization
    # ------------------------------------------------------------
    phase_management = [
        "Stepped_Care_Level_Assignment",
        "Care_Pathway_Low_Intensity_To_High_Intensity_Selection",
        "Acuity_Triage_And_Priority_Setting",
        "Stabilize_Before_Process_Decision_Routine",
        "Phase_Readiness_Screening_Routine",
        "Phase_Gate_Checklist_Use",
        "Dose_Pacing_And_Titration_Planning",
        "Microstep_Chunking_And_Grading_Plan",
        "Complexity_And_Comorbidity_Staging_Check",
        "Barrier_Friction_Audit_And_Removal_Plan",
        "Practice_Quality_Check_And_Coaching",
        "Mechanism_Target_Selection_Routine",
        "Intervention_Selection_By_Context_Sensitivity",
        "Cultural_Context_And_Identity_Sensitive_Adaptation_Routine",
        "Measurement_Based_Iteration_Weekly_Review",
        "Nonresponse_Algorithm_Reformulate_Adapt",
        "Rupture_Repair_Interrupt_And_Return_Routine",
        "Crisis_Interrupt_Stabilize_Then_Return_Routine",
        "Treatment_Dose_Adjustment_Routine",
        "Treatment_Focus_Shift_Decision_Routine",
        "Modality_Switch_And_Augmentation_Decision_Routine",
        "Homework_Design_Assign_Review_Routine",
        "Between_Session_Support_Design_Routine",
        "Generalization_Planning_Routine",
        "Termination_And_Transition_Planning_Routine",
        "Maintenance_Minimum_Viable_Practice_Design",
        "Wellbeing_Optimization_Path_Selection",
        "Clinical_Symptom_Stabilization_Path_Selection",
        "Performance_And_Resilience_Path_Selection",
        "Hybrid_Path_Selection_Wellbeing_And_Clinical",
    ]
    add_leaves(change_process, ["Phase_Management_and_Sequencing"], phase_management)

    case_formulation_and_planning = [
        "Idiosyncratic_Case_Formulation_Sprint",
        "Maintenance_Cycle_Map_Construction",
        "Functional_Analysis_Situations_Thoughts_Emotions_Behavior",
        "Trigger_Vulnerability_Protective_Factors_Map",
        "Mechanism_Hypothesis_Testing_Plan",
        "Target_Prioritization_By_Impact_And_Feasibility",
        "Define_Success_Criteria_And_Operational_Indicators",
        "Define_Early_Warning_Signs_And_Risk_Indicators",
        "Shared_Treatment_Plan_Drafting_Routine",
        "Phase_Assignment_By_Primary_Bottleneck",
        "Therapeutic_Levers_LowRisk_HighYield_Selection",
        "Constraint_And_Capacity_Check_Routine",
        "Practice_Dose_And_Feedback_Loop_Setup",
    ]
    add_leaves(change_process, ["Phase_Management_and_Sequencing", "Formulation_and_Treatment_Planning"],
               case_formulation_and_planning)

    stabilization_and_safety_crosscutting = [
        "Stabilization_Plan_When_Arousal_Or_Chaos_High",
        "Sleep_Stability_Firstline_Plan_When_Appropriate",
        "Crisis_Safety_Plan_Activation_Protocol",
        "Reduce_Acute_Avoidance_That_Increases_Risk_Plan",
        "Substance_Risk_Harm_Reduction_Referral_Linkage",
        "Dissociation_Management_Protocol_When_Present",
        "Self_Harm_Urge_Surfing_And_Means_Safety_Plan",
        "Grounding_First_Response_Protocol",
        "Co_Regulation_Support_Activation_Plan",
    ]
    add_leaves(change_process, ["Phase_Management_and_Sequencing", "Stabilization_and_Safety_Crosscutting"],
               stabilization_and_safety_crosscutting)

    optimization_and_flourishing_crosscutting = [
        "Strengths_First_Case_Conceptualization_Routine",
        "PERMA_Domain_Target_Selection_Routine",
        "Values_To_Time_Budget_Reallocation_Plan",
        "Flow_Opportunity_Design_Routine",
        "Positive_Emotion_Micropractice_Schedule",
        "Social_Connection_Investment_Plan",
        "Meaning_Purpose_Statement_And_Action_Plan",
        "Self_Compassion_And_Dignity_Protection_Plan",
        "Resilience_Buffer_Building_Plan",
    ]
    add_leaves(change_process, ["Phase_Management_and_Sequencing", "Flourishing_and_Prevention_Crosscutting"],
               optimization_and_flourishing_crosscutting)

    # ------------------------------------------------------------
    # Phase 1: Readiness and Engagement
    # ------------------------------------------------------------
    onboarding = [
        "Intake_Orientation_Session_Structure",
        "Explain_Process_And_Roles_Quickstart",
        "Expectation_Alignment_Conversation",
        "Therapy_Rationale_Link_To_Goals",
        "Preferred_Format_And_Channel_Setup",
        "Time_Energy_Budget_Planning",
        "Engagement_Support_And_Reminder_Plan",
        "Initial_Practice_Readiness_Check",
        "Logistics_Barrier_Solving_Plan",
        "First_Session_Win_Planning",
    ]
    add_leaves(change_process, ["Readiness_and_Engagement", "Access_Orientation_and_Onboarding"], onboarding)

    alliance_and_contract = [
        "Collaborative_Goal_Setting_Protocol",
        "Shared_Case_Frame_Starter_Map",
        "Session_Agenda_Collaboration_Routine",
        "Feedback_Informed_Checkin_Setup",
        "Rupture_Prediction_And_Repair_Plan",
        "Boundary_And_Frame_Clarification",
        "Treatment_Priorities_Triage_Conversation",
        "Shared_Decision_Making_Setup",
        "Therapy_Preferences_And_Fit_Conversation",
        "Therapeutic_Stance_Agreement_Supportive_Directive_Balance",
    ]
    add_leaves(change_process, ["Readiness_and_Engagement", "Alliance_Contracting_and_Working_Agreement"],
               alliance_and_contract)

    motivation_and_ambivalence = [
        "Readiness_And_Confidence_Ruler_Protocol",
        "Ambivalence_Exploration_Dialogue",
        "Decisional_Balance_Worksheet_Use",
        "Values_Discrepancy_Elicitation",
        "Change_Talk_Elicitation_Sequence",
        "Sustain_Talk_Softening_Sequence",
        "Commitment_Step_Negotiation",
        "Motivation_Renewal_Checkpoints",
        "Self_Efficacy_Priming_Protocol",
        "Hope_And_Expectancy_Shaping_Protocol",
    ]
    add_leaves(change_process, ["Readiness_and_Engagement", "Motivation_Readiness_and_Ambivalence_Work"],
               motivation_and_ambivalence)

    safety_and_consent = [
        "Informed_Consent_Review_And_Clarification",
        "Confidentiality_Limits_Explanation",
        "Risk_Screening_Routine",
        "Protective_Factors_Map_Creation",
        "Crisis_Plan_Creation",
        "Means_Safety_Conversation_Framework",
        "Stabilization_Need_Assessment",
        "Referral_And_Scope_Alignment_Check",
        "Crisis_Contact_List_And_Steps_Setup",
        "Stop_Rules_And_Pacing_Agreement_For_Intense_Work",
    ]
    add_leaves(change_process, ["Readiness_and_Engagement", "Safety_Consent_and_Risk_Setup"], safety_and_consent)

    baseline_and_success = [
        "Baseline_Symptom_Function_Wellbeing_Profile",
        "Context_And_Trigger_Inventory",
        "Strengths_And_Resources_Inventory",
        "Success_Criteria_Operationalization",
        "Primary_Mechanism_Hypothesis_Setup",
        "Initial_Target_Prioritization_Routine",
        "Measurement_Plan_Setup",
        "First_Week_Action_Plan",
        "Daily_Context_Tagging_Routine",
        "Initial_Skills_And_Tools_Selection",
    ]
    add_leaves(change_process, ["Readiness_and_Engagement", "Baseline_Assessment_and_Success_Definition"],
               baseline_and_success)

    adherence_support = [
        "Dropout_Risk_Conversation_Protocol",
        "Practice_Friction_Audit",
        "Environment_Support_Setup",
        "Accountability_Structure_Selection",
        "Between_Session_Support_Structure_Setup",
        "Early_Wins_Planning",
        "Implementation_Intention_For_Attendance_And_Practice",
        "Practice_Anchor_Selection_Routine",
        "Reminder_And_Prompting_System_Setup",
    ]
    add_leaves(change_process, ["Readiness_and_Engagement", "Engagement_Friction_and_Adherence_Support"],
               adherence_support)

    wellbeing_entry = [
        "Wellbeing_Goal_Clarification_Session",
        "Lifestyle_Stability_Check_Sleep_Stress_Support",
        "Strengths_Deployment_Plan",
        "Positive_Emotion_Micropractices_Plan",
        "Values_Aligned_Time_Allocation_Plan",
        "Meaning_And_Purpose_Checkin",
        "Preventive_Resilience_Routine_Setup",
        "Maintain_Gains_And_Prevent_Drift_Plan",
        "Community_And_Belonging_Entry_Plan",
        "Flow_And_Engagement_Entry_Plan",
    ]
    add_leaves(change_process, ["Readiness_and_Engagement", "Wellbeing_Optimization_Entry_Path"], wellbeing_entry)

    # ------------------------------------------------------------
    # Phase 2: Skill Acquisition
    # ------------------------------------------------------------
    psychoeducation = [
        "Personalized_Maintenance_Cycle_Map",
        "Normalize_Symptoms_And_Setbacks_Learning_Frame",
        "Explain_Practice_And_Generalization_Rationale",
        "Mechanism_Target_Mapping_Session",
        "Explain_Emotion_Learning_And_Avoidance",
        "Explain_Attention_Worry_Rumination_Loops",
        "Explain_Habit_And_Context_Dependence",
        "Explain_Values_Goals_And_Choice_Points",
        "Explain_Inhibitory_Learning_For_Exposure",
        "Explain_Reinforcement_And_Behavioral_Activation",
    ]
    add_leaves(change_process, ["Skill_Acquisition", "Model_Orientation_and_Psychoeducation"], psychoeducation)

    foundational_regulation = [
        "Grounding_And_Orienting_Practice",
        "Breath_Pacing_And_Downshift_Practice",
        "Arousal_Labeling_And_Reappraisal_Practice",
        "Sleep_Protection_Basics_Implementation",
        "Impulse_Pause_And_Urge_Surfing_Practice",
        "Self_Soothing_Sensory_Toolkit_Setup",
        "Micro_Recovery_Routines_Design",
        "Coping_Plan_Structure_NonCrisis",
        "Attention_Reset_60_Second_Practice",
        "Self_Compassion_First_Aid_Practice",
    ]
    add_leaves(change_process, ["Skill_Acquisition", "Foundational_Self_Regulation_Skills"], foundational_regulation)

    emotion_skills = [
        "Emotion_Labeling_Granularity_Training",
        "Primary_Secondary_Emotion_Discrimination_Practice",
        "Emotion_Function_Analysis_Practice",
        "Opposite_Action_Selection_And_Planning",
        "Allowing_Emotion_Waves_Practice",
        "Distress_Tolerance_Core_Skills_Training",
        "Aftercare_For_Intense_Emotion_Plan",
        "Repair_After_Dysregulation_Plan",
        "Compassionate_Self_Talk_Training",
        "Reappraisal_And_Reframing_Practice",
    ]
    add_leaves(change_process, ["Skill_Acquisition", "Emotion_Regulation_Skill_Building"], emotion_skills)

    cognitive_attention_skills = [
        "Thought_Monitoring_Log_Setup",
        "Automatic_Thought_Identification_Practice",
        "Decentering_And_Defusion_Practice",
        "Socratic_Questioning_Basics_Practice",
        "Probability_Reestimation_Practice",
        "Decatastrophizing_Practice",
        "Worry_Rumination_Discrimination_Practice",
        "Attention_Flexibility_Training_Routine",
        "Cognitive_Restructuring_Worksheet_Practice",
        "Rumination_Interruption_Protocol",
    ]
    add_leaves(change_process, ["Skill_Acquisition", "Cognitive_and_Attentional_Skill_Building"],
               cognitive_attention_skills)

    interpersonal_skills = [
        "Active_Listening_Training",
        "Validation_Skills_Training",
        "Assertiveness_Request_Making_Training",
        "Refusal_And_Boundary_Setting_Training",
        "Conflict_Deescalation_Scripts_Practice",
        "Repair_Attempt_Practice",
        "Feedback_Give_Receive_Training",
        "Support_Seeking_Scripts_Practice",
        "Difficult_Conversation_Rehearsal",
        "Co_Regulation_Request_Practice",
    ]
    add_leaves(change_process, ["Skill_Acquisition", "Interpersonal_and_Communication_Skill_Building"],
               interpersonal_skills)

    behavioral_planning = [
        "SMART_Goal_Specification_Practice",
        "Implementation_Intention_IfThen_Plans_Creation",
        "Graded_Task_Assignment_Planning",
        "Behavioral_Activation_Scheduling",
        "Stimulus_Control_And_Friction_Design",
        "Reward_Engineering_And_MicroWins_Planning",
        "Homework_Planning_And_Review_System",
        "Practice_Logging_And_Learning_Summary",
        "Time_Blocking_For_Practice_And_Routine",
        "Coping_Planning_For_Barriers_Practice",
    ]
    add_leaves(change_process, ["Skill_Acquisition", "Behavioral_Planning_and_Execution_Skills"], behavioral_planning)

    fluency_generalization = [
        "InSession_Rehearsal_And_Roleplay_Routine",
        "Between_Session_Practice_Schedule_Build",
        "Practice_Quality_Check_And_Coaching",
        "Context_Variation_Practice_Plan",
        "Teach_Back_And_Summarize_Learning",
        "Skill_Selection_Decision_Tree_Practice",
        "Early_Warning_Signs_Recognition_Framework",
        "Therapy_Toolbox_Consolidation",
        "Skill_Combination_Stacking_Practice",
        "Generalization_Test_Assignment",
    ]
    add_leaves(change_process, ["Skill_Acquisition", "Skill_Fluency_and_Generalization_Training"],
               fluency_generalization)

    # ------------------------------------------------------------
    # Phase 3: Experiential Disconfirmation
    # ------------------------------------------------------------
    experiential_readiness = [
        "Window_Of_Tolerance_Check_And_Adjust",
        "Anchoring_Skills_Verification",
        "Consent_Stop_Rules_And_Pacing_Agreement",
        "Safety_Behavior_And_Avoidance_Audit",
        "Dissociation_Risk_Check_And_Plan",
        "Aftercare_Plan_For_Intense_Work",
        "Crisis_Contingency_Plan_Activation_Routine",
        "Therapeutic_Pacing_Adjustment_Routine",
        "Titration_And_Resourcing_Before_Exposure_Practice",
        "Overwhelm_Recovery_Protocol_After_Session",
    ]
    add_leaves(change_process, ["Experiential_Disconfirmation", "Experiential_Work_Readiness_and_Safety"],
               experiential_readiness)

    behavioral_experiments = [
        "Prediction_And_Alternative_Prediction_Setup",
        "Single_Variable_Change_Experiment_Design",
        "Measurement_And_Rating_Setup",
        "Attention_Manipulation_Experiment",
        "Safety_Behavior_Drop_Experiment",
        "Interpersonal_Signaling_Experiment",
        "Post_Experiment_Learning_Summary",
        "Generalization_Experiment_Planning",
        "Behavioral_Experiment_Failure_Mode_Review",
        "Experiment_Repetition_With_Variability_Plan",
    ]
    add_leaves(change_process, ["Experiential_Disconfirmation", "Behavioral_Experiment_Protocol"],
               behavioral_experiments)

    exposure_inhibitory_learning = [
        "Exposure_Hierarchy_Construction",
        "Expectancy_Violation_Planning",
        "Multiple_Context_Exposure_Planning",
        "Variability_Exposure_Planning",
        "Response_Prevention_Planning",
        "Post_Exposure_Processing_Guidelines",
        "Reinstatement_Prevention_Planning",
        "Exposure_Learning_Log_Use",
        "Interoceptive_Exposure_When_Indicated",
        "Imaginal_Exposure_When_Indicated",
    ]
    add_leaves(change_process, ["Experiential_Disconfirmation", "Exposure_and_Inhibitory_Learning_Protocol"],
               exposure_inhibitory_learning)

    somatic_experiential = [
        "Somatic_Tracking_Practice",
        "Pendulation_Practice",
        "Titration_Of_Activation_Practice",
        "Orienting_Response_Practice",
        "Completion_Of_Defensive_Response_Imagery",
        "Embodied_Grounding_Practice",
        "Body_Map_Of_Emotion_Practice",
        "Interoceptive_Sensations_Reappraisal_Practice",
        "Movement_Based_Discharge_Practice",
        "Somatic_Aftercare_Routine",
    ]
    add_leaves(change_process, ["Experiential_Disconfirmation", "Interoceptive_and_Somatic_Experiential_Work"],
               somatic_experiential)

    emotion_processing = [
        "Primary_Emotion_Accessing_Practice",
        "Two_Chair_Self_Criticism_Work",
        "Compassionate_Chair_Work",
        "Imagery_Rescripting_Framework",
        "Corrective_Emotional_Experience_Facilitation",
        "Transform_Shame_With_Compassion_Practice",
        "Transform_Fear_With_Security_Practice",
        "Transform_Anger_With_Boundaries_Practice",
        "Grief_Processing_And_Integration_Practice",
        "Self_Compassion_Repair_After_Shame_Practice",
    ]
    add_leaves(change_process, ["Experiential_Disconfirmation", "Emotion_Processing_and_Transformational_Experiences"],
               emotion_processing)

    relational_experiential = [
        "InSession_Pattern_Evocation_And_Noticing",
        "Therapeutic_Honesty_Practice",
        "Interpersonal_Feedback_Loops_Practice",
        "Boundary_Enactment_Practice",
        "Rupture_Recognition_And_Repair_Practice",
        "Authentic_Self_Disclosure_Practice",
        "Social_Signaling_Skills_Experiments",
        "New_Relational_Roles_Tryout_Practice",
        "Corrective_Relational_Experience_Planning",
        "Relational_Exposure_To_Conflict_Or_Intimacy_When_Indicated",
    ]
    add_leaves(change_process, ["Experiential_Disconfirmation", "Relational_and_InSession_Experiential_Work"],
               relational_experiential)

    trauma_informed_optional = [
        "Stabilization_Readiness_Check_Framework",
        "Dual_Attention_Grounding_Practice",
        "Titration_And_Pacing_Principle_Use",
        "Target_Memory_Selection_Protocol",
        "Narrative_Processing_Structure",
        "Hotspot_Processing_Guidance",
        "Closure_Stabilization_Procedure",
        "Future_Template_Rehearsal_Practice",
        "Trigger_Linked_Somatic_Cue_Processing_Practice",
        "Post_Processing_Self_Care_And_Support_Plan",
    ]
    add_leaves(change_process, ["Experiential_Disconfirmation", "Trauma_Informed_Experiential_Processing_Optional"],
               trauma_informed_optional)

    # ------------------------------------------------------------
    # Phase 4: Meaning Update
    # ------------------------------------------------------------
    learning_consolidation = [
        "Session_Learning_Summary_Routine",
        "Mechanism_Change_Attribution_Review",
        "Update_Formulation_Map_Routine",
        "Teach_Back_What_Changed_And_Why",
        "Learning_To_Action_Bridge_Planning",
        "Integrate_Setbacks_As_Learning_Frame",
        "Update_Rules_And_Shoulds_Rewrite",
        "Create_Personal_Playbook_Document",
        "Write_If_Then_Lessons_Learned_Rules",
        "Create_Trigger_To_Response_Scripts",
    ]
    add_leaves(change_process, ["Meaning_Update", "Learning_Extraction_and_Consolidation"], learning_consolidation)

    belief_rule_updating = [
        "Core_Belief_Update_With_Evidence_Log",
        "Belief_Conditionality_Softening_Practice",
        "Reattribute_Threat_To_Arousal_Sensitivity_Practice",
        "Responsibility_Reappraisal_Practice",
        "Uncertainty_As_Survivable_Reframe_Practice",
        "Perfectionism_Standards_Cycle_Rewrite",
        "Self_Critical_Thought_Restructure_Practice",
        "Alternative_Explanations_Generation_Practice",
        "Cost_Benefit_Analysis_For_Rules_Practice",
        "Compassionate_Reframe_Of_Self_Judgment_Practice",
    ]
    add_leaves(change_process, ["Meaning_Update", "Cognitive_Belief_and_Rule_Updating"], belief_rule_updating)

    metacognitive_update = [
        "Beliefs_About_Thinking_Danger_Update_Practice",
        "Thought_Suppression_Rebound_Learning_Integration",
        "Threat_Monitoring_Reduction_Plan",
        "Rumination_Worry_Utility_Reappraisal_Practice",
        "Cognitive_Confidence_Recalibration_Practice",
        "Attention_Set_Shifting_Practice",
        "Detached_Mindfulness_Practice",
        "Meta_Awareness_Of_Mental_Events_Practice",
        "Worry_Postponement_Scheduling_Practice",
        "Attention_Training_Technique_Practice",
    ]
    add_leaves(change_process, ["Meaning_Update", "Metacognitive_and_Attentional_Meaning_Updating"],
               metacognitive_update)

    narrative_identity_update = [
        "Identify_Dominant_Problem_Story_Practice",
        "Thicken_Preferred_Story_With_Evidence_Practice",
        "Values_And_Agency_In_Story_Practice",
        "Identity_Based_Goal_Linking_Practice",
        "Write_New_Self_Narrative_Summary",
        "Future_Self_Letter_Practice",
        "Meaning_Making_From_Adversity_Practice",
        "Life_Timeline_Integration_Practice",
        "Identity_Evidence_Compilation_Practice",
        "Integrate_Contradictory_Self_Aspects_Practice",
    ]
    add_leaves(change_process, ["Meaning_Update", "Narrative_and_Identity_Updating"], narrative_identity_update)

    values_existential = [
        "Values_Clarification_Deepening_Practice",
        "Values_Conflict_Clarification_Practice",
        "Commitment_To_Chosen_Value_Path_Practice",
        "Moral_Dilemma_Processing_Practice",
        "Guilt_Repair_And_Restitution_Planning",
        "Self_Respect_Decision_Practice",
        "Boundary_Justification_By_Values_Practice",
        "Purpose_And_Meaning_Integration_Practice",
        "Role_Telos_Clarification_Practice",
        "Service_And_Contribution_Plan",
    ]
    add_leaves(change_process, ["Meaning_Update", "Values_Moral_and_Existential_Integration"], values_existential)

    relational_meaning_update = [
        "Relational_Pattern_Reframe_Practice",
        "Repair_As_Safety_Learning_Integration",
        "Ask_For_Needs_With_Self_Respect_Practice",
        "Co_Regulation_Request_Script_Practice",
        "Secure_Base_Building_Practice",
        "Relational_Safety_Cues_Identification_Practice",
        "Trust_As_Predictable_Repair_Practice",
        "Shame_And_Connection_Dynamics_Update_Practice",
        "Attachment_Narrative_Update_Practice",
        "Reduce_Mindreading_And_Assumption_Check_Practice",
    ]
    add_leaves(change_process, ["Meaning_Update", "Relational_Meaning_and_Attachment_Updating"],
               relational_meaning_update)

    # ------------------------------------------------------------
    # Phase 5: Behavioral Integration
    # ------------------------------------------------------------
    implementation_routines = [
        "Daily_Routine_Embedding_Plan",
        "Behavioral_Activation_Routine_Embedding",
        "Time_Blocking_For_Valued_Actions",
        "Minimum_Viable_Action_Dose_Definition",
        "Practice_Anchors_And_Cues_Setup",
        "Accountability_Checkin_Routine",
        "Practice_Tracking_And_Adjustment_Routine",
        "Consistency_Over_Intensity_Reframe_Practice",
        "Behavioral_Repeatability_Design_Practice",
        "Friction_And_Fatigue_Proofing_Plan",
    ]
    add_leaves(change_process, ["Behavioral_Integration", "Implementation_and_Routine_Building"],
               implementation_routines)

    environment_social_design = [
        "Environmental_Restructuring_For_Action",
        "Reduce_Friction_For_Target_Behavior",
        "Increase_Friction_For_Undesired_Behavior",
        "Remove_Temptations_Precommitment_Plan",
        "Social_Support_Mobilization_Plan",
        "Boundary_With_External_Stressors_Plan",
        "Communication_Pattern_Change_Practice",
        "Support_Seeking_And_Receiving_Plan",
        "Identity_Supporting_Environment_Redesign",
        "Commitment_Device_Setup_When_Appropriate",
    ]
    add_leaves(change_process, ["Behavioral_Integration", "Environmental_and_Social_System_Design"],
               environment_social_design)

    generalization_context = [
        "Multiple_Context_Practice_Schedule",
        "Trigger_Gradient_Practice_Plan",
        "State_Dependent_Practice_Planning",
        "Work_Home_Community_Transfer_Practice",
        "Relapse_Context_Rehearsal_Practice",
        "Novel_Context_Proof_Test_Practice",
        "Skill_Generalization_Plan",
        "Generalization_Review_And_Adjust",
        "High_Stress_Context_Training_Assignment",
        "Social_Context_Variation_Practice_Assignment",
    ]
    add_leaves(change_process, ["Behavioral_Integration", "Generalization_and_Context_Variation"],
               generalization_context)

    habit_automation = [
        "Habit_Stacking_And_Anchoring_Practice",
        "Cue_Response_Reward_Loop_Redesign",
        "Reward_Signal_Enhancement_Practice",
        "Identity_Evidence_Collection_Practice",
        "Implementation_Intention_Refresh_Routine",
        "Habit_Drift_Detection_And_Reset",
        "Maintenance_Behavior_Minimum_Dose_Practice",
        "Lapse_Interrupt_And_Return_Routine",
        "Context_Stability_For_Automaticity_Plan",
        "Replacement_Habit_Substitution_Plan",
    ]
    add_leaves(change_process, ["Behavioral_Integration", "Habit_Formation_and_Automation"], habit_automation)

    flexible_skill_use = [
        "Choice_Point_Identification_Practice",
        "Skill_Matching_To_Function_Practice",
        "Acceptance_vs_Change_Selection_Practice",
        "Cognition_vs_Action_Selection_Practice",
        "Self_Soothe_vs_Problem_Solve_Selection_Practice",
        "Stop_Rules_For_Maladaptive_Coping_Practice",
        "Early_Warning_Signs_To_Action_Routine",
        "Rapid_Recovery_Scripts_Practice",
        "Coping_Repertoire_Expansion_Practice",
        "Skills_Under_Time_Pressure_Rehearsal",
    ]
    add_leaves(change_process, ["Behavioral_Integration", "Flexible_Skill_Selection_in_Context"], flexible_skill_use)

    wellbeing_integration = [
        "Sleep_Protection_And_Recovery_Routine",
        "Stress_Load_Boundary_Setting_Practice",
        "Meaning_And_Purpose_Time_Allocation_Practice",
        "Positive_Emotion_And_Savoring_Routine",
        "Strengths_In_New_Context_Experiment",
        "Flow_Seeking_Activity_Design",
        "Social_Connection_Investment_Plan",
        "Self_Compassion_After_Failure_Practice",
        "Digital_Hygiene_And_Attention_Protection_Plan",
        "Weekly_Wellbeing_Review_And_Tuning_Routine",
    ]
    add_leaves(change_process, ["Behavioral_Integration", "Lifestyle_Performance_and_Wellbeing_Integration"],
               wellbeing_integration)

    # ------------------------------------------------------------
    # Phase 6: Identity Consolidation
    # ------------------------------------------------------------
    identity_story_stabilization = [
        "New_Identity_Statement_Creation",
        "Identity_Evidence_Log_Maintenance",
        "Integrate_Old_And_New_Self_Narrative",
        "Reduce_Shame_Based_Identity_Practice",
        "Agency_And_Competence_Identity_Reinforcement",
        "Values_To_Identity_Linking_Practice",
        "Meaning_Making_From_Change_Effort_Practice",
        "Self_Coaching_Voice_Installation_Practice",
        "Internal_Standards_And_Self_Respect_Clarification",
        "Identity_Consistent_Daily_Commitment_Practice",
    ]
    add_leaves(change_process, ["Identity_Consolidation", "Identity_Evidence_and_Story_Stabilization"],
               identity_story_stabilization)

    roles_belonging = [
        "Role_Transition_Support_Plan",
        "Boundary_As_Identity_Expression_Practice",
        "Repair_And_Trust_Repetition_Practice",
        "Attachment_Needs_Expression_Practice",
        "Care_Seeking_Care_Giving_Balance_Practice",
        "Community_Belonging_Investment_Plan",
        "Prosocial_Contribution_Planning",
        "Relational_Safety_Stabilization_Practice",
        "Social_Role_Selection_And_Prioritization_Practice",
        "Belonging_And_Values_Aligned_Community_Search_Plan",
    ]
    add_leaves(change_process, ["Identity_Consolidation", "Roles_Relationships_and_Belonging"], roles_belonging)

    parts_self_leadership = [
        "Parts_Mapping_And_Relationship_Practice",
        "Self_Leadership_Practice",
        "Protector_Role_Renegotiation_Practice",
        "Vulnerable_Part_Compassion_Practice",
        "Inner_Critic_Transformation_Practice",
        "Unblending_Skills_Consolidation_Practice",
        "Internal_Cooperation_Agreement_Practice",
        "Self_Respect_And_Dignity_Boundaries_Practice",
        "Compassionate_Internal_Dialogue_Practice",
        "Internal_Conflict_Mediation_Practice",
    ]
    add_leaves(change_process, ["Identity_Consolidation", "Parts_Integration_and_Self_Leadership"],
               parts_self_leadership)

    values_integrity = [
        "Values_Consistency_Check_Practice",
        "Commitment_Renewal_Practice",
        "Moral_Repair_Action_Plan",
        "Guilt_Repair_Planning_Practice",
        "Forgiveness_With_Boundaries_Practice",
        "Integrity_Under_Pressure_Rehearsal",
        "Dignity_Preserving_No_Practice",
        "Service_And_Contribution_As_Meaning_Practice",
        "Self_Respect_Based_Decision_Rule_Practice",
        "Avoid_Moral_Rigidity_With_Compassionate_Flexibility_Practice",
    ]
    add_leaves(change_process, ["Identity_Consolidation", "Values_Integrity_and_Moral_Repair"], values_integrity)

    autonomy_growth = [
        "Independent_Problem_Solving_Routine",
        "Personal_Review_Rituals_Setup",
        "Flexible_Goal_Resetting_Practice",
        "Self_Efficacy_Mastery_Recall_Practice",
        "Future_Self_Alignment_Practice",
        "Practice_As_Lifestyle_Routine",
        "Therapy_Toolbox_Personal_Manual_Finalize",
        "Self_Maintenance_Identity_Commitment_Practice",
        "Self_Directed_Learning_And_Experimentation_Routine",
        "Personal_Operating_System_Design_Practice",
    ]
    add_leaves(change_process, ["Identity_Consolidation", "Autonomy_and_Self_Directed_Growth"], autonomy_growth)

    # ------------------------------------------------------------
    # Phase 7: Maintenance and Transfer
    # ------------------------------------------------------------
    relapse_prevention = [
        "Relapse_Prevention_Framework_Implementation",
        "High_Risk_Situation_PrePlan_Practice",
        "Early_Warning_Signs_Action_Playbook",
        "Lapse_Response_Plan_NoShame_Practice",
        "Rapid_Return_To_Routine_Protocol",
        "Seemingly_Irrelevant_Decisions_Audit_Practice",
        "Lifestyle_Balance_And_Buffer_Building",
        "Booster_Session_Target_List_Creation",
        "Relapse_Drill_Rehearsal_Practice",
        "Recovery_Self_Efficacy_After_Setback_Practice",
    ]
    add_leaves(change_process, ["Maintenance_and_Transfer", "Relapse_Prevention_and_Stability"], relapse_prevention)

    long_term_upkeep = [
        "Maintenance_Minimum_Viable_Practices_Schedule",
        "Seasonal_And_Contextual_Adjustment_Routine",
        "Periodic_Values_Checkpoint_Routine",
        "Environmental_Friction_Audit_Refresh",
        "Support_Network_Maintenance_Routine",
        "Skill_Rotation_And_Refresh_Plan",
        "Personal_Progress_Dashboard_Review",
        "Longitudinal_Self_Monitoring_Plan",
        "Quarterly_System_Review_And_Refactor",
        "Annual_Identity_And_Purpose_Review_Routine",
    ]
    add_leaves(change_process, ["Maintenance_and_Transfer", "Long_Term_Transfer_and_Upkeep"], long_term_upkeep)

    setback_recovery = [
        "Setback_Normalization_And_Learning_Frame_Practice",
        "Post_Setback_Review_Protocol",
        "Update_Formulation_After_Setback_Practice",
        "Rebuild_Momentum_MicroSteps_Protocol",
        "Self_Compassion_After_Setback_Practice",
        "Return_To_Phase_Decision_Rule_Use",
        "Stress_Test_Rehearsal_And_Contingency_Update",
        "Optionality_And_Recovery_Buffer_Building",
        "Rupture_Repair_After_Setback_Protocol",
        "Re_Engagement_After_Dropout_Return_Protocol",
    ]
    add_leaves(change_process, ["Maintenance_and_Transfer", "Setback_Recovery_and_Resilience_Expansion"],
               setback_recovery)

    termination_continuity = [
        "Consolidate_Gains_And_Lessons_Practice",
        "Therapy_Toolbox_Consolidation_Final",
        "Future_Risk_Scenarios_Rehearsal_Practice",
        "Booster_Session_Plan_Setup",
        "Followup_Checkpoint_Schedule_Setup",
        "Reentry_Plan_If_Symptoms_Return",
        "Referral_Back_Paths_If_Needed_Setup",
        "Relational_Closure_And_Goodbye_Practice",
        "Self_Reliance_And_Support_Balance_Plan",
        "Transition_To_Self_Guided_Maintenance_Plan",
    ]
    add_leaves(change_process, ["Maintenance_and_Transfer", "Termination_and_Continuity_Planning"],
               termination_continuity)

    flourishing_maintenance = [
        "Strengths_Use_Maintenance_Routine",
        "Flow_And_Engagement_Maintenance_Routine",
        "Meaning_And_Purpose_Maintenance_Routine",
        "Connection_And_Community_Maintenance_Routine",
        "Gratitude_And_Savoring_Maintenance_Routine",
        "Growth_Goals_Quarterly_Reset_Practice",
        "Resilience_Lesson_Learning_Review_Routine",
        "Prosocial_Acts_Planning_Routine",
        "Joy_And_Play_Reinstatement_Practice",
        "Long_Term_Contribution_Project_Planning",
    ]
    add_leaves(change_process, ["Maintenance_and_Transfer", "Flourishing_and_Nonclinical_Optimization_Maintenance"],
               flourishing_maintenance)

    # ------------------------------------------------------------
    # Secondary Node: Behavioral Change Theories and Modern Protocol Arcs
    # (encoded as actionable, phase-usable solution entities)
    # ------------------------------------------------------------
    hapa_solutions = [
        "HAPA_Risk_Perception_Clarification_Interview",
        "HAPA_Outcome_Expectancy_Elicitation",
        "HAPA_Task_Self_Efficacy_Building_Plan",
        "HAPA_Intention_Formation_Commitment_Step",
        "HAPA_Action_Planning_Worksheet",
        "HAPA_Coping_Planning_Worksheet",
        "HAPA_Action_Control_Self_Monitoring_Routine",
        "HAPA_Maintenance_Self_Efficacy_Reinforcement",
        "HAPA_Recovery_Self_Efficacy_After_Lapse_Practice",
        "HAPA_Plan_Revision_After_Context_Shift_Practice",
    ]
    add_leaves(change_process,
               ["Behavior_Change_Theories_and_Modern_Protocol_Arcs", "HAPA_Health_Action_Process_Approach_Solutions"],
               hapa_solutions)

    ttm_solutions = [
        "TTM_Stage_Assessment_Routine",
        "TTM_Precontemplation_Engagement_Conversation",
        "TTM_Contemplation_Ambivalence_Work",
        "TTM_Preparation_Action_Plan_Setup",
        "TTM_Action_Support_And_Adherence_Plan",
        "TTM_Maintenance_Relapse_Prevention_Plan",
        "TTM_Process_Matching_Intervention_Selection",
        "TTM_Decisional_Balance_Update_Routine",
        "TTM_Self_Liberation_Commitment_Practice",
        "TTM_Stimulus_Control_And_Reinforcement_Plan",
    ]
    add_leaves(change_process, ["Behavior_Change_Theories_and_Modern_Protocol_Arcs", "TTM_Stages_of_Change_Solutions"],
               ttm_solutions)

    comb_bcw_solutions = [
        "COMB_Capability_Barrier_Assessment",
        "COMB_Opportunity_Barrier_Assessment",
        "COMB_Motivation_Barrier_Assessment",
        "BCW_Intervention_Function_Selection_Routine",
        "BCW_Environmental_Restructuring_Plan",
        "BCW_Training_And_Skills_Build_Plan",
        "BCW_Enablement_And_Support_Plan",
        "BCW_Incentive_And_Reinforcement_Plan",
        "BCW_Modeling_And_Social_Proof_Plan",
        "BCW_Prompting_And_Cueing_System_Setup",
    ]
    add_leaves(change_process,
               ["Behavior_Change_Theories_and_Modern_Protocol_Arcs", "COMB_and_Behavior_Change_Wheel_Solutions"],
               comb_bcw_solutions)

    sdt_solutions = [
        "SDT_Autonomy_Support_Language_Practice",
        "SDT_Choice_And_Ownership_Design",
        "SDT_Competence_Mastery_Scaffolding",
        "SDT_Relatedness_And_Belonging_Build",
        "SDT_Internalization_Link_Goals_To_Values",
        "SDT_Motivation_Quality_Checkin",
        "SDT_Avoid_Control_Language_Replacement_Practice",
        "SDT_Self_Concordant_Goal_Selection_Practice",
    ]
    add_leaves(change_process,
               ["Behavior_Change_Theories_and_Modern_Protocol_Arcs", "Self_Determination_Theory_Solutions"],
               sdt_solutions)

    sct_solutions = [
        "SCT_Self_Efficacy_Mastery_Recall_Practice",
        "SCT_Modeling_And_Observational_Learning_Plan",
        "SCT_Barrier_Solving_And_Skills_Build",
        "SCT_Self_Monitoring_Goals_And_Feedback_Loop",
        "SCT_Outcome_Expectancy_Reality_Testing",
        "SCT_Incremental_Mastery_Task_Ladder",
        "SCT_Social_Support_As_Opportunity_Design",
    ]
    add_leaves(change_process,
               ["Behavior_Change_Theories_and_Modern_Protocol_Arcs", "Social_Cognitive_Theory_Solutions"],
               sct_solutions)

    tpb_solutions = [
        "TPB_Attitude_Clarification_Conversation",
        "TPB_Norms_And_Support_Map_Practice",
        "TPB_Perceived_Control_Barrier_Solving",
        "TPB_Intention_Strengthening_Commitment_Step",
        "TPB_Intention_Behavior_Gap_Closure_Plan",
        "TPB_Belief_Elicitation_And_Targeting_Routine",
    ]
    add_leaves(change_process,
               ["Behavior_Change_Theories_and_Modern_Protocol_Arcs", "Theory_of_Planned_Behavior_Solutions"],
               tpb_solutions)

    pmt_solutions = [
        "PMT_Threat_Appraisal_Clarification",
        "PMT_Response_Efficacy_And_Feasibility_Check",
        "PMT_Self_Efficacy_Build_Plan",
        "PMT_Response_Cost_Reduction_Plan",
        "PMT_Maladaptive_Reward_Audit_And_Substitution",
        "PMT_Fear_To_Action_Channeling_Plan",
    ]
    add_leaves(change_process,
               ["Behavior_Change_Theories_and_Modern_Protocol_Arcs", "Protection_Motivation_Theory_Solutions"],
               pmt_solutions)

    control_theory_solutions = [
        "Control_Theory_Goal_Comparator_Setup",
        "Control_Theory_Feedback_Loop_Dashboard_Setup",
        "Control_Theory_Discrepancy_Reduction_Action_Routine",
        "Control_Theory_Reference_Value_Adjustment_Routine",
        "Implementation_Intentions_IfThen_Design_Routine",
        "Self_Regulation_Friction_Audit_And_Adjustment",
    ]
    add_leaves(change_process,
               ["Behavior_Change_Theories_and_Modern_Protocol_Arcs", "Control_Theory_and_Self_Regulation_Solutions"],
               control_theory_solutions)

    habit_dual_process_solutions = [
        "Habit_Cue_Identification_And_Stabilization",
        "Habit_Reward_Reengineering_Practice",
        "Habit_Context_Redesign_For_Automaticity",
        "Dual_Process_Reflective_Override_Practice",
        "Dual_Process_Automatic_Trigger_Disruption_Plan",
        "Identity_Cue_Alignment_Practice",
    ]
    add_leaves(change_process,
               ["Behavior_Change_Theories_and_Modern_Protocol_Arcs", "Habit_and_Dual_Process_Solutions"],
               habit_dual_process_solutions)

    relapse_theory_solutions = [
        "Relapse_High_Risk_Situation_Mapping",
        "Relapse_Coping_Response_Rehearsal",
        "Relapse_Self_Efficacy_Reinforcement_Practice",
        "Relapse_Lapse_Management_AV_Effect_Reframe",
        "Relapse_Lifestyle_Balance_And_Buffer_Plan",
        "Relapse_Seemingly_Irrelevant_Decisions_Review",
    ]
    add_leaves(change_process,
               ["Behavior_Change_Theories_and_Modern_Protocol_Arcs", "Relapse_Prevention_Theory_Solutions"],
               relapse_theory_solutions)

    modern_protocol_arcs = [
        "MI_Engaging_Process_Practice",
        "MI_Focusing_Process_Practice",
        "MI_Evoking_Process_Practice",
        "MI_Planning_Process_Practice",

        "Unified_Protocol_Module_Motivation_Goals",
        "Unified_Protocol_Module_Emotion_Awareness",
        "Unified_Protocol_Module_Cognitive_Flexibility",
        "Unified_Protocol_Module_Reduce_Avoidance_EDBs",
        "Unified_Protocol_Module_Emotion_Exposure",
        "Unified_Protocol_Module_Relapse_Prevention",

        "ACT_Process_Creative_Hopelessness_Work",
        "ACT_Process_Values_Clarification_Work",
        "ACT_Process_Acceptance_And_Defusion_Work",
        "ACT_Process_Self_As_Context_Work",
        "ACT_Process_Committed_Action_Work",
        "ACT_Process_Flexibility_Maintenance_Work",

        "DBT_Stage1_Safety_And_Behavioral_Stabilization",
        "DBT_Stage2_Trauma_And_Emotional_Experiencing",
        "DBT_Stage3_Goals_And_Self_Respect_Build",
        "DBT_Stage4_Freedom_And_Joy_Expansion",

        "CBT_Generic_Assessment_And_Formulation_Arc",
        "CBT_Generic_Skills_And_Behavior_Change_Arc",
        "CBT_Generic_Cognitive_Change_And_Experiments_Arc",
        "CBT_Generic_Exposure_When_Indicated_Arc",
        "CBT_Generic_Relapse_Prevention_Arc",

        "CBTI_Stimulus_Control_Implementation",
        "CBTI_Sleep_Restriction_Implementation",
        "CBTI_Sleep_Cognition_Update_Implementation",
        "CBTI_Sleep_Maintenance_Plan_Implementation",

        "EMDR_History_Taking_And_Planning_Implementation",
        "EMDR_Preparation_And_Resourcing_Implementation",
        "EMDR_Desensitization_And_Installation_Implementation",
        "EMDR_Closure_And_Reevaluation_Implementation",

        "PE_InVivo_Exposure_Implementation",
        "PE_Imaginal_Exposure_Implementation",
        "PE_Processing_And_Generalization_Implementation",

        "CPT_Stuck_Point_Work_Implementation",
        "CPT_Theme_Work_Safety_Trust_Power_Esteem_Intimacy",
        "CPT_Integration_And_Future_Planning_Implementation",

        "RODBT_Radical_Openness_Implementation",
        "RODBT_Social_Signaling_And_Connection_Implementation",
        "RODBT_Novelty_And_Flexibility_Implementation",

        "MBCT_Autopilot_And_Attention_Training_Implementation",
        "MBCT_Relating_Differently_To_Thoughts_Implementation",
        "MBCT_Turning_Toward_Difficulty_Implementation",
        "MBCT_Maintenance_Practice_Plan_Implementation",

        "SFBT_Preferred_Future_Construction_Implementation",
        "SFBT_Exception_Finding_And_Amplification_Implementation",
        "SFBT_Scaling_And_Microsteps_Implementation",
        "SFBT_Consolidation_And_Self_Sufficiency_Implementation",

        # Added: commonly used additional arcs (as actionable protocol-stage entities)
        "IPT_Interpersonal_Inventory_And_Target_Area_Selection",
        "IPT_Role_Transition_Work_Implementation",
        "IPT_Interpersonal_Dispute_Work_Implementation",
        "Schema_Therapy_Mode_Map_And_Mode_Work_Implementation",
        "CFT_Compassionate_Mind_Training_Implementation",
        "Metacognitive_Therapy_Detached_Mindfulness_And_Worry_Control_Implementation",
        "Problem_Solving_Therapy_Seven_Step_Implementation",
    ]
    add_leaves(change_process,
               ["Behavior_Change_Theories_and_Modern_Protocol_Arcs", "Modern_Clinical_Protocol_Arcs_Solutions"],
               modern_protocol_arcs)

    # Attach as a primary PSYCHO node
    PSYCHO["Change_Process_and_Therapeutic_Phases"] = change_process

    # ============================================================
    # Cognitive Capacity Enhancement & Stimulation (NEW DIMENSION)
    # (mechanism-level, schedule-free, psycho-only; not BIO/SOCIAL)
    # Leaf nodes = actionable cognitive training / strategy / task formats
    # ============================================================

    cognitive_capacity_enhancement = OntologyBuilder().root

    # ------------------------------------------------------------
    # 0) Setup, Personalization, Transfer (cognitive-only)
    # ------------------------------------------------------------
    add_leaves(
        cognitive_capacity_enhancement,
        ["Program_Setup_and_Transfer", "Assessment_Goals_and_Task_Selection", "Personalization"],
        [
            "Cognitive_Strengths_Weaknesses_Profile_Mapping",
            "Task_Difficulty_Calibration_Framework",
            "Cognitive_Domain_Target_Selection",
            "Error_Pattern_Analysis_For_Targeting",
            "Strategy_Selection_Based_On_Error_Signature",
            "Transfer_Goal_Definition_RealWorld_Function",
            "Generalization_Plan_Context_Variation",
            "Cognitive_Task_Self_Efficacy_Building",
            "Frustration_Tolerance_For_Cognitive_Training",
            "Motivation_Priming_For_Practice",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Program_Setup_and_Transfer", "Metacognitive_Scaffolding", "Monitor_and_Adjust"],
        [
            "Metacognitive_Monitoring_Checkpoints",
            "Strategy_Shift_When_Stuck_Routine",
            "Error_Awareness_Then_Correction_Routine",
            "Speed_Accuracy_Tradeoff_Awareness_Training",
            "Reflect_On_What_Worked_Short_Debrief",
            "Transfer_Bridge_Explicit_Link_To_Daily_Tasks",
            "Cognitive_Training_Log_Qualitative",
        ],
    )

    # ------------------------------------------------------------
    # 1) Attention Systems Training
    # ------------------------------------------------------------
    add_leaves(
        cognitive_capacity_enhancement,
        ["Attention_Systems_Training", "Sustained_Attention", "Vigilance_and_Consistency"],
        [
            "Sustained_Attention_Practice",
            "Vigilance_To_Targets_Practice",
            "Mind_Wandering_Detection_Practice",
            "Return_To_Task_Cueing_Practice",
            "Consistency_Monitoring_Practice",
            "Task_Goal_Reminder_Self_Cueing",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Attention_Systems_Training", "Selective_Attention", "Filtering_and_Focus"],
        [
            "Selective_Attention_Practice",
            "Distractor_Filtering_Practice",
            "Salience_Control_Practice",
            "Goal_Relevant_Feature_Selection_Practice",
            "Noise_Resistance_Focus_Practice",
            "Ignore_Irrelevant_Stimuli_Drills",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Attention_Systems_Training", "Divided_and_Shared_Attention", "Dual_Tasking"],
        [
            "Divided_Attention_Practice",
            "Dual_Task_Coordination_Practice",
            "Prioritize_Primary_Task_Routine",
            "Switch_Attention_Between_Streams_Practice",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Attention_Systems_Training", "Orienting_and_Shifting", "Flexibly_Move_Focus"],
        [
            "Attention_Shifting_Practice",
            "Orienting_To_Relevant_Cues_Practice",
            "Disengage_From_Sticky_Cues_Practice",
            "Set_Shifting_From_Internal_To_External_Attention",
            "Attentional_Refocusing_Routine",
        ],
    )

    # ------------------------------------------------------------
    # 2) Working Memory Capacity & Updating
    # ------------------------------------------------------------
    add_leaves(
        cognitive_capacity_enhancement,
        ["Working_Memory_and_Updating", "Maintenance", "Hold_Information_Online"],
        [
            "Working_Memory_Maintenance_Practice",
            "Verbal_Working_Memory_Rehearsal_Strategy",
            "Visuospatial_Working_Memory_Maintenance_Practice",
            "Chunking_Strategy",
            "Grouping_By_Meaning_Strategy",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Working_Memory_and_Updating", "Updating", "Replace_And_Refresh"],
        [
            "Working_Memory_Updating_Practice",
            "NBack_Style_Updating_Framework",
            "Refresh_And_Replace_Practice",
            "Keep_Relevant_Drop_Irrelevant_Practice",
            "Interference_Resistance_In_WM_Practice",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Working_Memory_and_Updating", "Manipulation", "Operate_On_Contents"],
        [
            "Mental_Reordering_Practice",
            "Mental_Arithmetic_Strategy_Practice",
            "Backward_Sequencing_Practice",
            "Mental_Rotation_With_Hold_Practice",
            "Rule_Application_With_WM_Load_Practice",
        ],
    )

    # ------------------------------------------------------------
    # 3) Executive Functions (Inhibition, Switching, Planning)
    # ------------------------------------------------------------
    add_leaves(
        cognitive_capacity_enhancement,
        ["Executive_Function_Training", "Inhibitory_Control", "Stop_Suppress_Resist"],
        [
            "Response_Inhibition_Practice",
            "Interference_Inhibition_Practice",
            "Prepotent_Response_Suppression_Drills",
            "GoNoGo_Style_Control_Framework",
            "Stop_Signal_Style_Control_Framework",
            "Cognitive_Impulse_Control_Self_Talk",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Executive_Function_Training", "Cognitive_Flexibility", "Switch_Set_Shift_Reframe"],
        [
            "Task_Switching_Practice",
            "Rule_Switching_Practice",
            "Category_Switching_Practice",
            "Alternative_Rule_Generation_Practice",
            "Flexible_Problem_Set_Reframing_Practice",
            "Cognitive_Flexibility_Practice",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Executive_Function_Training", "Planning_and_Sequencing", "Organize_Steps"],
        [
            "Executive_Planning_Skills_Practice",
            "Goal_Decomposition_Practice",
            "Sequence_Building_Practice",
            "Plan_then_Simulate_then_Execute_Routine",
            "Contingency_Thinking_Practice",
            "IfThen_Planning_For_Cognitive_Tasks",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Executive_Function_Training", "Error_Monitoring", "Detect_Correct_Learn"],
        [
            "Error_Awareness_Training",
            "Conflict_Monitoring_Practice",
            "Post_Error_Adjustment_Practice",
            "Slow_Down_Check_Routine",
            "Accuracy_Checklist_Use_Practice",
        ],
    )

    # ------------------------------------------------------------
    # 4) Processing Speed & Cognitive Efficiency
    # ------------------------------------------------------------
    add_leaves(
        cognitive_capacity_enhancement,
        ["Processing_Speed_and_Efficiency", "Speeded_Response", "Faster_Accurate_Output"],
        [
            "Processing_Speed_Strategy_Training",
            "Rapid_Classification_Practice",
            "Speeded_Decision_Practice",
            "Automatization_Of_Basic_Skills_Practice",
            "Fluency_Building_Practice",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Processing_Speed_and_Efficiency", "Efficiency_Strategies", "Reduce_Cognitive_Load"],
        [
            "Cognitive_Load_Management",
            "Simplify_Steps_Strategy",
            "Template_Use_For_Repeated_Tasks",
            "Precommit_To_Decision_Rules_Practice",
            "Batching_And_Grouping_Workflow_Practice",
        ],
    )

    # ------------------------------------------------------------
    # 5) Learning & Memory Skill Training (strategy-based stimulation)
    # ------------------------------------------------------------
    add_leaves(
        cognitive_capacity_enhancement,
        ["Learning_and_Memory_Strategies", "Encoding_Strategies", "Make_It_Stick"],
        [
            "Elaboration_Strategy",
            "Self_Explanation_Practice",
            "Dual_Coding_Strategy",
            "Keyword_Method_Strategy",
            "Method_Of_Loci_Strategy",
            "Story_Link_Method_Strategy",
            "Semantic_Clustering_Strategy",
            "Generate_Examples_Strategy",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Learning_and_Memory_Strategies", "Retrieval_Practice", "Strengthen_Recall"],
        [
            "Retrieval_Practice_Framework",
            "Free_Recall_Practice",
            "Cued_Recall_Practice",
            "Recognition_vs_Recall_Discrimination",
            "Errorful_Learning_With_Correction_Framework",
            "Feedback_Integrated_Retrieval_Practice",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Learning_and_Memory_Strategies", "Prospective_Memory", "Remember_To_Do"],
        [
            "Prospective_Memory_Aids_Training",
            "Implementation_Intention_For_Prospective_Memory",
            "Event_Based_Prospective_Memory_Strategy",
            "Time_Based_Prospective_Memory_Strategy_ScheduleFree",
            "Environmental_Cue_Placement_Strategy",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Learning_and_Memory_Strategies", "Metamemory", "Confidence_and_Calibration"],
        [
            "Judgment_Of_Learning_Calibration_Practice",
            "Feeling_Of_Knowing_Calibration_Practice",
            "Overconfidence_Check_Practice",
            "Study_Strategy_Adjustment_From_Test_Results",
        ],
    )

    # ------------------------------------------------------------
    # 6) Reasoning, Problem Solving, and Decision Skill Stimulation
    # ------------------------------------------------------------
    add_leaves(
        cognitive_capacity_enhancement,
        ["Reasoning_and_Problem_Solving_Stimulation", "Logical_and_Analytical_Reasoning", "Rule_Based_Thinking"],
        [
            "Deductive_Reasoning_Practice",
            "Inductive_Reasoning_Practice",
            "Hypothesis_Testing_Practice",
            "Base_Rate_Use_Practice",
            "Causal_Reasoning_Practice",
            "Argument_Evaluation_Practice",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Reasoning_and_Problem_Solving_Stimulation", "Cognitive_Bias_Resistance", "Debiasing_Drills"],
        [
            "Consider_The_Opposite_Practice",
            "Alternative_Hypotheses_Drill",
            "Evidence_Weighting_Practice",
            "Jumping_To_Conclusions_Check",
            "Confirmation_Bias_Check_Practice",
            "Hindsight_Bias_Check_Practice",
            "Availability_Bias_Check_Practice",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Reasoning_and_Problem_Solving_Stimulation", "Creative_Cognition", "Divergent_and_Flexible_Thinking"],
        [
            "Divergent_Thinking_Prompting",
            "Constraint_Removal_Ideation_Practice",
            "Perspective_Shift_Creative_Practice",
            "Analogical_Thinking_Practice",
            "Remote_Association_Practice",
            "Recombination_Ideation_Practice",
        ],
    )

    # ------------------------------------------------------------
    # 7) Language, Fluency, and Communication Cognition (psycho-only)
    # ------------------------------------------------------------
    add_leaves(
        cognitive_capacity_enhancement,
        ["Language_and_Verbal_Cognition", "Lexical_and_Semantic_Fluency", "Access_And_Retrieve_Words"],
        [
            "Verbal_Fluency_Semantic_Category_Practice",
            "Verbal_Fluency_Letter_Cue_Practice",
            "Word_Retrieval_Strategy_Practice",
            "TipOfTongue_Resolution_Strategy",
            "Semantic_Network_Expansion_Practice",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Language_and_Verbal_Cognition", "Comprehension_and_Expression", "Structure_And_Clarity"],
        [
            "Summarization_Practice",
            "Paraphrasing_Practice",
            "Main_Idea_Extraction_Practice",
            "Argument_Structure_Building_Practice",
            "Narrative_Coherence_Practice",
        ],
    )

    # ------------------------------------------------------------
    # 8) Visuospatial Skills and Mental Imagery Training
    # ------------------------------------------------------------
    add_leaves(
        cognitive_capacity_enhancement,
        ["Visuospatial_and_Imagery_Cognition", "Spatial_Manipulation", "Rotate_Transform_Track"],
        [
            "Mental_Rotation_Practice",
            "Spatial_Working_Memory_Practice",
            "Pattern_Completion_Practice",
            "Visual_Search_Strategy_Practice",
            "Figure_Ground_Discrimination_Practice",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Visuospatial_and_Imagery_Cognition", "Mental_Imagery_Control", "Generate_Maintain_Modify"],
        [
            "Imagery_Vividness_Training",
            "Imagery_Stability_Training",
            "Imagery_Manipulation_Practice",
            "Perspective_Shift_In_Imagery_Practice",
        ],
    )

    # ------------------------------------------------------------
    # 9) Social-Cognitive Processes (kept cognitive; not “social intervention”)
    # ------------------------------------------------------------
    add_leaves(
        cognitive_capacity_enhancement,
        ["Social_Cognitive_Processing", "Perspective_Taking", "Mental_State_Inference"],
        [
            "Perspective_Taking_Practice",
            "Multiple_Viewpoints_Generation_Practice",
            "Intent_Inference_Practice",
            "Ambiguous_Social_Cue_Interpretation_Flexibility",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Social_Cognitive_Processing", "Emotion_Recognition", "Perception_and_Labeling"],
        [
            "Facial_Affect_Recognition_Practice",
            "Prosody_Emotion_Recognition_Practice",
            "Contextual_Emotion_Inference_Practice",
        ],
    )

    # ------------------------------------------------------------
    # 10) Cognitive Offloading & External Aids (psychological strategy use)
    # ------------------------------------------------------------
    add_leaves(
        cognitive_capacity_enhancement,
        ["Cognitive_Offloading_and_External_Aids", "External_Memory_Systems", "Design_And_Use"],
        [
            "External_Aids_Use_Training",
            "Capture_Then_Clarify_Then_Organize_Workflow",
            "Checklists_For_Error_Prevention",
            "Templates_For_Repeated_Cognitive_Tasks",
            "Prospective_Memory_Cue_System_Design",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Cognitive_Offloading_and_External_Aids", "Attention_Support_Systems", "Reduce_Distraction"],
        [
            "Environmental_Stimulus_Minimization_Strategy",
            "Single_Task_Context_Setup_Strategy",
            "Cue_Control_For_Attention_Strategy",
            "Interruptions_Handling_Routine",
        ],
    )

    # ------------------------------------------------------------
    # 11) Enrichment & Stimulation Formats (content-agnostic, schedule-free)
    # ------------------------------------------------------------
    add_leaves(
        cognitive_capacity_enhancement,
        ["Cognitive_Enrichment_and_Stimulation_Formats", "Task_Based_Training_Formats", "Structured_Practice"],
        [
            "Computerized_Cognitive_Training_Framework",
            "Paper_Pencil_Cognitive_Drills_Framework",
            "Gamified_Cognitive_Challenge_Framework",
            "Adaptive_Difficulty_Training_Framework",
            "Strategy_Coached_Training_Framework",
        ],
    )

    add_leaves(
        cognitive_capacity_enhancement,
        ["Cognitive_Enrichment_and_Stimulation_Formats", "Everyday_Cognition_Stimulation", "RealWorld_Activities"],
        [
            "Novelty_Seeking_Cognitive_Stimulation_Framework",
            "Complex_Skill_Learning_Framework_NonBio",
            "Cognitive_Challenge_Selection_Framework",
            "Cognitive_Variety_Planning_Framework",
            "Transfer_Practice_Into_Daily_Tasks_Framework",
        ],
    )

    # Attach to PSYCHO
    PSYCHO["Cognitive_Capacity_Enhancement_and_Stimulation"] = cognitive_capacity_enhancement

    # ============================================================
    # 11) Substance_Use_Recovery_and_Behavior_Change (PSYCHO)
    # Design notes:
    # - PSYCHO-only solution variables (skills, interventions, therapy components, delivery/process choices)
    # - No disorder-labeled branches
    # - No dosing/frequency/duration/intensity nodes
    # - Avoid BIO/SOCIAL predictors (no meds/labs; no housing/finances/policy)
    # ============================================================
    substance_use_recovery = OntologyBuilder().root

    # ------------------------------------------------------------
    # 11.1 Engagement, Motivation, and Readiness (MI-aligned)
    # ------------------------------------------------------------
    add_leaves(substance_use_recovery, ["Engagement_and_Motivation", "Ambivalence_and_Discrepancy"], [
        "Ambivalence_Exploration_Dialogue",
        "Values_Behavior_Discrepancy_Conversation_Substance",
        "Decisional_Balance_Exploration_Substance",
        "Explore_Pros_Cons_Of_Change_Substance",
        "Evoke_Personal_Reasons_For_Change",
        "Elicit_Change_Talk_Substance",
        "Amplify_Change_Talk_Bundles",
        "Soften_Sustain_Talk_With_Reflection",
        "Roll_With_Resistance_Substance",
        "Autonomy_Support_Language_Substance",
    ])

    add_leaves(substance_use_recovery, ["Engagement_and_Motivation", "Commitment_and_Confidence"], [
        "Confidence_Ruler_And_Barrier_Exploration_Substance",
        "Importance_Ruler_And_Values_Link_Substance",
        "Strengthen_Self_Efficacy_Substance",
        "Commitment_Language_Strengthening_Substance",
        "Change_Plan_Development_Substance",
        "Explore_Identity_Consistent_Commitment",
        "Reinforce_MicroCommitments_And_Next_Steps",
    ])

    # ------------------------------------------------------------
    # 11.2 Functional Assessment and Case Formulation (CBT / learning model)
    # ------------------------------------------------------------
    add_leaves(substance_use_recovery, ["Assessment_and_Formulation", "Functional_Analysis_of_Use"], [
        "Use_Episode_Chain_Analysis",
        "Trigger_Cue_Context_Mapping_Substance",
        "Antecedent_Behavior_Consequence_Map_Substance",
        "Reinforcement_Loop_Identification_Positive",
        "Reinforcement_Loop_Identification_Negative",
        "Emotion_To_Use_Link_Mapping",
        "Thought_To_Use_Link_Mapping",
        "Interpersonal_Trigger_Mapping_Substance",
        "High_Risk_Situation_Profile",
        "Protective_Factors_Map_Substance",
    ])

    add_leaves(substance_use_recovery, ["Assessment_and_Formulation", "Goal_Setting_and_Change_Targets"], [
        "Recovery_Goal_Selection_Framework",
        "Clarify_NonNegotiables_And_Boundaries_Substance",
        "Identify_Priority_Use_Contexts_To_Target",
        "Skills_Gap_Analysis_For_Recovery",
        "Select_Mechanism_Targets_For_Intervention_Substance",
        "Recovery_Rationale_And_Personal_Narrative_Clarity",
    ])

    # ------------------------------------------------------------
    # 11.3 Craving, Urges, and Impulse Control (psycho-only)
    # ------------------------------------------------------------
    add_leaves(substance_use_recovery, ["Craving_and_Urge_Management", "Mindfulness_and_Surfing"], [
        "Urge_Surfing_Practice_Substance",
        "Craving_As_Wave_Reframe",
        "Mindful_Noting_Of_Craving_Sensations",
        "Allowing_Urges_Without_Acting_Practice",
        "RAIN_Framework_For_Craving",
        "Decentering_From_Craving_Thoughts",
    ])

    add_leaves(substance_use_recovery, ["Craving_and_Urge_Management", "Cognitive_Skills"], [
        "Permission_Giving_Thoughts_Identification",
        "Cognitive_Reframe_Craving_Is_TimeLimited",
        "Future_Consequences_Visualization_Practice",
        "Values_Linked_Choice_Point_During_Craving",
        "Self_Talk_For_Delay_And_Choice",
        "Cognitive_Compassion_For_Urges_Without_Indulgence",
    ])

    add_leaves(substance_use_recovery, ["Craving_and_Urge_Management", "Behavioral_Skills"], [
        "Delay_Distract_Decide_Skill",
        "Competing_Response_Planning_Substance",
        "Exit_And_Reset_Strategy_For_HighRisk_Moments",
        "Alternative_Reward_Substitution_Planning",
        "Urge_Triggered_IfThen_Plan_Substance",
        "Remove_Self_From_Cue_Rich_Context_Skill",
    ])

    # ------------------------------------------------------------
    # 11.4 CBT-Based Recovery Skills (coping, problem solving, decisions)
    # ------------------------------------------------------------
    add_leaves(substance_use_recovery, ["CBT_Based_Recovery_Skills", "Coping_Skills_Training"], [
        "Coping_Skills_Menu_Construction_Substance",
        "Coping_Cards_For_HighRisk_Situations",
        "Coping_With_Boredom_Without_Use",
        "Coping_With_Stress_Without_Use",
        "Coping_With_Anger_Without_Use",
        "Coping_With_Shame_Without_Use",
        "Coping_With_Social_Pressure_Without_Use",
        "Coping_With_Celebration_Triggers_Without_Use",
    ])

    add_leaves(substance_use_recovery, ["CBT_Based_Recovery_Skills", "Problem_Solving_and_Decision_Skills"], [
        "Problem_Solving_For_Recovery_Barriers",
        "Decision_Matrix_For_Risk_Situations_Substance",
        "Precommitment_Decision_Rules_Substance",
        "Alternative_Paths_If_Urge_Strengthens",
        "Repair_After_Setback_Problem_Solve_Not_Punish",
        "HighRisk_Situation_Rehearsal_And_Simulation",
    ])

    add_leaves(substance_use_recovery, ["CBT_Based_Recovery_Skills", "Skills_Rehearsal_and_Generalization"], [
        "Behavioral_Rehearsal_Refusal_Scripts",
        "Role_Play_HighRisk_Conversation",
        "Behavioral_Experiment_Test_Coping_Beliefs",
        "Plan_Do_Review_Cycle_For_Recovery_Skills",
        "Generalize_Skills_To_New_Contexts_Framework",
    ])

    # ------------------------------------------------------------
    # 11.5 Stimulus Control and Cue Management (psycho-only framing)
    # ------------------------------------------------------------
    add_leaves(substance_use_recovery, ["Cue_And_Context_Management", "Stimulus_Control_Principles"], [
        "Identify_And_Remove_Use_Cues_Framework",
        "Reduce_Exposure_To_Triggering_Contexts_Strategy",
        "Reconfigure_Routines_To_Avoid_Cue_Chains",
        "Environmental_Friction_Increase_For_Use_Behavior",
        "Environmental_Friction_Decrease_For_Recovery_Behavior",
        "Plan_Safe_Exits_From_Risky_Situations",
    ])

    add_leaves(substance_use_recovery, ["Cue_And_Context_Management", "Cue_Exposure_and_Response_Prevention"], [
        "Cue_Exposure_Framework_Substance",
        "Response_Prevention_Principles_Substance",
        "Drop_Safety_Behaviors_That_Enable_Use",
        "Expectancy_Violation_Planning_Substance_Cues",
        "Post_Exposure_Learning_Summary_Substance",
    ])

    # ------------------------------------------------------------
    # 11.6 Emotion Regulation and Distress Tolerance for Recovery
    # ------------------------------------------------------------
    add_leaves(substance_use_recovery, ["Affect_Regulation_for_Recovery", "Distress_Tolerance"], [
        "Distress_Tolerance_Toolkit_For_Recovery",
        "Grounding_Practices_For_Craving_And_Affect",
        "Impulse_Pause_Practice_For_Use_Urges",
        "Self_Soothing_Sensory_Toolkit_For_Recovery",
        "Radical_Acceptance_For_Urge_And_Discomfort",
        "Turning_Toward_Emotion_Instead_Of_Use",
    ])

    add_leaves(substance_use_recovery, ["Affect_Regulation_for_Recovery", "Shame_Guilt_and_Self_Attack_Work"], [
        "Compassionate_Response_To_Lapse",
        "Reduce_Self_Criticism_After_Setback",
        "Shame_Resilience_Practices_Substance",
        "Self_Forgiveness_Practice_Substance",
        "Repair_And_Restore_Dignity_After_Setback",
    ])

    add_leaves(substance_use_recovery, ["Affect_Regulation_for_Recovery", "Interpersonal_Affect_Triggers"], [
        "Deescalation_Scripts_For_Conflict_Triggers",
        "Boundary_Setting_To_Reduce_Affect_Drivers",
        "Co_Regulation_Request_Scripts_Recovery",
        "Repair_After_Conflict_To_Prevent_Use_Urges",
    ])

    # ------------------------------------------------------------
    # 11.7 Interpersonal and Group-Based Recovery Skills (PSYCHO-only)
    # ------------------------------------------------------------
    add_leaves(substance_use_recovery, ["Interpersonal_and_Group_Recovery_Skills", "Refusal_and_Assertiveness"], [
        "Refusal_Skills_Substance",
        "Refusal_With_Warmth_And_Boundary",
        "Prepare_And_Practice_Exit_Lines",
        "Assertive_Request_For_Support_Scripts",
        "Boundary_Setting_With_Enabling_Dynamics",
    ])

    add_leaves(substance_use_recovery, ["Interpersonal_and_Group_Recovery_Skills", "Support_Engagement_Skills"], [
        "Support_Seeking_Scripts_Recovery",
        "Disclose_Recovery_Goals_With_Boundaries",
        "Ask_For_Accountability_Support_Scripts",
        "Mutual_Help_Facilitation_Framework",
        "Group_Participation_Skills",
        "Receive_Feedback_Nondefensively_Practice",
        "Offer_Support_To_Others_As_Commitment_Practice",
    ])

    # ------------------------------------------------------------
    # 11.8 Recovery Identity, Meaning, and Values (third-wave compatible)
    # ------------------------------------------------------------
    add_leaves(substance_use_recovery, ["Recovery_Identity_and_Meaning", "Values_and_Purpose"], [
        "Values_Clarification_For_Recovery",
        "Values_Based_Choice_Point_Training_Recovery",
        "Meaning_Making_From_Change_Effort",
        "Future_Self_Alignment_Practice_Recovery",
        "Identity_Based_Commitment_Language_Recovery",
    ])

    add_leaves(substance_use_recovery, ["Recovery_Identity_and_Meaning", "Narrative_and_Self_Concept_Work"], [
        "Recovery_Narrative_Reauthoring_Practice",
        "Integrate_Setbacks_Into_Learning_Story",
        "Strengths_Based_Recovery_Identity_Map",
        "Self_Respect_Practice_In_Recovery_Context",
    ])

    # ------------------------------------------------------------
    # 11.9 Relapse Prevention and Lapse Management (learning-first)
    # ------------------------------------------------------------
    add_leaves(substance_use_recovery, ["Relapse_Prevention_and_Lapse_Management", "HighRisk_Planning"], [
        "HighRisk_Situations_PrePlan_Framework",
        "Early_Warning_Signs_Recognition_Substance",
        "Relapse_Process_Map_And_Interruptions",
        "Alternative_Action_Menu_For_Risky_Moments",
        "Recovery_Contingency_Plans_For_Common_Triggers",
    ])

    add_leaves(substance_use_recovery, ["Relapse_Prevention_and_Lapse_Management", "Lapse_Response_and_Repair"], [
        "Lapse_Response_Plan_NoShame",
        "Lapse_As_Data_Learning_Frame",
        "Abstinence_Violation_Effect_Reframe",
        "Rapid_Return_To_Recovery_Steps_Framework",
        "Repair_Routine_After_Setback",
        "Update_Triggers_And_Skills_After_Lapse_Review",
    ])

    # ------------------------------------------------------------
    # 11.10 Rehabilitation Program Engagement Skills (PSYCHO-only)
    # ------------------------------------------------------------
    add_leaves(substance_use_recovery, ["Rehabilitation_Engagement_and_Skill_Use", "Program_Utilization_Skills"], [
        "Use_Structure_As_Support_Framework",
        "Engage_In_Groups_With_Intention_Practice",
        "Identify_And_Practice_Core_Skills_In_Program",
        "Ask_For_Help_Early_In_Program_Context",
        "Feedback_Integration_Practice_Rehab_Context",
        "Build_Routine_Anchors_From_Program_Structure",
    ])

    add_leaves(substance_use_recovery, ["Rehabilitation_Engagement_and_Skill_Use", "Aftercare_and_Continuation_Planning"], [
        "Aftercare_Plan_Creation_Psychological",
        "Identify_Recovery_Risks_After_Transition",
        "Continuation_Skills_Plan_Post_Program",
        "Maintain_Motivation_After_Intensity_Shifts",
        "Plan_For_Unexpected_Triggers_Post_Transition",
    ])

    # ------------------------------------------------------------
    # 11.11 Contingency Management and Reinforcement Design (behavioral)
    # ------------------------------------------------------------
    add_leaves(substance_use_recovery, ["Reinforcement_and_Contingency_Design", "Contingency_Management_Frameworks"], [
        "Contingency_Management_Framework",
        "Define_Target_Behaviors_For_Reinforcement",
        "Immediate_Natural_Rewards_Identification",
        "Build_Alternative_Reinforcement_System",
        "Response_Cost_For_Use_Enabling_Patterns_Self_Administered",
        "Commitment_Device_Design_For_Recovery",
    ])

    # Attach to PSYCHO as a primary domain
    PSYCHO["Substance_Use_Recovery_and_Behavior_Change"] = substance_use_recovery

    # ============================================================
    # Classical- and Modern Lifestyle Philosophies
    # (psycho-only; not BIO/SOCIAL)
    # ============================================================
    lifestyle_philosophies = OntologyBuilder().root

    # ------------------------------------------------------------
    # A) Greco-Roman and Western Classical Practical Philosophies
    # ------------------------------------------------------------
    add_leaves(lifestyle_philosophies, ["Greco_Roman_Practical_Philosophies", "Stoicism"], [
        "Stoic_Dichotomy_Of_Control_Practice",
        "Stoic_Trichotomy_Of_Control_Practice",
        "Stoic_View_From_Above_Practice",
        "Stoic_Negative_Visualization_Practice",
        "Stoic_Premeditatio_Malorum_Practice",
        "Stoic_Virtue_Focus_Wisdom_Justice_Courage_Temperance",
        "Stoic_Values_Aligned_Action_Principle",
        "Stoic_Emotion_As_Judgment_Reframe",
        "Stoic_Adversity_As_Training_Frame",
        "Stoic_Amor_Fati_Attitude_Practice",
        "Stoic_Memento_Mori_Perspective_Practice",
        "Stoic_Equanimity_Training",
        "Stoic_Compassionate_Firmness_Stance",
        "Stoic_Inner_Citadel_Metaphor_Practice",
        "Stoic_Stoic_Journaling_Reflection_Practice",
        "Stoic_Role_Ethics_Duty_Reflection",
        "Stoic_Gratitude_For_Present_Goods_Practice",
        "Stoic_Prosoche_Attention_To_Mind_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Greco_Roman_Practical_Philosophies", "Epicureanism"], [
        "Epicurean_Ataraxia_Tranquility_Orientation",
        "Epicurean_Desire_Classification_Natural_Necessary_Vain",
        "Epicurean_Simple_Pleasures_Savoring_Practice",
        "Epicurean_Prudence_As_Guide_Practice",
        "Epicurean_Fear_Of_Death_Reframe_Practice",
        "Epicurean_Pain_Avoidance_With_Wisdom_Principle",
        "Epicurean_Friendship_As_Flourishing_Reflection",
        "Epicurean_Contentment_With_Enough_Practice",
        "Epicurean_Gratitude_For_Sufficiency_Practice",
        "Epicurean_Moderation_And_Desire_Limiting_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Greco_Roman_Practical_Philosophies", "Aristotelian_Virtue_Ethics"], [
        "Aristotelian_Eudaimonia_Flourishing_Orientation",
        "Aristotelian_Golden_Mean_Practice",
        "Aristotelian_Practical_Wisdom_Phronesis_Reflection",
        "Aristotelian_Character_Habit_Cultivation_Principle",
        "Aristotelian_Virtue_As_Skill_Practice_Frame",
        "Aristotelian_Telos_Purpose_Clarification_Practice",
        "Aristotelian_Friendship_And_Character_Reflection",
        "Aristotelian_Emotions_As_Educable_Practice_Frame",
        "Aristotelian_Excellence_Over_Outcomes_Principle",
    ])

    add_leaves(lifestyle_philosophies, ["Greco_Roman_Practical_Philosophies", "Pyrrhonian_Skepticism"], [
        "Skeptic_Epoche_Suspension_Of_Judgment_Practice",
        "Skeptic_Fallibilism_Cognitive_Humility_Practice",
        "Skeptic_Distinguish_Appearance_From_Assertion_Practice",
        "Skeptic_Reduce_Dogmatism_Through_Alternatives_Practice",
        "Skeptic_Tranquility_Through_NonAttachment_To_Opinions",
        "Skeptic_Equipollence_Balancing_Considerations_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Greco_Roman_Practical_Philosophies", "Cynicism_Voluntary_Simplicity"], [
        "Cynic_Voluntary_Simplicity_Practice",
        "Cynic_Convention_Defusion_Practice",
        "Cynic_Independence_From_Status_Attachment_Practice",
        "Cynic_Ask_What_Is_Necessary_Reflection",
        "Cynic_Integrity_Over_Approval_Principle",
        "Cynic_Truth_Telling_With_Courage_Practice",
    ])

    add_leaves(lifestyle_philosophies,
               ["Greco_Roman_Practical_Philosophies", "Classical_Practical_Wisdom_StoicAdjacent"], [
                   "Praxis_Over_Theory_Principle",
                   "Cultivate_Inner_Freedom_Principle",
                   "Reduce_External_Validation_Dependency_Practice",
                   "Honor_Role_And_Duty_Without_Self_Erasure_Practice",
                   "Equanimity_Under_Change_Practice",
               ])

    # ------------------------------------------------------------
    # B) Indian and Yogic Philosophical Lifestyles (psychological framing)
    # ------------------------------------------------------------
    add_leaves(lifestyle_philosophies, ["Indian_and_Yogic_Philosophies", "Yoga_Sutras_Ethical_And_Mental_Discipline"], [
        "Yoga_Yamas_Ethical_Commitments_Practice",
        "Yoga_Niyamas_Personal_Observances_Practice",
        "Yoga_Ahimsa_NonHarming_In_SelfTalk_Practice",
        "Yoga_Satya_Truthfulness_With_Kindness_Practice",
        "Yoga_Aparigraha_NonGrasping_Practice",
        "Yoga_Santosha_Contentment_Practice",
        "Yoga_Tapas_Disciplined_Effort_Practice",
        "Yoga_Svadhyaya_Self_Study_Practice",
        "Yoga_IshvaraPranidhana_Let_Go_Of_Control_Practice",
        "Yoga_Practice_As_Mental_Training_Orientation",
        "Yoga_Citta_Vritti_Nirodha_Mind_Wave_Observation_Practice",
        "Yoga_Witness_Consciousness_Practice",
        "Yoga_Vairagya_NonAttachment_Practice",
        "Yoga_Abhyasa_Perseverance_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Indian_and_Yogic_Philosophies", "Bhagavad_Gita_Action_And_Equanimity"], [
        "Gita_Karma_Yoga_Duty_With_NonAttachment_Practice",
        "Gita_Focus_On_Effort_Not_Fruit_Practice",
        "Gita_Equanimity_In_Success_And_Setback_Practice",
        "Gita_Self_Mastery_Over_Reactivity_Practice",
        "Gita_Values_Aligned_Action_Under_Conflict_Practice",
        "Gita_Service_Orientation_As_Meaning_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Indian_and_Yogic_Philosophies", "Vedanta_NonDual_Self_Inquiry"], [
        "Vedanta_Self_Inquiry_WhoAmI_Practice",
        "Vedanta_Neti_Neti_Disidentification_Practice",
        "Vedanta_Witnessing_Thoughts_As_NotSelf_Practice",
        "Vedanta_NonAttachment_To_Roles_Practice",
        "Vedanta_Equanimity_Through_Self_As_Awareness_Practice",
    ])

    # ------------------------------------------------------------
    # C) Buddhist and Contemplative Philosophical Lifestyles
    # (kept as philosophy frames; overlaps with mindfulness skills is acceptable)
    # ------------------------------------------------------------
    add_leaves(lifestyle_philosophies, ["Buddhist_and_Contemplative_Philosophies", "Secular_Buddhism_Core_Frames"], [
        "Buddhist_Dukkha_As_Universal_Struggle_Reframe",
        "Buddhist_Craving_Aversion_Delusion_Map_Practice",
        "Buddhist_Impermanence_Anicca_Contemplation_Practice",
        "Buddhist_NonSelf_Anatta_Perspective_Practice",
        "Buddhist_Compassion_And_Wisdom_Balance_Practice",
        "Buddhist_Right_View_As_Lens_Practice",
        "Buddhist_Ethical_Living_As_Mental_Training_Principle",
        "Buddhist_Middle_Way_Moderation_Practice",
        "Buddhist_Karma_As_Learning_Consequences_Frame",
        "Buddhist_Three_Jewels_Refuge_Secularized_Practice",
    ])

    add_leaves(lifestyle_philosophies,
               ["Buddhist_and_Contemplative_Philosophies", "Eightfold_Path_As_Lifestyle_Framework"], [
                   "Eightfold_Right_Intention_Values_Alignment_Practice",
                   "Eightfold_Right_Speech_Mindful_Communication_Practice",
                   "Eightfold_Right_Action_Ethical_Consistency_Practice",
                   "Eightfold_Right_Livelihood_Meaning_Alignment_Reflection",
                   "Eightfold_Right_Effort_Wholesome_Cultivation_Practice",
                   "Eightfold_Right_Mindfulness_Continuous_Awareness_Practice",
                   "Eightfold_Right_Concentration_Stability_Practice",
               ])

    add_leaves(lifestyle_philosophies, ["Buddhist_and_Contemplative_Philosophies", "Zen_And_Beginner_Mind"], [
        "Zen_Shikantaza_Just_Sitting_Orientation",
        "Zen_Beginner_Mind_Practice",
        "Zen_NonDual_Ordinary_Mind_Is_Way_Frame",
        "Zen_Questioning_Koan_Style_Open_Inquiry",
        "Zen_Let_Go_Of_Conceptual_Grasping_Practice",
        "Zen_Simplicity_And_Direct_Experience_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Buddhist_and_Contemplative_Philosophies", "Tibetan_Lojong_Mind_Training"], [
        "Lojong_Reframe_Adversity_As_Path_Practice",
        "Lojong_Exchange_Self_And_Other_Perspective_Practice",
        "Lojong_Transform_Blame_Into_Curiosity_Practice",
        "Lojong_Compassionate_Response_To_Irritation_Practice",
        "Lojong_Tonglen_Informed_Giving_Receiving_Practice",
        "Lojong_Train_In_Gratitude_For_Teachers_And_Challenges",
    ])

    # ------------------------------------------------------------
    # D) Chinese and East Asian Philosophical Lifestyles
    # ------------------------------------------------------------
    add_leaves(lifestyle_philosophies, ["Chinese_and_East_Asian_Philosophies", "Taoism_Dao_De_Jing"], [
        "Taoism_WuWei_Effortless_Action_Principle",
        "Taoism_Ziran_Naturalness_Practice",
        "Taoism_Softness_Over_Force_Practice",
        "Taoism_Begin_With_Stillness_Practice",
        "Taoism_Yin_Yang_Balance_Reflection",
        "Taoism_Empty_Mind_Open_Space_Practice",
        "Taoism_Paradox_Tolerance_Practice",
        "Taoism_NonContention_Practice",
        "Taoism_Simplicity_And_Uncluttering_Of_Desires_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Chinese_and_East_Asian_Philosophies", "Confucian_Role_Ethics_And_Cultivation"],
               [
                   "Confucian_Ren_Humaneness_Practice",
                   "Confucian_Li_Ritual_As_Self_Regulation_Practice",
                   "Confucian_Xiao_Respect_And_Gratitude_Practice",
                   "Confucian_Self_Cultivation_Junzi_Orientation",
                   "Confucian_Shame_As_Moral_Compass_Reframe",
                   "Confucian_Reciprocity_Golden_Rule_Practice",
                   "Confucian_Learning_As_Character_Building_Practice",
               ])

    add_leaves(lifestyle_philosophies, ["Chinese_and_East_Asian_Philosophies", "Bushido_And_Integrity_Codes"], [
        "Bushido_Honor_Integrity_Alignment_Practice",
        "Bushido_Courage_Under_Fear_Practice",
        "Bushido_Discipline_As_Freedom_Principle",
        "Bushido_Respectful_Conduct_Practice",
        "Bushido_Readiness_For_Loss_Perspective_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Chinese_and_East_Asian_Philosophies", "Wabi_Sabi_And_Imperfection"], [
        "Wabi_Sabi_Imperfection_Acceptance_Practice",
        "Wabi_Sabi_Simplicity_And_Enoughness_Practice",
        "Wabi_Sabi_Aging_As_Beauty_Reframe",
        "Kintsugi_Brokenness_As_Story_Reframe",
        "Mono_No_Aware_Tenderness_For_Transience_Practice",
    ])

    # ------------------------------------------------------------
    # E) Abrahamic and Contemplative Ethical Traditions (psycho framing)
    # ------------------------------------------------------------
    add_leaves(lifestyle_philosophies, ["Abrahamic_And_Contemplative_Ethics", "Ignatian_Examen_And_Discernment"], [
        "Ignatian_Examen_Reflection_Practice",
        "Ignatian_Notice_Consolation_Desolation_Practice",
        "Ignatian_Discernment_Of_Motives_Practice",
        "Ignatian_Gratitude_First_Orientation_Practice",
        "Ignatian_Values_Aligned_Choice_Discernment_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Abrahamic_And_Contemplative_Ethics", "Jewish_Mussar_Character_Practice"], [
        "Mussar_Middot_Trait_Selection_Practice",
        "Mussar_Cheshbon_HaNefesh_Self_Accounting_Practice",
        "Mussar_Anavah_Humility_Practice",
        "Mussar_Savlanut_Patience_Practice",
        "Mussar_Emet_Truth_And_Integrity_Practice",
        "Mussar_Compassionate_Self_Correction_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Abrahamic_And_Contemplative_Ethics", "Sufi_And_Heart_Discipline"], [
        "Sufi_Dhikr_Mindful_Remembrance_Practice",
        "Sufi_Purify_Intention_Niyyah_Practice",
        "Sufi_Love_As_Transformative_Orientation",
        "Sufi_Ego_Softening_Practice",
        "Sufi_Surrender_And_Trust_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Abrahamic_And_Contemplative_Ethics", "Christian_Contemplative_Practices"], [
        "Lectio_Divina_Slow_Reading_Reflection_Practice",
        "Centering_Prayer_Stillness_Practice",
        "Rule_Of_Life_Values_Rhythm_Design_Practice",
        "Humility_And_Service_Orientation_Practice",
        "Forgiveness_As_Freedom_Practice",
    ])

    # ------------------------------------------------------------
    # F) Existential, Humanistic, and Modern Western Philosophies
    # ------------------------------------------------------------
    add_leaves(lifestyle_philosophies, ["Existential_And_Humanistic_Philosophies", "Existentialism_And_Authenticity"], [
        "Existential_Authenticity_Practice",
        "Existential_Freedom_And_Responsibility_Orientation",
        "Existential_Choice_Clarification_Practice",
        "Existential_Meaning_Making_In_Suffering_Practice",
        "Existential_Courage_To_Be_Practice",
        "Existential_Isolation_As_Connection_Invitation_Reframe",
        "Existential_Uncertainty_Tolerance_Practice",
        "Existential_Life_As_Project_Orientation",
    ])

    add_leaves(lifestyle_philosophies, ["Existential_And_Humanistic_Philosophies", "Logotherapy_And_Purpose"], [
        "Logotherapy_Will_To_Meaning_Orientation",
        "Logotherapy_Attitudinal_Change_In_Unchangeable_Context",
        "Logotherapy_Values_Creation_Experience_Attitude_Map",
        "Logotherapy_Self_Transcendence_Practice",
        "Logotherapy_Paradoxical_Intention_Attitude",
        "Logotherapy_Dereflection_Shift_From_Self_Focus_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Existential_And_Humanistic_Philosophies", "Pragmatism_And_Experimentation"], [
        "Pragmatism_Truth_As_Workability_Check_Practice",
        "Pragmatism_Learning_By_Doing_Orientation",
        "Pragmatism_Hypothesis_Testing_Life_Experiments_Practice",
        "Pragmatism_Flexible_Beliefs_Update_Practice",
        "Pragmatism_Iterate_And_Adapt_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Existential_And_Humanistic_Philosophies", "Absurdism_And_Revolt"], [
        "Absurdism_Accept_Absurdity_Without_Despair_Practice",
        "Absurdism_Revolt_And_Engagement_Practice",
        "Absurdism_Create_Meaning_Without_Certainty_Practice",
        "Absurdism_Humor_As_Defiance_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Existential_And_Humanistic_Philosophies", "Secular_Humanism_And_Dignity"], [
        "Humanism_Dignity_And_Agency_Orientation",
        "Humanism_Compassionate_Ethics_Practice",
        "Humanism_Critical_Thinking_As_Self_Care_Practice",
        "Humanism_Service_And_Contribution_As_Meaning_Practice",
        "Humanism_Prosocial_Responsibility_Practice",
    ])

    # ------------------------------------------------------------
    # G) Modern Psychology-Informed Lifestyle Frameworks
    # (positioned as life philosophies, not clinical protocols)
    # ------------------------------------------------------------
    add_leaves(lifestyle_philosophies, ["Modern_Psychology_Informed_Lifestyles", "Self_Determination_Theory_Lifestyle"],
               [
                   "SDT_Autonomy_Supporting_Self_Talk_Practice",
                   "SDT_Competence_Building_Mastery_Orientation",
                   "SDT_Relatedness_And_Belonging_Intention_Practice",
                   "SDT_Values_Congruence_Check_Practice",
                   "SDT_Internal_Motivation_Cultivation_Practice",
                   "SDT_Choose_And_Own_Goals_Practice",
               ])

    add_leaves(lifestyle_philosophies, ["Modern_Psychology_Informed_Lifestyles", "Growth_Mindset_Lifestyle"], [
        "Growth_Mindset_Learnability_Belief_Practice",
        "Growth_Mindset_Effort_As_Information_Reframe",
        "Growth_Mindset_Mistakes_As_Feedback_Reframe",
        "Growth_Mindset_Process_Praise_Self_Talk_Practice",
        "Growth_Mindset_Yet_Language_Practice",
        "Growth_Mindset_Challenge_Seeking_Orientation",
    ])

    add_leaves(lifestyle_philosophies,
               ["Modern_Psychology_Informed_Lifestyles", "Psychological_Flexibility_As_Lifestyle"], [
                   "Flexibility_Values_Guided_Living_Practice",
                   "Flexibility_Acceptance_Of_Internal_Experience_Practice",
                   "Flexibility_Defusion_From_Rules_And_Stories_Practice",
                   "Flexibility_Present_Moment_Return_Practice",
                   "Flexibility_Self_As_Context_Perspective_Practice",
                   "Flexibility_Commitment_To_Action_Practice",
               ])

    add_leaves(lifestyle_philosophies, ["Modern_Psychology_Informed_Lifestyles", "PERMA_Flourishing_Lifestyle"], [
        "PERMA_Positive_Emotion_Cultivation_Practice",
        "PERMA_Engagement_Flow_Seeking_Practice",
        "PERMA_Relationships_Nurture_Intention_Practice",
        "PERMA_Meaning_Contribution_Orientation_Practice",
        "PERMA_Accomplishment_Mastery_Orientation_Practice",
        "PERMA_Balance_Domains_Reflection_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Modern_Psychology_Informed_Lifestyles", "Psychological_Capital_Lifestyle"], [
        "PsyCap_Hope_Pathways_Thinking_Practice",
        "PsyCap_Hope_Agency_Self_Talk_Practice",
        "PsyCap_Efficacy_Mastery_Recall_Practice",
        "PsyCap_Resilience_Learning_From_Setback_Practice",
        "PsyCap_Optimism_Balanced_Realism_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Modern_Psychology_Informed_Lifestyles", "Self_Compassion_As_Life_Stance"], [
        "Self_Compassion_Kindness_In_Failure_Practice",
        "Self_Compassion_Common_Humanity_Reframe_Practice",
        "Self_Compassion_Mindful_Awareness_Of_Pain_Practice",
        "Self_Compassion_Supportive_Inner_Voice_Practice",
        "Self_Compassion_Dignity_Preserving_Boundaries_Practice",
    ])

    # ------------------------------------------------------------
    # H) Minimalism, Simplicity, and Enoughness Movements (psycho framing)
    # ------------------------------------------------------------
    add_leaves(lifestyle_philosophies, ["Minimalism_And_Simplicity_Movements", "Minimalism_Attentional_Clarity"], [
        "Minimalism_Attention_On_Essentials_Principle",
        "Minimalism_Intentional_Choice_Filter_Practice",
        "Minimalism_Reduce_Cluttered_Commitments_Practice",
        "Minimalism_Enoughness_Mindset_Practice",
        "Minimalism_Less_But_Better_Reframe",
        "Minimalism_Possession_Attachment_Defusion_Practice",
    ])

    add_leaves(lifestyle_philosophies,
               ["Minimalism_And_Simplicity_Movements", "Essentialism_Prioritization_Philosophy"], [
                   "Essentialism_Yes_By_Default_No_By_Design_Practice",
                   "Essentialism_Tradeoff_Acceptance_Practice",
                   "Essentialism_Protect_What_Matters_Practice",
                   "Essentialism_Eliminate_NonEssentials_Practice",
                   "Essentialism_Clarity_Then_Commitment_Practice",
               ])

    add_leaves(lifestyle_philosophies, ["Minimalism_And_Simplicity_Movements", "Slow_Living_And_Presence"], [
        "Slow_Living_Intentional_Pacing_Practice",
        "Slow_Living_Savoring_Ordinary_Moments_Practice",
        "Slow_Living_Single_Task_Attention_Practice",
        "Slow_Living_Quality_Over_Hurry_Practice",
        "Slow_Living_Choose_Less_Do_Deeper_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Minimalism_And_Simplicity_Movements", "Nordic_Balance_Philosophies"], [
        "Lagom_Just_Enough_Balance_Practice",
        "Hygge_Cozy_Safety_And_Savoring_Practice",
        "Friluftsliv_Nature_Connection_As_Presence_Practice",
        "Janteloven_Humility_And_Community_Mindedness_Practice",
    ])

    # ------------------------------------------------------------
    # I) Purpose, Meaning, and Life-Design Philosophies
    # ------------------------------------------------------------
    add_leaves(lifestyle_philosophies, ["Purpose_And_Life_Design_Philosophies", "Ikigai_And_Purpose_Crafting"], [
        "Ikigai_Purpose_Intersection_Reflection_Practice",
        "Ikigai_Values_Skills_Service_Alignment_Practice",
        "Ikigai_Meaning_Through_Contribution_Practice",
        "Ikigai_MicroPurpose_Selection_Practice",
        "Ikigai_Sustained_Engagement_Orientation",
    ])

    add_leaves(lifestyle_philosophies, ["Purpose_And_Life_Design_Philosophies", "Life_Design_Prototyping"], [
        "Life_Design_Curiosity_Orientation",
        "Life_Design_Reframe_Dysfunction_As_Data_Practice",
        "Life_Design_Prototype_Possible_Selves_Practice",
        "Life_Design_Experiment_Mindset_Practice",
        "Life_Design_Wayfinding_Over_Planning_Practice",
        "Life_Design_Rewrite_Limiting_Beliefs_Practice",
        "Life_Design_Multiple_Option_Generation_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Purpose_And_Life_Design_Philosophies", "Craftsmanship_And_Mastery_Ethos"], [
        "Craftsmanship_Pride_In_Work_Practice",
        "Craftsmanship_Process_Over_Applause_Practice",
        "Craftsmanship_Deliberate_Refinement_Practice",
        "Craftsmanship_Standards_With_Kindness_Practice",
        "Craftsmanship_Focus_On_Skill_And_Service_Practice",
    ])

    # ------------------------------------------------------------
    # J) Attention, Focus, and Information Ecology Philosophies
    # ------------------------------------------------------------
    add_leaves(lifestyle_philosophies, ["Attention_And_Focus_Philosophies", "Digital_Minimalism_And_Intentional_Tech"],
               [
                   "Digital_Minimalism_Intentional_Technology_Use_Principle",
                   "Digital_Minimalism_Value_First_Tool_Second_Principle",
                   "Digital_Minimalism_Choose_High_Quality_Inputs_Practice",
                   "Digital_Minimalism_Reduce_Compulsive_Checking_Practice",
                   "Digital_Minimalism_Reclaim_Offline_Joy_Practice",
                   "Digital_Minimalism_Attention_As_Sacred_Resource_Practice",
               ])

    add_leaves(lifestyle_philosophies, ["Attention_And_Focus_Philosophies", "Deep_Work_And_Cognitive_Craft"], [
        "Deep_Work_Focus_As_Skill_Principle",
        "Deep_Work_Distraction_Resistance_Identity_Practice",
        "Deep_Work_Attention_Protection_Practice",
        "Deep_Work_Produce_Then_Improve_Practice",
        "Deep_Work_Quality_And_Concentration_Link_Belief",
        "Deep_Work_Boredom_Tolerance_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Attention_And_Focus_Philosophies", "Information_Diet_And_Mental_Inputs"], [
        "Information_Diet_Selectivity_Practice",
        "Information_Diet_Replace_Doomscrolling_With_Intentional_Reading",
        "Information_Diet_Emotional_Contagion_Awareness_Practice",
        "Information_Diet_Choose_Inputs_Aligned_With_Values",
        "Information_Diet_Reflection_Over_Reactivity_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Attention_And_Focus_Philosophies", "Flow_Orientation_As_Lifestyle"], [
        "Flow_Skill_Challenge_Balance_Reflection_Practice",
        "Flow_Clear_Goals_And_Feedback_Seeking_Practice",
        "Flow_Deep_Immersion_Allowance_Practice",
        "Flow_Reduce_Context_Switching_Practice",
        "Flow_Choose_Absorbing_Activities_Practice",
    ])

    # ------------------------------------------------------------
    # K) Habit, Self-Management, and Practical Systems Philosophies
    # ------------------------------------------------------------
    add_leaves(lifestyle_philosophies, ["Habit_And_Systems_Philosophies", "Kaizen_Continuous_Improvement"], [
        "Kaizen_Small_Improvement_Orientation",
        "Kaizen_Reflect_Adjust_Iterate_Practice",
        "Kaizen_Process_Focus_Over_Perfection_Practice",
        "Kaizen_Experiment_And_Learn_Practice",
        "Kaizen_Standardize_What_Works_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Habit_And_Systems_Philosophies", "Behavior_Design_And_Tiny_Habits"], [
        "Behavior_Design_Motivation_Ability_Prompt_Model",
        "Tiny_Habits_Anchor_Behavior_Method",
        "Tiny_Habits_Celebration_Reinforcement_Practice",
        "Tiny_Habits_Shrink_The_Behavior_Practice",
        "Tiny_Habits_Focus_On_Consistency_Over_Intensity_Practice",
        "Tiny_Habits_Identity_Through_Repetition_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Habit_And_Systems_Philosophies", "Identity_Based_Habit_Philosophy"], [
        "Identity_Based_Habits_Become_The_Type_Of_Person_Practice",
        "Identity_Evidence_Collection_Practice",
        "Systems_Over_Goals_Principle",
        "Focus_On_Trajectory_Not_Perfection_Practice",
        "Make_Desired_Behavior_Easier_Principle",
        "Reduce_Friction_For_Valued_Actions_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Habit_And_Systems_Philosophies", "Getting_Things_Done_GTD"], [
        "GTD_Capture_Everything_Practice",
        "GTD_Clarify_Next_Action_Practice",
        "GTD_Organize_By_Context_Practice",
        "GTD_Reflect_To_Trust_System_Practice",
        "GTD_Engage_With_Clear_Mind_Practice",
        "GTD_Next_Action_Thinking_Practice",
        "GTD_Projects_As_Outcomes_Definition_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Habit_And_Systems_Philosophies", "Bullet_Journaling_As_Intentional_Living"], [
        "Bullet_Journal_Rapid_Logging_Practice",
        "Bullet_Journal_Reflection_Then_Migration_Practice",
        "Bullet_Journal_Intention_Setting_Practice",
        "Bullet_Journal_Future_Log_As_Commitment_Practice",
        "Bullet_Journal_Review_And_Align_With_Values_Practice",
    ])

    # ------------------------------------------------------------
    # L) Resilience, Strength, and Adversity Philosophies
    # ------------------------------------------------------------
    add_leaves(lifestyle_philosophies,
               ["Resilience_And_Adversity_Philosophies", "Antifragility_And_Optionality_Mindset"], [
                   "Antifragile_Seek_Learning_From_Volatility_Practice",
                   "Antifragile_Stressors_As_Information_Practice",
                   "Antifragile_Build_Optionality_In_Choices_Practice",
                   "Antifragile_Avoid_Fragile_Overcertainty_Practice",
                   "Antifragile_Skin_In_The_Game_Integrity_Practice",
               ])

    add_leaves(lifestyle_philosophies, ["Resilience_And_Adversity_Philosophies", "Nietzschean_Self_Overcoming"], [
        "Nietzsche_Self_Overcoming_Practice",
        "Nietzsche_Amor_Fati_As_Yes_To_Life_Practice",
        "Nietzsche_Create_Values_Not_Just_Follow_Practice",
        "Nietzsche_Transform_Suffering_Into_Strength_Reframe",
        "Nietzsche_Responsibility_For_Interpretation_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Resilience_And_Adversity_Philosophies", "Stoic_Buddhist_Synthesis_Frames"], [
        "Synthesis_Accept_What_Cannot_Be_Changed_Practice",
        "Synthesis_Act_On_What_Can_Be_Influenced_Practice",
        "Synthesis_NonAttachment_To_Outcomes_Practice",
        "Synthesis_Compassion_With_Firm_Boundaries_Practice",
        "Synthesis_Equanimity_And_Engagement_Practice",
    ])

    add_leaves(lifestyle_philosophies, ["Resilience_And_Adversity_Philosophies", "Nordic_Sisu_And_Grit"], [
        "Sisu_Steadfast_Courage_Practice",
        "Sisu_Persist_Through_Discomfort_Practice",
        "Sisu_Integrity_Under_Pressure_Practice",
        "Sisu_Practical_Optimism_Practice",
    ])

    # ------------------------------------------------------------
    # M) Relational Ethics as Lifestyle (psycho framing; not “social policy”)
    # ------------------------------------------------------------
    add_leaves(lifestyle_philosophies, ["Relational_Ethics_As_Lifestyle", "Compassionate_Conduct_Traditions"], [
        "Compassion_First_Interpretation_Practice",
        "Assume_Good_Intent_Until_Data_Practice",
        "Truth_With_Kindness_Principle",
        "Repair_Over_Righteousness_Practice",
        "Forgiveness_With_Boundaries_Practice",
        "Gratitude_And_Appreciation_Expression_Practice",
        "Radical_Candor_With_Care_Principle",
    ])

    add_leaves(lifestyle_philosophies, ["Relational_Ethics_As_Lifestyle", "Nonviolent_Communication_As_Stance"], [
        "NVC_Observation_Without_Evaluation_Practice",
        "NVC_Feeling_Needs_Linking_Practice",
        "NVC_Request_Not_Demand_Practice",
        "NVC_Empathic_Listening_Practice",
        "NVC_Self_Empathy_Practice",
        "NVC_Conflict_As_Needs_Clash_Reframe",
    ])

    # Attach as a primary PSYCHO node
    PSYCHO["Lifestyle_Philosophies"] = lifestyle_philosophies

    return {"PSYCHO": PSYCHO}


# ------------------------ Compatibility helpers (same signatures as your original) ------------------------

def add_path(tree: dict, path: List[str]) -> None:
    node = tree
    for p in path:
        if not isinstance(node, dict):
            raise TypeError(f"Ontology corruption: expected dict at {p}, got {type(node)}")
        node = node.setdefault(p, {})

def add_leaves(tree: dict, base_path: List[str], leaves: List[str]) -> None:
    for leaf in leaves:
        add_path(tree, base_path + [leaf])


# ------------------------ Writer + metadata ------------------------

def write_outputs(ontology: dict, out_json_path: str) -> Tuple[str, str, dict]:
    out_json_path = os.path.expanduser(out_json_path)
    out_dir = os.path.dirname(out_json_path)

    # If path is not viable, fall back to a local PSYCHO/ folder next to this script
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        out_dir = os.path.join(os.path.dirname(__file__), "PSYCHO")
        os.makedirs(out_dir, exist_ok=True)
        out_json_path = os.path.join(out_dir, "PSYCHO.json")

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(ontology, f, ensure_ascii=False, indent=2)

    leaf_paths = list(iter_leaf_paths(ontology["PSYCHO"]))
    leaf_count = count_leaves(ontology["PSYCHO"])
    node_count = count_nodes(ontology["PSYCHO"])
    depth = max_depth(ontology["PSYCHO"])
    top_counts = subtree_leaf_counts(ontology["PSYCHO"])
    depths = [len(p) for p in leaf_paths]

    # Hash minified JSON for versioning
    minified = json.dumps(ontology, ensure_ascii=False, separators=(",", ":"))
    sha256 = hashlib.sha256(minified.encode("utf-8")).hexdigest()

    metadata = {
        "generated_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "leaf_count": leaf_count,
        "node_count": node_count,
        "max_depth": depth,
        "avg_leaf_depth": sum(depths) / max(1, len(depths)),
        "median_leaf_depth": statistics.median(depths) if depths else 0,
        "sha256_minified_json": sha256,
        "leaf_count_by_primary_node": dict(sorted(top_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "output_json_path": os.path.abspath(out_json_path),
    }

    meta_lines = [
        f"generated_utc: {metadata['generated_utc']}",
        f"leaf_count: {metadata['leaf_count']}",
        f"node_count: {metadata['node_count']}",
        f"max_depth: {metadata['max_depth']}",
        f"avg_leaf_depth: {metadata['avg_leaf_depth']:.2f}",
        f"median_leaf_depth: {metadata['median_leaf_depth']}",
        f"sha256_minified_json: {metadata['sha256_minified_json']}",
        "",
        "leaf_count_by_primary_node:",
    ]
    for k, v in metadata["leaf_count_by_primary_node"].items():
        meta_lines.append(f"  - {k}: {v}")

    meta_path = os.path.join(out_dir, "metadata.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines))

    return out_json_path, meta_path, metadata


# ------------------------ Validation ------------------------

def validate_key_style(key: str) -> None:
    # allow A-Z a-z 0-9 underscore; must start with letter
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key):
        raise ValueError(f"Invalid key style: '{key}' (must match [A-Za-z][A-Za-z0-9_]*)")

def validate_ontology(ontology: dict) -> None:
    """
    Guardrails:
    - reject explicit schedules/frequency/duration/intensity tokens as node labels
    - reject numeric dose tokens (minutes, sessions, Hz, per week/day/month)
    - reject explicit disorder-labeled branch tokens (*_Disorder, *_Syndrome, DSM, ICD) anywhere
    """
    forbidden_patterns = [
        # scheduling / dosing
        r"\b\d+\s*x\s*week\b",
        r"\b\d+x_week\b",
        r"\bper_week\b",
        r"\bweekly\b",
        r"\bdaily\b",
        r"\bper_day\b",
        r"\bper_month\b",
        r"\b\d+\s*min\b",
        r"\b\d+\s*minutes\b",
        r"\b\d+\s*hour\b",
        r"\b\d+\s*hours\b",
        r"\b\d+\s*hz\b",
        r"\b\d+\s*weeks\b",
        r"\b\d+\s*days\b",
        r"\b\d+\s*sessions\b",
        r"\bminutes_per\b",
        r"\bhz\b",
        # explicit disorder-label tokens
        r"_Disorder\b",
        r"_Syndrome\b",
        r"\bDSM\b",
        r"\bICD\b",
    ]

    if "PSYCHO" not in ontology or not isinstance(ontology["PSYCHO"], dict):
        raise ValueError("Ontology must be shaped as {'PSYCHO': { ... }}")

    leaf_paths = list(iter_leaf_paths(ontology["PSYCHO"]))
    bad = scan_forbidden_tokens(leaf_paths, forbidden_patterns)
    if bad:
        example_path, example_pat = bad[0]
        raise ValueError(
            "Forbidden token detected in ontology labels. "
            f"pattern={example_pat!r} in path={' / '.join(example_path)}"
        )

    # Ensure every node is a dict and every leaf is an empty dict
    def _check(node: Any, prefix: Tuple[str, ...] = ()) -> None:
        if isinstance(node, dict):
            # leaf
            if not node:
                return
            for k, v in node.items():
                if not isinstance(k, str) or not k:
                    raise ValueError(f"Invalid key at {' / '.join(prefix)}")
                validate_key_style(k)
                _check(v, prefix + (k,))
        else:
            raise ValueError(f"Non-dict node at {' / '.join(prefix)}: {type(node)}")

    _check(ontology["PSYCHO"])


# ------------------------ Main ------------------------

def main() -> None:
    # Override via PSYCHO_OUT_PATH env var.
    default_out = os.path.join(os.path.dirname(__file__), "PSYCHO.json")
    out_json_path = os.environ.get("PSYCHO_OUT_PATH", default_out)

    ontology = build_psycho_ontology()
    validate_ontology(ontology)

    out_json_path, meta_path, metadata = write_outputs(ontology, out_json_path)

    print("Leaf nodes:", metadata["leaf_count"])
    print("Nodes total:", metadata["node_count"])
    print("Max depth:", metadata["max_depth"])
    print("Wrote JSON:", out_json_path)
    print("Wrote metadata:", meta_path)


if __name__ == "__main__":
    main()
