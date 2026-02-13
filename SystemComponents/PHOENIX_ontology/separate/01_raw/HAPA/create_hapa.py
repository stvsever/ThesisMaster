#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAPA PHOENIX_ontology Generator (Solution-Variable-Oriented; Mapping-Free)

Goal
- Formal HAPA-aligned, therapy-ready ontology for behavior change messaging.
- Leaf nodes represent actionable "solution variables" (interventions, supports, strategies, tools, scripts)
  and (for the dedicated barrier library) high-resolution barrier descriptors.
- STRICTLY NO mapping layers in this script:
  - No Barrier→Predictor mapping
  - No Coping→Barrier mapping
  - No crosswalks / edges / adjacency lists between BARRIERS and COPING_STRATEGIES

Primary nodes (top-level under HAPA):
1) Motivation_Phase
2) Volition_Phase
4) BARRIERS                      (high-resolution barrier ontology)
5) COPING_STRATEGIES             (high-resolution coping strategy ontology)

Constraints
- No disorder-labeled branches.
- No explicit schedule/frequency/duration/intensity parameters as nodes.

Writes
1) HAPA.json
2) metadata.txt (same folder)

Override output path:
  HAPA_OUT_PATH="/path/to/HAPA/HAPA.json" python generate_hapa_ontology.py
"""

import os
import json
import re
import hashlib
import datetime
import statistics


# ------------------------ Tree helpers ------------------------

def add_path(tree: dict, path: list[str]) -> None:
    node = tree
    for p in path:
        node = node.setdefault(p, {})

def add_leaves(tree: dict, base_path: list[str], leaves: list[str]) -> None:
    for leaf in leaves:
        add_path(tree, base_path + [leaf])

def count_leaves(node) -> int:
    # Leaves are empty dicts
    if isinstance(node, dict):
        if not node:
            return 1
        return sum(count_leaves(v) for v in node.values())
    return 1

def count_nodes(node) -> int:
    if isinstance(node, dict):
        return 1 + sum(count_nodes(v) for v in node.values())
    return 1

def max_depth(node, depth: int = 0) -> int:
    if isinstance(node, dict) and node:
        return max(max_depth(v, depth + 1) for v in node.values())
    return depth

def iter_leaf_paths(node, prefix: tuple[str, ...] = ()):
    if isinstance(node, dict):
        if not node:
            yield prefix
        else:
            for k, v in node.items():
                yield from iter_leaf_paths(v, prefix + (k,))
    else:
        yield prefix

def subtree_leaf_counts(root: dict) -> dict[str, int]:
    return {k: count_leaves(v) for k, v in root.items()}

def scan_forbidden_tokens(paths: list[tuple[str, ...]], forbidden_patterns: list[str], limit: int = 50):
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


# ------------------------ PHOENIX_ontology builder ------------------------

def build_hapa_ontology() -> dict:
    HAPA: dict = {}

    # ============================================================
    # 1) Motivation Phase: risk perception, outcome expectancies,
    #    task self-efficacy -> intention
    # ============================================================
    motivation: dict = {}

    # --- Risk perception / threat appraisal
    risk: dict = {}
    risk_assessment = [
        "Personalized_Risk_Appraisal_Feedback",
        "Risk_Factor_Inventory_Review",
        "Trigger_And_Context_Risk_Mapping",
        "Vulnerability_Clarification_Exercise",
        "Severity_Clarification_Exercise",
        "Risk_Normalization_And_Stigma_Safe_Frames",
        "Misconception_Correction_Risk_Education",
        "Unrealistic_Optimism_Check_Exercise",
        "Comparative_Risk_Reflection_Exercise",
        "Probability_Literacy_Support_Explainer",
        "Uncertainty_Tolerance_Frame_Risk_Info",
        "Risk_To_Values_Relevance_Link_Exercise",
        "Teach_Back_Comprehension_Check",
    ]
    risk_communication = [
        "Gain_Framed_Risk_Communication",
        "Loss_Framed_Risk_Communication_Cautious",
        "Narrative_Risk_Story_Reflection",
        "Future_Self_Simulation_Risk_Focused",
        "Mental_Imagery_Of_Consequences_Safe",
        "Values_Based_Risk_Relevance_Link",
        "Motivational_Norms_Feedback_Safe",
        "Prospective_Regret_Frame_Safe",
        "Control_Focus_Reorientation",
        "Hope_Induction_Frame",
    ]
    risk_emotion_regulation = [
        "Fear_Arousal_Containment_Script",
        "Anxiety_Tolerance_Brief_Skill",
        "Threat_To_Coping_Reframe",
        "Self_Compassion_After_Threat_Info",
        "Grounding_After_Threat_Info_Skill",
        "Cognitive_Diffusion_From_Catastrophic_Imagery",
    ]
    add_leaves(risk, ["Assessment_and_Feedback"], risk_assessment)
    add_leaves(risk, ["Communication_and_Framing"], risk_communication)
    add_leaves(risk, ["Affect_Regulation_After_Risk_Info"], risk_emotion_regulation)
    motivation["Risk_Perception_and_Threat_Appraisal"] = risk

    # --- Outcome expectancies: pros/cons, benefits, costs
    expectancies: dict = {}
    expectancy_elicitation = [
        "Decisional_Balance_Pros_Cons_Map",
        "Expected_Benefits_List_Generation",
        "Expected_Costs_Barriers_List_Generation",
        "Short_Term_vs_Long_Term_Consequences_Map",
        "Identity_Consistent_Benefits_Reflection",
        "Social_Consequences_Reflection",
        "Affective_Outcomes_Reflection",
        "Functional_Outcomes_Reflection",
        "Opportunity_Cost_Reflection",
        "Cost_Realism_Check_Exercise",
        "Benefit_Realism_Check_Exercise",
    ]
    expectancy_enhancement = [
        "Benefit_Visualization_Guided",
        "Contrast_Current_vs_Desired_State_Reflection",
        "Best_Possible_Self_Exercise",
        "Behavioral_Experiment_To_Test_Benefit",
        "Testimonial_And_Role_Model_Exposure",
        "Social_Proof_Information_Safe",
        "Reappraise_Costs_Cognitive_Reframe",
        "Expectation_Management_Realism_Frame",
        "Affective_Benefit_Noticing_Prompt",
        "Functional_Gains_Spotting_Prompt",
    ]
    expectancy_commitment = [
        "Personal_Why_Statement_Creation",
        "Values_Clarification_Link_To_Behavior",
        "Meaning_Making_Narrative_Rewrite",
        "Regret_Minimization_Frame",
        "Legacy_And_Role_Model_Frame",
        "Commitment_Contract_Self_Only",
        "Identity_Aligned_Benefit_Statement",
    ]
    add_leaves(expectancies, ["Elicitation"], expectancy_elicitation)
    add_leaves(expectancies, ["Enhancement_and_Recalibration"], expectancy_enhancement)
    add_leaves(expectancies, ["Commitment_Anchors"], expectancy_commitment)
    motivation["Outcome_Expectancies"] = expectancies

    # --- Task self-efficacy: ability to start/initiate
    task_se: dict = {}
    mastery = [
        "Graded_Task_Ladder_Design",
        "Minimum_Viable_Action_Definition",
        "Microcommitment_Pledge",
        "Behavioral_Rehearsal_Script",
        "Skills_Practice_With_Feedback",
        "Success_Log_Creation",
        "Evidence_Of_Capability_Review",
        "Obstacle_Rehearsal_Safe",
        "Start_Ritual_Design",
        "Past_Success_Generalization_Exercise",
        "Strengths_Inventory_Activation",
    ]
    vicarious = [
        "Role_Model_Selection",
        "Peer_Modeling_Examples",
        "Video_Modeling_Examples",
        "Social_Learning_Observation_Plan",
        "Ask_A_Successful_Peer_Interview",
        "Mentorship_Linkage",
        "Case_Vignette_Selection",
    ]
    persuasion = [
        "Strengths_Based_Feedback_Script",
        "Verbal_Persuasion_Clinician_Message",
        "Motivational_Affirmations_Personalized",
        "Self_Talk_Script_Constructive",
        "Identity_Based_Affirmation_Script",
        "Autonomy_Supportive_Language_Frame",
        "Compassionate_Encouragement_Script",
    ]
    phys_affect = [
        "Arousal_Labeling_And_Reframe",
        "Energy_Management_Plan_ScheduleFree",
        "Stress_Buffering_Plan",
        "Fatigue_Compassionate_Start_Plan",
        "Somatic_Calming_PreStart",
        "Sleep_Deprivation_Self_Kindness_Frame",
        "Pain_Or_Discomfort_Acceptance_Skill_Brief",
    ]
    add_leaves(task_se, ["Mastery_Experiences"], mastery)
    add_leaves(task_se, ["Vicarious_Experience_and_Modeling"], vicarious)
    add_leaves(task_se, ["Social_Persuasion_and_Affirmation"], persuasion)
    add_leaves(task_se, ["Physiological_and_Affective_Interpretation"], phys_affect)
    motivation["Task_Self_Efficacy_Enhancement"] = task_se

    # --- Intention formation
    intention: dict = {}
    intention_building = [
        "Goal_Selection_From_Menu",
        "Personal_Goal_Specification",
        "Commitment_Statement_Written",
        "Public_Commitment_Safe_Sharing",
        "Implementation_Intention_Seed_Statement",
        "Identity_Linked_Goal_Statement",
        "Self_Concordance_Check",
        "Choice_Architecture_Offer_Set",
        "Autonomy_Supportive_Language_Frame",
        "Values_Aligned_Goal_Message",
        "Barrier_Anticipation_Prompt_Intention",
    ]
    ambivalence_resolution = [
        "Motivational_Interviewing_Elicit_Change_Talk",
        "Importance_Confidence_Ruler_Conversation",
        "Values_Discrepancy_Reflection",
        "Cost_Of_Inaction_Reflection",
        "Barriers_To_Change_Compassionate_Review",
        "Choice_Point_Clarification",
        "Preference_Stability_Check",
        "Internal_Objections_Listing_Exercise",
        "Commitment_Readiness_Check",
    ]
    add_leaves(intention, ["Intention_Formation_and_Commitment"], intention_building)
    add_leaves(intention, ["Ambivalence_Resolution"], ambivalence_resolution)
    motivation["Intention"] = intention

    HAPA["Motivation_Phase"] = motivation

    # ============================================================
    # 2) Volition Phase: planning -> action -> maintenance -> recovery
    # ============================================================
    volition: dict = {}

    # --- Planning phase: action planning + coping planning
    planning: dict = {}

    action_planning: dict = {}
    action_plan_templates = [
        "Action_Plan_Template_Where_When_WithWhom",
        "Action_Plan_Template_How_Steps",
        "Resource_Checklist_For_Action",
        "Environment_Setup_Checklist",
        "Start_Cue_Definition",
        "End_Cue_Definition",
        "Subgoal_Ladder_Definition",
        "Decision_Rules_If_Then_Behavioral",
        "Plan_For_Tracking_And_Feedback",
        "Plan_For_Reward_And_Reinforcement",
        "Default_Option_Design",
        "Boundary_Conditions_Definition",
        "Plan_Branching_Options",
    ]
    plan_quality = [
        "Specificity_Upgrade_Check",
        "Feasibility_Check",
        "Opportunity_Cost_Check",
        "Alternative_Options_List",
        "Plan_Simplification_Pass",
        "Friction_Reduction_Pass",
        "Alignment_With_Values_Check",
        "Reduce_Choice_Load_Strategy",
        "Precommitment_Device_Selection",
        "Plan_Ownership_Check",
    ]
    add_leaves(action_planning, ["Templates_and_Artifacts"], action_plan_templates)
    add_leaves(action_planning, ["Plan_Quality_Optimization"], plan_quality)
    planning["Action_Planning"] = action_planning

    coping_planning: dict = {}
    coping_identification = [
        "Barrier_Brainstorm_Worksheet",
        "High_Risk_Situations_Map",
        "Competing_Goals_Map",
        "Social_Pressure_Scenarios_List",
        "Mood_Driven_Risk_Scenarios_List",
        "Fatigue_And_Low_Energy_Scenarios_List",
        "Access_And_Logistics_Risk_Map",
        "Travel_And_Disruption_Risk_Map",
        "Slip_Pattern_Review",
        "Early_Warning_Signs_Checklist",
        "Trigger_Chain_Analysis",
        "Functional_Analysis_ABC_Worksheet",
    ]
    coping_tools = [
        "If_Then_Coping_Plan_Cards",
        "Coping_Scripts_For_Urges",
        "Coping_Scripts_For_Stress",
        "Coping_Scripts_For_Social_Pressure",
        "Alternative_Action_Menu",
        "Problem_Solving_Steps_Guide",
        "Support_Seeking_Script",
        "Stimulus_Control_Plan",
        "Delay_And_Distract_Toolkit",
        "Self_Compassion_After_Slip_Script",
        "Crisis_Plan_If_Overwhelmed",
        "Cognitive_Reframe_In_Moment",
        "Attention_Redirection_Skill",
    ]
    add_leaves(coping_planning, ["Barrier_Identification"], coping_identification)
    add_leaves(coping_planning, ["Coping_Plan_Tools"], coping_tools)
    planning["Coping_Planning"] = coping_planning

    # --- Planning self-efficacy
    planning_se: dict = {}
    planning_se_leaves = [
        "Plan_With_Clinician_CoDesign",
        "Plan_With_Peer_CoDesign",
        "Plan_Demo_Examples_Exposure",
        "Plan_Rehearsal_Mental_Simulation",
        "Plan_Rehearsal_Role_Play",
        "Plan_Confidence_Check_And_Revise",
        "Barrier_Rehearsal_And_Response_Practice",
        "Implementation_Intention_Practice",
        "Plan_Skill_Building_MicroLessons",
    ]
    add_leaves(planning_se, ["Planning_Self_Efficacy_Enhancement"], planning_se_leaves)
    planning["Planning_Self_Efficacy_Support"] = planning_se

    volition["Planning_Phase"] = planning

    # --- Action phase: action initiation + action control
    action_phase: dict = {}

    initiation: dict = {}
    initiation_support = [
        "Action_Initiation_Cue_Design",
        "Start_Ritual_Protocol",
        "Reduce_Activation_Energy_Plan",
        "Precommitment_Device_Selection",
        "Accountability_Activation_Message",
        "Implementation_Intention_Deployment",
        "Context_Selection_And_Anchor",
        "Contingency_Start_Option",
        "Starter_Step_Fallback_Option",
        "Behavioral_Activation_Microstep",
        "Two_Minute_Starter_Script",
        "Remove_Startup_Friction_Checklist",
    ]
    add_leaves(initiation, ["Action_Initiation_Support"], initiation_support)
    action_phase["Action_Initiation"] = initiation

    action_control: dict = {}
    self_monitoring = [
        "Self_Monitoring_Log_Simple",
        "Self_Monitoring_Log_Contextual",
        "Self_Monitoring_With_Technology_Tool",
        "Check_In_Questions_Set",
        "Progress_Signal_Visualization",
        "Feedback_From_Trusted_Person",
        "Reflective_Review_Prompt_Set",
        "Goal_Standard_Reminder_Card",
        "Short_Term_Milestones_Definition",
        "Implementation_Fidelity_Checklist",
    ]
    awareness_of_standards = [
        "Define_Success_Criteria_Card",
        "Define_Minimum_Standard_Card",
        "Define_Stretch_Standard_Card",
        "Value_Based_Standard_Statement",
        "Identity_Based_Standard_Statement",
        "Boundary_Conditions_Definition",
        "NonNegotiables_Clarification",
    ]
    regulatory_effort = [
        "Urge_Surfing_Skill",
        "Delay_Tactic_Tool",
        "Distraction_Toolkit_Action",
        "Cognitive_Reframe_In_Moment",
        "Self_Talk_In_Moment",
        "Attention_Redirection_Skill",
        "Competing_Intention_Inhibition_Script",
        "Micro_Recovery_Pause_Skill",
        "Emotion_Regulation_Brief_Skill",
        "Values_Reminder_In_Moment_Card",
        "Self_Compassion_During_Difficulty_Script",
    ]
    add_leaves(action_control, ["Self_Monitoring"], self_monitoring)
    add_leaves(action_control, ["Awareness_of_Standards"], awareness_of_standards)
    add_leaves(action_control, ["Regulatory_Effort_and_Inhibition"], regulatory_effort)
    action_phase["Action_Control"] = action_control

    situational: dict = {}
    supports = [
        "Environmental_Restructuring_Action_Context",
        "Remove_Friction_From_Action",
        "Add_Friction_To_Competing_Behavior",
        "Cue_Placement_And_Salience_Design",
        "Tool_And_Resource_Availability_Setup",
        "Social_Support_Request_Action",
        "Accountability_Partner_Linkage",
        "Coach_Or_Therapist_CheckIn_Structure",
        "Peer_Support_Linkage",
        "Change_Social_Context_Option",
        "Immediate_Reward_Design_NonFood",
        "Access_Alternative_Option_Setup",
    ]
    add_leaves(situational, ["Situational_Barriers_and_Resources_Support"], supports)
    action_phase["Situational_Level_Supports"] = situational

    volition["Action_Phase"] = action_phase

    # --- Maintenance phase
    maintenance: dict = {}

    maintenance_se: dict = {}
    maintenance_se_leaves = [
        "Streak_And_Progress_Identity_Reinforcement",
        "Resilience_Narrative_Training",
        "Obstacle_Confidence_Rehearsal",
        "Coping_Success_Review",
        "Recommitment_Statement_Refresh",
        "Skill_Upgrading_Plan",
        "Social_Proof_And_Belonging_Reinforcement",
        "Autonomy_And_Ownership_Reinforcement",
        "Long_Horizon_Goal_Refresh",
        "Sustainability_Check_Reflection",
    ]
    add_leaves(maintenance_se, ["Maintenance_Self_Efficacy_Enhancement"], maintenance_se_leaves)
    maintenance["Maintenance_Self_Efficacy"] = maintenance_se

    stabilization: dict = {}
    stabilization_leaves = [
        "Routine_Stabilization_Design",
        "Habit_Loop_Design_Cue_Action_Reward",
        "Identity_Based_Habit_Statement",
        "Environment_Defaults_Setting",
        "Relapse_Triggers_Removal_Plan",
        "Plateau_Management_Reframe",
        "Boredom_Antidote_Variation_Menu",
        "Sustainable_Enjoyment_Design",
        "Maintenance_Review_Protocol",
        "Early_Warning_Signs_Checklist",
        "Meaning_Making_Narrative_Rewrite",
    ]
    add_leaves(stabilization, ["Habit_and_Routine_Stabilization"], stabilization_leaves)
    maintenance["Maintenance_Stabilization"] = stabilization

    volition["Maintenance_Phase"] = maintenance

    # --- Recovery phase
    recovery: dict = {}

    recovery_se: dict = {}
    recovery_se_leaves = [
        "Self_Compassion_After_Lapse_Script",
        "Normalize_Lapse_As_Learning_Frame",
        "Rapid_Reengagement_Confidence_Script",
        "Rebuild_Trust_With_Self_Protocol",
        "Setback_To_Data_Reframe",
        "Recommitment_With_Choice_Frame",
        "Hope_Restoration_Script",
        "Shame_Reduction_Script",
        "Compassionate_Closure_Script",
        "AllOrNothing_Thinking_Reset_Script",
    ]
    add_leaves(recovery_se, ["Recovery_Self_Efficacy_Enhancement"], recovery_se_leaves)
    recovery["Recovery_Self_Efficacy"] = recovery_se

    lapse_management: dict = {}
    lapse_tools = [
        "Lapse_Analysis_Worksheet",
        "Trigger_Chain_Analysis",
        "Competing_Goal_Analysis",
        "Barrier_Update_And_Replan",
        "Coping_Plan_Update",
        "Support_Activation_After_Lapse",
        "Reduce_AllOrNothing_Thinking_Script",
        "Repair_Ritual_Design",
        "Restart_Cue_Definition",
        "Protective_Factors_Inventory",
        "Lesson_Learned_Summary_Card",
    ]
    add_leaves(lapse_management, ["Lapse_and_Setback_Management"], lapse_tools)
    recovery["Lapse_Management"] = lapse_management

    relapse_prevention: dict = {}
    relapse_leaves = [
        "High_Risk_Context_Avoidance_Plan",
        "Stimulus_Control_Relapse_Prevention",
        "Social_Pressure_Boundary_Script",
        "Contingency_Plan_For_Disruption",
        "Temptation_Management_Toolkit",
        "Recovery_Plan_If_Slip",
        "Support_Network_Relapse_Plan",
        "Identity_Protection_Script",
        "Values_Protection_Script",
        "Relapse_Risk_Audit_Checklist",
    ]
    add_leaves(relapse_prevention, ["Relapse_Prevention_Structures"], relapse_leaves)
    recovery["Relapse_Prevention"] = relapse_prevention

    volition["Recovery_Phase"] = recovery

    HAPA["Volition_Phase"] = volition

    # ============================================================
    # 4) BARRIERS (Primary Node; ultra high-resolution; taxonomy only)
    #    NOTE: No links/mappings to coping strategies here.
    # ============================================================
    barriers: dict = {}

    # ----------------------------
    # MOTIVATION / APPRAISAL
    # ----------------------------

    # A) Risk perception barriers (deeper clustering)
    add_leaves(barriers, ["Motivation", "Risk_Perception", "Salience_and_Attention"], [
        "Low_Risk_Salience",
        "Risk_Not_On_Mind",
        "Attention_Captured_By_Other_Concerns",
        "Habituation_To_Risk_Messages",
        "Desensitization_To_Threat",
        "Low_Symptom_Salience",
        "Low_Body_Awareness_Interoception",
    ])

    add_leaves(barriers, ["Motivation", "Risk_Perception", "Interpretation_and_Attribution"], [
        "Misinterpretation_Of_Symptoms",
        "Normalization_Of_Problems",
        "Benign_Attribution_Bias",
        "Externalization_Of_Risk_Cause",
        "Symptom_Attribution_To_Personality",
        "Attribution_To_Temporary_Stress_Only",
        "Attribution_To_Others_Not_Self",
        "Confusion_About_Diagnosis_Or_Label",
    ])

    add_leaves(barriers, ["Motivation", "Risk_Perception", "Cognitive_Biases"], [
        "Unrealistic_Optimism",
        "Comparative_Risk_Discounting",
        "Base_Rate_Neglect",
        "Availability_Heuristic_Risk",
        "Representativeness_Bias_Risk",
        "Confirmation_Bias_Risk",
        "Motivated_Reasoning_Risk",
        "Omission_Bias_Risk",
    ])

    add_leaves(barriers, ["Motivation", "Risk_Perception", "Information_Trust_and_Source"], [
        "Distrust_Of_Information_Source",
        "Authority_Mistrust",
        "Distrust_Due_To_Past_Harm",
        "Conflicting_Advice_Paralysis",
        "Perceived_Manipulation_Of_Messaging",
        "Low_Trust_In_Healthcare_System",
        "Healthcare_Discrimination_Expectation",
    ])

    add_leaves(barriers, ["Motivation", "Risk_Perception", "Information_Processing_and_Literacy"], [
        "Low_Comprehension_Of_Risk_Info",
        "Low_Health_Literacy_Risk",
        "Numeracy_Difficulty_Risk",
        "Risk_Probability_Confusion",
        "Ambiguous_Risk_Message_Confusion",
        "Cognitive_Load_Reduces_Comprehension",
        "Language_Barrier_In_Risk_Info",
    ])

    add_leaves(barriers, ["Motivation", "Risk_Perception", "Threat_Response_and_Tolerance"], [
        "Information_Avoidance",
        "Threat_Denial_Minimization",
        "Threat_Overwhelm_Shutdown",
        "Uncertainty_Intolerance_Risk_Info",
        "Trauma_Triggered_Risk_Shutdown",
        "Health_Anxiety_Amplification",
        "Catastrophic_Risk_Interpretation",
        "Hypervigilance_Threat_Scanning",
    ])

    # B) Outcome expectancies barriers (deeper clustering)
    add_leaves(barriers, ["Motivation", "Outcome_Expectancies", "Benefit_Appraisal"], [
        "Low_Perceived_Benefit",
        "Benefit_Feels_Abstract",
        "Low_Benefit_Immediacy",
        "Low_Social_Benefit_Expectation",
        "Belief_Action_Wont_Matter",
        "Perceived_Ineffectiveness_Belief",
        "Low_Outcome_Certainty",
        "Low_Control_Over_Outcomes",
        "Outcome_Fatalism",
    ])

    add_leaves(barriers, ["Motivation", "Outcome_Expectancies", "Cost_Appraisal"], [
        "High_Perceived_Costs",
        "Time_Cost_Overestimation",
        "Financial_Cost_Overestimation",
        "Effort_Cost_Overestimation",
        "Anticipated_Discomfort_Aversion",
        "Low_Enjoyment_Expectation",
        "Opportunity_Cost_Inflation",
    ])

    add_leaves(barriers, ["Motivation", "Outcome_Expectancies", "Social_and_Status_Consequences"], [
        "Social_Rejection_Expectation",
        "Fear_Of_Judgment",
        "Stigma_Cost_Expectation",
        "Reputation_Concern",
        "Fear_Of_Being_Treated_Differently",
        "Workplace_Stigma",
        "Community_Stigma",
    ])

    add_leaves(barriers, ["Motivation", "Outcome_Expectancies", "Identity_and_Self_Concept_Consequences"], [
        "Identity_Threat_Expectation",
        "Change_Is_Dangerous_Belief",
        "Fear_Of_Success_Change",
        "Fear_Of_Losing_Self_Definitions",
        "Perceived_Inauthenticity_Of_Change",
        "Loss_Of_Coping_Or_Numbing_Fear",
        "Pleasure_Loss_Fear",
    ])

    add_leaves(barriers, ["Motivation", "Outcome_Expectancies", "Side_Effects_and_Safety_Beliefs"], [
        "Side_Effect_Fear",
        "Treatment_Harm_Fear",
        "Medication_Safety_Fear",
        "Fear_Of_Dependence_On_Treatment",
        "Fear_Of_Withdrawal_Or_Symptom_Rebound",
        "Fear_Of_Therapy_Dredging_Up_Pain",
    ])

    add_leaves(barriers, ["Motivation", "Outcome_Expectancies", "Temporal_Discounting_and_Present_Bias"], [
        "Delayed_Consequences_Discounting",
        "Immediate_Reward_Bias",
        "Short_Term_Mood_Over_Goals",
        "Preference_For_Quick_Fixes",
        "Low_Tolerance_For_Delayed_Gains",
    ])

    # C) Values & identity barriers (deeper clustering)
    add_leaves(barriers, ["Motivation", "Values_Identity", "Values_Clarity_and_Priorities"], [
        "Low_Value_Clarity",
        "Conflicting_Values",
        "Low_Priority_Assigned_To_Health",
        "Values_Not_Linked_To_Action",
        "Competing_Values_Unresolved",
    ])

    add_leaves(barriers, ["Motivation", "Values_Identity", "Identity_Coherence_and_Alignment"], [
        "Identity_Misalignment",
        "Rigid_Self_Story",
        "Identity_Foreclosure",
        "Fear_Of_Identity_Change",
        "Role_Conflict",
        "Role_Identity_Entrenchment",
        "Perceived_Incompatibility_With_Self_Image",
    ])

    add_leaves(barriers, ["Motivation", "Values_Identity", "Autonomy_Control_and_Reactance"], [
        "Low_Autonomy_Sense",
        "Reactance_To_Pressure",
        "Perceived_Control_By_Others",
        "Autonomy_Threat_From_Recommendations",
        "Resistance_To_Being_Told_What_To_Do",
    ])

    add_leaves(barriers, ["Motivation", "Values_Identity", "Stigma_Shame_and_Self_Worth"], [
        "Internalized_Stigma_Shame",
        "Low_Sense_Of_Deservingness",
        "Low_Self_Worth_Beliefs",
        "Perceived_Unlovability_Belief",
        "Self_Punishment_Motive",
        "Moral_Injury_Or_Guilt_Block",
        "Fear_Of_Disclosure_Shame",
    ])

    add_leaves(barriers, ["Motivation", "Values_Identity", "Cultural_and_Role_Norms"], [
        "Cultural_Value_Misalignment",
        "Masculinity_Femininity_Norm_Conflict",
        "Family_Culture_Opposition",
        "Values_Conflict_With_Social_Group",
        "Norms_Against_Disclosure",
    ])

    # D) Self-efficacy barriers (split into capability domains)
    add_leaves(barriers, ["Motivation", "Self_Efficacy", "Task_And_Skill_Self_Efficacy"], [
        "Low_Task_Self_Efficacy",
        "Low_Confidence_In_Starting",
        "Perceived_Lack_Of_Skills",
        "Perceived_Lack_Of_Knowledge",
        "Low_Problem_Solving_Confidence",
        "Fear_Of_Failure",
        "Past_Failure_Overgeneralization",
        "Learned_Helplessness_Schema",
    ])

    add_leaves(barriers, ["Motivation", "Self_Efficacy", "Coping_Self_Efficacy"], [
        "Low_Emotion_Regulation_Self_Efficacy",
        "Low_Distress_Tolerance_Self_Efficacy",
        "Fear_Of_Symptoms_During_Action",
        "Fear_Of_Panic_During_Action",
        "Low_Confidence_In_Managing_Cravings",
        "Low_Confidence_In_Handling_Setbacks",
    ])

    add_leaves(barriers, ["Motivation", "Self_Efficacy", "Social_And_Communication_Self_Efficacy"], [
        "Low_Social_Self_Efficacy",
        "Low_Communication_Self_Efficacy",
        "Boundary_Setting_Difficulty",
        "Communication_Difficulty_Requesting_Support",
        "Fear_Of_Advocating_For_Needs",
    ])

    add_leaves(barriers, ["Motivation", "Self_Efficacy", "Executive_Function_Self_Efficacy"], [
        "Low_Executive_Function_Confidence",
        "Low_Control_Beliefs",
        "Low_Confidence_In_Planning",
        "Low_Confidence_In_FollowThrough",
        "Low_Confidence_In_Adjusting_Plan",
    ])

    add_leaves(barriers, ["Motivation", "Self_Efficacy", "Maintenance_And_Recovery_Self_Efficacy"], [
        "Low_Confidence_In_Consistency",
        "Low_Maintenance_Self_Efficacy",
        "Low_Recovery_Self_Efficacy",
        "Low_Confidence_In_Recovery_After_Slip",
        "Overconfidence_Risk",
    ])

    # E) Intention / commitment barriers (deeper clustering)
    add_leaves(barriers, ["Motivation", "Ambivalence_Commitment", "Ambivalence_and_ApproachAvoidance"], [
        "Ambivalence",
        "Approach_Avoidance_Conflict",
        "Mixed_Motives_Conflict",
        "Fear_Of_Change_Ambivalence",
    ])

    add_leaves(barriers, ["Motivation", "Ambivalence_Commitment", "Commitment_Stability_and_Readiness"], [
        "Low_Commitment",
        "Low_Readiness_To_Change",
        "Preference_Instability",
        "Commitment_Discounting_Over_Time",
        "Conditional_Commitment_Only",
    ])

    add_leaves(barriers, ["Motivation", "Ambivalence_Commitment", "Goal_Competition_and_Substitution"], [
        "Competing_Goals_Salient",
        "Goal_Substitution_Risk",
        "Overchoice_Too_Many_Options",
        "Competing_Identity_Goals",
    ])

    add_leaves(barriers, ["Motivation", "Ambivalence_Commitment", "Decision_Process_Barriers"], [
        "Decision_Paralysis",
        "Fear_Of_Regret_Paralysis",
        "Choice_Overload_Analysis_Paralysis",
        "Low_Decision_Confidence",
    ])

    # F) Norms / beliefs / misinformation (deeper clustering)
    add_leaves(barriers, ["Motivation", "Social_Norms_And_Beliefs", "Normative_Pressure_and_Conformity"], [
        "Perceived_Norm_Against_Change",
        "Fear_Of_Social_Deviation",
        "Peer_Group_Reinforces_Risk",
        "Norms_Against_Help_Seeking",
    ])

    add_leaves(barriers, ["Motivation", "Social_Norms_And_Beliefs", "Stigma_Climate"], [
        "Workplace_Stigma",
        "Community_Stigma",
        "Family_Stigma",
        "Healthcare_Discrimination_Expectation",
    ])

    add_leaves(barriers, ["Motivation", "Social_Norms_And_Beliefs", "Belief_Ecosystem_and_Misinformation"], [
        "Misinformation_Exposure",
        "Myths_About_Mental_Health",
        "Anti_Treatment_Belief_System",
        "Conspiracy_Beliefs_About_Care",
        "Conflicting_Advice_Paralysis",
    ])

    # G) Knowledge / psychoeducation barriers (deeper clustering)
    add_leaves(barriers, ["Motivation", "Knowledge_And_Psychoeducation", "Knowledge_Of_Action_and_HowTo"], [
        "Low_Knowledge_Of_What_To_Do",
        "Low_Knowledge_Of_How_To_Start",
        "Low_Knowledge_Of_How_To_Practice_Skills",
        "Low_Knowledge_Of_Safe_Exposure_Steps",
    ])

    add_leaves(barriers, ["Motivation", "Knowledge_And_Psychoeducation", "Symptom_Recognition_and_Labeling"], [
        "Low_Recognition_Of_Warning_Signs",
        "Low_Emotion_Labeling_Ability",
        "Confusion_Between_Stress_And_Disorder",
        "Misattribution_Of_Cause",
    ])

    add_leaves(barriers, ["Motivation", "Knowledge_And_Psychoeducation", "Resource_Knowledge_and_Navigation"], [
        "Low_Knowledge_Of_Resources",
        "Low_Knowledge_Of_Crisis_Options",
        "Unaware_Of_Eligibility_Or_Pathways",
        "Unaware_Of_Low_Cost_Options",
    ])

    # ----------------------------
    # VOLITION / SELF-REGULATION CAPACITY
    # ----------------------------

    # A) Planning barriers (deeper clustering)
    add_leaves(barriers, ["Volition", "Planning", "Goal_Definition_Quality"], [
        "Vague_Intention",
        "Low_Goal_Clarity",
        "Overambitious_Goal_Setting",
        "Unrealistic_Timeline",
        "Unrealistic_Frequency_Target",
        "Ambiguous_Success_Criteria",
    ])

    add_leaves(barriers, ["Volition", "Planning", "Action_Specification_and_Steps"], [
        "No_Action_Plan",
        "Poor_Task_Sequencing",
        "Unclear_Next_Action",
        "Too_Many_Steps_Before_Start",
        "Plan_Not_Actionable",
    ])

    add_leaves(barriers, ["Volition", "Planning", "Cue_And_Context_Design"], [
        "No_Cue_Design",
        "No_Routine_Anchor",
        "Cue_Conflict_Competing_Routines",
        "Start_Signal_Not_Noticed",
        "Context_Unstable_For_Action",
    ])

    add_leaves(barriers, ["Volition", "Planning", "Resource_And_Tool_Planning"], [
        "No_Resource_Plan",
        "Low_Access_To_Tools",
        "Tool_Unavailability",
        "Resource_Constraints",
        "Missing_Required_Materials",
    ])

    add_leaves(barriers, ["Volition", "Planning", "Contingency_And_Barrier_Planning"], [
        "No_Contingency_Plan",
        "Underestimated_Barriers",
        "Low_Contingency_Specificity",
        "No_Recovery_Plan_For_Slip",
    ])

    add_leaves(barriers, ["Volition", "Planning", "Tracking_And_Feedback_Planning"], [
        "No_Measurement_Plan",
        "No_Feedback_Loop",
        "No_Progress_Review_Routine",
        "Tracking_Tool_Not_Set_Up",
    ])

    add_leaves(barriers, ["Volition", "Planning", "Complexity_Overplanning_and_Ownership"], [
        "Overly_Complex_Plan",
        "Excessive_Planning_No_Action",
        "Planning_Avoidance",
        "Low_Plan_Ownership",
        "No_Prioritization",
    ])

    # B) Time / organization barriers (deeper clustering)
    add_leaves(barriers, ["Volition", "Time_And_Organization", "Scheduling_and_Calendar"], [
        "Calendar_Overload",
        "No_Scheduling_Block",
        "Time_Pressure",
        "Shift_Work_Disruption",
        "Unpredictable_Routine",
    ])

    add_leaves(barriers, ["Volition", "Time_And_Organization", "Time_Estimation_and_Pacing"], [
        "Poor_Time_Estimation",
        "Underestimates_Task_Duration",
        "Overestimates_Task_Duration",
        "No_Breaks_Planned",
    ])

    add_leaves(barriers, ["Volition", "Time_And_Organization", "Organization_Systems_and_Externalization"], [
        "Cluttered_Task_System",
        "Lost_Notes_Or_Plans",
        "Poor_Reminder_Use",
        "No_Single_Source_Of_Truth_System",
    ])

    add_leaves(barriers, ["Volition", "Time_And_Organization", "Decision_Fatigue_and_Choice_Load"], [
        "Decision_Fatigue_Risk",
        "Too_Many_Daily_Decisions",
        "Overchoice_Too_Many_Options",
    ])

    # C) Action initiation barriers (deeper clustering)
    add_leaves(barriers, ["Volition", "Action_Initiation", "Memory_and_Reminder_Failures"], [
        "Forgetfulness",
        "Intention_Forgets_In_Moment",
        "Reminder_Ignored_Habitually",
        "Prospective_Memory_Failure",
    ])

    add_leaves(barriers, ["Volition", "Action_Initiation", "Activation_Energy_and_Fatigue"], [
        "Low_Energy_Or_Fatigue",
        "Low_Momentum_Inertia",
        "Psychomotor_Retardation",
        "Sleep_Deprivation_Impairment",
        "Pain_Or_Discomfort_Interference",
    ])

    add_leaves(barriers, ["Volition", "Action_Initiation", "Avoidance_and_Procrastination"], [
        "Avoidance_Procrastination",
        "Task_Aversion",
        "Boredom_Intolerance",
        "Initiation_Anxiety",
        "Planning_As_Avoidance",
    ])

    add_leaves(barriers, ["Volition", "Action_Initiation", "Affective_Blocks_Anxiety_Shame"], [
        "Anxiety_Anticipation_Block",
        "Shame_Anticipation_Block",
        "Fear_Of_Evaluative_Outcomes",
        "Fear_Of_Judgment",
        "Negative_Mood_Interference",
    ])

    add_leaves(barriers, ["Volition", "Action_Initiation", "Perfectionism_and_Overthinking"], [
        "Perfectionism_AllOrNothing",
        "Overthinking_Before_Starting",
        "AllOrNothing_Commitment_Style",
        "Fear_Of_Failure",
    ])

    add_leaves(barriers, ["Volition", "Action_Initiation", "Executive_Initiation_Dysfunction"], [
        "Executive_Dysfunction_Initiation",
        "ADHD_Inattention_Initiation",
        "Unclear_Starting_Point",
        "Start_Requires_Too_Many_Steps",
    ])

    # D) Action control barriers (deeper clustering)
    add_leaves(barriers, ["Volition", "Action_Control", "Self_Monitoring_Deficits"], [
        "Low_Self_Monitoring",
        "Low_Awareness_Of_Standards",
        "Poor_Progress_Awareness",
        "Avoiding_Feedback_Data",
    ])

    add_leaves(barriers, ["Volition", "Action_Control", "Inhibitory_Control_and_Urges"], [
        "Low_Inhibitory_Control",
        "Urge_Driven_Choice",
        "Craving_Driven_Interference",
        "Immediate_Reward_Bias",
        "Impulsivity_Under_Stress",
    ])

    add_leaves(barriers, ["Volition", "Action_Control", "Attentional_Control_and_Distraction"], [
        "Attentional_Capture_Distraction",
        "Low_Focus_Sustenance",
        "Mind_Wandering_Interference",
        "Multitasking_Habit",
        "Environmental_Distraction_High",
    ])

    add_leaves(barriers, ["Volition", "Action_Control", "Emotion_Dysregulation_In_Moment"], [
        "Emotion_Dysregulation_In_Moment",
        "Low_Distress_Tolerance",
        "Stress_Overload",
        "Hyperarousal_Interference",
        "Sensory_Overload_Interference",
    ])

    add_leaves(barriers, ["Volition", "Action_Control", "Cognitive_Interference_Rumination_Worry"], [
        "Rumination_During_Action",
        "Worry_Interference",
        "Obsessive_Checking_Interference",
        "Compulsive_Ritual_Interference",
        "Cognitive_Overload",
        "Working_Memory_Limits",
    ])

    add_leaves(barriers, ["Volition", "Action_Control", "Persistence_and_Tolerance_of_Difficulty"], [
        "Low_Persistence_When_Difficult",
        "Low_Distress_Tolerance",
        "Stops_When_Not_Perfect",
        "Frustration_Intolerance",
    ])

    add_leaves(barriers, ["Volition", "Action_Control", "Error_Correction_and_Adjustment"], [
        "Poor_Error_Detection",
        "Slow_Recalibration_After_Deviation",
        "Low_Confidence_In_Adjusting_Plan",
        "Rigidity_Cant_Adapt",
    ])

    # E) Digital distraction barriers (deeper clustering)
    add_leaves(barriers, ["Volition", "Attention_And_Digital_Distraction", "Device_And_Notification_Load"], [
        "Problematic_Smartphone_Use",
        "Notification_Interruptions",
        "Always_On_Responsiveness_Pressure",
    ])

    add_leaves(barriers, ["Volition", "Attention_And_Digital_Distraction", "Content_Loops_And_Information_Overload"], [
        "Social_Media_Scroll_Loop",
        "Doomscrolling_Anxiety_Loop",
        "Information_Overload",
        "News_Anxiety_Consumption",
    ])

    # F) Symptom interference barriers (deep clinical/non-clinical coverage)
    add_leaves(barriers, ["Volition", "Symptom_Interference", "Depressive_Anhedonic_Apathy"], [
        "Anhedonia_Reduces_Initiation",
        "Apathy_Or_Amotivation",
        "Hopelessness",
        "Low_Dopamine_Reward_Sensitivity",
    ])

    add_leaves(barriers, ["Volition", "Symptom_Interference", "Anxiety_Panic_OCD"], [
        "Panic_Attacks_Interference",
        "Generalized_Anxiety_Interference",
        "Health_Anxiety_Amplification",
        "Intrusive_Thoughts_Interference",
        "Obsessive_Checking_Interference",
    ])

    add_leaves(barriers, ["Volition", "Symptom_Interference", "Trauma_Dissociation_Hyperarousal"], [
        "PTSD_Flashbacks_Interference",
        "Dissociation_Interference",
        "Dissociation_Or_Numbing_Block",
        "Hyperarousal_Interference",
        "Trauma_Triggered_Risk_Shutdown",
    ])

    add_leaves(barriers, ["Volition", "Symptom_Interference", "Psychosis_Mania_Impulsivity"], [
        "Psychosis_Suspiciousness_Interference",
        "Mania_Hypomania_Impulsivity",
        "Irritability_Impulsivity",
        "Poor_Insight_Into_Need_For_Action",
    ])

    add_leaves(barriers, ["Volition", "Symptom_Interference", "Neurocognitive_Executive_Attention"], [
        "Cognitive_Fog_Interference",
        "Working_Memory_Limits",
        "Executive_Dysfunction_Initiation",
        "Low_Attention_Control",
    ])

    add_leaves(barriers, ["Volition", "Symptom_Interference", "Sleep_Circadian_Fatigue"], [
        "Sleep_Deprivation_Impairment",
        "Unstable_Sleep_Wake_Cycle",
        "Insomnia_Interference",
        "Nightmares_Interference",
    ])

    add_leaves(barriers, ["Volition", "Symptom_Interference", "Pain_Somatic_Medication_Effects"], [
        "Pain_Or_Discomfort_Interference",
        "Medication_Sedation_Interference",
        "Medication_Side_Effects_Interference",
        "Somatic_Symptom_Burden_Interference",
    ])

    add_leaves(barriers, ["Volition", "Symptom_Interference", "Substance_Use_Withdrawal_Craving"], [
        "Substance_Withdrawal_Interference",
        "Craving_Driven_Interference",
        "Substance_Use_Escalation",
    ])

    # ----------------------------
    # CONTEXT (SOCIAL / ENVIRONMENT / SYSTEMS)
    # ----------------------------

    # A) Social/interpersonal barriers (deeper clustering)
    add_leaves(barriers, ["Context", "Social_Interpersonal", "Support_Availability_and_Quality"], [
        "Low_Support_Availability",
        "Unsupportive_Social_Environment",
        "Support_Dropoff",
        "Loss_Of_Key_Support",
        "Lack_Of_Role_Models",
    ])

    add_leaves(barriers, ["Context", "Social_Interpersonal", "Social_Pressure_and_Sabotage"], [
        "Social_Pressure",
        "Peer_Reinforcement_Of_Maladaptive_Coping",
        "Pressure_To_Maintain_Status_Quo",
        "Social_Invitations_Compete_With_Action",
    ])

    add_leaves(barriers, ["Context", "Social_Interpersonal", "Relationship_Conflict_and_Violence"], [
        "Conflict_With_Close_Others",
        "Relationship_Instability",
        "Domestic_Conflict_Or_Violence",
        "Caretaking_Burden",
    ])

    add_leaves(barriers, ["Context", "Social_Interpersonal", "Communication_and_Assertiveness"], [
        "Communication_Difficulty_Requesting_Support",
        "Boundary_Setting_Difficulty",
        "Low_Communication_Self_Efficacy",
        "Fear_Of_Advocating_For_Needs",
    ])

    add_leaves(barriers, ["Context", "Social_Interpersonal", "Isolation_and_Belonging"], [
        "Isolation_Loneliness",
        "Group_Belongingness_Low",
        "Social_Withdrawal_After_Slip",
        "Limited_Community_Integration",
    ])

    add_leaves(barriers, ["Context", "Social_Interpersonal", "Attachment_Trust_and_Alliance"], [
        "Mistrust_And_Attachment_Fear",
        "Therapeutic_Alliance_Difficulty",
        "Fear_Of_Disclosure_Shame",
    ])

    # B) Environmental/logistics barriers (deeper clustering)
    add_leaves(barriers, ["Context", "Environmental_Logistics", "Time_Pressure_and_Role_Demands"], [
        "Time_Pressure",
        "Conflicting_Role_Demands",
        "Work_Academic_Overload",
        "Caregiving_Schedule_Disruption",
        "Shift_Work_Disruption",
    ])

    add_leaves(barriers, ["Context", "Environmental_Logistics", "Financial_and_Resource_Constraints"], [
        "Resource_Constraints",
        "Financial_Barriers_To_Care",
        "Food_Insecurity",
        "Housing_Instability",
        "Tool_Unavailability",
    ])

    add_leaves(barriers, ["Context", "Environmental_Logistics", "Access_Transport_and_Geography"], [
        "Transport_And_Access",
        "Geographic_Access_Barrier",
        "Travel_Disruption",
        "Mobility_Limitations_Access",
    ])

    add_leaves(barriers, ["Context", "Environmental_Logistics", "Home_Environment_Privacy_Safety"], [
        "Lack_Of_Safe_Space",
        "Privacy_Lack",
        "Unsafe_Home_Environment",
        "Noise_Crowding_Stress",
        "High_Stimulus_Environment",
    ])

    add_leaves(barriers, ["Context", "Environmental_Logistics", "Routine_Unpredictability_and_Disruption"], [
        "Unpredictable_Routine",
        "Life_Event_Disruption",
        "Weather_Or_External_Disruption",
        "Context_Change_Disruption",
    ])

    add_leaves(barriers, ["Context", "Environmental_Logistics", "Digital_Infrastructure_and_Accessibility"], [
        "Digital_Access_Barrier",
        "Device_Or_Internet_Unreliable",
        "Low_Access_To_Tools",
        "Accessibility_Barriers_Interfaces",
    ])

    add_leaves(barriers, ["Context", "Environmental_Logistics", "Administrative_and_Bureaucracy"], [
        "Administrative_Bureaucracy_Barrier",
        "Complex_Forms_And_Paperwork",
        "Navigation_Confusion_Services",
    ])

    # C) Healthcare/services access barriers (deeper clustering)
    add_leaves(barriers, ["Context", "Healthcare_And_Services_Access", "Availability_and_Waitlists"], [
        "No_Provider_Available",
        "Long_Waitlists",
        "Appointment_Scheduling_Difficulty",
        "Limited_AfterHours_Access",
    ])

    add_leaves(barriers, ["Context", "Healthcare_And_Services_Access", "Cost_Insurance_and_Coverage"], [
        "Cost_Of_Care",
        "Insurance_Coverage_Gaps",
        "High_Out_Of_Pocket_Costs",
        "Prior_Authorization_Barrier",
    ])

    add_leaves(barriers, ["Context", "Healthcare_And_Services_Access", "Continuity_and_Coordination"], [
        "Low_Continuity_Of_Care",
        "Fragmented_Services",
        "Poor_Care_Coordination",
        "Conflicting_Provider_Guidance",
    ])

    add_leaves(barriers, ["Context", "Healthcare_And_Services_Access", "Language_Cultural_and_Discrimination"], [
        "Language_Barrier_In_Care",
        "Cultural_Incompetence_In_Care",
        "Discrimination_In_Care",
        "Low_Cultural_Safety",
    ])

    add_leaves(barriers, ["Context", "Healthcare_And_Services_Access", "Trust_Privacy_and_Records_Concerns"], [
        "Confidentiality_Concerns",
        "Fear_Of_Diagnosis_Or_Records",
        "Low_Trust_In_Healthcare_System",
        "Authority_Mistrust",
    ])

    add_leaves(barriers, ["Context", "Healthcare_And_Services_Access", "Medication_Access_and_Adherence_System"], [
        "Medication_Access_Issues",
        "Pharmacy_Access_Barrier",
        "Prescription_Refill_Barriers",
        "Medication_Complexity_Regimen",
    ])

    # D) Cultural/structural barriers (deeper clustering)
    add_leaves(barriers, ["Context", "Cultural_Structural", "Stigma_Norms_and_Disclosure"], [
        "Cultural_Stigma",
        "Norms_Against_Disclosure",
        "Norms_Against_Help_Seeking",
        "Religious_Or_Cultural_Explanations_Conflict",
    ])

    add_leaves(barriers, ["Context", "Cultural_Structural", "Minority_Stress_and_Discrimination"], [
        "Minority_Stress_Chronic",
        "Acculturation_Stress",
        "Workplace_Harassment",
        "Community_Violence_Exposure",
    ])

    add_leaves(barriers, ["Context", "Cultural_Structural", "Legal_Status_and_Policy"], [
        "Legal_Status_Stress",
        "Fear_Of_Authority_Contact",
        "Policy_Barriers_To_Access",
    ])

    add_leaves(barriers, ["Context", "Cultural_Structural", "Workplace_and_Academic_Policy"], [
        "Workplace_Policy_Barriers",
        "Academic_Policy_Barriers",
        "Lack_Of_Accommodations",
        "Punitive_Attendance_Policies",
    ])

    add_leaves(barriers, ["Context", "Cultural_Structural", "Socioeconomic_Strain"], [
        "Socioeconomic_Strain",
        "Job_Insecurity",
        "Debt_Stress",
        "Unstable_Living_Conditions",
    ])

    # E) Safety/trauma exposure barriers (deeper clustering)
    add_leaves(barriers, ["Context", "Safety_And_Trauma_Exposure", "Ongoing_Threat_and_Unsafe_Settings"], [
        "Ongoing_Trauma_Exposure",
        "Unsafe_Home_Environment",
        "Unpredictable_Threat_Environment",
        "Workplace_Harassment",
    ])

    add_leaves(barriers, ["Context", "Safety_And_Trauma_Exposure", "Trigger_Rich_Environments"], [
        "Trigger_Rich_Environment",
        "Community_Violence_Exposure",
        "Frequent_Exposure_To_Triggers",
    ])

    add_leaves(barriers, ["Context", "Safety_And_Trauma_Exposure", "Sleep_Safety_and_Nighttime_Threat"], [
        "Sleep_Environment_Unsafe",
        "Nighttime_Hypervigilance",
    ])

    # ----------------------------
    # MAINTENANCE
    # ----------------------------

    add_leaves(barriers, ["Maintenance", "Sustaining_Change", "Motivation_Drift_and_Meaning_Loss"], [
        "Loss_Of_Meaning",
        "Decreased_Salience_Of_Goal",
        "Identity_Drift",
        "Low_Priority_Creep",
    ])

    add_leaves(barriers, ["Maintenance", "Sustaining_Change", "Reward_Habituation_and_Boredom"], [
        "Reward_Habituation",
        "Boredom_And_Novelty_Seeking",
        "Low_Reward_Salience",
        "Routine_Saturation",
    ])

    add_leaves(barriers, ["Maintenance", "Sustaining_Change", "Support_and_Accountability_Dropoff"], [
        "Support_Dropoff",
        "Accountability_Dropoff",
        "Loss_Of_Key_Support",
    ])

    add_leaves(barriers, ["Maintenance", "Sustaining_Change", "Burnout_Overload_and_Monitoring_Fatigue"], [
        "Burnout_From_Effort",
        "Monitoring_Fatigue",
        "Compassion_Fatigue_Self",
        "Stress_Overload",
    ])

    add_leaves(barriers, ["Maintenance", "Sustaining_Change", "Context_Transitions_and_Generalization_Failures"], [
        "Context_Change_Disruption",
        "Life_Event_Disruption",
        "Seasonal_Mood_Disruption",
        "Generalization_Failure_New_Context",
    ])

    add_leaves(barriers, ["Maintenance", "Sustaining_Change", "Standards_Creep_and_Perfectionism"], [
        "Escalating_Standards_Perfectionism",
        "AllOrNothing_Thinking",
        "Low_Tolerance_For_Imperfection",
    ])

    add_leaves(barriers, ["Maintenance", "Habit_Formation", "Cue_Association_and_Automaticity"], [
        "Weak_Cue_Association",
        "Cue_Unreliable",
        "Skills_Not_Automated",
    ])

    add_leaves(barriers, ["Maintenance", "Habit_Formation", "Competing_Habits_and_Interference"], [
        "Competing_Habits_Strong",
        "Competing_Routines_Strong",
        "Environmental_Cues_For_Old_Habit",
    ])

    add_leaves(barriers, ["Maintenance", "Habit_Formation", "Context_Consistency_and_Routine_Stability"], [
        "Inconsistent_Context",
        "Low_Routine_Stability",
        "Unpredictable_Routine",
    ])

    add_leaves(barriers, ["Maintenance", "Habit_Formation", "Reward_Timing_and_Salience"], [
        "Low_Reward_Salience",
        "Reward_Delayed_Too_Long",
        "Reward_Not_Noticeable",
    ])

    add_leaves(barriers, ["Maintenance", "Habit_Formation", "SleepWake_Stability"], [
        "Unstable_Sleep_Wake_Cycle",
        "Sleep_Deprivation_Impairment",
    ])

    add_leaves(barriers, ["Maintenance", "Skill_Consolidation", "Generalization_and_Transfer"], [
        "Skills_Not_Generalized",
        "Generalization_Failure_New_Context",
        "Context_Dependency_Too_Narrow",
    ])

    add_leaves(barriers, ["Maintenance", "Skill_Consolidation", "Refresh_Practice_and_Boosters"], [
        "No_Refresh_Practice",
        "Loss_Of_Scaffolding",
        "No_Booster_Sessions",
    ])

    add_leaves(barriers, ["Maintenance", "Skill_Consolidation", "Feedback_and_Self_Correction"], [
        "Inconsistent_Coaching_Feedback",
        "Low_Confidence_In_Self_Correction",
        "Reduced_Feedback_Visibility",
    ])

    # ----------------------------
    # RECOVERY / RELAPSE
    # ----------------------------

    add_leaves(barriers, ["Recovery", "After_Setback", "Shame_Self_Criticism_and_Self_Attack"], [
        "Shame_After_Slip",
        "Self_Blame_Attacks",
        "Punitive_Self_Talk",
        "Self_Punishment_Motive",
    ])

    add_leaves(barriers, ["Recovery", "After_Setback", "Catastrophizing_Hopelessness_and_Globalizing"], [
        "Catastrophizing_After_Slip",
        "Hopelessness",
        "Setback_Attribution_Internal_Stable_Global",
        "AllOrNothing_Thinking",
    ])

    add_leaves(barriers, ["Recovery", "After_Setback", "Rumination_Worry_and_Cognitive_Stuckness"], [
        "Rumination",
        "Worry_Interference",
        "Replay_And_Regret_Loops",
    ])

    add_leaves(barriers, ["Recovery", "After_Setback", "Avoidance_Disengagement_and_Withdrawal"], [
        "Avoidance_After_Setback",
        "Social_Withdrawal_After_Slip",
        "Avoiding_Measurement_After_Slip",
        "Escalation_Of_Coping_By_Avoidance",
    ])

    add_leaves(barriers, ["Recovery", "After_Setback", "Rebound_And_Overcorrection"], [
        "Rebound_Overcorrection",
        "Compensatory_Overcontrol",
        "Overcommitment_After_Slip",
    ])

    add_leaves(barriers, ["Recovery", "Reengagement_Blocks", "Restart_Cues_and_Repair_Rituals_Missing"], [
        "Delay_In_Restart",
        "No_Restart_Cue",
        "No_Repair_Ritual",
        "Loss_Of_Structure_After_Slip",
    ])

    add_leaves(barriers, ["Recovery", "Reengagement_Blocks", "Replanning_and_Learning_Avoidance"], [
        "Replan_Avoidance",
        "Barrier_Update_Avoidance",
        "Belief_Setback_Proves_Incapacity",
    ])

    add_leaves(barriers, ["Recovery", "Reengagement_Blocks", "Fear_Of_Reexperiencing_Difficulty"], [
        "Fear_Of_Reexperiencing_Difficulty",
        "Avoids_Triggering_Context_Entirely",
    ])

    add_leaves(barriers, ["Recovery", "Reengagement_Blocks", "Support_Recontact_Failure"], [
        "Support_Not_Recontacted",
        "Fear_Of_Telling_Support",
    ])

    add_leaves(barriers, ["Recovery", "Crisis_Escalation_Risk", "Overwhelm_Panic_and_Dissociation_Spirals"], [
        "Overwhelm_Spiral",
        "Panic_Spiral",
        "Dissociation_Spiral",
    ])

    add_leaves(barriers, ["Recovery", "Crisis_Escalation_Risk", "Substance_Escalation_and_Impulse_Risk"], [
        "Substance_Use_Escalation",
        "Craving_Escalation",
        "Risk_Taking_Escalation",
    ])

    add_leaves(barriers, ["Recovery", "Crisis_Escalation_Risk", "Self_Harm_Risk_and_HelpSeeking_Inhibition"], [
        "Self_Harm_Urges_Escalation",
        "Help_Seeking_Inhibition",
        "Isolation_Escalation",
    ])

    add_leaves(barriers, ["Recovery", "Crisis_Escalation_Risk", "Sleep_Collapse_and_Functional_Deterioration"], [
        "Sleep_Collapse",
        "Functional_Collapse_Activities_Of_Daily_Living",
    ])

    HAPA["BARRIERS"] = barriers

    # ============================================================
    # 5) COPING_STRATEGIES (Primary Node; ultra high-resolution; taxonomy only)
    #    NOTE: No links/mappings to barriers here.
    # ============================================================
    coping: dict = {}

    # ----------------------------
    # A) ASSESSMENT / FORMULATION TOOLS
    # ----------------------------

    add_leaves(coping, ["Assessment_and_Formulation", "Functional_Analysis", "Trigger_and_Antecedent_Mapping"], [
        "Trigger_And_Context_Mapping",
        "Trigger_Chain_Analysis",
        "High_Risk_Situations_Map",
        "Context_Tagging_Log",
        "Early_Warning_Signs_Checklist",
        "Vulnerability_Factors_Checklist",
    ])

    add_leaves(coping, ["Assessment_and_Formulation", "Functional_Analysis", "Behavior_Consequence_and_Reinforcement"],
               [
                   "Functional_Analysis_ABC_Worksheet",
                   "Reinforcement_Contingency_Map",
                   "Slip_Pattern_Review",
                   "Maintaining_Factors_Map",
                   "Avoidance_Function_Map",
                   "Substance_Use_Function_Map",
               ])

    add_leaves(coping, ["Assessment_and_Formulation", "Functional_Analysis", "Emotion_Cognition_Behavior_Links"], [
        "Emotion_Behavior_Link_Map",
        "Sleep_Stress_Symptom_Map",
        "Rumination_And_Worry_Map",
        "Somatic_Symptom_Context_Map",
    ])

    add_leaves(coping, ["Assessment_and_Formulation", "Functional_Analysis", "Interpersonal_and_Systemic_Cycles"], [
        "Interpersonal_Cycle_Map",
        "Conflict_Pattern_Review",
        "Support_System_Map",
        "Social_Trigger_Mapping",
    ])

    add_leaves(coping, ["Assessment_and_Formulation", "Self_Insight_Prompts", "Values_Identity_and_Purpose"], [
        "Values_Clarification_Deep_Dive",
        "Identity_Alignment_Check",
        "Meaning_And_Purpose_Prompt",
        "Values_And_Needs_Mapping",
        "Legacy_Prompt",
    ])

    add_leaves(coping, ["Assessment_and_Formulation", "Self_Insight_Prompts", "Ambivalence_and_Decision_Processes"], [
        "Decisional_Balance_Pros_Cons_Map",
        "Opportunity_Cost_Reflection",
        "Preference_Stability_Check",
        "Regret_Minimization_Reflection",
        "Goal_Conflict_Resolution_Prompt",
    ])

    add_leaves(coping, ["Assessment_and_Formulation", "Self_Insight_Prompts", "Beliefs_Stigma_and_Control"], [
        "Stigma_Beliefs_Audit",
        "Autonomy_And_Control_Reflection",
        "Narrative_Beliefs_Inventory",
        "Self_Efficacy_Belief_Audit",
    ])

    add_leaves(coping, ["Assessment_and_Formulation", "Self_Insight_Prompts", "Strengths_Resources_and_Past_Success"], [
        "Strengths_And_Past_Success_Recall",
        "Protective_Factors_Inventory",
        "Coping_Skills_Inventory",
        "Resource_Map_Personal",
    ])

    add_leaves(coping, ["Assessment_and_Formulation", "Monitoring_and_Tracking", "Symptom_Tracking"], [
        "Mood_Tracking_Daily_CheckIn",
        "Anxiety_Tracking_SUDS",
        "Panic_Attack_Log",
        "Craving_Log",
        "Intrusion_Flashback_Log",
        "Irritability_Stress_Log",
    ])

    add_leaves(coping,
               ["Assessment_and_Formulation", "Monitoring_and_Tracking", "Sleep_Energy_and_Physiology_Tracking"], [
                   "Sleep_Log_Simple",
                   "Energy_And_Fatigue_Log",
                   "Pain_Somatic_Symptom_Log",
                   "Caffeine_Alcohol_Intake_Log",
               ])

    add_leaves(coping, ["Assessment_and_Formulation", "Monitoring_and_Tracking", "Behavior_and_Habit_Tracking"], [
        "Behavior_Frequency_Log",
        "Habit_Completion_Checklist",
        "Exposure_Practice_Log",
        "Skills_Practice_Log",
        "Protective_Actions_Log",
    ])

    add_leaves(coping, ["Assessment_and_Formulation", "Monitoring_and_Tracking", "Adherence_and_Care_Tracking"], [
        "Medication_Adherence_Log",
        "Appointment_Attendance_Log",
        "Therapy_Homework_Completion_Log",
    ])

    add_leaves(coping, ["Assessment_and_Formulation", "Risk_and_Safety_Assessment", "Crisis_Risk_Screening"], [
        "Crisis_Risk_Screening_Checklist",
        "Means_Safety_Audit",
        "Risk_Factors_And_Protective_Factors_Check",
    ])

    add_leaves(coping, ["Assessment_and_Formulation", "Risk_and_Safety_Assessment", "Safety_Planning"], [
        "Safety_Contacts_List_Build",
        "Early_Warning_Signs_Safety_Map",
        "Coping_Card_For_Crisis",
        "Share_Safety_Plan_With_Support",
    ])

    add_leaves(coping, ["Assessment_and_Formulation", "Risk_and_Safety_Assessment", "Care_Resources_and_Consent"], [
        "Professional_Resources_Map",
        "Support_Permissions_And_Consent_Plan",
        "Confidentiality_Preferences_Plan",
    ])

    add_leaves(coping, ["Assessment_and_Formulation", "Clarification_and_Summary", "Problem_Definition_and_Targeting"],
               [
                   "Problem_Clarification_Summary",
                   "Barrier_Brainstorm_Worksheet",
                   "Goal_To_Barrier_Map",
                   "Define_One_Target_Behavior",
               ])

    # ----------------------------
    # B) PLANNING & IMPLEMENTATION TOOLS
    # ----------------------------

    add_leaves(coping, ["Planning_and_Implementation", "Goal_Setting_and_Prioritization", "Goal_Definition"], [
        "Goal_Specification_SMART",
        "Avoidance_Goal_To_Approach_Goal_Rewrite",
        "Minimum_Viable_Goal_Definition",
        "Good_Better_Best_Goal_Tiers",
        "Goal_To_Motivation_Link_Map",
    ])

    add_leaves(coping, ["Planning_and_Implementation", "Goal_Setting_and_Prioritization", "Metrics_and_Criteria"], [
        "Success_Metrics_Definition",
        "Define_Success_Criteria_Card",
        "Define_Minimum_Standard_Card",
        "Define_Stretch_Standard_Card",
        "Setback_Tolerant_Standard_Card",
    ])

    add_leaves(coping,
               ["Planning_and_Implementation", "Goal_Setting_and_Prioritization", "Priorities_and_Review_Routines"], [
                   "Priority_Ranking_Top3",
                   "Weekly_Goal_Review",
                   "Monthly_Goal_Refresh",
                   "NonNegotiables_Clarification",
               ])

    add_leaves(coping, ["Planning_and_Implementation", "Action_Planning", "Where_When_WithWhom"], [
        "Action_Plan_Template_Where_When_WithWhom",
        "Timeboxing_Block_Schedule",
        "Calendar_Blocking_Session",
        "Routine_Anchor_Placement",
    ])

    add_leaves(coping, ["Planning_and_Implementation", "Action_Planning", "How_Steps_and_Sequencing"], [
        "Action_Plan_Template_How_Steps",
        "Subgoal_Ladder_Definition",
        "One_Thing_Next_Action_Card",
        "Task_Splitting_One_Micro_Action",
        "Plan_Quality_Scorecard",
    ])

    add_leaves(coping, ["Planning_and_Implementation", "Action_Planning", "Cue_Design_and_Boundaries"], [
        "Start_Cue_Definition",
        "End_Cue_Definition",
        "Boundary_Conditions_Definition",
        "Cue_Placement_Plan",
    ])

    add_leaves(coping, ["Planning_and_Implementation", "Action_Planning", "Resources_Tools_and_Setup"], [
        "Resource_Checklist_For_Action",
        "Environment_Setup_Checklist",
        "Tool_And_Resource_Availability_Setup",
        "Remove_Startup_Friction_Checklist",
    ])

    add_leaves(coping, ["Planning_and_Implementation", "Action_Planning", "Simplification_and_Choice_Load"], [
        "Specificity_Upgrade_Check",
        "Feasibility_Check",
        "Plan_Simplification_Pass",
        "Reduce_Choice_Load_Strategy",
        "Default_Option_Design",
    ])

    add_leaves(coping, ["Planning_and_Implementation", "Action_Planning", "Tracking_Feedback_and_Reinforcement_Plans"],
               [
                   "Plan_For_Tracking_And_Feedback",
                   "Plan_For_Reward_And_Reinforcement",
                   "Progress_Signal_Visualization_Tracking",
                   "Milestones_And_Checkpoints_Plan",
               ])

    add_leaves(coping, ["Planning_and_Implementation", "Coping_Planning", "IfThen_Response_Library"], [
        "If_Then_Coping_Plan_Cards",
        "Trigger_Specific_IfThen_Library",
        "Alternative_Action_Menu",
        "Plan_Branching_Options",
    ])

    add_leaves(coping, ["Planning_and_Implementation", "Coping_Planning", "Symptom_Flare_Contingencies"], [
        "Symptom_Flare_Contingency",
        "Panic_Flare_Plan",
        "Low_Mood_Day_Plan",
        "Craving_Flare_Plan",
        "Trauma_Trigger_Plan",
    ])

    add_leaves(coping, ["Planning_and_Implementation", "Coping_Planning", "Context_Disruption_Contingencies"], [
        "Travel_Disruption_IfThen_Set",
        "Sleep_Disruption_IfThen_Set",
        "Work_Overload_IfThen_Set",
        "Family_Crisis_IfThen_Set",
        "Weather_Disruption_IfThen_Set",
    ])

    add_leaves(coping, ["Planning_and_Implementation", "Coping_Planning", "Digital_Distraction_Contingencies"], [
        "Digital_Distraction_IfThen_Set",
        "Notification_Interruptions_IfThen_Set",
        "Doomscrolling_Interrupt_IfThen_Set",
    ])

    add_leaves(coping, ["Planning_and_Implementation", "Coping_Planning", "Relapse_and_Recovery_Plans"], [
        "Recovery_Plan_If_Slip",
        "Relapse_Risk_Audit_Checklist",
        "Relapse_Prevention_Structure_Plan",
        "Barrier_Update_And_Replan",
        "Coping_Plan_Update",
    ])

    add_leaves(coping,
               ["Planning_and_Implementation", "Precommitment_and_Commitment", "Commitment_Writing_and_Reminders"], [
                   "Microcommitment_Pledge",
                   "Commitment_Statement_Written",
                   "Commitment_Reminder_Cue",
                   "Identity_And_Values_Pledge",
               ])

    add_leaves(coping,
               ["Planning_and_Implementation", "Precommitment_and_Commitment", "Accountability_and_Safe_Sharing"], [
                   "Public_Commitment_Safe_Sharing",
                   "Accountability_Partner_Linkage",
                   "Buddy_System_CheckIn_Schedule",
                   "Coach_Or_Therapist_CheckIn_Structure",
               ])

    add_leaves(coping,
               ["Planning_and_Implementation", "Precommitment_and_Commitment", "Precommitment_Devices_and_Friction"], [
                   "Precommitment_Device_Selection",
                   "Remove_Escape_Hatches_Protocol",
                   "Add_Friction_To_Competing_Behavior",
                   "Temptation_Removal_Plan",
               ])

    add_leaves(coping,
               ["Planning_and_Implementation", "Care_Coordination_and_Navigation", "Appointments_and_Preparation"], [
                   "Default_Appointment_Booking",
                   "Appointment_Booking_Workflow",
                   "Prepare_Questions_For_Provider",
                   "Symptom_Summary_For_Visit",
               ])

    add_leaves(coping,
               ["Planning_and_Implementation", "Care_Coordination_and_Navigation", "Medication_Regimen_Support"], [
                   "Medication_Reminder_Scheduling",
                   "Refill_Reminder_Workflow",
                   "Side_Effect_Tracking_For_Clinician",
                   "Medication_Safety_Check_Questions_List",
               ])

    add_leaves(coping,
               ["Planning_and_Implementation", "Care_Coordination_and_Navigation", "Access_and_Administrative_Steps"], [
                   "Insurance_Coverage_Checklist",
                   "Low_Cost_Resource_Search_Plan",
                   "Paperwork_Batching_Session",
                   "Transport_Plan_For_Visits",
               ])

    # ----------------------------
    # C) ENVIRONMENTAL & SITUATIONAL ENGINEERING
    # ----------------------------

    add_leaves(coping, ["Environment_and_Context_Engineering", "Stimulus_Control", "Remove_Triggers_and_Temptations"], [
        "Stimulus_Control_Plan",
        "High_Risk_Context_Avoidance_Plan",
        "Temptation_Removal_Plan",
        "Remove_Trigger_Leave_Situation_Script",
    ])

    add_leaves(coping, ["Environment_and_Context_Engineering", "Stimulus_Control", "Cue_Placement_and_Salience"], [
        "Cue_Placement_And_Salience_Design",
        "Start_Cue_Definition",
        "Goal_Standard_Reminder_Card",
        "Visual_Cue_Signage_Setup",
    ])

    add_leaves(coping, ["Environment_and_Context_Engineering", "Stimulus_Control", "Reduce_Friction_For_Target_Action"],
               [
                   "Remove_Friction_From_Action",
                   "Tool_And_Resource_Availability_Setup",
                   "Environmental_Restructuring_Action_Context",
                   "Prep_The_Night_Before_Setup",
               ])

    add_leaves(coping,
               ["Environment_and_Context_Engineering", "Stimulus_Control", "Increase_Friction_For_Competing_Behavior"],
               [
                   "Add_Friction_To_Competing_Behavior",
                   "Delay_Device_Access_Rule",
                   "Temptation_Gates_Passworded",
               ])

    add_leaves(coping, ["Environment_and_Context_Engineering", "Context_Selection", "Safe_Space_and_Privacy"], [
        "Safe_Space_Selection_Protocol",
        "Privacy_And_Safe_Space_Setup",
        "Workstation_Or_Clinic_Setup",
    ])

    add_leaves(coping, ["Environment_and_Context_Engineering", "Context_Selection", "Anchor_Context_and_Routine"], [
        "Context_Selection_And_Anchor",
        "Anchor_Context_Routine_Build",
        "Commute_Based_Anchor",
    ])

    add_leaves(coping, ["Environment_and_Context_Engineering", "Context_Selection", "Disruption_Readiness"], [
        "Disruption_Readiness_Setup",
        "Travel_Ready_Action_Kit",
        "Backup_Context_Option_List",
    ])

    add_leaves(coping,
               ["Environment_and_Context_Engineering", "Digital_Environment_Design", "Notifications_and_Interruptions"],
               [
                   "Notification_Diet_Plan",
                   "Focus_Mode_Protocol",
                   "Do_Not_Disturb_Schedule",
               ])

    add_leaves(coping,
               ["Environment_and_Context_Engineering", "Digital_Environment_Design", "Blocking_and_Friction_Tools"], [
                   "App_Blocker_Configuration",
                   "Doomscrolling_Friction_Setup",
                   "Website_Blocklist_Setup",
               ])

    add_leaves(coping, ["Environment_and_Context_Engineering", "Digital_Environment_Design", "Interface_Minimalism"], [
        "Home_Screen_Minimalism_Setup",
        "Remove_Attention_Grabbing_Apps",
        "Single_Purpose_Device_Mode",
    ])

    add_leaves(coping,
               ["Environment_and_Context_Engineering", "Digital_Environment_Design", "Automations_and_Reminders"], [
                   "Digital_Reminder_Scheduling",
                   "Automation_Trigger_For_Start",
                   "Calendar_Reminder_Stacking",
               ])

    add_leaves(coping,
               ["Environment_and_Context_Engineering", "Social_Environment_Engineering", "Body_Doubling_and_CoWorking"],
               [
                   "CoWorking_Body_Double_Setup",
                   "Ask_For_Body_Double_Script",
                   "Shared_Focus_Session_Routine",
               ])

    add_leaves(coping,
               ["Environment_and_Context_Engineering", "Social_Environment_Engineering", "Household_and_Shared_Rules"],
               [
                   "Shared_Household_Rules_Agreement",
                   "Household_Quiet_Hours_Agreement",
                   "Shared_Trigger_Reduction_Plan",
               ])

    add_leaves(coping, ["Environment_and_Context_Engineering", "Social_Environment_Engineering",
                        "Boundary_With_Unsupportive_Contexts"], [
                   "Remove_Unsupportive_Contact_Temporarily",
                   "Change_Social_Context_Option",
                   "Values_Based_Boundary_Script",
               ])

    add_leaves(coping,
               ["Environment_and_Context_Engineering", "Social_Environment_Engineering", "Supportive_Space_Scouting"], [
                   "Supportive_Spaces_Scouting",
                   "Community_Support_Group_Scouting",
                   "Clinic_Resource_Scouting",
               ])

    add_leaves(coping,
               ["Environment_and_Context_Engineering", "Accessibility_and_Accommodations", "Sensory_Accommodations"], [
                   "Sensory_Load_Reduction",
                   "Noise_Reduction_Headphones_Plan",
                   "Lighting_Adjustment_Plan",
               ])

    add_leaves(coping,
               ["Environment_and_Context_Engineering", "Accessibility_and_Accommodations", "Cognitive_Accommodations"],
               [
                   "Working_Memory_Offload_Checklist",
                   "Visual_Checklist_Printouts",
                   "Simplify_Instructions_One_Step",
               ])

    add_leaves(coping, ["Environment_and_Context_Engineering", "Accessibility_and_Accommodations",
                        "Mobility_and_Access_Planning"], [
                   "Transport_Backup_Plan",
                   "Home_Based_Alternative_Action_Set",
                   "Telehealth_Setup_Checklist",
               ])

    # ----------------------------
    # D) ACTION INITIATION & ACTIVATION
    # ----------------------------

    add_leaves(coping, ["Action_Initiation_and_Activation", "Starting_Skills", "Start_Rituals_and_Entry"], [
        "Start_Ritual_Protocol",
        "Start_Ritual_Design",
        "Just_Open_The_Door_Step",
        "First_Then_Script",
        "Count_Down_5_Second_Rule",
    ])

    add_leaves(coping, ["Action_Initiation_and_Activation", "Starting_Skills", "Microstarts_and_Minimum_Viable_Action"],
               [
                   "Two_Minute_Starter_Script",
                   "Minimum_Viable_Action_Definition",
                   "Behavioral_Activation_Microstep",
                   "Starter_Step_Fallback_Option",
                   "Task_Splitting_One_Micro_Action",
               ])

    add_leaves(coping, ["Action_Initiation_and_Activation", "Starting_Skills", "Startup_Friction_Removal"], [
        "Remove_Startup_Friction_Checklist",
        "Prep_Tools_In_Advance",
        "Preload_Environment_For_Start",
    ])

    add_leaves(coping,
               ["Action_Initiation_and_Activation", "Procrastination_and_Avoidance_Tools", "Timeboxing_and_Trials"], [
                   "Timebox_10_Minute_Trial",
                   "Focus_Sprint_25_Minutes",
                   "Commit_To_One_Unit_Only",
               ])

    add_leaves(coping, ["Action_Initiation_and_Activation", "Procrastination_and_Avoidance_Tools",
                        "Lowering_The_Bar_and_Anti_Perfectionism"], [
                   "Lower_The_Bar_Script",
                   "Anti_Perfectionism_Start_Script",
                   "Good_Enough_Rule",
               ])

    add_leaves(coping,
               ["Action_Initiation_and_Activation", "Procrastination_and_Avoidance_Tools", "Emotion_Label_and_Move"], [
                   "Avoidance_Label_And_Name",
                   "Name_The_Feelings_Then_Start",
                   "Start_Before_Ready_Script",
                   "Action_First_Mood_Follows_Frame",
               ])

    add_leaves(coping, ["Action_Initiation_and_Activation", "Behavioral_Activation", "Activity_Scheduling"], [
        "Positive_Event_Scheduling",
        "Mastery_And_Pleasure_Balance",
        "Values_Based_Activation_Plan",
    ])

    add_leaves(coping, ["Action_Initiation_and_Activation", "Behavioral_Activation", "Social_Activation"], [
        "Social_Contact_Scheduling",
        "Buddy_Walk_Or_CheckIn_Plan",
        "Join_Group_Activity_Microstep",
    ])

    add_leaves(coping, ["Action_Initiation_and_Activation", "Exposure_and_Approach", "Graded_Exposure_Ladders"], [
        "Graded_Exposure_Ladder_Definition",
        "Social_Exposure_Microstep",
        "Interoceptive_Exposure_Mini",
    ])

    add_leaves(coping, ["Action_Initiation_and_Activation", "Energy_and_Fatigue_Support", "Pacing_and_Microbreaks"], [
        "Fatigue_Compassionate_Start_Plan",
        "Pacing_And_Rest_Break_Schedule",
        "Micro_Recovery_Pause_Skill",
    ])

    add_leaves(coping, ["Action_Initiation_and_Activation", "Energy_and_Fatigue_Support",
                        "Sleep_Nutrition_Movement_MicroSupport"], [
                   "Sleep_Support_Micro_Plan",
                   "Nutrition_Hydration_Check",
                   "Light_Movement_Activation",
               ])

    add_leaves(coping, ["Action_Initiation_and_Activation", "Energy_and_Fatigue_Support", "Somatic_PreStart_Calming"], [
        "Somatic_Calming_PreStart",
        "Pain_Or_Discomfort_Acceptance_Skill_Brief",
    ])

    # ----------------------------
    # E) SELF-REGULATION / ACTION CONTROL / MONITORING
    # ----------------------------

    add_leaves(coping, ["Self_Regulation_and_Action_Control", "Self_Monitoring", "Logs_and_Checkins"], [
        "Self_Monitoring_Log_Simple",
        "Self_Monitoring_Log_Contextual",
        "Check_In_Questions_Set",
        "Daily_Review_2_Minutes",
        "Weekly_Retrospective",
    ])

    add_leaves(coping, ["Self_Regulation_and_Action_Control", "Self_Monitoring", "Technology_Aided_Monitoring"], [
        "Self_Monitoring_With_Technology_Tool",
        "Habit_Tracker_Setup",
        "Reminder_And_CheckIn_Automation",
    ])

    add_leaves(coping, ["Self_Regulation_and_Action_Control", "Self_Monitoring", "Progress_Signals_and_Milestones"], [
        "Short_Term_Milestones_Definition",
        "Micro_Wins_Capture",
        "Successes_Inventory",
        "Progress_Signal_Visualization_Tracking",
    ])

    add_leaves(coping, ["Self_Regulation_and_Action_Control", "Standards_and_Goals", "Standards_Definition"], [
        "Define_Success_Criteria_Card",
        "Define_Minimum_Standard_Card",
        "Define_Stretch_Standard_Card",
        "Setback_Tolerant_Standard_Card",
    ])

    add_leaves(coping, ["Self_Regulation_and_Action_Control", "Standards_and_Goals", "Identity_and_Values_Standards"], [
        "Value_Based_Standard_Statement",
        "Identity_Based_Standard_Statement",
        "NonNegotiables_Clarification",
    ])

    add_leaves(coping, ["Self_Regulation_and_Action_Control", "Standards_and_Goals", "Consistency_Frames"], [
        "Consistency_Over_Intensity_Frame",
        "Repair_Not_Punish_Rule",
        "If_Slip_Then_Smaller_Step_Rule",
    ])

    add_leaves(coping, ["Self_Regulation_and_Action_Control", "Inhibition_and_Urges", "Urge_Surfing_and_Delay"], [
        "Urge_Surfing_Skill",
        "Delay_Tactic_Tool",
        "Delay_And_Distract_Toolkit",
    ])

    add_leaves(coping, ["Self_Regulation_and_Action_Control", "Inhibition_and_Urges",
                        "Competing_Responses_and_Temptation_Management"], [
                   "Competing_Intention_Inhibition_Script",
                   "Temptation_Management_Toolkit",
                   "Distraction_Toolkit_Action",
                   "Craving_SOS_Plan",
               ])

    add_leaves(coping, ["Self_Regulation_and_Action_Control", "Attention_and_Focus", "Focus_Structure"], [
        "Single_Tasking_Protocol",
        "Focus_Sprint_25_Minutes",
        "Task_Resumption_Cue",
    ])

    add_leaves(coping, ["Self_Regulation_and_Action_Control", "Attention_and_Focus", "Cognitive_Offload_and_Simplify"],
               [
                   "Working_Memory_Offload_Checklist",
                   "Next_Action_PostIt",
                   "Environment_Scan_Remove_Distractions",
               ])

    add_leaves(coping, ["Self_Regulation_and_Action_Control", "Attention_and_Focus", "Mind_Wandering_Return_Skills"], [
        "Mind_Wandering_Label_Return",
        "Attention_Reset_Breath",
        "Orienting_Response_Reset",
    ])

    add_leaves(coping, ["Self_Regulation_and_Action_Control", "Decision_Support", "Defaults_and_Predecisions"], [
        "Predecide_Rules_IfThen",
        "Default_Decision_Rule",
        "Two_Options_Only_Rule",
    ])

    add_leaves(coping, ["Self_Regulation_and_Action_Control", "Decision_Support", "Deadlines_and_Satisficing"], [
        "Decision_Deadline_Set",
        "Satisficing_Rule",
        "Regret_Minimization_Question",
        "Consult_Past_Self_Note",
    ])

    add_leaves(coping, ["Self_Regulation_and_Action_Control", "Error_Correction_and_Adjustment", "Rapid_Resets"], [
        "Mini_Reset_Routine",
        "AllOrNothing_Thinking_Reset_Script",
        "Plan_Adjustment_Protocol",
        "Barrier_Update_And_Replan",
    ])

    # ----------------------------
    # F) COGNITIVE STRATEGIES (MEANING / SELF-TALK / DEFUSION / PROBLEM SOLVING)
    # ----------------------------

    add_leaves(coping, ["Cognitive_Strategies", "Reappraisal_and_Reframing", "Cost_Benefit_Reframes"], [
        "Reappraise_Costs_Cognitive_Reframe",
        "Benefit_Realism_Check_Exercise",
        "Cost_Realism_Check_Exercise",
        "Regret_Minimization_Frame",
    ])

    add_leaves(coping, ["Cognitive_Strategies", "Reappraisal_and_Reframing", "Setback_and_Threat_Reframes"], [
        "Setback_To_Data_Reframe",
        "Threat_To_Coping_Reframe",
        "Plateau_Management_Reframe",
        "Hope_Induction_Frame",
    ])

    add_leaves(coping, ["Cognitive_Strategies", "Reappraisal_and_Reframing", "Growth_and_Experiment_Frames"], [
        "Growth_Mindset_Reframe",
        "Experiment_Mindset_Reframe",
        "Curiosity_Instead_Of_Judgment_Reframe",
        "Control_What_You_Can_Reframe",
    ])

    add_leaves(coping, ["Cognitive_Strategies", "Self_Talk_and_Identity", "Coach_Voice_and_Constructive_Scripts"], [
        "Self_Talk_Script_Constructive",
        "Self_Talk_In_Moment",
        "Coach_Voice_Script",
        "Permission_To_Be_Average_Script",
    ])

    add_leaves(coping, ["Cognitive_Strategies", "Self_Talk_and_Identity", "Values_and_Identity_Affirmations"], [
        "Identity_Based_Affirmation_Script",
        "Values_Protection_Script",
        "Identity_Protection_Script",
        "Streak_And_Progress_Identity_Reinforcement",
    ])

    add_leaves(coping, ["Cognitive_Strategies", "Self_Talk_and_Identity", "Anti_Shame_Scripts"], [
        "Anti_Shame_Self_Talk",
        "Compassionate_Interpretation_Reframe",
        "Normalize_Lapse_As_Learning_Frame",
    ])

    add_leaves(coping, ["Cognitive_Strategies", "Prospection_and_Visualization", "Future_Self_and_Benefit_Imagery"], [
        "Future_Self_Simulation_Risk_Focused",
        "Benefit_Visualization_Guided",
        "Best_Possible_Self_Exercise",
    ])

    add_leaves(coping, ["Cognitive_Strategies", "Prospection_and_Visualization", "WOOP_and_Obstacle_Imagery"], [
        "WOOP_Wish_Outcome_Obstacle_Plan",
        "Process_Visualization_Steps",
        "Coping_Imagery_For_Setbacks",
    ])

    add_leaves(coping, ["Cognitive_Strategies", "Defusion_and_Metacognition", "Defusion_Exercises"], [
        "Thought_Labeling_Just_A_Thought",
        "Name_The_Story_Defusion",
        "Leaves_On_A_Stream_Exercise",
        "Cognitive_Defusion_Silly_Voice",
    ])

    add_leaves(coping, ["Cognitive_Strategies", "Defusion_and_Metacognition", "Worry_Rumination_Management"], [
        "Worry_Postponement_Window",
        "Rumination_Stop_Signal_Then_Action",
        "Meta_Awareness_Check_What_Mode_Am_I_In",
    ])

    add_leaves(coping,
               ["Cognitive_Strategies", "Defusion_and_Metacognition", "Reality_Testing_and_Reappraisal_Questions"], [
                   "Reality_Testing_Questions_Set",
                   "Check_Facts_Protocol",
                   "Alternative_Interpretations_Prompt",
               ])

    add_leaves(coping,
               ["Cognitive_Strategies", "Problem_Solving_and_Cognitive_Skills", "Problem_Definition_and_Options"], [
                   "Problem_Definition_One_Sentence",
                   "Brainstorm_Solutions_Timer",
                   "Option_Evaluation_ProsCons",
                   "Pick_One_Option_Test",
               ])

    add_leaves(coping,
               ["Cognitive_Strategies", "Problem_Solving_and_Cognitive_Skills", "Implementation_and_FollowThrough"], [
                   "Convert_Solution_To_Next_Action",
                   "Barrier_Anticipation_For_Solution",
                   "Plan_Branching_Options",
                   "Default_Option_Design",
               ])

    # ----------------------------
    # G) EMOTION REGULATION & DISTRESS TOLERANCE
    # ----------------------------

    add_leaves(coping,
               ["Emotion_Regulation_and_Distress_Tolerance", "Downregulation_Skills", "Breathing_and_CO2_Tolerance"], [
                   "Box_Breathing_Skill",
                   "Physiological_Sigh_Skill",
                   "Paced_Breathing_Extended_Exhale",
               ])

    add_leaves(coping,
               ["Emotion_Regulation_and_Distress_Tolerance", "Downregulation_Skills", "Grounding_and_Orientation"], [
                   "Grounding_After_Threat_Info_Skill",
                   "Sensory_Grounding_5_4_3_2_1",
                   "Orienting_Response_Scan",
                   "Grounding_Acute_5_Senses",
               ])

    add_leaves(coping,
               ["Emotion_Regulation_and_Distress_Tolerance", "Downregulation_Skills", "Relaxation_and_Somatic_Reset"], [
                   "Progressive_Muscle_Relaxation_Brief",
                   "Vagus_Nerve_Toning_Brief",
                   "Cold_Splash_TIPP_Brief",
                   "Cold_Water_Reset_Brief",
               ])

    add_leaves(coping, ["Emotion_Regulation_and_Distress_Tolerance", "Downregulation_Skills", "Labeling_and_Reframe"], [
        "Arousal_Labeling_And_Reframe",
        "Anxiety_Tolerance_Brief_Skill",
        "Emotion_Regulation_Brief_Skill",
        "Fear_Arousal_Containment_Script",
    ])

    add_leaves(coping,
               ["Emotion_Regulation_and_Distress_Tolerance", "Distress_Tolerance_DBT_Tools", "Pause_and_Choice"], [
                   "STOP_Skill",
                   "Wise_Mind_Pause",
                   "Pros_Cons_Urge_Action",
               ])

    add_leaves(coping,
               ["Emotion_Regulation_and_Distress_Tolerance", "Distress_Tolerance_DBT_Tools", "Distraction_and_Soothe"],
               [
                   "ACCEPTS_Distraction_Set",
                   "IMPROVE_Moment_Set",
                   "Self_Soothe_5_Senses_Set",
               ])

    add_leaves(coping,
               ["Emotion_Regulation_and_Distress_Tolerance", "Distress_Tolerance_DBT_Tools", "Acceptance_Skills"], [
                   "Radical_Acceptance_Brief",
                   "Half_Smile_Willing_Hands",
               ])

    add_leaves(coping,
               ["Emotion_Regulation_and_Distress_Tolerance", "Self_Compassion_and_Shame", "Self_Compassion_Scripts"], [
                   "Self_Compassion_After_Lapse_Script",
                   "Self_Compassion_During_Difficulty_Script",
                   "Self_Kindness_In_Crisis_Script",
                   "Compassionate_Closure_Script",
               ])

    add_leaves(coping, ["Emotion_Regulation_and_Distress_Tolerance", "Self_Compassion_and_Shame",
                        "Shame_Reduction_and_Forgiveness"], [
                   "Shame_Reduction_Script",
                   "Self_Forgiveness_Protocol",
                   "Common_Humanity_Reminder",
                   "Compassionate_Letter_To_Self",
               ])

    add_leaves(coping, ["Emotion_Regulation_and_Distress_Tolerance", "Upregulation_and_Positive_Affect",
                        "Savoring_and_Gratitude"], [
                   "Savoring_Practice_Brief",
                   "Gratitude_Three_Good_Things",
                   "Affective_Benefit_Noticing_Prompt",
               ])

    add_leaves(coping, ["Emotion_Regulation_and_Distress_Tolerance", "Upregulation_and_Positive_Affect",
                        "Activation_and_Connection"], [
                   "Kindness_Action_Prompt",
                   "Music_Mood_Shift_Playlist",
                   "Light_And_Outdoor_Exposure",
               ])

    add_leaves(coping, ["Emotion_Regulation_and_Distress_Tolerance", "Sleep_and_Stress_Physiology",
                        "WindDown_and_Stimulus_Control"], [
                   "Sleep_Environment_Optimization",
                   "Wind_Down_Routine_Script",
                   "Stimulus_Control_Sleep_Rule_Set",
               ])

    add_leaves(coping, ["Emotion_Regulation_and_Distress_Tolerance", "Sleep_and_Stress_Physiology",
                        "Nightmare_and_Night_Anxiety_Support"], [
                   "Nightmare_Coping_Brief",
                   "Nighttime_Grounding_Protocol",
                   "Reduce_Night_Stimulation_Plan",
               ])

    # ----------------------------
    # H) SOCIAL SUPPORT & COMMUNICATION
    # ----------------------------

    add_leaves(coping, ["Social_Support_and_Communication", "Support_Activation", "Support_Seeking_Scripts"], [
        "Support_Seeking_Script",
        "Nonjudgmental_Support_Request_Script",
        "Confidential_Support_Option",
        "Support_Activation_After_Lapse",
    ])

    add_leaves(coping, ["Social_Support_and_Communication", "Support_Activation", "Structured_CheckIns"], [
        "Buddy_System_CheckIn_Schedule",
        "Coach_Or_Therapist_CheckIn_Structure",
        "Accountability_CheckIn_With_Kindness",
    ])

    add_leaves(coping, ["Social_Support_and_Communication", "Support_Activation", "Peer_and_Group_Support"], [
        "Peer_Support_Linkage",
        "Community_Support_Group_Linkage",
        "Online_Peer_Support_Safe_Option",
    ])

    add_leaves(coping, ["Social_Support_and_Communication", "Boundaries_and_Pressure", "Refusal_and_Exit_Scripts"], [
        "Assertive_Refusal_Script",
        "Social_Pressure_Boundary_Script",
        "Exit_Plan_For_Unsafe_Situations",
    ])

    add_leaves(coping, ["Social_Support_and_Communication", "Boundaries_and_Pressure", "Values_Based_Boundaries"], [
        "Values_Based_Boundary_Script",
        "Limit_Setting_Rehearsal",
        "Grey_Rock_Response_Script",
    ])

    add_leaves(coping,
               ["Social_Support_and_Communication", "Interpersonal_Effectiveness_Skills", "Communication_MicroSkills"],
               [
                   "Active_Listening_Micro_Skills",
                   "I_Statements_Script",
                   "Repair_Conversation_Script",
               ])

    add_leaves(coping,
               ["Social_Support_and_Communication", "Interpersonal_Effectiveness_Skills", "Conflict_Deescalation"], [
                   "Conflict_Deescalation_Steps",
                   "Timeout_And_Return_Agreement",
                   "Validation_First_Protocol",
               ])

    add_leaves(coping, ["Social_Support_and_Communication", "Interpersonal_Effectiveness_Skills", "DBT_Structures"], [
        "DEAR_MAN_Script",
        "GIVE_Skill",
        "FAST_Skill",
    ])

    add_leaves(coping,
               ["Social_Support_and_Communication", "Disclosure_and_Stigma_Management", "Disclosure_Decision_Aids"], [
                   "Disclosure_Decision_Aid",
                   "What_To_Share_And_With_Whom_Map",
                   "Confidentiality_Boundary_Script",
               ])

    add_leaves(coping,
               ["Social_Support_and_Communication", "Disclosure_and_Stigma_Management", "Stigma_Response_Scripts"], [
                   "Stigma_Response_Short_Script",
                   "Correct_Misinformation_Script",
                   "Self_Advocacy_Script",
               ])

    # ----------------------------
    # I) REINFORCEMENT, REWARDS, ENJOYMENT, HABIT DESIGN
    # ----------------------------

    add_leaves(coping, ["Reinforcement_and_Enjoyment", "Reward_Design", "Immediate_Rewards_and_Tokenization"], [
        "Immediate_Reward_Design_NonFood",
        "Token_Reward_System_Simple",
        "Reward_Schedule_Variable",
        "Pair_Action_With_Pleasure",
    ])

    add_leaves(coping, ["Reinforcement_and_Enjoyment", "Reward_Design", "Progress_Feedback_as_Reward"], [
        "Progress_Signal_Visualization_Reward",
        "Micro_Wins_Capture",
        "Successes_Inventory",
    ])

    add_leaves(coping, ["Reinforcement_and_Enjoyment", "Enjoyment_and_Novelty", "Variation_and_Boredom_Antidotes"], [
        "Boredom_Antidote_Variation_Menu",
        "Sustainable_Enjoyment_Design",
        "Novelty_Injection_One_Change",
    ])

    add_leaves(coping, ["Reinforcement_and_Enjoyment", "Habit_and_Routine", "Habit_Loop_Design"], [
        "Habit_Loop_Design_Cue_Action_Reward",
        "Habit_Stacking_Protocol",
        "Cue_Reliability_Upgrade",
    ])

    add_leaves(coping, ["Reinforcement_and_Enjoyment", "Habit_and_Routine", "Routine_Stabilization"], [
        "Routine_Stabilization_Design",
        "Context_Consistency_Plan",
        "Frictionless_Defaults_Routine",
    ])

    add_leaves(coping, ["Reinforcement_and_Enjoyment", "Habit_and_Routine", "Maintenance_Review"], [
        "Maintenance_Review_Protocol",
        "Weekly_Maintenance_Audit",
        "Long_Horizon_Goal_Refresh",
    ])

    add_leaves(coping, ["Reinforcement_and_Enjoyment", "Meaning_and_Values_Connection", "Meaning_Linkage"], [
        "Values_Link_Reminder_Card",
        "Purpose_Statement_One_Liner",
        "Meaning_Making_Narrative_Rewrite",
    ])

    add_leaves(coping, ["Reinforcement_and_Enjoyment", "Meaning_and_Values_Connection", "Impact_and_Legacy"], [
        "Impact_Visualization_Who_Benefits",
        "Legacy_Prompt",
        "Self_Respect_Reinforcement_Frame",
    ])

    # ----------------------------
    # J) RECOVERY & RELAPSE PREVENTION TOOLKIT
    # ----------------------------

    add_leaves(coping, ["Recovery_and_Relapse_Prevention", "Restart_and_Repair", "Restart_Cues_and_Rituals"], [
        "Restart_Cue_Definition",
        "Repair_Ritual_Design",
        "Tiny_Restart_Within_24_Hours",
    ])

    add_leaves(coping, ["Recovery_and_Relapse_Prevention", "Restart_and_Repair", "Recommitment_and_Choice_Frames"], [
        "Recommitment_With_Choice_Frame",
        "Rapid_Reengagement_Confidence_Script",
        "Rebuild_Trust_With_Self_Protocol",
    ])

    add_leaves(coping, ["Recovery_and_Relapse_Prevention", "Learning_From_Slips", "Slip_Analysis_and_Lessons"], [
        "Lesson_Learned_Summary_Card",
        "Slip_Pattern_Review",
        "Setback_To_Data_Reframe",
    ])

    add_leaves(coping, ["Recovery_and_Relapse_Prevention", "Learning_From_Slips", "Plan_Update_Workflows"], [
        "Slip_To_Plan_Update_Workflow",
        "Barrier_Update_And_Replan",
        "Coping_Plan_Update",
    ])

    add_leaves(coping, ["Recovery_and_Relapse_Prevention", "Relapse_Prevention_Structures", "Trigger_Risk_Audits"], [
        "Relapse_Risk_Audit_Checklist",
        "Early_Warning_Signs_Action_List",
        "High_Risk_Situations_Map",
    ])

    add_leaves(coping, ["Recovery_and_Relapse_Prevention", "Relapse_Prevention_Structures",
                        "Stimulus_Control_And_Context_Plans"], [
                   "Relapse_Triggers_Removal_Plan",
                   "Stimulus_Control_Relapse_Prevention",
                   "High_Risk_Context_Avoidance_Plan",
               ])

    add_leaves(coping,
               ["Recovery_and_Relapse_Prevention", "Relapse_Prevention_Structures", "Disruption_Transition_Plans"], [
                   "Contingency_Plan_For_Disruption",
                   "Plan_For_Context_Transitions",
                   "Travel_Disruption_IfThen_Set",
               ])

    add_leaves(coping, ["Recovery_and_Relapse_Prevention", "Relapse_Prevention_Structures", "Support_Network_Plans"], [
        "Support_Network_Relapse_Plan",
        "Support_Activation_After_Lapse",
        "Booster_Sessions_Scheduling",
    ])

    add_leaves(coping, ["Recovery_and_Relapse_Prevention", "Self_Compassionate_Accountability", "No_Shame_Review"], [
        "No_Shame_Data_Review",
        "Accountability_CheckIn_With_Kindness",
        "Repair_Not_Punish_Rule",
    ])

    add_leaves(coping, ["Recovery_and_Relapse_Prevention", "Self_Compassionate_Accountability", "Standards_Reset"], [
        "Compassionate_Standard_Reset",
        "If_Slip_Then_Smaller_Step_Rule",
        "Good_Enough_Rule",
    ])

    # ----------------------------
    # K) CRISIS & OVERWHELM SUPPORT
    # ----------------------------

    add_leaves(coping, ["Crisis_and_Overwhelm_Support", "Safety_and_Stabilization", "Crisis_Planning"], [
        "Crisis_Plan_If_Overwhelmed",
        "Emergency_Threshold_Rules",
        "Reduce_Commitment_Load_Temporary_Script",
    ])

    add_leaves(coping, ["Crisis_and_Overwhelm_Support", "Safety_and_Stabilization", "Immediate_Support_Activation"], [
        "Reach_Out_Now_Script_Safe",
        "Safety_Contacts_List_Build",
        "Share_Safety_Plan_With_Support",
    ])

    add_leaves(coping, ["Crisis_and_Overwhelm_Support", "Safety_and_Stabilization", "Overwhelm_Triage"], [
        "Overwhelm_Triage_Script",
        "One_Breath_One_Action_Rule",
        "Write_One_Line_Now_What",
    ])

    add_leaves(coping, ["Crisis_and_Overwhelm_Support", "Acute_Deescalation_Tools", "Physiological_Deescalation"], [
        "TIPP_Temperature_Intense_Exercise_Brief",
        "Paced_Breathing_Extended_Exhale",
        "Cold_Splash_TIPP_Brief",
    ])

    add_leaves(coping, ["Crisis_and_Overwhelm_Support", "Acute_Deescalation_Tools", "Cognitive_Somatic_Containment"], [
        "Containment_Imagery_Skill",
        "Find_Safe_Place_Protocol",
        "Grounding_Acute_5_Senses",
    ])

    add_leaves(coping, ["Crisis_and_Overwhelm_Support", "Acute_Deescalation_Tools", "Self_Soothe_and_Stabilize"], [
        "Self_Soothe_Kit_Access",
        "Self_Kindness_In_Crisis_Script",
        "Compassionate_Closure_Script",
    ])

    add_leaves(coping, ["Crisis_and_Overwhelm_Support", "Care_Navigation", "Crisis_Lines_and_Urgent_Care"], [
        "Find_Local_Crisis_Line_Step",
        "Emergency_Services_Decision_Tree",
        "Call_Text_Professional_Resources_Step",
    ])

    add_leaves(coping, ["Crisis_and_Overwhelm_Support", "Care_Navigation", "Clinical_FollowUp_and_Booking"], [
        "Contact_Primary_Care_Or_Psychiatry_Step",
        "Appointment_Booking_In_Crisis_Support",
        "Symptom_Summary_For_Visit",
    ])

    add_leaves(coping, ["Crisis_and_Overwhelm_Support", "Post_Crisis_Stabilization", "Basic_Needs_and_Routine_Reentry"],
               [
                   "Basic_Needs_Check_Food_Water_Sleep",
                   "Sleep_Reentry_Plan",
                   "Micro_Routine_Rebuild_Protocol",
               ])

    add_leaves(coping, ["Crisis_and_Overwhelm_Support", "Post_Crisis_Stabilization", "Supportive_Contact_Routine"], [
        "Planned_Support_CheckIn_After_Crisis",
        "Buddy_System_CheckIn_Schedule",
        "Coach_Or_Therapist_CheckIn_Structure",
    ])

    HAPA["COPING_STRATEGIES"] = coping

    return {"HAPA": HAPA}

# ------------------------ Writer + metadata ------------------------

def write_outputs(ontology: dict, out_json_path: str) -> tuple[str, str, dict]:
    out_json_path = os.path.expanduser(out_json_path)
    out_dir = os.path.dirname(out_json_path)

    # If path is not viable, fall back to a local HAPA/ folder next to this script
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        out_dir = os.path.join(os.path.dirname(__file__), "HAPA")
        os.makedirs(out_dir, exist_ok=True)
        out_json_path = os.path.join(out_dir, "HAPA.json")

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(ontology, f, ensure_ascii=False, indent=2)

    leaf_paths = list(iter_leaf_paths(ontology["HAPA"]))
    leaf_count = count_leaves(ontology["HAPA"])
    node_count = count_nodes(ontology["HAPA"])
    depth = max_depth(ontology["HAPA"])
    top_counts = subtree_leaf_counts(ontology["HAPA"])
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


def main():
    default_out = "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/01_raw/HAPA/HAPA.json"
    out_json_path = os.environ.get("HAPA_OUT_PATH", default_out)

    ontology = build_hapa_ontology()

    # Guardrail: reject explicit schedule/frequency/duration tokens in leaf paths
    forbidden_patterns = [
        r"\b\d+\s*x\s*week\b",
        r"\b\d+x_week\b",
        r"\bper_week\b",
        r"\bweekly\b",
        r"\bdaily\b",
        r"\bper_day\b",
        r"\bper_month\b",
        r"\b\d+\s*min\b",
        r"\b\d+\s*minutes\b",
        r"\b\d+\s*hz\b",
        r"\b\d+\s*weeks\b",
        r"\b\d+\s*days\b",
        r"\b\d+\s*sessions\b",
    ]
    leaf_paths = list(iter_leaf_paths(ontology["HAPA"]))
    bad = scan_forbidden_tokens(leaf_paths, forbidden_patterns)
    if bad:
        example_path, example_pat = bad[0]
        raise ValueError(
            "Forbidden schedule/frequency token detected. "
            f"pattern={example_pat!r} in path={' / '.join(example_path)}"
        )

    out_json_path, meta_path, metadata = write_outputs(ontology, out_json_path)

    print("Leaf nodes:", metadata["leaf_count"])
    print("Wrote JSON:", out_json_path)
    print("Wrote metadata:", meta_path)


if __name__ == "__main__":
    main()

# TODO: further expand barrier granularity and coping strategy granularity if needed (mapping remains out-of-scope here)
# TODO: further optimize content
# TODO: ignore focussed library
# TODO: improve high-level structure (so mapped BARRIERS and COPINGS are inside a separate top-level nodes)
