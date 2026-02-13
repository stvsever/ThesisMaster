#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHOENIX Context Sub-Ontology (Internal + External Environment)
- Entity-only hierarchical taxonomy (nested dict; leaf nodes are {}).
- Builds an OWL ontology using rdflib and serializes:
  1) context.owl               (RDF/XML)
  2) CONTEXT.json              (entity-only nested dict)

Usage:
  python build_context_ontology.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Set

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL


# =============================================================================
# 1) Entity-only taxonomy (high-resolution; internal + external contextual state)
# =============================================================================

CONTEXT_TAXONOMY: Dict[str, Any] = {
    "Context": {
        # ---------------------------------------------------------------------
        # INTERNAL ENVIRONMENT (person-level momentary state + near-term modifiers)
        # ---------------------------------------------------------------------
        "Internal_Environment": {
            # 1) Affect / mood / emotion context
            "Affect_And_Mood_Context": {
                "Affect_Valence": {
                    "Momentary_Positive_Affect": {},
                    "Momentary_Negative_Affect": {},
                    "Neutral_Affect_Level": {},
                    "Emotional_Numbness_Level": {},
                    "Mixed_Affect_Level": {},
                    "Baseline_Mood_Today": {},
                },
                "Arousal_And_Tension": {
                    "Physiological_Arousal_Level": {},
                    "Subjective_Tension_Level": {},
                    "Agitation_Level": {},
                    "Calmness_Level": {},
                    "Startle_Sensitivity_Level": {},
                    "Restlessness_Urge_Level": {},
                },
                "Emotion_Specific_Intensity": {
                    "Anxiety_Intensity": {},
                    "Fear_Intensity": {},
                    "Sadness_Intensity": {},
                    "Grief_Intensity": {},
                    "Anger_Intensity": {},
                    "Irritability_Intensity": {},
                    "Shame_Intensity": {},
                    "Guilt_Intensity": {},
                    "Disgust_Intensity": {},
                    "Joy_Intensity": {},
                    "Contentment_Intensity": {},
                    "Loneliness_Intensity": {},
                    "Connectedness_Felt_Intensity": {},
                    "Hopefulness_Intensity": {},
                    "Hopelessness_Intensity": {},
                    "Compassion_Toward_Self_Intensity": {},
                },
                "Affect_Dynamics": {
                    "Mood_Lability_Index": {},
                    "Emotional_Inertia_Index": {},
                    "Affect_Recovery_Speed": {},
                    "Peak_Affect_Intensity_Today": {},
                    "Affect_Variability_Today": {},
                    "Contextual_Affect_Reactivity_Level": {},
                    "Diurnal_Mood_Slope": {},
                },
                "Affect_Regulation_State": {
                    "Emotion_Regulation_Success_Level": {},
                    "Emotion_Regulation_Effort_Level": {},
                    "Suppression_Use_Level": {},
                    "Reappraisal_Use_Level": {},
                    "Acceptance_Use_Level": {},
                    "Avoidance_Use_Level": {},
                    "Self_Soothing_Use_Level": {},
                },
                "Social_Affect_State": {
                    "Social_Anxiety_Affect_Level": {},
                    "Interpersonal_Warmth_Level": {},
                    "Rejection_Sensitivity_Affect_Level": {},
                },
            },

            # 2) Motivation / goals / values context
            "Motivation_And_Goal_Context": {
                "Motivation_State": {
                    "Motivation_Level": {},
                    "Initiation_Readiness": {},
                    "Persistence_Readiness": {},
                    "Effort_Willingness": {},
                    "Task_Engagement_Level": {},
                    "Procrastination_Pressure_Level": {},
                    "Self_Control_Resource_Level": {},
                },
                "Approach_Avoidance_Dynamics": {
                    "Approach_Motivation_Level": {},
                    "Avoidance_Motivation_Level": {},
                    "Avoidance_Urge_Level": {},
                    "Safety_Seeking_Urge_Level": {},
                    "Exploration_Urge_Level": {},
                },
                "Goal_Structure": {
                    "Goal_Salience": {},
                    "Goal_Clarity": {},
                    "Goal_Conflict_Level": {},
                    "Goal_Progress_Perception": {},
                    "Goal_Importance_Perception": {},
                    "Goal_Difficulty_Perception": {},
                    "Goal_Feasibility_Perception": {},
                },
                "Behavior_Change_Readiness": {
                    "Stage_Of_Change": {},
                    "Intention_Strength": {},
                    "Commitment_Level": {},
                    "Implementation_Intention_Strength": {},
                    "Plan_Quality_Level": {},
                    "Barrier_Anticipation_Level": {},
                },
                "Self_Determination_Satisfaction": {
                    "Autonomy_Satisfaction_Level": {},
                    "Competence_Satisfaction_Level": {},
                    "Relatedness_Satisfaction_Level": {},
                },
                "Meaning_And_Values_Alignment": {
                    "Values_Alignment_Level": {},
                    "Sense_Of_Meaning_Level": {},
                    "Purposefulness_Level": {},
                    "Moral_Injury_Distress_Level": {},
                },
                "Reward_And_Incentive_Sensitivity": {
                    "Reward_Responsiveness": {},
                    "Anhedonia_Level": {},
                    "Anticipatory_Pleasure_Level": {},
                    "Consummatory_Pleasure_Level": {},
                    "Reward_Learning_Confidence": {},
                    "Punishment_Sensitivity_Level": {},
                },
            },

            # 3) Cognitive state / executive / metacognitive context
            "Cognitive_State_And_Load_Context": {
                "Attention_And_Focus": {
                    "Attention_Stability": {},
                    "Distractibility_Level": {},
                    "Sustained_Attention_Capacity": {},
                    "Selective_Attention_Bias_To_Threat": {},
                    "Selective_Attention_Bias_To_Rejection": {},
                    "Mind_Wandering_Level": {},
                    "Flow_State_Level": {},
                },
                "Cognitive_Load": {
                    "Working_Memory_Load": {},
                    "Decision_Fatigue_Level": {},
                    "Information_Overload_Level": {},
                    "Multitasking_Level": {},
                    "Cognitive_Effort_Cost_Level": {},
                },
                "Executive_Control_State": {
                    "Inhibitory_Control_Level": {},
                    "Set_Shifting_Ability": {},
                    "Planning_Capacity_Level": {},
                    "Cognitive_Flexibility_Level": {},
                    "Cognitive_Rigidity_Level": {},
                    "Goal_Shielding_Level": {},
                },
                "Memory_And_Processing": {
                    "Processing_Speed_Level": {},
                    "Short_Term_Memory_Confidence": {},
                    "Prospective_Memory_Confidence": {},
                    "Memory_Intrusion_Level": {},
                    "Cognitive_Slippage_Level": {},
                },
                "Appraisal_And_Interpretation_State": {
                    "Threat_Appraisal_Level": {},
                    "Control_Appraisal_Level": {},
                    "Uncertainty_Intolerance_Level": {},
                    "Perceived_Injustice_Level": {},
                    "Perfectionism_Pressure_Level": {},
                    "Self_Standard_Discrepancy_Level": {},
                },
                "Repetitive_Negative_Thinking": {
                    "Rumination_Level": {},
                    "Worry_Level": {},
                    "Intrusive_Thought_Frequency": {},
                    "Catastrophic_Thinking_Level": {},
                    "Self_Criticism_Level": {},
                    "Counterfactual_Thinking_Level": {},
                },
                "Metacognitive_And_Awareness_State": {
                    "Meta_Awareness_Level": {},
                    "Mindfulness_State_Level": {},
                    "Decentering_Ability_Level": {},
                    "Cognitive_Defusion_Ability_Level": {},
                    "Thought_Believability_Level": {},
                    "Interoceptive_Interpretation_Bias_Level": {},
                },
                "Dissociation_And_Altered_Consciousness": {
                    "Depersonalization_Level": {},
                    "Derealization_Level": {},
                    "Absorption_Level": {},
                    "Time_Distortion_Level": {},
                    "Emotional_Detachment_Level": {},
                },
                "Cognitive_Bias_Proxies": {
                    "Negativity_Bias_Level": {},
                    "Interpretation_Bias_To_Threat_Level": {},
                    "Confirmation_Bias_Threat_Level": {},
                    "Availability_Bias_Stress_Level": {},
                },
            },

            # 4) Stress / coping / resilience context
            "Stress_And_Coping_Context": {
                "Stress_Exposure": {
                    "Acute_Stressor_Presence": {},
                    "Chronic_Stress_Load": {},
                    "Social_Evaluative_Threat_Level": {},
                    "Uncontrollability_Exposure_Level": {},
                    "Unpredictability_Exposure_Level": {},
                    "Role_Overload_Exposure_Level": {},
                },
                "Stress_Response": {
                    "Perceived_Stress_Level": {},
                    "Overwhelm_Level": {},
                    "Irritability_Level": {},
                    "Hypervigilance_Level": {},
                    "Shutdown_Freeze_Level": {},
                    "Emotional_Flooding_Level": {},
                },
                "Coping_Capacity": {
                    "Coping_Confidence_Level": {},
                    "Coping_Self_Efficacy": {},
                    "Coping_Repertoire_Availability": {},
                    "Problem_Solving_Capacity_Level": {},
                    "Emotion_Regulation_Capacity_Level": {},
                    "Social_Problem_Solving_Capacity_Level": {},
                },
                "Coping_Mode": {
                    "Problem_Focused_Coping_Level": {},
                    "Emotion_Focused_Coping_Level": {},
                    "Avoidant_Coping_Level": {},
                    "Seeking_Support_Coping_Level": {},
                    "Meaning_Focused_Coping_Level": {},
                    "Self_Compassionate_Coping_Level": {},
                },
                "Recovery_State": {
                    "Psychological_Recovery_Level": {},
                    "Need_For_Recovery_Level": {},
                    "Decompression_Availability": {},
                    "Restoration_Level": {},
                    "Allostatic_Load_Proxy_Level": {},
                },
                "Resilience_Proxies": {
                    "Bouncing_Back_Confidence_Level": {},
                    "Stress_Tolerance_Level": {},
                    "Frustration_Tolerance_Level": {},
                    "Adaptive_Flexibility_Level": {},
                },
            },

            # 5) Physiological / somatic / interoceptive context
            "Physiological_And_Somatic_Context": {
                "Autonomic_State": {
                    "Cardiovascular_State": {
                        "Heart_Rate": {},
                        "Heart_Rate_Variability": {},
                        "Blood_Pressure_Level": {},
                        "Peripheral_Temperature": {},
                    },
                    "Respiratory_State": {
                        "Respiration_Rate": {},
                        "Respiratory_Variability_Level": {},
                    },
                    "Electrodermal_State": {
                        "Electrodermal_Activity_Level": {},
                        "Skin_Conductance_Reactivity_Level": {},
                    },
                },
                "Breathing_State": {
                    "Breath_Holding_Urge_Level": {},
                    "Hyperventilation_Tendency_Level": {},
                    "Breathing_Discomfort_Level": {},
                    "Dyspnea_Sensation_Level": {},
                },
                "Somatic_Symptom_Burden": {
                    "Somatic_Discomfort_Level": {},
                    "Pain_Intensity": {},
                    "Headache_Intensity": {},
                    "GI_Discomfort_Level": {},
                    "Nausea_Level": {},
                    "Dizziness_Level": {},
                    "Palpitations_Level": {},
                    "Chest_Tightness_Level": {},
                },
                "Muscle_And_Body_Tension": {
                    "Muscle_Tension_Level": {},
                    "Jaw_Clenching_Level": {},
                    "Restlessness_Level": {},
                    "Tremor_Level": {},
                    "Shoulder_Neck_Tension_Level": {},
                },
                "Sickness_And_Inflammatory_Signal_Proxies": {
                    "Malaise_Level": {},
                    "Feverishness_Level": {},
                    "Chills_Level": {},
                    "Sore_Throat_Level": {},
                    "Fatigue_Sickness_Like_Level": {},
                },
                "Interoceptive_Awareness": {
                    "Interoceptive_Sensitivity_Level": {},
                    "Body_Scan_Clarity_Level": {},
                    "Heartbeat_Perception_Accuracy_Level": {},
                    "Interoceptive_Confidence_Level": {},
                },
                "Menstrual_Somatic_Proxies": {
                    "Cramps_Intensity_Level": {},
                    "Breast_Tenderness_Level": {},
                    "Bloating_Level": {},
                },
            },

            # 6) Sleep / circadian / chronobiology context
            "Sleep_And_Circadian_Context": {
                "Sleep_Quantity": {
                    "Sleep_Duration_Last_Night": {},
                    "Time_In_Bed_Last_Night": {},
                    "Total_Sleep_Time_7Day_Average": {},
                    "Nap_Duration_Today": {},
                    "Sleep_Opportunity_Window_Length": {},
                },
                "Sleep_Quality": {
                    "Sleep_Quality_Self_Report": {},
                    "Sleep_Fragmentation_Index": {},
                    "Sleep_Onset_Latency": {},
                    "Wake_After_Sleep_Onset": {},
                    "Early_Morning_Awakening_Level": {},
                    "Nonrestorative_Sleep_Level": {},
                },
                "Sleep_Timing": {
                    "Bedtime_Clock_Time": {},
                    "Wake_Time_Clock_Time": {},
                    "Mid_Sleep_Time": {},
                    "Sleep_Midpoint_Regularity_Level": {},
                },
                "Circadian_Phase_And_Regularity": {
                    "Sleep_Regularity_Index": {},
                    "Social_Jetlag_Index": {},
                    "Circadian_Misalignment_Level": {},
                    "Shift_Work_Circadian_Impact_Level": {},
                    "Chronobiological_Stability_Level": {},
                },
                "Sleep_Debt_And_Sleepiness": {
                    "Sleep_Debt_Level": {},
                    "Daytime_Sleepiness_Level": {},
                    "Alertness_Level": {},
                    "Microsleep_Risk_Level": {},
                },
                "Dream_And_Night_Disturbance": {
                    "Nightmare_Frequency": {},
                    "Night_Terror_Indicator": {},
                    "Restless_Sleep_Level": {},
                    "Nocturnal_Panic_Indicator": {},
                },
                "Chronotype_Context": {
                    "Morningness_Eveningness_Index": {},
                    "Peak_Energy_Time_Window": {},
                    "Preferred_Bedtime_Window": {},
                },
            },

            # 7) Energy / fatigue / activation context
            "Energy_And_Fatigue_Context": {
                "Energy_State": {
                    "Energy_Level": {},
                    "Vitality_Level": {},
                    "Mental_Energy_Level": {},
                    "Physical_Energy_Level": {},
                    "Social_Energy_Level": {},
                },
                "Fatigue_State": {
                    "Fatigue_Level": {},
                    "Mental_Fatigue_Level": {},
                    "Physical_Fatigue_Level": {},
                    "Exhaustion_Level": {},
                    "Burnout_Proxy_Level": {},
                },
                "Activation_State": {
                    "Psychomotor_Slowness_Level": {},
                    "Psychomotor_Agitation_Level": {},
                    "Activation_Readiness_Level": {},
                    "Activation_Drive_Level": {},
                },
                "Recovery_And_Rest_State": {
                    "Restedness_Level": {},
                    "Need_For_Rest_Level": {},
                    "Rest_Opportunity_Availability_Level": {},
                },
            },

            # 8) Behavioral / lifestyle / routines (action-relevant state)
            "Behavioral_And_Lifestyle_Context": {
                "Physical_Activity_Behavior": {
                    "Step_Count_Today": {},
                    "Moderate_Vigorous_Activity_Minutes_Today": {},
                    "Exercise_Session_Indicator": {},
                    "Strength_Training_Indicator": {},
                    "Outdoor_Physical_Activity_Indicator": {},
                },
                "Sedentary_Behavior": {
                    "Sedentary_Duration_Today": {},
                    "Prolonged_Sitting_Bouts_Count": {},
                    "Breaks_From_Sitting_Count": {},
                },
                "Eating_And_Nutrition_Behavior": {
                    "Meal_Count_Today": {},
                    "Meal_Timing_Regularity_Level": {},
                    "Late_Night_Eating_Indicator": {},
                    "Appetite_Stability_Level": {},
                },
                "Hydration_Behavior": {
                    "Water_Intake_Proxy_Level": {},
                    "Hydration_Routine_Stability_Level": {},
                },
                "Social_Behavior": {
                    "Social_Contact_Count_Today": {},
                    "Face_To_Face_Contact_Indicator": {},
                    "Support_Seeking_Behavior_Indicator": {},
                    "Social_Withdrawal_Behavior_Level": {},
                },
                "Self_Care_Behavior": {
                    "Hygiene_Self_Care_Indicator": {},
                    "Relaxation_Practice_Indicator": {},
                    "Mindfulness_Practice_Indicator": {},
                    "Breathing_Practice_Indicator": {},
                    "Journaling_Indicator": {},
                    "Time_For_Self_Care_Today_Level": {},
                },
                "Cognitive_And_Productive_Behavior": {
                    "Deep_Work_Time_Today": {},
                    "Task_Completion_Count_Today": {},
                    "Avoidance_Behavior_Level": {},
                },
                "Leisure_And_Pleasure_Behavior": {
                    "Hobby_Time_Today": {},
                    "Pleasant_Activity_Indicator": {},
                    "Playfulness_Level": {},
                },
            },
            # 9) Exposure / triggers / vulnerability / safety / protective context
            # (NO symptom clusters here; those live in your other ontology)
            "Exposure_Trigger_Vulnerability_And_Safety_Context": {
                # A) Proximal trigger exposure (momentary / last hours-days)
                "Proximal_Trigger_Exposure": {
                    "Trauma_Cue_Exposure": {
                        "Trauma_Cue_Exposure_Level": {},
                        "Anniversary_Cue_Exposure_Level": {},
                        "Grief_Cue_Exposure_Level": {},
                        "Medical_Trauma_Cue_Exposure_Level": {},
                        "Interpersonal_Trauma_Cue_Exposure_Level": {},
                    },
                    "Sensory_And_Environmental_Triggers": {
                        "Sensory_Overload_Exposure_Level": {},
                        "Noise_Trigger_Exposure_Level": {},
                        "Light_Trigger_Exposure_Level": {},
                        "Crowding_Trigger_Exposure_Level": {},
                        "Confinement_Trigger_Exposure_Level": {},
                        "Odor_Trigger_Exposure_Level": {},
                    },
                    "Interpersonal_Triggers": {
                        "Interpersonal_Trigger_Exposure_Level": {},
                        "Rejection_Cue_Exposure_Level": {},
                        "Conflict_Trigger_Exposure_Level": {},
                        "Criticism_Trigger_Exposure_Level": {},
                        "Boundary_Violation_Trigger_Exposure_Level": {},
                        "Betrayal_Cue_Exposure_Level": {},
                    },
                    "Performance_And_Evaluation_Triggers": {
                        "Performance_Trigger_Exposure_Level": {},
                        "Public_Speaking_Trigger_Exposure_Level": {},
                        "Test_Exam_Trigger_Exposure_Level": {},
                        "Work_Evaluation_Trigger_Exposure_Level": {},
                        "Social_Evaluative_Threat_Exposure_Level": {},
                    },
                    "Health_And_Body_Triggers": {
                        "Health_Anxiety_Trigger_Exposure_Level": {},
                        "Interoceptive_Trigger_Exposure_Level": {},
                        "Pain_Trigger_Exposure_Level": {},
                        "Somatic_Sensation_Trigger_Exposure_Level": {},
                        "Medical_Setting_Trigger_Exposure_Level": {},
                    },
                    "Substance_And_Cue_Triggers": {
                        "Substance_Cue_Exposure_Level": {},
                        "Alcohol_Cue_Exposure_Level": {},
                        "Nicotine_Cue_Exposure_Level": {},
                        "Drug_Cue_Exposure_Level": {},
                    },
                    "Digital_And_Media_Triggers": {
                        "News_Trigger_Exposure_Level": {},
                        "Conflictual_Content_Exposure_Level": {},
                        "Social_Comparison_Trigger_Exposure_Level": {},
                        "Online_Harassment_Exposure_Level": {},
                        "Graphic_Content_Exposure_Level": {},
                    },
                    "Sleep_And_Fatigue_Triggers": {
                        "Sleep_Loss_Trigger_Exposure_Level": {},
                        "Circadian_Disruption_Trigger_Exposure_Level": {},
                        "Fatigue_Trigger_Exposure_Level": {},
                    },
                },

                # B) Life event exposure (objective events; not symptoms)
                # Use as contextual modifiers (recent/past, acute/chronic, expected/unexpected).
                "Life_Event_Exposure": {
                    "Bereavement_And_Loss_Exposure": {
                        "Death_Of_Close_Person_Exposure": {},
                        "Sudden_Death_Exposure": {},
                        "Expected_Death_Exposure": {},
                        "Suicide_Bereavement_Exposure": {},
                        "Pregnancy_Loss_Exposure": {},
                        "Pet_Loss_Exposure": {},
                        "Loss_Of_Friendship_Exposure": {},
                    },
                    "Relationship_And_Family_Disruption_Exposure": {
                        "Breakup_Or_Divorce_Exposure": {},
                        "Separation_Exposure": {},
                        "Infidelity_Exposure": {},
                        "Family_Conflict_Escalation_Exposure": {},
                        "Estrangement_Exposure": {},
                        "Custody_Dispute_Exposure": {},
                        "Domestic_Violence_Exposure": {},
                        "Stalking_By_Ex_Partner_Exposure": {},
                    },
                    "Interpersonal_Violence_And_Abuse_Exposure": {
                        "Physical_Assault_Exposure": {},
                        "Sexual_Assault_Exposure": {},
                        "Sexual_Harassment_Exposure": {},
                        "Emotional_Abuse_Exposure": {},
                        "Psychological_Control_Exposure": {},
                        "Neglect_Exposure": {},
                        "Intimate_Partner_Violence_Exposure": {},
                        "Childhood_Maltreatment_Exposure": {
                            "Childhood_Physical_Abuse_Exposure": {},
                            "Childhood_Sexual_Abuse_Exposure": {},
                            "Childhood_Emotional_Abuse_Exposure": {},
                            "Childhood_Neglect_Exposure": {},
                        },
                    },
                    "Accident_And_Injury_Exposure": {
                        "Motor_Vehicle_Accident_Exposure": {},
                        "Serious_Injury_Exposure": {},
                        "Workplace_Accident_Exposure": {},
                        "Sport_Injury_Exposure": {},
                        "Burn_Exposure": {},
                        "Near_Drowning_Exposure": {},
                        "Fall_Exposure": {},
                    },
                    "Medical_And_Health_Trauma_Exposure": {
                        "Life_Threatening_Illness_Diagnosis_Exposure": {},
                        "Acute_Medical_Emergency_Exposure": {},
                        "ICU_Admission_Exposure": {},
                        "Surgery_Exposure": {},
                        "Painful_Procedure_Exposure": {},
                        "Childbirth_Trauma_Exposure": {},
                        "Medical_Error_Exposure": {},
                        "Chronic_Illness_Burden_Exposure": {},
                    },
                    "Disaster_And_Environmental_Event_Exposure": {
                        "Fire_Disaster_Exposure": {},
                        "Flood_Disaster_Exposure": {},
                        "Earthquake_Disaster_Exposure": {},
                        "Severe_Storm_Disaster_Exposure": {},
                        "Heatwave_Exposure": {},
                        "Wildfire_Smoke_Exposure": {},
                        "Evacuation_Exposure": {},
                    },
                    "War_And_Political_Violence_Exposure": {
                        "Warzone_Exposure": {},
                        "Bombing_Exposure": {},
                        "Terror_Attack_Exposure": {},
                        "Armed_Conflict_Witnessing_Exposure": {},
                        "Forced_Displacement_Exposure": {},
                        "Refugee_Transit_Exposure": {},
                    },
                    "Witnessing_And_Secondary_Trauma_Exposure": {
                        "Witnessing_Violence_Exposure": {},
                        "Witnessing_Serious_Accident_Exposure": {},
                        "Witnessing_Death_Exposure": {},
                        "Secondary_Trauma_From_Close_Other_Exposure": {},
                        "Vicarious_Trauma_From_Work_Exposure": {},
                    },
                    "Occupational_And_Institutional_Exposure": {
                        "Workplace_Bullying_Exposure": {},
                        "Workplace_Harassment_Exposure": {},
                        "Moral_Injury_Work_Exposure": {},
                        "Unjust_Disciplinary_Action_Exposure": {},
                        "Institutional_Betrayal_Exposure": {},
                        "Academic_Dismissal_Exposure": {},
                    },
                    "Legal_And_System_Exposure": {
                        "Arrest_Or_Detention_Exposure": {},
                        "Court_Proceedings_Exposure": {},
                        "Legal_Threat_Exposure": {},
                        "Immigration_Detention_Exposure": {},
                        "Deportation_Risk_Exposure": {},
                    },
                    "Financial_And_Housing_Exposure": {
                        "Job_Loss_Exposure": {},
                        "Major_Financial_Loss_Exposure": {},
                        "Debt_Crisis_Exposure": {},
                        "Eviction_Exposure": {},
                        "Homelessness_Exposure": {},
                        "Housing_Insecurity_Exposure": {},
                    },
                    "Identity_Based_And_Social_Adversity_Exposure": {
                        "Discrimination_Exposure": {},
                        "Racism_Exposure": {},
                        "Sexism_Exposure": {},
                        "Homophobia_Transphobia_Exposure": {},
                        "Ableism_Exposure": {},
                        "Religious_Discrimination_Exposure": {},
                        "Hate_Crime_Exposure": {},
                    },
                    "Community_And_Societal_Stressors_Exposure": {
                        "Community_Violence_Exposure": {},
                        "Neighborhood_Unsafety_Exposure": {},
                        "Pandemic_Restriction_Exposure": {},
                        "Mass_Unemployment_Shock_Exposure": {},
                        "Civil_Unrest_Exposure": {},
                    },
                    "Developmental_And_Caregiving_Exposure": {
                        "Caregiving_Burden_Exposure": {},
                        "Caring_For_Seriously_Ill_Person_Exposure": {},
                        "Child_Special_Needs_Care_Exposure": {},
                        "Parental_Illness_Exposure": {},
                        "Parental_Substance_Use_Exposure": {},
                    },
                },

                # C) Exposure attributes / qualifiers (adds a depth layer to interpret event impact)
                "Exposure_Qualifiers": {
                    "Exposure_Timing": {
                        "Exposure_Recency_Level": {},
                        "Exposure_Chronicity_Level": {},
                        "Exposure_Duration_Level": {},
                        "Exposure_Frequency_Level": {},
                        "Anniversary_Proximity_Level": {},
                    },
                    "Exposure_Severity_And_Proximity": {
                        "Objective_Severity_Level": {},
                        "Perceived_Severity_Level": {},
                        "Threat_To_Life_Indicator": {},
                        "Physical_Injury_Indicator": {},
                        "Direct_Victim_Indicator": {},
                        "Witness_Indicator": {},
                        "Indirect_Exposure_Indicator": {},
                    },
                    "Predictability_And_Control": {
                        "Unexpectedness_Level": {},
                        "Controllability_Level": {},
                        "Preparedness_Level": {},
                    },
                    "Interpersonal_And_Moral_Dimensions": {
                        "Betrayal_Level": {},
                        "Intentionality_Indicator": {},
                        "Humiliation_Exposure_Level": {},
                        "Moral_Conflict_Exposure_Level": {},
                    },
                    "Contextual_Resources_At_Time_Of_Exposure": {
                        "Support_Availability_During_Exposure_Level": {},
                        "Safety_Availability_During_Exposure_Level": {},
                        "Information_Clarity_During_Exposure_Level": {},
                        "Access_To_Care_During_Exposure_Level": {},
                    },
                },

                # D) Vulnerability state (non-symptom predisposition / readiness-to-decompensate)
                "Vulnerability_State": {
                    "General_Vulnerability": {
                        "Relapse_Vulnerability_Level": {},
                        "Dysregulation_Vulnerability_Level": {},
                        "Overwhelm_Vulnerability_Level": {},
                        "Shutdown_Vulnerability_Level": {},
                    },
                    "Behavioral_Vulnerability": {
                        "Urge_To_Avoid_Level": {},
                        "Safety_Seeking_Urge_Level": {},
                        "Compulsion_Urge_Level": {},
                        "Craving_Intensity": {},
                        "Impulsivity_Vulnerability_Level": {},
                    },
                    "Physiological_Vulnerability": {
                        "Sleep_Loss_Vulnerability_Level": {},
                        "Pain_Vulnerability_Level": {},
                        "Illness_Vulnerability_Level": {},
                        "Substance_Withdrawal_Vulnerability_Level": {},
                    },
                    "Social_Vulnerability": {
                        "Rejection_Sensitivity_Vulnerability_Level": {},
                        "Conflict_Vulnerability_Level": {},
                        "Isolation_Vulnerability_Level": {},
                    },
                    "Cognitive_Vulnerability": {
                        "Uncertainty_Intolerance_Vulnerability_Level": {},
                        "Threat_Bias_Vulnerability_Level": {},
                        "Self_Criticism_Vulnerability_Level": {},
                    },
                },

                # E) Safety and acute risk (kept variable-only; no instructions)
                "Safety_And_Risk_Internal": {
                    "Perceived_Safety_Internal": {},
                    "Threat_Immediacy_Internal_Level": {},
                    "Perceived_Entrapment_Level": {},
                    "Perceived_Burdensomeness_Level": {},
                    "Acute_Agitation_Risk_Level": {},
                    "Non_Suicidal_Self_Injury_Urge_Level": {},
                    "Self_Harm_Urge_Level": {},
                    "Suicidal_Ideation_Level": {},
                    "Suicidal_Intent_Level": {},
                    "Suicide_Plan_Indicator": {},
                    "Access_To_Lethal_Means_Indicator": {},
                    "Substance_Intoxication_Risk_Level": {},
                },

                # F) Protective state (momentary buffers; still variable-only)
                "Protective_State": {
                    "Grounding_And_Presence": {
                        "Groundedness_Level": {},
                        "Orientation_Clarity_Level": {},
                        "Present_Moment_Contact_Level": {},
                    },
                    "Emotion_And_Arousal_Buffering": {
                        "Self_Soothe_Capacity_Level": {},
                        "Calming_Capacity_Level": {},
                        "Distress_Tolerance_Capacity_Level": {},
                    },
                    "Coping_Access_And_Readiness": {
                        "Access_To_Coping_Strategies_Internal_Level": {},
                        "Skill_Recall_Clarity_Level": {},
                        "Help_Seeking_Readiness_Level": {},
                    },
                    "Social_Protection": {
                        "Trusted_Contact_Reachability_Level": {},
                        "Perceived_Support_Availability_Level": {},
                        "Belongingness_Protection_Level": {},
                    },
                    "Meaning_And_Commitment_Protection": {
                        "Reasons_For_Living_Clarity_Level": {},
                        "Values_Commitment_Level": {},
                        "Future_Orientation_Level": {},
                    },
                },
            },
            # 10) Agency / self-efficacy / confidence context
            "Agency_And_Self_Efficacy_Context": {
                "Agency_State": {
                    "Sense_Of_Control_Level": {},
                    "Autonomy_Level": {},
                    "Decisional_Agency_Level": {},
                    "Behavioral_Activation_Capacity_Level": {},
                    "Initiative_Capacity_Level": {},
                },
                "Self_Efficacy_State": {
                    "Task_Self_Efficacy": {},
                    "Emotion_Regulation_Self_Efficacy": {},
                    "Social_Self_Efficacy": {},
                    "Health_Behavior_Self_Efficacy": {},
                    "Coping_Self_Efficacy": {},
                },
                "Confidence_And_Competence": {
                    "Competence_Belief_Level": {},
                    "Performance_Confidence_Level": {},
                    "Learning_Confidence_Level": {},
                    "Communication_Confidence_Level": {},
                },
                "Hope_And_Expectancy": {
                    "Outcome_Expectancy_Positive": {},
                    "Outcome_Expectancy_Negative": {},
                    "Hopefulness_Level": {},
                    "Efficacy_Expectancy_Level": {},
                },
                "Self_Concept_State": {
                    "Self_Esteem_Level": {},
                    "Self_Worth_Level": {},
                    "Identity_Stability_Level": {},
                    "Self_Compassion_Level": {},
                },
            },

            # 11) Substances / medication / nutrition-state proxies
            "Substance_And_Medication_Context": {
                "Substance_Use_State": {
                    "Caffeine_Intake_Level": {},
                    "Alcohol_Intake_Level": {},
                    "Nicotine_Use_Level": {},
                    "Cannabis_Use_Level": {},
                    "Other_Substance_Use_Level": {},
                    "Supplement_Use_Indicator": {},
                },
                "Withdrawal_And_Aftereffects": {
                    "Caffeine_Withdrawal_Symptoms_Level": {},
                    "Alcohol_Hangover_Level": {},
                    "Nicotine_Withdrawal_Level": {},
                    "Cannabis_Aftereffect_Level": {},
                },
                "Medication_State": {
                    "Medication_Adherence_Level": {},
                    "Medication_Dose_Taken_Today": {},
                    "Medication_Dose_Timing_Deviation": {},
                    "Medication_Side_Effect_Burden": {},
                    "PRN_Medication_Use_Indicator": {},
                    "Medication_Efficacy_Belief_Level": {},
                },
                "Nutrition_And_Hydration_Proxies": {
                    "Time_Since_Last_Meal": {},
                    "Meal_Regularity_Level": {},
                    "Hunger_Level": {},
                    "Satiety_Level": {},
                    "Hydration_Level": {},
                    "Dehydration_Symptom_Level": {},
                },
                "Glycemic_And_Appetite_Proxies": {
                    "Sugar_Crash_Likelihood_Level": {},
                    "Craving_For_Sweets_Level": {},
                    "Craving_For_Salt_Fat_Level": {},
                },
            },

            # 12) Sensory sensitivity / comfort (internal processing of stimuli)
            "Sensory_And_Comfort_Context": {
                "Sensory_Sensitivity": {
                    "Noise_Sensitivity_Level": {},
                    "Light_Sensitivity_Level": {},
                    "Touch_Sensitivity_Level": {},
                    "Smell_Sensitivity_Level": {},
                    "Crowding_Sensitivity_Level": {},
                },
                "Sensory_Seeking": {
                    "Sensory_Seeking_Level": {},
                    "Need_For_Sensory_Reduction_Level": {},
                    "Stimulation_Need_Level": {},
                },
                "Comfort_State": {
                    "Thermal_Comfort_Level": {},
                    "Physical_Comfort_Level": {},
                    "Clothing_Comfort_Level": {},
                    "Postural_Comfort_Level": {},
                },
                "Sensory_Modulation_Resources": {
                    "Access_To_Headphones_Indicator": {},
                    "Access_To_Sunglasses_Indicator": {},
                    "Access_To_Quiet_Object_Indicator": {},
                },
            },

            # 13) Interpersonal internal (attachment / trust / social-cognitive state)
            "Interpersonal_Internal_Context": {
                "Attachment_And_Security": {
                    "Attachment_Security_Level": {},
                    "Fear_Of_Rejection_Level": {},
                    "Fear_Of_Abandonment_Level": {},
                    "Attachment_Activation_Level": {},
                },
                "Trust_And_Safety_With_Others": {
                    "Interpersonal_Trust_Level": {},
                    "Perceived_Social_Safety_Level": {},
                    "Perceived_Betrayal_Sensitivity_Level": {},
                    "Vigilance_To_Social_Threat_Level": {},
                },
                "Social_Cognitive_State": {
                    "Perceived_Judgment_Sensitivity_Level": {},
                    "Threat_Interpretation_In_Social_Cues_Level": {},
                    "Belongingness_Level": {},
                    "Mentalizing_Clarity_Level": {},
                    "Empathy_Capacity_Level": {},
                },
                "Conflict_Readiness": {
                    "Conflict_Anticipation_Level": {},
                    "Boundary_Setting_Readiness_Level": {},
                    "Assertiveness_Readiness_Level": {},
                },
            },

            # 14) Capability / functional capacity (momentary)
            "Functional_Capacity_Context": {
                "Physical_Functioning": {
                    "Mobility_Capacity_Level": {},
                    "Endurance_Level": {},
                    "Coordination_Level": {},
                    "Pain_Interference_Level": {},
                },
                "Cognitive_Functioning": {
                    "Comprehension_Capacity_Level": {},
                    "Concentration_Capacity_Level": {},
                    "Learning_Capacity_Level": {},
                    "Decision_Making_Capacity_Level": {},
                },
                "Social_Functioning": {
                    "Social_Engagement_Capacity_Level": {},
                    "Communication_Capacity_Level": {},
                    "Conflict_Management_Capacity_Level": {},
                },
                "Occupational_Functioning": {
                    "Work_Ability_Level": {},
                    "Role_Performance_Capacity_Level": {},
                },
            },

            # 15) Hormonal / cyclic state
            "Hormonal_And_Cyclic_Context": {
                "Menstrual_Cycle_Context": {
                    "Menstrual_Cycle_Phase": {},
                    "Premenstrual_Symptom_Burden_Level": {},
                    "Ovulation_Window_Indicator": {},
                },
                "Reproductive_State_Context": {
                    "Pregnancy_Indicator": {},
                    "Postpartum_Indicator": {},
                    "Lactation_Indicator": {},
                },
                "Menopause_Context": {
                    "Perimenopause_Indicator": {},
                    "Hot_Flash_Intensity_Level": {},
                    "Sleep_Disruption_Menopause_Level": {},
                },
                "Endocrine_Modulator_Proxies": {
                    "Thyroid_Dysregulation_Proxy_Level": {},
                    "Cortisol_Arousal_Proxy_Level": {},
                },
            },

            # 16) Treatment / intervention engagement (personalization feedback loops)
            "Treatment_And_Intervention_Engagement_Context": {
                "Therapy_Engagement": {
                    "Session_Attendance_Consistency_Level": {},
                    "Homework_Completion_Level": {},
                    "Therapeutic_Alliance_Level": {},
                    "Between_Session_Practice_Level": {},
                },
                "Skill_Use_Recency": {
                    "CBT_Skill_Use_Level": {},
                    "ACT_Skill_Use_Level": {},
                    "DBT_Skill_Use_Level": {},
                    "Mindfulness_Skill_Use_Level": {},
                    "Behavioral_Activation_Skill_Use_Level": {},
                },
                "Intervention_Response_History": {
                    "Perceived_Effectiveness_Last_Intervention_Level": {},
                    "Adherence_To_Last_Recommendation_Level": {},
                    "Acceptability_Last_Recommendation_Level": {},
                    "Burden_Last_Recommendation_Level": {},
                },
                "Care_Beliefs_And_Expectations": {
                    "Treatment_Outcome_Expectancy_Level": {},
                    "Treatment_Credibility_Level": {},
                    "Medication_Attitude_Level": {},
                    "Side_Effect_Concern_Level": {},
                },
            },

            # 17) Preferences / personalization modifiers (constraints without actions)
            "Preference_And_Personalization_Modifiers": {
                "Notification_Preferences": {
                    "Preferred_Notification_Channel": {},
                    "Preferred_Notification_Timing_Window": {},
                    "Notification_Frequency_Tolerance_Level": {},
                    "Notification_Intrusiveness_Tolerance_Level": {},
                },
                "Privacy_And_Sharing_Preferences": {
                    "Data_Sharing_Comfort_Level": {},
                    "Location_Sharing_Consent_Indicator": {},
                    "Social_Sharing_Comfort_Level": {},
                },
                "Intervention_Format_Preferences": {
                    "Text_Based_Intervention_Preference_Level": {},
                    "Audio_Guided_Preference_Level": {},
                    "Video_Guided_Preference_Level": {},
                    "Social_Intervention_Preference_Level": {},
                    "Self_Directed_Preference_Level": {},
                },
                "Accessibility_Needs": {
                    "Visual_Accessibility_Needs_Indicator": {},
                    "Auditory_Accessibility_Needs_Indicator": {},
                    "Motor_Accessibility_Needs_Indicator": {},
                    "Cognitive_Accessibility_Needs_Indicator": {},
                },
                "Cultural_And_Language_Preferences": {
                    "Preferred_Language": {},
                    "Cultural_Context_Relevance_Level": {},
                    "Religious_Sensitivity_Preference_Level": {},
                },
            },
        },

        # ---------------------------------------------------------------------
        # EXTERNAL ENVIRONMENT (situational + environmental + service context)
        # ---------------------------------------------------------------------
        "External_Environment": {
            # 1) Temporal context (calendar + cycles + day structure)
            "Temporal_Context": {
                "Clock_Time": {
                    "Local_Time_Of_Day": {
                        "Night": {},
                        "Early_Morning": {},
                        "Morning": {},
                        "Midday": {},
                        "Afternoon": {},
                        "Evening": {},
                        "Late_Night": {},
                    },
                    "Hour_Of_Day": {
                        "Hour_00": {}, "Hour_01": {}, "Hour_02": {}, "Hour_03": {},
                        "Hour_04": {}, "Hour_05": {}, "Hour_06": {}, "Hour_07": {},
                        "Hour_08": {}, "Hour_09": {}, "Hour_10": {}, "Hour_11": {},
                        "Hour_12": {}, "Hour_13": {}, "Hour_14": {}, "Hour_15": {},
                        "Hour_16": {}, "Hour_17": {}, "Hour_18": {}, "Hour_19": {},
                        "Hour_20": {}, "Hour_21": {}, "Hour_22": {}, "Hour_23": {},
                    },
                    "Minute_Of_Hour": {},
                },
                "Calendar_Time": {
                    "Day_Of_Week": {
                        "Monday": {},
                        "Tuesday": {},
                        "Wednesday": {},
                        "Thursday": {},
                        "Friday": {},
                        "Saturday": {},
                        "Sunday": {},
                    },
                    "Weekend_Indicator": {},
                    "Week_Of_Year": {},
                    "Month_Of_Year": {
                        "January": {}, "February": {}, "March": {}, "April": {},
                        "May": {}, "June": {}, "July": {}, "August": {},
                        "September": {}, "October": {}, "November": {}, "December": {},
                    },
                    "Season_Of_Year": {
                        "Winter": {},
                        "Spring": {},
                        "Summer": {},
                        "Autumn": {},
                    },
                    "Year": {},
                    "Daylight_Saving_Time_Indicator": {},
                },
                "Temporal_Landmarks": {
                    "Start_Of_Day_Indicator": {},
                    "End_Of_Day_Indicator": {},
                    "Start_Of_Week_Indicator": {},
                    "End_Of_Week_Indicator": {},
                    "Start_Of_Month_Indicator": {},
                    "End_Of_Month_Indicator": {},
                    "Pay_Day_Proximity_Level": {},
                },
                "Holiday_And_Break_Context": {
                    "Public_Holiday_Indicator": {},
                    "Bank_Holiday_Indicator": {},
                    "School_Holiday_Indicator": {},
                    "Religious_Holiday_Indicator": {},
                    "Cultural_Festival_Indicator": {},
                    "Holiday_Eve_Indicator": {},
                    "Vacation_Period_Indicator": {},
                    "Exam_Period_Indicator": {},
                    "End_Of_Term_Indicator": {},
                    "New_Year_Period_Indicator": {},
                },
                "Time_Since_Events": {
                    "Time_Since_Waking": {},
                    "Time_Since_Last_Sleep": {},
                    "Time_Since_Last_Meal": {},
                    "Time_Since_Last_Social_Contact": {},
                    "Time_Since_Last_Exercise": {},
                    "Time_Since_Last_Outdoor_Exposure": {},
                    "Time_Since_Last_Therapy_Session": {},
                    "Time_Since_Last_Medication_Dose": {},
                },
                "Temporal_Pressure": {
                    "Deadline_Proximity_Level": {},
                    "Schedule_Tightness_Level": {},
                    "Time_Availability_Level": {},
                    "Time_Buffer_Level": {},
                    "Appointment_Imminence_Level": {},
                    "Next_Commitment_Urgency_Level": {},
                },
                "Routine_And_Structure": {
                    "Routine_Stability_Level": {},
                    "Routine_Disruption_Indicator": {},
                    "Morning_Routine_Window_Indicator": {},
                    "Workday_Start_Window_Indicator": {},
                    "Evening_Wind_Down_Window_Indicator": {},
                    "Bedtime_Window_Indicator": {},
                    "Meal_Window_Indicator": {},
                },
            },

            # 2) Geospatial / mobility / transportation context
            "Geospatial_Context": {
                "Place_Type": {
                    "At_Home": {},
                    "At_Work": {},
                    "At_School": {},
                    "At_University": {},
                    "In_Transit": {},
                    "Outdoors": {},
                    "Indoors": {},
                    "Healthcare_Setting": {},
                    "Gym_Or_Sports_Facility": {},
                    "Retail_Setting": {},
                    "Public_Space": {},
                    "Nature_Setting": {},
                    "Religious_Setting": {},
                },
                "Micro_Location_Context": {
                    "Private_Room_Indicator": {},
                    "Shared_Room_Indicator": {},
                    "Open_Plan_Office_Indicator": {},
                    "Kitchen_Indicator": {},
                    "Bedroom_Indicator": {},
                    "Bathroom_Indicator": {},
                    "Desk_Indicator": {},
                    "Vehicle_Interior_Indicator": {},
                },
                "Urbanicity_And_Terrain": {
                    "Urban_Indicator": {},
                    "Suburban_Indicator": {},
                    "Rural_Indicator": {},
                    "Altitude_Level": {},
                    "Hilliness_Level": {},
                },
                "Mobility_Patterns": {
                    "Distance_Traveled_Today": {},
                    "Step_Count_Today_External": {},
                    "Sedentary_Duration_Today_External": {},
                    "Location_Entropy": {},
                    "Home_Stay_Duration": {},
                    "Number_Of_Location_Transitions": {},
                    "Radius_Of_Gyration": {},
                },
                "Transport_Mode": {
                    "Walking_Indicator": {},
                    "Cycling_Indicator": {},
                    "Car_Indicator": {},
                    "Public_Transport_Indicator": {},
                    "Rideshare_Indicator": {},
                },
                "Accessibility": {
                    "Access_To_Green_Space_Level": {},
                    "Access_To_Grocery_Level": {},
                    "Access_To_Safe_Walking_Routes_Level": {},
                    "Access_To_Quiet_Space_Level": {},
                    "Commute_Burden_Level": {},
                    "Barrier_Free_Access_Level": {},
                },
                "Proximity_Indicators": {
                    "Distance_To_Home": {},
                    "Distance_To_Work": {},
                    "Distance_To_Healthcare": {},
                    "Distance_To_Support_Network": {},
                },
            },

            # 3) Weather / light / air quality / natural environment context
            "Weather_And_Natural_Environment_Context": {
                "Weather_Conditions": {
                    "Temperature_Context": {
                        "Ambient_Temperature": {},
                        "Feels_Like_Temperature": {},
                        "Heat_Index_Level": {},
                        "Wind_Chill_Level": {},
                    },
                    "Moisture_Context": {
                        "Humidity": {},
                        "Dew_Point": {},
                        "Precipitation_Intensity": {},
                        "Precipitation_Probability_Level": {},
                        "Snow_Or_Ice_Indicator": {},
                    },
                    "Wind_And_Sky_Context": {
                        "Wind_Speed": {},
                        "Wind_Gust_Speed": {},
                        "Cloud_Cover": {},
                        "Visibility_Level": {},
                        "Fog_Indicator": {},
                        "Thunderstorm_Indicator": {},
                    },
                    "Pressure_Context": {
                        "Barometric_Pressure": {},
                        "Pressure_Change_Rate_Level": {},
                    },
                },
                "Light_And_Daylight": {
                    "Daylight_Duration": {},
                    "Sunrise_Time": {},
                    "Sunset_Time": {},
                    "Sunlight_Exposure_Level": {},
                    "Illuminance_Outdoor_Level": {},
                    "UV_Index": {},
                    "Blue_Light_Exposure_Proxy_Level": {},
                },
                "Air_Quality_And_Allergens": {
                    "Air_Quality_Index": {},
                    "PM25_Level": {},
                    "PM10_Level": {},
                    "Ozone_Level": {},
                    "NO2_Level": {},
                    "Pollen_Count_Level": {},
                    "Allergen_Exposure_Level": {},
                },
                "Environmental_Disruption": {
                    "Severe_Weather_Indicator": {},
                    "Weather_Travel_Disruption_Level": {},
                    "Extreme_Heat_Indicator": {},
                    "Extreme_Cold_Indicator": {},
                    "Wildfire_Smoke_Indicator": {},
                },
                "Seasonal_And_Phenology_Context": {
                    "Seasonal_Affective_Risk_Level": {},
                    "Day_Length_Change_Rate_Level": {},
                },
            },

            # 4) Built / physical environment / indoor context
            "Physical_Environment_Context": {
                "Ambient_Stimuli": {
                    "Noise_Level": {},
                    "Light_Level": {},
                    "Crowding_Level": {},
                    "Privacy_Level": {},
                    "Vibration_Level": {},
                    "Odor_Intensity_Level": {},
                },
                "Acoustic_And_Lighting_Profile": {
                    "Traffic_Noise_Indicator": {},
                    "Speech_Noise_Level": {},
                    "Sudden_Noise_Indicator": {},
                    "Natural_Light_Level": {},
                    "Artificial_Light_Level": {},
                    "Glare_Indicator": {},
                    "Flicker_Indicator": {},
                },
                "Indoor_Air_And_Comfort": {
                    "Indoor_Temperature": {},
                    "Indoor_Humidity": {},
                    "Indoor_CO2_Level": {},
                    "Indoor_PM25_Level": {},
                    "Ventilation_Quality_Level": {},
                    "Airflow_Level": {},
                },
                "Environmental_Quality": {
                    "Cleanliness_Level": {},
                    "Clutter_Level": {},
                    "Ergonomic_Suitability_Level": {},
                    "Seating_Comfort_Level": {},
                    "Workspace_Organization_Level": {},
                },
                "Safety_Of_Space": {
                    "Safe_Space_Availability_Level": {},
                    "Escape_Route_Availability_Level": {},
                    "Lighting_Safety_Level": {},
                    "Trip_Hazard_Level": {},
                },
                "Nature_Exposure": {
                    "Green_Exposure_Level": {},
                    "Blue_Exposure_Level": {},
                    "Time_In_Nature_Today": {},
                    "View_Of_Nature_Indicator": {},
                },
            },

            # 5) Social environment (objective + situational)
            "Social_Environment_Context": {
                "Co_Presence": {
                    "Alone_Indicator": {},
                    "With_Partner_Indicator": {},
                    "With_Family_Indicator": {},
                    "With_Friends_Indicator": {},
                    "With_Colleagues_Indicator": {},
                    "With_Strangers_Indicator": {},
                    "Group_Size": {},
                    "Social_Density_Level": {},
                },
                "Interaction_Dynamics": {
                    "Social_Interaction_Frequency": {},
                    "Social_Interaction_Duration": {},
                    "Interaction_Valence_Level": {},
                    "Perceived_Social_Support_Level": {},
                    "Perceived_Social_Conflict_Level": {},
                    "Argument_Indicator": {},
                    "Reassurance_Seeking_Opportunity_Level": {},
                    "Interaction_Channel_Context": {
                        "In_Person_Interaction_Indicator": {},
                        "Phone_Interaction_Indicator": {},
                        "Text_Interaction_Indicator": {},
                        "Video_Interaction_Indicator": {},
                    },
                },
                "Social_Demand_And_Evaluation": {
                    "Perceived_Judgment_Level": {},
                    "Social_Performance_Demand_Level": {},
                    "Need_To_Impression_Manage_Level": {},
                    "Social_Role_Strictness_Level": {},
                },
                "Social_Isolation_And_Connection": {
                    "Social_Isolation_Level": {},
                    "Connectedness_Level": {},
                    "Belonging_Context_Level": {},
                    "Loneliness_Context_Level": {},
                },
                "Network_Availability": {
                    "Trusted_Contact_Availability_Level": {},
                    "Partner_Availability_Indicator": {},
                    "Peer_Support_Availability_Level": {},
                    "Family_Support_Availability_Level": {},
                },
                "Cultural_And_Community_Context": {
                    "Language_Context_Match_Level": {},
                    "Cultural_Minority_Stress_Level": {},
                    "Community_Event_Indicator": {},
                    "Religious_Community_Event_Indicator": {},
                },
            },

            # 6) Task / role / workload / obligations
            "Task_And_Role_Demand_Context": {
                "Work_And_Academic_Demands": {
                    "Workday_Indicator": {},
                    "Workload_Intensity_Level": {},
                    "Meeting_Indicator": {},
                    "Meeting_Burden_Level": {},
                    "Deep_Work_Block_Indicator": {},
                    "Cognitive_Task_Demand_Level": {},
                    "Academic_Demand_Level": {},
                    "Exam_Today_Indicator": {},
                    "Presentation_Today_Indicator": {},
                    "Performance_Evaluation_Proximity_Level": {},
                },
                "Demand_Profile": {
                    "Social_Demand_Level": {},
                    "Physical_Demand_Level": {},
                    "Sustained_Attention_Demand_Level": {},
                    "Emotional_Labor_Demand_Level": {},
                },
                "Domestic_And_Care_Demands": {
                    "Household_Task_Burden_Level": {},
                    "Caregiving_Demand_Level": {},
                    "Childcare_Demand_Level": {},
                    "Eldercare_Demand_Level": {},
                },
                "Role_Transitions": {
                    "Role_Switching_Frequency": {},
                    "Context_Switching_Frequency": {},
                    "Unexpected_Task_Interruptions_Level": {},
                },
                "Obligation_And_Admin": {
                    "Administrative_Task_Burden_Level": {},
                    "Financial_Admin_Burden_Level": {},
                    "Errand_Burden_Level": {},
                },
                "Recovery_Opportunities": {
                    "Break_Opportunity_Level": {},
                    "Lunch_Break_Indicator": {},
                    "Microbreak_Opportunity_Level": {},
                    "Quiet_Time_Opportunity_Level": {},
                },
                "Task_Control_And_Predictability": {
                    "Task_Control_Level": {},
                    "Task_Predictability_Level": {},
                    "Task_Clarity_Level": {},
                },
            },

            # 7) Digital / device / connectivity / media exposure
            "Digital_And_Technology_Context": {
                "Device_State": {
                    "Battery_Level": {},
                    "Low_Battery_Indicator": {},
                    "Connectivity_Quality_Level": {},
                    "Offline_Indicator": {},
                    "Do_Not_Disturb_Indicator": {},
                    "Notification_Volume_Level": {},
                    "Sensor_Availability_Level": {},
                },
                "Usage_Patterns": {
                    "Screen_Time_Today": {},
                    "Screen_Time_Last_Hour": {},
                    "Nighttime_Phone_Use_Level": {},
                    "App_Switching_Rate": {},
                    "Phone_Unlock_Frequency": {},
                    "Typing_Activity_Level": {},
                },
                "Interruption_And_Attention_Disruption": {
                    "Notification_Count_Last_Hour": {},
                    "Interruptions_During_Task_Level": {},
                    "Context_Switching_Digital_Level": {},
                },
                "Communication_Load": {
                    "Message_Inflow_Rate": {},
                    "Message_Outflow_Rate": {},
                    "Call_Frequency": {},
                    "Email_Inflow_Rate": {},
                    "Calendar_Alert_Frequency": {},
                },
                "Content_Exposure": {
                    "News_Exposure_Level": {},
                    "Social_Media_Exposure_Level": {},
                    "Conflictual_Content_Exposure_Level": {},
                    "Health_Information_Exposure_Level": {},
                    "Triggering_Content_Exposure_Level": {},
                },
                "Online_Social_Context": {
                    "Online_Interaction_Valence_Level": {},
                    "Cyberbullying_Exposure_Indicator": {},
                    "Social_Comparison_Exposure_Level": {},
                },
                "Digital_Friction": {
                    "App_Crash_Indicator": {},
                    "Login_Friction_Level": {},
                    "Notification_Overload_Level": {},
                    "Recommendation_Fatigue_Level": {},
                },
            },

            # 8) Resource / finance / material constraints
            "Resource_And_Financial_Context": {
                "Financial_Pressure": {
                    "Financial_Stress_Level": {},
                    "Bill_Payment_Pressure_Level": {},
                    "Income_Uncertainty_Level": {},
                    "Unexpected_Expense_Indicator": {},
                    "Debt_Pressure_Level": {},
                },
                "Material_Access": {
                    "Food_Access_Level": {},
                    "Food_Insecurity_Level": {},
                    "Medication_Affordability_Level": {},
                    "Housing_Stability_Level": {},
                    "Heating_Affordability_Level": {},
                    "Clothing_Affordability_Level": {},
                },
                "Time_And_Service_Resources": {
                    "Childcare_Availability_Level": {},
                    "Transportation_Affordability_Level": {},
                    "Internet_Affordability_Level": {},
                    "Device_Access_Level": {},
                    "Private_Space_Access_Level": {},
                },
                "Logistical_Resources": {
                    "Transportation_Availability_Level": {},
                    "Delivery_Service_Access_Level": {},
                    "Local_Services_Access_Level": {},
                    "Financial_Advice_Access_Level": {},
                },
            },

            # 9) Healthcare / service availability / treatment logistics
            "Healthcare_And_Services_Context": {
                "Care_Contact_State": {
                    "Therapy_Session_Today_Indicator": {},
                    "Appointment_Upcoming_Indicator": {},
                    "Recent_Clinician_Contact_Indicator": {},
                    "Crisis_Checkin_Indicator": {},
                },
                "Care_Access": {
                    "Access_To_Provider_Level": {},
                    "Waiting_Time_Burden_Level": {},
                    "Prescription_Access_Level": {},
                    "Medication_Refill_Needed_Indicator": {},
                    "Telehealth_Access_Level": {},
                    "In_Person_Care_Access_Level": {},
                },
                "Coverage_And_Cost_Context": {
                    "Insurance_Coverage_Indicator": {},
                    "Out_Of_Pocket_Cost_Burden_Level": {},
                },
                "Self_Management_Support": {
                    "Peer_Support_Access_Level": {},
                    "Group_Therapy_Indicator": {},
                    "Digital_Care_Tool_Access_Level": {},
                    "Psychoeducation_Access_Level": {},
                },
                "Crisis_And_Safety_Net": {
                    "Crisis_Service_Access_Level": {},
                    "Emergency_Support_Availability_Level": {},
                    "Trusted_Contact_Availability_Level_External": {},
                },
            },

            # 10) Safety / constraints / disruptions / policy-like constraints
            "Safety_And_Constraint_Context": {
                "Perceived_Safety": {
                    "Perceived_Personal_Safety_Level": {},
                    "Perceived_Environmental_Safety_Level": {},
                    "Safe_Exit_Availability_Level": {},
                },
                "Exposure_To_Risk": {
                    "Violence_Exposure_Indicator": {},
                    "Harassment_Exposure_Indicator": {},
                    "Accident_Risk_Exposure_Level": {},
                    "Substance_Cue_Exposure_Level_External": {},
                    "Crime_Exposure_Proxy_Level": {},
                },
                "Constraints_And_Disruptions": {
                    "Travel_Disruption_Level": {},
                    "Access_Restriction_Level": {},
                    "Unexpected_Event_Disruption_Level": {},
                    "Service_Outage_Indicator": {},
                    "Curfew_Indicator": {},
                },
                "Policy_And_Legal_Constraints": {
                    "Workplace_Policy_Constraint_Level": {},
                    "School_Policy_Constraint_Level": {},
                    "Public_Health_Restriction_Indicator": {},
                },
                "Environmental_Stability": {
                    "Housing_Stability_External_Level": {},
                    "Neighborhood_Stability_Level": {},
                },
            },

            # 11) Life events / milestones / anniversaries (contextual modifiers)
            "Life_Event_Context": {
                "Personal_Milestones": {
                    "Birthday_Period_Indicator": {},
                    "Anniversary_Indicator": {},
                    "Grief_Anniversary_Indicator": {},
                    "Major_Life_Transition_Indicator": {},
                },
                "Interpersonal_Events": {
                    "Relationship_Conflict_Event_Indicator": {},
                    "Breakup_Indicator": {},
                    "Reunion_Indicator": {},
                    "Bereavement_Event_Indicator": {},
                },
                "Work_And_Academic_Events": {
                    "Job_Change_Indicator": {},
                    "Performance_Review_Indicator": {},
                    "Graduation_Period_Indicator": {},
                    "Relocation_For_Work_Indicator": {},
                },
                "Health_Events": {
                    "Medical_Appointment_Today_Indicator": {},
                    "Symptom_Flare_Event_Indicator": {},
                    "Injury_Event_Indicator": {},
                },
                "Legal_And_Housing_Events": {
                    "Housing_Move_Indicator": {},
                    "Eviction_Risk_Indicator": {},
                    "Legal_Proceeding_Indicator": {},
                },
            },

            # 12) Travel / displacement / jetlag context
            "Travel_And_Displacement_Context": {
                "Travel_Status": {
                    "Traveling_Indicator": {},
                    "Overnight_Trip_Indicator": {},
                    "International_Travel_Indicator": {},
                    "Long_Commute_Indicator": {},
                },
                "Jetlag_And_Timezone": {
                    "Timezone_Change_Indicator": {},
                    "Jetlag_Likelihood_Level": {},
                    "Circadian_Resync_Burden_Level": {},
                },
                "Accommodation_Context": {
                    "Sleeping_Away_From_Home_Indicator": {},
                    "Hotel_Indicator": {},
                    "Staying_With_Others_Indicator": {},
                    "Sleeping_Environment_Novelty_Level": {},
                },
                "Travel_Stressors": {
                    "Travel_Uncertainty_Level": {},
                    "Travel_Fatigue_Level": {},
                    "Travel_Safety_Concern_Level": {},
                },
            },

            # 13) Community / societal / macro context (optional external modifiers)
            "Community_And_Societal_Context": {
                "Public_Health_Context": {
                    "Infectious_Disease_Risk_Level": {},
                    "Local_Health_Advisory_Indicator": {},
                    "Healthcare_System_Strain_Proxy_Level": {},
                },
                "Economic_Macro_Context": {
                    "Local_Unemployment_Proxy_Level": {},
                    "Inflation_Pressure_Proxy_Level": {},
                    "Cost_Of_Living_Pressure_Level": {},
                },
                "Civic_And_Political_Context": {
                    "Election_Period_Indicator": {},
                    "Political_Unrest_Indicator": {},
                    "Community_Tension_Level": {},
                },
                "Community_Safety_Context": {
                    "Crime_Rate_Proxy_Level": {},
                    "Community_Violence_News_Indicator": {},
                },
                "Environmental_Event_Context": {
                    "Flood_Risk_Indicator": {},
                    "Earthquake_Event_Indicator": {},
                    "Heatwave_Event_Indicator": {},
                },
            },
        },
    }
}


# =============================================================================
# 2) OWL builder (classes + subclass edges)
# =============================================================================

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _as_label(name: str) -> str:
    return name.replace("_", " ")


def _walk_taxonomy(parent: str, subtree: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    for child, child_subtree in subtree.items():
        yield (parent, child)
        if isinstance(child_subtree, dict) and child_subtree:
            yield from _walk_taxonomy(child, child_subtree)


def _all_nodes(taxonomy: Dict[str, Any]) -> Set[str]:
    nodes: Set[str] = set()

    def rec(d: Dict[str, Any]) -> None:
        for k, v in d.items():
            nodes.add(k)
            if isinstance(v, dict) and v:
                rec(v)

    rec(taxonomy)
    return nodes


def _validate_names(taxonomy: Dict[str, Any]) -> None:
    """
    Validates that every node name is a safe compact identifier for URI fragments.
    """
    bad = [n for n in _all_nodes(taxonomy) if not _NAME_RE.match(n)]
    if bad:
        preview = ", ".join(bad[:25])
        more = "" if len(bad) <= 25 else f" (+{len(bad) - 25} more)"
        raise ValueError(
            "Invalid taxonomy node identifiers (must match ^[A-Za-z_][A-Za-z0-9_]*$): "
            f"{preview}{more}"
        )


def build_context_owl(
    taxonomy: Dict[str, Any],
    base_iri: str = "http://example.org/phoenix/context#",
) -> Graph:
    """
    Creates an OWL ontology where every entity is an owl:Class and hierarchy is rdfs:subClassOf.
    """
    if len(taxonomy.keys()) != 1:
        raise ValueError("Taxonomy must have exactly one root key (e.g., {'Context': {...}}).")

    _validate_names(taxonomy)

    g = Graph()
    NS = Namespace(base_iri)

    ontology_iri = URIRef(base_iri.rstrip("#"))
    g.add((ontology_iri, RDF.type, OWL.Ontology))

    # Declare each node as an owl:Class with a human label
    for node in _all_nodes(taxonomy):
        node_uri = NS[node]
        g.add((node_uri, RDF.type, OWL.Class))
        g.add((node_uri, RDFS.label, Literal(_as_label(node))))

    # Add subclass edges
    root = next(iter(taxonomy.keys()))
    root_subtree = taxonomy[root]
    for parent, child in _walk_taxonomy(root, root_subtree):
        g.add((NS[child], RDFS.subClassOf, NS[parent]))

    return g


# =============================================================================
# 3) Serialization helpers
# =============================================================================

def save_json(taxonomy: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(taxonomy, indent=2, ensure_ascii=False), encoding="utf-8")


def save_owl_xml(graph: Graph, path: Path) -> None:
    graph.serialize(destination=str(path), format="xml")


# =============================================================================
# 4) Main
# =============================================================================

def main() -> None:
    out_dir = Path(".").resolve()
    owl_path = out_dir / "context.owl"
    json_path = out_dir / "CONTEXT.json"

    g = build_context_owl(CONTEXT_TAXONOMY, base_iri="http://example.org/phoenix/context#")

    #save_owl_xml(g, owl_path)
    save_json(CONTEXT_TAXONOMY, json_path)

    print(f"[OK] Wrote OWL:  {owl_path}")
    print(f"[OK] Wrote JSON: {json_path}")
    print(f"[INFO] Total classes: {len(_all_nodes(CONTEXT_TAXONOMY))}")


if __name__ == "__main__":
    main()
