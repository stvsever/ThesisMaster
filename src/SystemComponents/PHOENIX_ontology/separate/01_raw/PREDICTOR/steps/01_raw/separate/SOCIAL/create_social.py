#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOCIAL Predictor PHOENIX_ontology Generator (Solution-Variable-Oriented)

Design principles:
- Leaf nodes represent actionable SOCIAL "solution variables" (services, supports, roles, structures, access pathways).
- No disorder-labeled branches (avoid disorder names as category labels).
- No frequency/duration/intensity parameters as nodes (no schedules, minutes, counts, Hz).
- High-resolution social + interpersonal therapeutic solutions, including intimacy/relationships.
- Writes:
  1) SOCIAL.json
  2) metadata.txt (same folder)

Override output path:
  SOCIAL_OUT_PATH="/path/to/SOCIAL/SOCIAL.json" python generate_social_ontology.py
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

def scan_forbidden_tokens(
    paths: list[tuple[str, ...]],
    forbidden_patterns: list[str],
    limit: int = 50
):
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

def build_social_ontology() -> dict:
    SOCIAL: dict = {}

    # ============================================================
    # 0) Care Navigation, Access, and Coordination (social enablers)
    # ============================================================
    access: dict = {}

    add_leaves(access, ["Care_Navigation_and_Access", "Navigation_Services"], [
        "Care_Navigation_Service",
        "Case_Management_Service",
        "Social_Work_Support_Service",
        "Community_Health_Worker_Support",
        "Peer_Navigator_Service",
        "Benefits_Navigator_Service",
        "Housing_Navigator_Service",
        "Employment_Navigator_Service",
        "Education_Navigator_Service",
        "Legal_Navigator_Service",
        "Disability_Services_Navigator",
        "Youth_Services_Navigator",
        "Older_Adult_Services_Navigator",
        "Family_Caregiver_Navigator_Service",
        "Transition_Age_Youth_Navigator",
        "Multisector_Service_Navigator",
    ])

    add_leaves(access, ["Care_Navigation_and_Access", "Service_Matching_and_Referral"], [
        "Needs_and_Goals_Intake_For_Service_Matching",
        "Strengths_And_Preferences_Informed_Service_Matching",
        "Culturally_Matched_Provider_Locating",
        "Language_Matched_Provider_Locating",
        "Trauma_Informed_Service_Option_Matching",
        "Accessibility_Compatible_Service_Matching",
        "Community_Program_Eligibility_Screening",
        "Warm_Introduction_To_Service_Staff",
        "Referral_Coordination_Service",
        "Resource_Directory_Curation_For_Client",
        "Cross_Sector_Referral_Bridging",
        "Education_Employment_And_Health_Linkage",
        "Community_Connector_Service",
        "Social_Prescribing_Referral",
    ])

    add_leaves(access, ["Care_Navigation_and_Access", "Coordination_and_Shared_Planning"], [
        "Appointment_Scheduling_Assistance",
        "Care_Plan_Shared_Documentation",
        "Care_Plan_Goal_Tracking_Support",
        "Release_Of_Information_Coordination",
        "Interagency_Case_Conference_Coordination",
        "Family_Inclusive_Planning_Consent_Process",
        "Shared_Decision_Making_Facilitation",
        "Service_Provider_Communication_Bridging",
        "Practical_Task_Splitting_And_FollowThrough_Support",
        "Aftercare_Continuity_Coordination",
        "Care_Transition_Planning",
        "Reentry_To_Community_Planning",
    ])

    add_leaves(access, ["Care_Navigation_and_Access", "Advocacy_and_Representation"], [
        "Advocacy_For_Service_Eligibility",
        "Reasonable_Adjustments_Advocacy",
        "School_Accommodations_Advocacy",
        "Workplace_Accommodations_Advocacy",
        "Anti_Discrimination_Complaints_Support",
        "Tenant_Rights_Advocacy_Support",
        "Healthcare_Billing_Dispute_Support",
        "Benefits_Appeal_Support",
        "Identity_Documentation_Advocacy_Support",
        "Victim_Advocacy_Service",
        "Confidential_Advocacy_Service",
    ])

    add_leaves(access, ["Care_Navigation_and_Access", "Communication_and_Language_Access"], [
        "Language_Interpretation_Service",
        "Translated_Materials_Access",
        "Plain_Language_Health_Literacy_Support",
        "Form_Completion_Support_Service",
        "Phone_Call_Support_For_Services",
        "Email_And_Portal_Message_Composition_Support",
        "Meeting_Preparation_And_Agenda_Setting",
        "Visit_Debrief_And_Action_Steps_Summarization",
        "Supported_Self_Advocacy_Script_Preparation",
    ])

    add_leaves(access, ["Care_Navigation_and_Access", "Crisis_and_Urgent_Access"], [
        "Crisis_Line_Access_Planning",
        "Emergency_Resource_Information_Plan",
        "Crisis_Safe_Location_Planning",
        "Crisis_Deescalation_Support_Linkage",
        "Urgent_Safety_Resource_Navigation",
        "Rapid_Service_Access_Escalation_Pathway",
        "Crisis_Contact_List_Setup",
    ])

    add_leaves(access, ["Access_Barrier_Reduction_Solutions", "Financial_and_Administrative"], [
        "Cost_Barrier_Reduction_Support",
        "Insurance_Navigation_Support",
        "Sliding_Scale_Clinic_Locating",
        "Public_Service_Eligibility_Screening",
        "Documentation_Assistance_For_Services",
        "Fee_Waiver_Application_Support",
        "Payment_Plan_Negotiation_Support",
        "Benefit_Enrollment_Assistance",
        "Transportation_Voucher_Navigation",
        "Childcare_Subsidy_Navigation",
    ])

    add_leaves(access, ["Access_Barrier_Reduction_Solutions", "Digital_Access_and_Technology"], [
        "Digital_Access_Support_For_TeleServices",
        "Device_Access_Support",
        "Connectivity_And_Data_Plan_Assistance",
        "Portal_Access_And_Login_Support",
        "TeleService_Setup_And_Test_Call_Support",
        "Assistive_Technology_Access_Support",
        "Digital_Literacy_For_Service_Access",
        "Privacy_Safe_Device_Sharing_Plan",
    ])

    add_leaves(access, ["Access_Barrier_Reduction_Solutions", "Logistics_and_Time"], [
        "Transportation_Coordination_For_Appointments",
        "Childcare_Coordination_For_Appointments",
        "Flexible_Appointment_Locating_Support",
        "Multi_Appointment_Trip_Chaining_Planning",
        "Work_Schedule_Coordination_For_Care",
        "Caregiver_Scheduling_Coordination",
        "Time_Constraints_Mitigation_Planning",
        "Home_Visit_Or_Mobile_Service_Locating",
    ])

    add_leaves(access, ["Access_Barrier_Reduction_Solutions", "Accessibility_and_Disability_Accommodations"], [
        "Accessibility_Accommodations_Coordination",
        "Mobility_Access_Planning",
        "Sensory_Friendly_Service_Locating",
        "Communication_Access_Support",
        "Support_Person_Attendance_Coordination",
        "Accessible_Transportation_Planning",
        "Service_Site_Physical_Access_Verification",
    ])

    add_leaves(access, ["Access_Barrier_Reduction_Solutions", "Trust_Stigma_and_Engagement"], [
        "Stigma_Reduction_Advocacy_Support",
        "Confidentiality_Privacy_Planning_For_Care",
        "Trusted_Provider_Matching_Support",
        "First_Visit_Support_Person_Planning",
        "Fear_And_Avoidance_Barrier_Support",
        "Service_Engagement_Building_Plan",
    ])

    add_leaves(access, ["Access_Barrier_Reduction_Solutions", "Cultural_Context_and_Safety"], [
        "Cultural_Broker_Service",
        "Community_Leader_Introduction_For_Trust",
        "Culturally_Safe_Service_Planning",
        "Faith_And_Cultural_Practice_Compatibility_Planning",
        "Migration_And_Resettlement_Context_Support",
    ])

    SOCIAL["Care_Navigation_Access_and_Coordination"] = access

    # ============================================================
    # 1) Social Support, Belonging, and Community Connection
    # ============================================================
    support: dict = {}

    add_leaves(support, ["Support_Network_Development", "Peer_and_Mutual_Aid_Services"], [
        "Peer_Support_Specialist_Service",
        "Peer_Run_DropIn_Space_Connection",
        "Mutual_Aid_Connection",
        "Shared_Experience_Peer_Matching",
        "Peer_Led_Support_Group_Enrollment",
        "Online_Peer_Community_Enrollment",
        "Telephone_Support_Service",
        "Text_Based_Peer_Support_Linkage",
        "Recovery_Or_Wellbeing_Community_Connection",
    ])

    add_leaves(support, ["Support_Network_Development", "Mentorship_and_Buddying"], [
        "Mentorship_Matching_Service",
        "Buddy_System_Program",
        "Workplace_Mentorship_Connection",
        "Campus_Mentorship_Connection",
        "Newcomer_Buddy_Program_Connection",
        "Skill_Sharing_Peer_Pairing",
        "Accountability_Partner_Matching",
    ])

    add_leaves(support, ["Support_Network_Development", "Befriending_and_Outreach"], [
        "Community_Befriending_Service",
        "Volunteer_Visitor_Program",
        "Social_Isolation_Outreach_Service",
        "Home_Visit_Social_Connection_Service",
        "Friendly_Caller_Program_Connection",
        "Supported_First_Social_Contact_Planning",
    ])

    add_leaves(support, ["Support_Network_Development", "Group_Based_Connection"], [
        "Group_Coaching_Community_Format",
        "Group_Skills_Workshop_Enrollment",
        "Community_CheckIn_Group_Enrollment",
        "Interest_Based_Group_Enrollment_Support",
        "Identity_Affirming_Group_Enrollment_Support",
        "Grief_Support_Group_Enrollment",
        "Life_Transition_Support_Group_Enrollment",
        "Caregiver_Support_Group_Enrollment",
    ])

    add_leaves(support, ["Support_Network_Development", "Family_Caregiver_and_Friends_Mobilization"], [
        "Family_And_Friends_Support_Circle_Mobilization",
        "Supportive_CheckIn_Network_Setup",
        "Trusted_Person_Emergency_Contact_Plan",
        "Household_Support_Task_Coordination",
        "Caregiver_Role_Clarification_And_Support_Plan",
        "Shared_Care_Plan_With_Family_Consent",
    ])

    add_leaves(support, ["Network_Building_And_Maintenance", "Mapping_and_Assessment"], [
        "Social_Network_Mapping_Exercise",
        "Identify_Supportive_Contacts_List",
        "Relationship_Strengths_And_Gaps_Assessment",
        "Support_Needs_By_Context_Assessment",
        "Belonging_And_Community_Fit_Assessment",
    ])

    add_leaves(support, ["Network_Building_And_Maintenance", "Reconnection_and_Repair"], [
        "Reconnection_With_Lapsed_Contacts",
        "Apology_And_Repair_Script_Preparation",
        "Boundary_Reset_Conversation_Planning",
        "Relationship_Expectation_Reset_Planning",
        "Conflict_Repair_And_Reengagement_Support",
    ])

    add_leaves(support, ["Network_Building_And_Maintenance", "Initiation_and_Invitation_Support"], [
        "Community_Introductions_Requesting",
        "Structured_Social_Invitation_Scripts",
        "Conversation_Starter_Toolkit",
        "Follow_Up_Message_Templates_For_Connection",
        "Event_Attendance_With_Support_Person",
        "Gradual_Social_Exposure_Plan",
    ])

    add_leaves(support, ["Network_Building_And_Maintenance", "Reciprocity_and_Contribution"], [
        "Reciprocity_Planning_Support_Giving",
        "Skill_Based_Helping_Role_Planning",
        "Volunteer_Role_Matching_For_Contribution",
        "Mutual_Support_Exchange_Agreement",
        "Community_Role_Adoption_Plan",
    ])

    add_leaves(support, ["Belonging_Isolation_and_Meaning", "Loneliness_Reduction_Planning"], [
        "Loneliness_Reduction_Action_Plan",
        "Micro_Connection_Opportunities_Plan",
        "Meaningful_Activity_With_Others_Planning",
        "Barrier_Analysis_For_Social_Withdrawal",
        "Reengagement_After_Isolation_Plan",
    ])

    add_leaves(support, ["Belonging_Isolation_and_Meaning", "Safety_and_Trust_in_Support"], [
        "Trusted_Person_Safety_Check_Plan",
        "Boundary_Setting_With_Support_Network",
        "Support_Network_Privacy_Plan",
        "Conflict_Deescalation_With_Close_Others",
        "Reduce_Toxic_Relationship_Exposure_Plan",
    ])

    SOCIAL["Social_Support_and_Belonging"] = support

    # ============================================================
    # 2) Community Participation and Social Prescribing (high-resolution, NON-combinatorial)
    #    -> NO domain × stage cross-product (no loops, no repeated action sets per domain)
    #    -> Domains are represented as joinable solution-options (leaves)
    #    -> Participation pathway supports are represented ONCE (domain-agnostic)
    # ============================================================
    community: dict = {}

    # ---------------------------------------------------------------------
    # 2A) Community Participation Library (JOINABLE OPTIONS as leaf solutions)
    #     Domains are “what you can join” (solutions), not multiplied by stages/actions.
    # ---------------------------------------------------------------------

    # Movement / sport
    add_leaves(community, ["Community_Participation_Library", "Movement_and_Sport", "Low_Barrier_Activity_Groups"], [
        "Walking_Group",
        "Running_Group",
        "Cycling_Group",
        "Hiking_Community",
        "Mindful_Movement_Community",
    ])

    add_leaves(community,
               ["Community_Participation_Library", "Movement_and_Sport", "Skill_Building_and_Practice_Communities"], [
                   "Yoga_Community",
                   "Pilates_Community",
                   "Swimming_Group",
                   "Strength_Training_Community",
                   "Martial_Arts_Community",
                   "Dance_Community",
               ])

    add_leaves(community, ["Community_Participation_Library", "Movement_and_Sport", "Team_and_League_Communities"], [
        "Team_Sport_Community",
        "Outdoor_Adventure_Community",
    ])

    # Arts / culture
    add_leaves(community, ["Community_Participation_Library", "Arts_and_Culture", "Performing_Arts_and_Music"], [
        "Choir_or_Singing_Community",
        "Music_Jam_Community",
        "Theatre_Community",
    ])

    add_leaves(community, ["Community_Participation_Library", "Arts_and_Culture", "Visual_Arts_and_Creative_Practice"],
               [
                   "Painting_Art_Community",
                   "Photography_Community",
                   "Crafts_Community",
                   "Creative_Writing_Community",
               ])

    add_leaves(community, ["Community_Participation_Library", "Arts_and_Culture", "Spaces_and_Programs"], [
        "Maker_Space_Community",
        "Film_Community",
        "Museum_Community_Program",
    ])

    # Learning / hobby
    add_leaves(community, ["Community_Participation_Library", "Learning_and_Hobby", "Language_and_Discussion"], [
        "Language_Exchange_Community",
        "Book_Club_Community",
        "Philosophy_Community",
        "History_Community",
    ])

    add_leaves(community, ["Community_Participation_Library", "Learning_and_Hobby", "Games_and_Strategy"], [
        "Board_Games_Community",
        "Chess_Community",
    ])

    add_leaves(community, ["Community_Participation_Library", "Learning_and_Hobby", "HandsOn_and_Lifestyle_Skills"], [
        "Cooking_Community",
        "Gardening_Community",
        "Nature_Community",
        "Meditation_Community",
    ])

    add_leaves(community, ["Community_Participation_Library", "Learning_and_Hobby", "STEM_and_Technology"], [
        "Science_Community",
        "Math_Community",
        "Coding_Community",
        "AI_Community",
    ])

    # Social / service
    add_leaves(community,
               ["Community_Participation_Library", "Social_and_Service", "Prosocial_Action_and_Volunteering"], [
                   "Volunteer_Service_Community",
                   "Animal_Shelter_Volunteer_Community",
                   "Food_Distribution_Volunteer_Community",
                   "Elder_Support_Volunteer_Community",
                   "Youth_Mentoring_Community",
               ])

    add_leaves(community, ["Community_Participation_Library", "Social_and_Service", "Mutual_Aid_and_Local_Action"], [
        "Mutual_Aid_Community",
        "Neighborhood_Improvement_Community",
        "Environmental_Action_Community",
    ])

    add_leaves(community, ["Community_Participation_Library", "Social_and_Service", "Civic_and_Advocacy"], [
        "Civic_Engagement_Community",
        "Advocacy_Community",
    ])

    # Identity / culture
    add_leaves(community,
               ["Community_Participation_Library", "Identity_and_Culture", "Cultural_and_Diaspora_Communities"], [
                   "Cultural_Heritage_Community",
                   "Immigrant_Integration_Community",
                   "International_Student_Community",
               ])

    add_leaves(community, ["Community_Participation_Library", "Identity_and_Culture", "Affirming_Communities"], [
        "Disability_Affirming_Community",
        "Neurodiversity_Affirming_Community",
        "LGBTQIA_Affirming_Community",
        "Mens_Community",
        "Womens_Community",
    ])

    add_leaves(community, ["Community_Participation_Library", "Identity_and_Culture", "Faith_and_Interfaith"], [
        "Interfaith_Dialogue_Community",
    ])

    # Life-stage / roles
    add_leaves(community, ["Community_Participation_Library", "Life_Stage_and_Roles", "Parenting_and_Care_Roles"], [
        "New_Parent_Community",
        "Caregiver_Community",
    ])

    add_leaves(community, ["Community_Participation_Library", "Life_Stage_and_Roles", "Age_Cohort_Communities"], [
        "Young_Adult_Community",
        "Midlife_Community",
        "Older_Adult_Community",
        "Retirement_Community",
    ])

    # Everyday connection (low-barrier third-place belonging)
    add_leaves(community,
               ["Community_Participation_Library", "Everyday_Connection", "Third_Place_Regulars_and_Routines"], [
                   "Community_Cafe_Regulars_Community",
                   "Local_Market_Community",
               ])

    add_leaves(community, ["Community_Participation_Library", "Everyday_Connection", "Activity_Routine_Anchors"], [
        "Dog_Walking_Community",
        "CoWorking_Community",
    ])

    # ---------------------------------------------------------------------
    # 2B) Participation Pathway Supports (DOMAIN-AGNOSTIC; defined ONCE)
    #     These are the actionable “how to join/maintain” solutions.
    # ---------------------------------------------------------------------

    add_leaves(community, ["Community_Participation_Pathway_Supports", "Discovery_and_Fit"], [
        "Option_Scouting_Assistance",
        "Community_Program_Matching_Service",
        "Interest_And_Values_Fit_Clarification",
        "Skill_Level_And_Comfort_Fit_Matching",
        "Beginner_Friendly_Option_Locating",
        "Trial_Attendance_Selection_Support",
        "Identity_And_Culture_Safety_Fit_Check",
        "Accessibility_Fit_Check_For_Community",
        "Sensory_Friendly_Option_Locating",
        "Cost_And_Equipment_Burden_Check",
    ])

    add_leaves(community, ["Community_Participation_Pathway_Supports", "Joining_and_Onboarding"], [
        "First_Attendance_Planning",
        "Registration_Or_Signup_Assistance",
        "Organizer_Or_Facilitator_Introduction",
        "Buddy_Or_Greeter_Connection",
        "Group_Norms_Orientation",
        "Arrival_And_Exit_Plan_For_Anxiety_Safety",
        "Self_Introduction_And_Sharing_Boundary_Toolkit",
        "Conversation_Starters_For_New_Groups",
        "Communication_Scripts_For_Group_Joining",
    ])

    add_leaves(community, ["Community_Participation_Pathway_Supports", "Sustaining_and_Integrating"], [
        "Participation_Consistency_Support_Plan",
        "Reengagement_After_Absence_Plan",
        "Dropout_Prevention_And_Reengagement_Plan",
        "Accountability_Partner_For_Community_Participation",
        "Social_Connection_Bridging_Within_Group",
        "Role_Clarity_As_New_Member_Coaching",
        "Energy_And_Pacing_Planning_For_Attendance",
        "Overcommitment_Boundary_Planning",
    ])

    add_leaves(community, ["Community_Participation_Pathway_Supports", "Contribution_and_Role_Growth"], [
        "Volunteer_Role_Onramp",
        "Skill_Sharing_Role_Onramp",
        "Mentoring_Others_Within_Community",
        "Co_Facilitation_Or_Leadership_Onramp",
        "Community_Project_Participation",
        "Prosocial_Contribution_Planning",
    ])

    add_leaves(community, ["Community_Participation_Pathway_Supports", "Access_Logistics_and_Practical_Barriers"], [
        "Transportation_And_Route_Planning_For_Community",
        "Transportation_And_Logistics_Support_For_Participation",
        "Equipment_And_Supplies_Access_For_Participation",
        "Membership_Fee_And_Cost_Barrier_Support",
        "Childcare_Coordination_For_Community_Participation",
        "Caregiver_Scheduling_Support_For_Community",
        "Language_Access_For_Community_Programs",
    ])

    add_leaves(community, ["Community_Participation_Pathway_Supports", "Safety_Boundaries_and_Conflict"], [
        "Boundary_Setting_For_Group_Participation",
        "Conflict_Resolution_Pathway_Within_Group",
        "Conflict_Mediation_Linkage_For_Community",
        "Harassment_And_Safety_Response_Pathway",
        "Privacy_And_Digital_Safety_For_Group_Engagement",
        "Substance_And_Risk_Environment_Safety_Planning",
    ])

    # ---------------------------------------------------------------------
    # 2C) Community Infrastructure Enablers (systems-level access supports)
    # ---------------------------------------------------------------------
    add_leaves(community, ["Community_Infrastructure_Enablers", "Access_Resources"], [
        "Membership_Fee_Assistance_Fund_Access",
        "Equipment_Lending_Program_Access",
        "Transportation_Voucher_For_Community_Access",
        "Accessible_Venue_Locating_Support",
        "Community_Space_Pass_Or_Entry_Support",
        "Sliding_Scale_Program_Locating_Support",
    ])

    # ---------------------------------------------------------------------
    # 2D) Social Prescribing and Community Linkage (kept separate; high resolution)
    # ---------------------------------------------------------------------
    add_leaves(community, ["Social_Prescribing_and_Community_Linkage", "Assessment_and_Plan"], [
        "Social_Prescribing_Assessment",
        "Values_And_Interests_Discovery_For_Prescribing",
        "Community_Role_And_Purpose_Assessment",
        "Barriers_To_Engagement_Assessment",
        "Community_Referral_Plan",
        "Stepwise_Engagement_Plan",
    ])

    add_leaves(community, ["Social_Prescribing_and_Community_Linkage", "Link_Worker_and_FollowThrough_Support"], [
        "Social_Prescribing_Link_Worker_Service",
        "Warm_Hand_Introduction_To_Community_Program",
        "First_Attendance_Support",
        "Problem_Solving_Barriers_During_Engagement",
        "Motivation_And_Confidence_Support_For_Engagement",
        "Reengagement_Support_After_Dropout",
    ])

    add_leaves(community, ["Social_Prescribing_and_Community_Linkage", "Practical_Barrier_Supports"], [
        "Transportation_Support_For_Community_Program",
        "Accessibility_Support_For_Community_Program",
        "Financial_Subsidy_For_Community_Program",
        "Childcare_Support_For_Community_Program",
        "Language_Access_Support_For_Community_Program",
    ])

    SOCIAL["Community_Participation_and_Social_Prescribing"] = community

    # ============================================================
    # 3) Romantic Relationships, Intimacy, and Sexual Health (social)
    #    Expanded for high coverage (depth + breadth)
    #    Leaf nodes = actionable solution-variables (interventions/services/supports/protocols)
    # ============================================================
    intimacy: dict = {}

    # ------------------------------------------------------------------
    # A) Romantic Relationship Skills and Support (formation -> maintenance)
    # ------------------------------------------------------------------

    # A1) Values, readiness, identity, and intention setting (pre-relationship + early)
    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Values_Readiness_and_Intentions", "Clarification_and_Preparation"], [
        "Values_Aligned_Relationship_Goals_Clarification",
        "Dealbreakers_And_Nonnegotiables_Clarification",
        "Relationship_Readiness_Self_Assessment_Toolkit",
        "Attachment_Preferences_And_Needs_Reflection_Exercise",
        "Healthy_Relationship_Modeling_Psychoeducation",
        "Dating_Purpose_And_Intention_Setting_Plan",
        "Self_Worth_And_Standards_Anchoring_Coaching",
        "Loneliness_Driven_Decision_Risk_Check_Tool",
        "Rebound_Risk_And_Impulse_Dating_Guardrail_Plan",
        "Emotional_Availability_Check_And_Support_Plan",
        "Boundary_Language_Practice_For_Early_Dating",
        "Communication_Style_Self_Profile_For_Relationships",
        "Sexual_Values_And_Safety_Preferences_Clarification",
        "Family_Planning_Intentions_Clarification_Tool",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Values_Readiness_and_Intentions", "Identity_and_Context_Affirmation"], [
        "Sexual_Orientation_Affirming_Relationship_Support",
        "Gender_Identity_Affirming_Relationship_Support",
        "Cultural_Values_Integration_For_Relationship_Goals",
        "Faith_And_Relationship_Compatibility_Clarification",
        "Migration_And_Relationship_Context_Support",
        "Language_And_CrossCultural_Communication_Planning",
        "Disability_Aware_Dating_Access_Planning",
        "Neurodiversity_Aware_Relationship_Communication_Planning",
        "Trauma_Informed_Dating_Readiness_Support",
        "Body_Image_And_Dating_Confidence_Coaching",
    ])

    # A2) Initiation and Dating (opportunities -> early-stage -> commitment)
    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Initiation_and_Dating", "Opportunity_Creation_and_Meeting"], [
        "Values_Aligned_Dating_Strategy",
        "Interest_Based_Partner_Meeting_Plan",
        "Social_Opportunity_Expansion_For_Dating_Plan",
        "Friendship_Network_Bridging_For_Romantic_Introductions",
        "Approach_Anxiety_Exposure_Ladder_For_Dating",
        "Flirting_Skills_Coaching_Respectful",
        "Asking_Someone_Out_Scripts_And_Practice",
        "Rejection_Resilience_After_Dating_Setbacks_Coaching",
        "Handling_Ghosting_And_Ambiguous_Loss_Support",
        "Dating_Communication_Etiquette_Toolkit",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Initiation_and_Dating", "Online_Dating_Profile_and_Messaging"], [
        "Online_Dating_Safety_Practices",
        "Dating_Profile_And_Messaging_Coaching",
        "Photo_And_Bio_Selection_Guidance_For_Authenticity",
        "Messaging_Skills_For_Reciprocity_And_Respect",
        "Screening_For_Values_Fit_Question_Prompts",
        "Early_Stage_Red_Flag_Screening_Checklist",
        "Catfishing_And_Scam_Avoidance_Safety_Plan",
        "Privacy_Preserving_Contact_Sharing_Plan",
        "Unwanted_Sexual_Messaging_Response_Scripts",
        "Disengagement_And_Blocking_Boundary_Scripts",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Initiation_and_Dating", "Early_Dating_and_First_Dates"], [
        "First_Date_Anxiety_Support_Plan",
        "First_Date_Safety_Logistics_Plan",
        "First_Date_Conversation_Starter_Toolkit",
        "Authentic_Self_Disclosure_Pacing_Guide",
        "Interest_And_Reciprocity_Check_Framework",
        "Consent_Forward_Dating_Communication_Scripts",
        "Physical_Intimacy_Pacing_And_Consent_Plan",
        "Alcohol_And_Risk_Context_Dating_Safety_Plan",
        "Post_Date_Reflection_And_Decision_Framework",
        "FollowUp_Message_Templates_For_Dating",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Initiation_and_Dating", "Transition_to_Commitment"], [
        "Defining_The_Relationship_Conversation_Planning",
        "Exclusive_Dating_Agreement_Drafting_Toolkit",
        "Friendship_To_Romance_Conversation_Planning",
        "Early_Conflict_And_Mismatch_Navigation_Coaching",
        "Boundary_And_Expectation_Setting_Early_Relationship",
        "Integrating_Into_Each_Others_Social_Circles_Plan",
        "Introducing_To_Family_And_Friends_Planning",
        "Attachment_Needs_Conversation_Guide_Early_Stage",
        "Values_Alignment_CheckIn_Protocol_Early_Stage",
    ])

    # A3) Relationship Maintenance and Enrichment (connection, meaning, rituals)
    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Relationship_Maintenance_and_Enrichment", "Connection_Rituals_and_Quality_Time"], [
        "Shared_Meaning_and_Rituals_Design",
        "Relationship_CheckIn_Protocol_Design",
        "Quality_Time_Planning_Framework",
        "Device_Free_Couple_Time_Ritual_Design",
        "Affection_And_Closeness_Planning",
        "Novelty_And_Playfulness_Injection_Plan",
        "Shared_Hobbies_And_Activities_Planning",
        "Micro_Connection_Habits_Design_For_Couples",
        "Goodbye_And_Reunion_Rituals_CoDesign",
        "Long_Term_Vision_Building_As_Couple",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Relationship_Maintenance_and_Enrichment", "Positive_Interactions_and_Appreciation"], [
        "Appreciation_And_Gratitude_Rituals",
        "Strengths_Spotting_And_Validation_Practice_As_Couple",
        "Fondness_And_Admiration_Building_Practice",
        "Positive_Override_And_Repair_Buffering_Plan",
        "Compliment_And_Affection_Language_Matching_Coaching",
        "Kindness_MicroBehaviors_Practice",
        "Celebration_And_Achievement_Sharing_Ritual",
        "Shared_Humor_And_Play_Cultivation",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Relationship_Maintenance_and_Enrichment", "Emotional_Intimacy_and_Vulnerability"], [
        "Vulnerability_Sharing_Scaffolding_Protocol",
        "Emotional_Safety_CoCreation_Plan",
        "Needs_And_Requests_Language_Framework",
        "Support_Seeking_And_Support_Giving_Scripts",
        "Empathic_Attunement_Practice_As_Couple",
        "Attachment_Injury_Repair_Conversation_Guide",
        "Shared_Stress_Debrief_Ritual",
        "Co_Regulation_Practice_As_Couple",
        "Self_Soothing_And_Couple_Soothing_Balancing_Plan",
        "Shame_Sensitive_Conversation_Planning",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Relationship_Maintenance_and_Enrichment", "Care_Roles_and_Emotional_Labor"], [
        "Invisible_Labor_And_Mental_Load_Mapping_Exercise",
        "Division_Of_Labor_Negotiation",
        "Support_Roles_And_Care_Expectations_Agreement",
        "Burnout_Prevention_In_Couple_Caregiving_Plan",
        "Compassion_Fatigue_Risk_Mitigation_Couple_Plan",
        "Partner_Support_Boundaries_For_Emotional_Labor",
        "Balancing_Independence_And_Togetherness_Coaching",
    ])

    # A4) Communication and Connection (skills, difficult talks, repair)
    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Communication_and_Connection", "Skills_and_Frameworks"], [
        "Relationship_Education_Workshop",
        "Communication_Skills_For_Couples_Workshop",
        "Active_Listening_Practice_As_Couple",
        "Validation_And_Reflective_Listening_Practice",
        "Nondefensive_Responding_Skills_Practice",
        "I_Statement_And_Impact_Language_Coaching",
        "Curiosity_And_Assumption_Checking_Practice_As_Couple",
        "Difficult_Conversation_Planning_Couple_Format",
        "Boundary_Communication_Scripts_For_Couples",
        "Repair_Attempt_Practice_For_Couples",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Communication_and_Connection", "CheckIns_and_Problem_Solving"], [
        "Structured_Problem_Solving_As_Couple_Protocol",
        "Agenda_Setting_For_Couple_Meetings_Toolkit",
        "Conflict_Trigger_Mapping_As_Couple_Exercise",
        "Expectation_Alignment_CheckIn_Protocol",
        "Appreciation_And_Request_Balance_CheckIn_Tool",
        "Decision_Making_Framework_For_Couples",
        "Tradeoff_Negotiation_Framework_For_Couples",
        "Feedback_Conversation_Support_As_Couple",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Communication_and_Connection", "Stress_and_CoRegulation"], [
        "Stress_Spillover_Prevention_Plan",
        "Work_Stress_Decompression_Ritual_CoDesign",
        "High_Arousal_Conversation_TimeOut_Protocol",
        "Deescalation_Cues_And_Signals_Agreement",
        "Grounding_And_Return_To_Connection_Steps",
        "Partner_Support_During_Panic_Or_Overwhelm_Plan",
        "Sleep_Fatigue_And_Irritability_Risk_Mitigation_Plan",
    ])

    # A5) Conflict, Boundaries, Trust (fair fighting, repair, jealousy, tech)
    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Conflict_Boundaries_and_Trust", "Conflict_Deescalation_and_Fair_Fighting"], [
        "Conflict_Deescalation_Routine_For_Couples",
        "Fair_Fighting_Rules_Agreement",
        "Soft_Startup_Practice_For_Complaints",
        "Escalation_Early_Warning_Signs_Identification",
        "Mutual_TimeOut_And_Return_Protocol",
        "Criticism_Defensiveness_Contempt_Stonewalling_Reduction_Plan",
        "Repair_Attempt_Menu_CoDesign",
        "Apology_And_Repair_Script_Preparation_As_Couple",
        "Post_Conflict_Reconnection_Ritual_Design",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Conflict_Boundaries_and_Trust", "Boundaries_Roles_and_Expectations"], [
        "Boundary_Setting_As_Couple",
        "Personal_Privacy_And_Sharing_Agreements",
        "Role_Clarity_And_Expectations_CoDesign",
        "Shared_Household_Rules_And_Routines_Agreement",
        "Financial_Transparency_Conversation_Guide",
        "Spending_And_Saving_Values_Alignment_Tool",
        "Extended_Family_Contact_Boundaries_Agreement",
        "In_Law_And_Family_Boundary_Planning",
        "Friendship_Boundaries_And_OppositeSex_Friendship_Agreement",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Conflict_Boundaries_and_Trust", "Trust_Building_and_Maintenance"], [
        "Trust_Building_MicroBehaviors_Practice",
        "Reliability_And_FollowThrough_Agreement",
        "Transparency_And_Reassurance_Behaviors_Plan",
        "Bids_For_Connection_Recognition_Practice",
        "Rupture_And_Repair_Tracking_Tool_For_Couples",
        "Forgiveness_Process_Support_Framework",
        "Trust_Rebuilding_Action_Plan_After_Rupture",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Conflict_Boundaries_and_Trust", "Jealousy_Insecurity_and_Reassurance"], [
        "Jealousy_And_Insecurity_Conversation_Framework",
        "Reassurance_Request_Scripts_And_Response_Scripts",
        "Comparison_And_Threat_Trigger_Management_Plan",
        "Attachment_Reassurance_Rituals_CoDesign",
        "Social_Media_Jealousy_Trigger_Reduction_Plan",
        "Managing_Trust_Triggers_From_Past_Relationships_Coaching",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Conflict_Boundaries_and_Trust", "Betrayal_And_Repair_Work"], [
        "Disclosure_And_Accountability_Conversation_Protocol",
        "Repair_After_Betrayal_Therapeutic_Program",
        "Affair_Impact_Processing_Support_Plan",
        "Transparency_Agreement_Post_Betrayal",
        "Boundaries_Restoration_After_Betrayal_Plan",
        "Triggers_And_Reassurance_Plan_Post_Betrayal",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Conflict_Boundaries_and_Trust", "Technology_Privacy_and_Digital_Life"], [
        "Technology_Use_Boundary_As_Couple",
        "Phone_Privacy_And_Password_Sharing_Agreement",
        "Social_Media_Sharing_And_Tagging_Boundary_Agreement",
        "Digital_Communication_Tone_And_Timing_Agreement",
        "Online_Flirting_Boundary_Negotiation_Coaching",
        "Pornography_Boundary_Negotiation_Coaching",
        "CoParenting_App_Use_Setup_For_Separated_Parents",
    ])

    # A6) Sexual communication, consent, pleasure, and intimacy practices
    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Sexual_Communication_and_Consent", "Consent_and_Communication"], [
        "Consent_And_Sexual_Communication_Coaching",
        "Consent_Forward_Dating_Communication_Scripts",
        "Yes_No_Maybe_Boundary_Exercise_Framework",
        "Asking_For_What_You_Want_Sexually_Coaching",
        "Responding_To_No_And_Boundary_Respect_Skills_Coaching",
        "Aftercare_And_Emotional_Processing_After_Sex_Planning",
        "Sexual_Frequency_Preference_Negotiation_Framework",
        "Initiation_And_Rejection_Sensitivity_Support_In_Sex",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Sexual_Communication_and_Consent", "Desire_Discrepancy_and_Arousal"], [
        "Desire_Discrepancy_Negotiation_Coaching",
        "Responsive_Desire_Education_And_Planning",
        "Desire_Building_And_Erotic_Context_Planning",
        "Stress_Fatigue_And_Desire_Interference_Plan",
        "Touch_And_Affection_Gradient_Planning",
        "Pressure_Reduction_And_Playful_Exploration_Plan",
        "TurnOns_TurnOffs_Communication_Exercise",
        "Sexual_Compatibility_Conversation_Guide",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Sexual_Communication_and_Consent", "Pleasure_Body_Confidence_and_Education"], [
        "Pleasure_And_Body_Communication_Education",
        "Anatomy_And_Pleasure_Education_Session",
        "Body_Image_And_Sexual_Confidence_Coaching",
        "Mindfulness_Based_Sensual_Awareness_Practice",
        "Erotic_Scripts_And_Fantasy_Communication_Guide",
        "Sensual_Touch_Skillbuilding_Framework",
        "Pleasure_Focused_Goals_Setting_As_Couple",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Sexual_Communication_and_Consent", "Intimacy_Exercises_and_Reconnection"], [
        "Sensate_Focus_Exercise_Framework",
        "Nonsexual_Intimacy_And_Touch_Rebuilding_Plan",
        "Intimacy_After_Conflict_Reconnection_Plan",
        "Rebuilding_Sexual_Connection_After_Long_Pause_Plan",
        "Intimacy_Recovery_After_Health_Event_Planning",
        "Gradual_Exposure_To_Touch_After_Aversive_Experiences_Plan",
        "Pleasure_First_No_Goal_Sex_Framework",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Sexual_Communication_and_Consent", "Boundaries_Safety_and_Exploration"], [
        "Sexual_Boundaries_And_Safety_Plan",
        "Condom_And_Barrier_Method_Negotiation_Scripts",
        "Safer_Sex_Agreement_CoDesign",
        "Substance_And_Sex_Risk_Reduction_Plan",
        "Kink_Boundary_And_Safety_Communication_Coaching",
        "Power_Dynamics_And_Consent_Clarity_Coaching",
        "Sexual_Curiosity_And_Experimentation_Guide_Consensual",
    ])

    # A7) Relationship structures and contexts (monogamy/nonmonogamy, distance, cohabitation)
    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Relationship_Structure_and_Life_Context", "Defining_Structure_and_Agreements"], [
        "Monogamy_Agreement_CoDesign",
        "Exclusive_Dating_Agreement_Drafting_Toolkit",
        "Ethical_Nonmonogamy_Orientation_Education",
        "Opening_Relationship_Conversation_Planning",
        "Nonmonogamy_Boundary_And_Agreement_CoDesign",
        "Metamour_And_Network_Communication_Planning",
        "Safer_Sex_Agreement_For_Nonmonogamy_CoDesign",
        "Jealousy_And_Compersion_Skills_Coaching",
        "Reclosing_Relationship_Or_Pause_Agreement_Planning",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Relationship_Structure_and_Life_Context", "Cohabitation_and_Shared_Life_Admin"], [
        "Cohabitation_Readiness_Assessment_Toolkit",
        "Moving_In_Together_Planning_Support",
        "Household_Roles_And_Routines_CoDesign_For_Couple",
        "Shared_Budgeting_And_Accounts_Options_Exploration",
        "Domestic_Labor_Equity_Planning",
        "Shared_Calendar_And_Task_System_Setup_As_Couple",
        "Guest_And_Hosting_Boundary_Agreement",
        "Shared_Space_Privacy_And_Alone_Time_Agreement",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Relationship_Structure_and_Life_Context", "Long_Distance_and_Travel_Context"], [
        "Long_Distance_Connection_Rituals_Design",
        "Long_Distance_Trust_And_Transparency_Agreement",
        "Visit_Planning_And_Expectations_Setting_Toolkit",
        "Time_Zone_And_Communication_Window_Planning",
        "Reunion_And_Separation_Transition_Support_Plan",
        "Travel_Separation_And_Work_Trip_Coping_Plan",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Relationship_Structure_and_Life_Context", "Cultural_Family_and_Social_Context"], [
        "Intercultural_Couple_Values_Integration_Coaching",
        "Interfaith_Couple_Conversation_Guide",
        "Language_Barrier_Communication_Strategy_For_Couples",
        "Family_Approval_And_Boundary_Navigation_Coaching",
        "Cultural_Role_Expectations_And_Gender_Norms_Negotiation",
        "Navigating_Stigma_And_Minority_Stress_As_Couple_Support",
    ])

    add_leaves(intimacy, ["Romantic_Relationship_Skills_and_Support", "Relationship_Structure_and_Life_Context", "Health_Disability_and_Care_Context"], [
        "Disability_Aware_Intimacy_Adaptation_Planning",
        "Chronic_Illness_Impact_On_Relationship_Communication_Plan",
        "Caregiving_Role_Negotiation_As_Couple",
        "Medical_Appointments_Attendance_And_Support_Role_Planning",
        "Fatigue_And_Pacing_Planning_For_Intimacy",
        "Medication_Impact_On_Desire_Conversation_Preparation",
        "Sensory_Aware_Intimacy_Planning_Coaching",
    ])

    # ------------------------------------------------------------------
    # B) Relational Therapeutic Services (clinical-grade, specific routing targets)
    # ------------------------------------------------------------------
    add_leaves(intimacy, ["Relational_Therapeutic_Services", "Couple_and_Relationship_Services", "Evidence_Informed_Modalities"], [
        "Couple_Therapy_Emotionally_Focused_Modality",
        "Couple_Therapy_Behavioral_Modality",
        "Couple_Therapy_Integrative_Modality",
        "Couple_Therapy_Gottman_Informed_Modality",
        "Couple_Therapy_Systemic_Modality",
        "Couple_Therapy_Solution_Focused_Modality",
        "Couple_Therapy_Narrative_Modality",
        "Couple_Therapy_Imago_Informed_Modality",
        "Couple_Therapy_Communication_Skills_Focused_Modality",
        "Couple_Therapy_Attachment_Based_Modality",
    ])

    add_leaves(intimacy, ["Relational_Therapeutic_Services", "Couple_and_Relationship_Services", "Decision_and_Discernment"], [
        "Discernment_Counseling_Service",
        "Stay_Leave_Decision_Support_Coaching",
        "Commitment_Alignment_Discussion_Facilitation",
        "Values_And_Life_Path_Divergence_Mediation",
        "Readiness_For_Cohabitation_Counseling_Service",
        "Pre_Marital_Or_Partnership_Preparation_Counseling",
    ])

    add_leaves(intimacy, ["Relational_Therapeutic_Services", "Intimacy_and_Sex_Therapy_Services", "Clinical_Services"], [
        "Sex_Therapy_Service_Clinician_Led",
        "Intimacy_Coaching_Service",
        "Sensate_Focus_Clinician_Guided_Program",
        "Desire_Discrepancy_Therapy_Service",
        "Sexual_Anxiety_And_Avoidance_Therapy_Service",
        "Pleasure_And_Orgasm_Skills_Therapy_Service",
        "Sexual_Pain_Functional_Assessment_And_Treatment_Planning",
        "Trauma_Informed_Sex_Therapy_Service",
    ])

    add_leaves(intimacy, ["Relational_Therapeutic_Services", "Specialized_Couple_Services", "Context_Specific"], [
        "Trauma_Informed_Couple_Therapy_Service",
        "Neurodiversity_Aware_Couple_Therapy_Service",
        "Disability_Aware_Couple_Therapy_Service",
        "Intercultural_Or_Interfaith_Couple_Therapy_Service",
        "LGBTQIA_Affirming_Couple_Therapy_Service",
        "High_Conflict_Couple_Therapy_Stabilization_Service",
        "Coercion_Risk_Informed_Relational_Consult_Service",
    ])

    add_leaves(intimacy, ["Relational_Therapeutic_Services", "Group_and_Workshop_Formats", "Couples_Programs"], [
        "Couples_Communication_Skills_Group_Program",
        "Couples_Conflict_Repair_Skills_Group_Program",
        "Couples_Intimacy_Enrichment_Workshop",
        "Couples_Transition_To_Parenthood_Workshop",
        "Couples_Rebuilding_After_Rupture_Group_Program",
        "Nonmonogamy_Agreement_Skills_Workshop",
    ])

    # ------------------------------------------------------------------
    # C) Separation, Breakup, and Co-Parenting Support
    # ------------------------------------------------------------------
    add_leaves(intimacy, ["Relational_Therapeutic_Services", "Separation_and_CoParenting_Support", "Separation_Planning_and_Communication"], [
        "Mediation_For_Relationship_Separation",
        "Separation_Communication_Protocol_Planning",
        "Separation_Boundary_And_Contact_Rules_Agreement",
        "Property_And_Belongings_Division_Conversation_Planning",
        "Mutual_Friends_And_Social_Circle_Navigation_Plan",
        "Breakup_Conversation_Safety_And_Deescalation_Plan",
        "Post_Separation_Housing_And_Logistics_Action_Plan",
    ])

    add_leaves(intimacy, ["Relational_Therapeutic_Services", "Separation_and_CoParenting_Support", "Emotional_Recovery_and_Adjustment"], [
        "Breakup_Grief_And_Loss_Processing_Support",
        "Rumination_And_Closure_Seeking_Reduction_Plan",
        "No_Contact_Or_Low_Contact_Boundary_Planning",
        "Co_Parenting_Or_Ex_Partner_Trigger_Management_Plan",
        "Self_Compassion_And_Self_Worth_Rebuilding_Coaching",
        "Rebuilding_Social_Support_After_Separation_Plan",
        "Dating_After_Separation_Readiness_Coaching",
    ])

    add_leaves(intimacy, ["Relational_Therapeutic_Services", "Separation_and_CoParenting_Support", "CoParenting_Services"], [
        "Co_Parenting_Mediation_Service",
        "Co_Parenting_Communication_Coaching",
        "Child_Focused_Separation_Planning",
        "Parenting_Plan_And_Routines_CoDesign_Support",
        "Conflict_Reduction_CoParenting_Protocol",
        "Parallel_Parenting_Protocol_Planning",
        "School_And_Healthcare_Communication_CoParenting_Plan",
    ])

    add_leaves(intimacy, ["Relational_Therapeutic_Services", "Separation_and_CoParenting_Support", "Practical_Navigation"], [
        "Housing_And_Finance_Separation_Navigation",
        "Benefits_And_Legal_Resource_Referral_For_Separation",
        "Budget_Rebuild_After_Separation_Support",
        "CoParenting_Scheduling_And_HandOff_Logistics_Planning",
        "Boundary_With_New_Partners_And_Blended_Family_Planning",
        "Social_Support_During_Separation_Plan",
    ])

    # ------------------------------------------------------------------
    # D) Sexual Health Access and Support (clinical + preventive + adjustment)
    # ------------------------------------------------------------------
    add_leaves(intimacy, ["Sexual_Health_Access_and_Support", "Clinical_Access_and_Care", "Primary_Sexual_Health_Services"], [
        "Sexual_Health_Clinician_Consult",
        "STI_Screening_Access_Support",
        "Contraception_Counseling_Access",
        "Emergency_Contraception_Access_Support",
        "Fertility_Counseling_Access",
        "Gynecological_Care_Access",
        "Urological_Care_Access",
        "Pelvic_Floor_Physiotherapy_Access",
        "Hormonal_And_Endocrine_Sexual_Health_Consult_Access",
    ])

    add_leaves(intimacy, ["Sexual_Health_Access_and_Support", "Clinical_Access_and_Care", "Function_and_Pain_Pathways"], [
        "Sexual_Pain_Specialist_Referral",
        "Pain_With_Intercourse_Assessment_Referral_Pathway",
        "Erection_Difficulty_Medical_Evaluation_Access",
        "Orgasm_Difficulty_Clinical_Assessment_Access",
        "Arousal_Difficulty_Clinical_Assessment_Access",
        "Post_Surgical_Or_Post_Treatment_Sexual_Function_Rehab_Referral",
        "Pelvic_Health_And_Bladder_Bowel_Interface_Assessment_Access",
    ])

    add_leaves(intimacy, ["Sexual_Health_Access_and_Support", "Prevention_and_Risk_Reduction", "Resources_and_Navigation"], [
        "Safer_Sex_Education_And_Skills_Support",
        "Condom_And_Barrier_Method_Access_Support",
        "STI_Partner_Notification_Support_Service",
        "HIV_PrEP_Access_Navigation",
        "HIV_PEP_Urgent_Access_Navigation",
        "HPV_Vaccination_Access_Support",
        "Hepatitis_Screening_And_Vaccination_Access_Support",
    ])

    add_leaves(intimacy, ["Sexual_Health_Access_and_Support", "Education_and_Adjustment_Support", "Psychoeducation_and_Skills"], [
        "Sexual_Function_Education_Session",
        "Consent_Education_Session",
        "Pleasure_And_Anatomy_Education_Session",
        "Trauma_Informed_Intimacy_Support_Service",
        "Body_Image_And_Sexual_Confidence_Coaching",
        "Medication_SideEffect_Conversation_Preparation",
        "Communication_About_STI_Status_Disclosure_Scripts",
        "Safer_Sex_Agreement_Coaching_As_Couple",
    ])

    add_leaves(intimacy, ["Sexual_Health_Access_and_Support", "Education_and_Adjustment_Support", "Life_Transitions_and_Changes"], [
        "Intimacy_After_Illness_Adjustment_Counseling",
        "Intimacy_After_Childbirth_Adjustment_Counseling",
        "Menopause_And_Aging_Sexual_Health_Adjustment_Counseling",
        "Intimacy_After_Loss_Or_Bereavement_Adjustment_Support",
        "Sexuality_And_Identity_Exploration_Support_Counseling",
        "Sexual_Health_Adjustment_After_Disability_Onset_Counseling",
        "Intimacy_After_Trauma_Reclamation_Counseling",
    ])

    add_leaves(intimacy, ["Sexual_Health_Access_and_Support", "Reproductive_and_Family_Planning_Support", "Navigation_and_Decision_Support"], [
        "Family_Planning_Decision_Support_Counseling",
        "Preconception_Health_Counseling_Access",
        "Pregnancy_And_Postpartum_Sexual_Health_Education",
        "Infertility_Stress_And_Relationship_Support_Counseling",
        "Pregnancy_Loss_Grief_And_Couple_Support",
        "Parenthood_Timing_And_Readiness_Couple_Coaching",
    ])

    # ------------------------------------------------------------------
    # E) Relationship Safety and Protection (screening, planning, linkage)
    # ------------------------------------------------------------------
    add_leaves(intimacy, ["Relationship_Safety_and_Protection", "Assessment_and_Planning", "Risk_Screening_and_Stabilization"], [
        "Relationship_Safety_Assessment_Service",
        "Safety_Planning_Service",
        "Interpersonal_Coercion_And_Control_Risk_Screening",
        "Sexual_Coercion_Risk_Screening_And_Response_Plan",
        "Reproductive_Coercion_Risk_Screening_And_Response_Plan",
        "Digital_Safety_Planning_For_Relationships",
        "Stalking_Safety_Planning_Support",
        "Safe_Contact_And_Boundary_Planning",
        "Exit_Plan_Logistics_And_Document_Safety_Planning",
    ])

    add_leaves(intimacy, ["Relationship_Safety_and_Protection", "Assessment_and_Planning", "Post_Separation_Safety"], [
        "Post_Separation_Safety_Planning_Service",
        "Child_Exchange_Safety_Logistics_Planning",
        "Home_And_Work_Safety_Routine_Planning",
        "Technology_And_Account_Security_Hardening_Support",
        "Harassment_Response_And_Documentation_Toolkit",
    ])

    add_leaves(intimacy, ["Relationship_Safety_and_Protection", "Protection_and_Resource_Linkage", "Advocacy_and_Emergency_Resources"], [
        "Domestic_Violence_Support_Service",
        "Emergency_Shelter_Access_Support",
        "Protective_Order_Navigation_Support",
        "Confidential_Advocacy_Service",
        "Court_Accompaniment_Service",
        "Victim_Advocacy_Service_Relationship_Context",
        "Financial_Independence_Planning_Support",
        "Workplace_And_School_Safety_Accommodations_Navigation",
        "Trauma_Informed_Crisis_Support_Linkage",
    ])

    add_leaves(intimacy, ["Relationship_Safety_and_Protection", "Protection_and_Resource_Linkage", "Community_and_Long_Term_Support"], [
        "Survivor_Peer_Support_Group_Linkage",
        "Trauma_Focused_Therapy_Referral_Pathway",
        "Legal_Aid_Linkage_For_Family_Safety",
        "Housing_Stability_Linkage_For_Safety",
        "Secure_Communication_Channel_Setup_Support",
    ])

    SOCIAL["Romantic_Intimacy_and_Relationships"] = intimacy


    # ============================================================
    # 4) Family, Parenting, and Household Systems (social solutions)
    # ============================================================
    family: dict = {}

    add_leaves(family, ["Family_and_Household_Services", "Family_Therapy_and_Mediation"], [
        "Family_Therapy_Systemic_Service",
        "Family_Therapy_Structural_Service",
        "Family_Therapy_Strategic_Service",
        "Family_Therapy_Communication_Focused_Service",
        "Family_Mediation_Service",
        "Household_Conflict_Resolution_Facilitation",
        "Restorative_Family_Conversation_Facilitation",
        "Sibling_Conflict_Mediation_Support",
    ])

    add_leaves(family, ["Family_and_Household_Services", "Family_Meetings_and_Care_Planning"], [
        "Care_Planning_Family_Meeting_Facilitation",
        "Family_Psychoeducation_Session",
        "Shared_Household_Goals_Setting",
        "Support_Role_Clarification_Meeting",
        "Reducing_Expressed_Emotion_Coaching",
        "Crisis_Family_Communication_Plan",
    ])

    add_leaves(family, ["Family_and_Household_Services", "Household_Systems_and_Routines"], [
        "Household_Roles_And_Rules_Redesign",
        "Chore_System_Implementation",
        "Family_Routine_Structure_Design",
        "Meal_Sharing_And_Household_Rhythm_Planning",
        "Shared_Calendar_And_Task_Management_Setup",
        "Household_Boundary_And_Privacy_Agreements",
        "Noise_And_Sensory_Environment_Home_Adjustment",
    ])

    add_leaves(family, ["Parenting_and_Caregiver_Support", "Parenting_Education_and_Coaching"], [
        "Parenting_Education_Class",
        "Parenting_Coaching_Service",
        "Emotion_Coaching_Parenting_Program",
        "Positive_Reinforcement_Parenting_Program",
        "Boundary_Setting_Parenting_Program",
        "Co_Parenting_Coordination_Service",
        "Parent_Stress_Support_Service",
        "Parent_Peer_Support_Group_Enrollment",
    ])

    add_leaves(family, ["Parenting_and_Caregiver_Support", "School_and_Community_Interface"], [
        "School_Home_Collaboration_Plan",
        "Teacher_Communication_Planning",
        "Special_Education_Services_Navigation",
        "After_School_Program_Access",
        "Youth_Mentoring_Program_Connection",
        "Family_Support_Worker_Service",
    ])

    add_leaves(family, ["Parenting_and_Caregiver_Support", "Practical_Parenting_Supports"], [
        "Childcare_Assistance_Navigation",
        "Respite_Care_For_Parents_Access",
        "Parenting_Task_Sharing_Planning",
        "Family_Logistics_And_Transport_Planning",
        "Crisis_Childcare_Contingency_Plan",
    ])

    add_leaves(family, ["Caregiving_and_Respite_Solutions", "Respite_and_Relief"], [
        "Caregiver_Support_Group_Enrollment",
        "Caregiver_Respite_Service_Access",
        "Caregiver_Burnout_Prevention_Planning",
        "Home_Care_Service_Navigation",
        "Day_Program_Access_For_Dependent",
        "Short_Term_Relief_Care_Navigation",
    ])

    add_leaves(family, ["Caregiving_and_Respite_Solutions", "Task_Sharing_and_Planning"], [
        "Family_Care_Task_Sharing_Planning",
        "Multi_Generational_Support_Planning",
        "Care_Role_Clarification_And_Boundary_Planning",
        "Care_Communication_Protocol_Planning",
        "Medication_And_Appointment_Task_Support_Planning",
    ])

    add_leaves(family, ["Caregiving_and_Respite_Solutions", "Care_Navigation_for_Dependent_Family_Member"], [
        "Care_Coordination_For_Dependent_Family_Member",
        "Assistive_Services_Navigation",
        "Accessibility_Equipment_Access_Support",
        "Advance_Care_Planning_Support",
        "Benefits_And_Disability_Support_For_Dependent",
    ])

    SOCIAL["Family_Parenting_and_Household_Systems"] = family

    # ============================================================
    # 5) Work, Education, and Role-Based Social Functioning
    # ============================================================
    roles: dict = {}

    add_leaves(roles, ["Work_and_Occupational_Solutions", "Access_Accommodations_and_Protection"], [
        "Workplace_Accommodations_Request_Support",
        "Reasonable_Adjustments_Advocacy",
        "Workplace_Bullying_Harassment_Support_Service",
        "Workplace_Grievance_Support_Service",
        "Union_or_Employee_Advocacy_Support",
        "Confidential_Disclosure_Planning",
    ])

    add_leaves(roles, ["Work_and_Occupational_Solutions", "Job_Retention_and_Support"], [
        "Return_To_Work_Planning_Service",
        "Occupational_Rehabilitation_Service",
        "Workplace_Onboarding_Support",
        "Job_Crafting_Support",
        "Workload_And_Role_Clarity_Negotiation_Support",
        "Energy_And_Pacing_For_Work_Planning",
        "Workplace_Wellbeing_Program_Enrollment",
        "Manager_Communication_Planning",
    ])

    add_leaves(roles, ["Work_and_Occupational_Solutions", "Workplace_Relationships_and_Team_Functioning"], [
        "Workplace_Conflict_Coaching",
        "Workplace_Mediation_Service",
        "Team_Communication_Workshop",
        "Feedback_Conversation_Support",
        "Boundary_Setting_For_Work_Communication",
        "Mentorship_And_Sponsorship_Connection",
        "Professional_Supervision_or_Consultation",
    ])

    add_leaves(roles, ["Work_and_Occupational_Solutions", "Career_Development_and_Employment_Access"], [
        "Career_Counseling_Service",
        "Vocational_Assessment_Service",
        "Job_Search_Support_Service",
        "CV_And_Interview_Coaching_Service",
        "Supported_Employment_Service",
        "Work_Training_Program_Navigation",
        "Professional_Networking_Support",
    ])

    add_leaves(roles, ["Education_and_Training_Context_Solutions", "Academic_Access_and_Accommodations"], [
        "Academic_Accommodations_Request_Support",
        "Disability_Services_Coordination",
        "Exam_And_Assessment_Accommodations_Support",
        "Note_Taking_And_Learning_Access_Support",
        "Accessible_Course_Materials_Support",
    ])

    add_leaves(roles, ["Education_and_Training_Context_Solutions", "Learning_Skills_and_Structure"], [
        "Study_Skills_Support_Service",
        "Learning_Support_Coaching_Service",
        "Tutoring_Service_Access",
        "Time_And_Task_Organization_Coaching",
        "Peer_Study_Group_Enrollment",
        "Advisor_Meeting_Support",
    ])

    add_leaves(roles, ["Education_and_Training_Context_Solutions", "Campus_Belonging_and_Support"], [
        "Campus_Support_Service_Linkage",
        "Mentorship_Program_Enrollment",
        "Student_Community_Group_Connection",
        "International_Student_Support_Service",
        "Financial_Aid_Navigation_Support",
    ])

    add_leaves(roles, ["Role_Transitions_and_Life_Changes", "Relocation_and_Integration"], [
        "Relocation_Settling_In_Support_Service",
        "Immigration_Integration_Service",
        "Language_And_Culture_Orientation_Linkage",
        "Newcomer_Community_Connection",
        "Local_Resource_Onboarding_Plan",
    ])

    add_leaves(roles, ["Role_Transitions_and_Life_Changes", "Life_Stage_Transitions"], [
        "Retirement_Transition_Support_Service",
        "New_Parent_Transition_Support",
        "Bereavement_Practical_Support_Service",
        "Relationship_Separation_Practical_Support",
        "Caregiver_Role_Transition_Support",
    ])

    add_leaves(roles, ["Role_Transitions_and_Life_Changes", "Reentry_and_Reintegration"], [
        "Reentry_To_Community_Support_Service",
        "Community_Reintegration_Service",
        "Housing_And_Employment_Reintegration_Support",
        "Social_Support_Rebuild_After_Transition",
        "Life_Coaching_Role_Transitions",
    ])

    SOCIAL["Work_Education_and_Role_Functioning"] = roles

    # ============================================================
    # 6) Socioeconomic, Housing, and Material Resource Supports
    # ============================================================
    resources: dict = {}

    add_leaves(resources, ["Financial_and_Benefits_Support", "Financial_Counseling_and_Protection"], [
        "Financial_Counseling_Service",
        "Budgeting_Support_Service",
        "Debt_Management_Service",
        "Banking_Access_Support",
        "Fraud_Protection_Support",
        "Credit_Report_And_Dispute_Support",
        "Consumer_Legal_Advice_Access",
    ])

    add_leaves(resources, ["Financial_and_Benefits_Support", "Benefits_and_Income_Support"], [
        "Benefits_Enrollment_Assistance",
        "Income_Support_Application_Assistance",
        "Disability_Benefit_Navigation",
        "Tax_Credit_Eligibility_Support",
        "Emergency_Financial_Assistance_Access",
        "Rent_Subsidy_Application_Assistance",
        "Utility_Assistance_Application",
    ])

    add_leaves(resources, ["Food_Transport_and_Childcare_Support", "Food_Security"], [
        "Food_Assistance_Enrollment",
        "Food_Pantry_Access",
        "Meal_Delivery_Service_Access",
        "Community_Kitchen_Program_Enrollment",
        "Nutrition_Budgeting_Support",
        "Culturally_Appropriate_Food_Resource_Locating",
    ])

    add_leaves(resources, ["Food_Transport_and_Childcare_Support", "Transportation_and_Mobility"], [
        "Transportation_Voucher_Access",
        "Public_Transport_Navigation_Support",
        "Mobility_Assistance_Service_Access",
        "Accessible_Transportation_Access_Support",
        "Travel_Training_For_Public_Transport",
    ])

    add_leaves(resources, ["Food_Transport_and_Childcare_Support", "Childcare_and_Dependent_Care"], [
        "Childcare_Subsidy_Navigation",
        "Childcare_Provider_Locating_Support",
        "After_School_Program_Access",
        "Respite_Care_Access_For_Dependents",
        "Dependent_Care_Contingency_Planning",
    ])

    add_leaves(resources, ["Housing_and_Neighborhood_Solutions", "Housing_Access_and_Stability"], [
        "Housing_Navigation_Service",
        "Housing_Stability_Planning",
        "Eviction_Prevention_Service",
        "Shelter_Access_Support",
        "Transitional_Housing_Access_Support",
        "Supported_Housing_Program_Enrollment",
        "Rapid_Rehousing_Service_Access",
        "Roommate_Agreement_Facilitation",
        "CoHousing_Community_Exploration",
    ])

    add_leaves(resources, ["Housing_and_Neighborhood_Solutions", "Housing_Quality_and_Safety"], [
        "Home_Safety_Modification_Service",
        "Accessibility_Housing_Modification_Support",
        "Home_Organization_And_Decluttering_Support",
        "Pest_Or_Environmental_Hazard_Navigation",
        "Utility_Setup_And_Account_Transfer_Support",
        "Neighborhood_Safety_Advocacy_Support",
    ])

    add_leaves(resources, ["Material_and_Digital_Needs_Support", "Basic_Needs_and_Supplies"], [
        "Clothing_Assistance_Access",
        "Hygiene_Supplies_Access",
        "School_Supplies_Access",
        "Work_Uniform_Or_Equipment_Access",
        "Assistive_Device_Access_Support",
    ])

    add_leaves(resources, ["Material_and_Digital_Needs_Support", "Digital_Inclusion"], [
        "Device_Access_Support",
        "Connectivity_And_Data_Plan_Assistance",
        "Digital_Literacy_Support_Service",
        "Safe_Public_Internet_Access_Locating",
        "Privacy_And_Security_Basics_Coaching",
    ])

    SOCIAL["Socioeconomic_and_Material_Resource_Solutions"] = resources

    # ============================================================
    # 7) Legal, Safety, and Rights-Based Supports
    # ============================================================
    legal: dict = {}

    add_leaves(legal, ["Legal_and_Rights_Support", "General_and_Civil_Legal_Aid"], [
        "General_Legal_Aid_Service",
        "Identity_Document_Replacement_Support",
        "Consumer_Law_Support_Service",
        "Debt_And_Collections_Legal_Support",
        "Family_Law_Support_Service",
        "Housing_Law_Support_Service",
        "Employment_Law_Support_Service",
        "Immigration_Legal_Support_Service",
        "Disability_Rights_Advocacy_Service",
        "Student_Rights_Advocacy_Service",
    ])

    add_leaves(legal, ["Legal_and_Rights_Support", "Justice_System_Navigation_and_Advocacy"], [
        "Police_Report_Support_Service",
        "Court_Accompaniment_Service",
        "Victim_Advocacy_Service",
        "Protective_Order_Navigation",
        "Restorative_Justice_Program_Access",
        "Mediation_Service_Community",
    ])

    add_leaves(legal, ["Safety_and_Protection_Solutions", "Personal_and_Home_Safety"], [
        "Personal_Safety_Planning_Service",
        "Home_Safety_Assessment_Service",
        "Emergency_Resource_Information_Plan",
        "Safe_Location_And_Escape_Planning",
        "Community_Safety_Resources_Linkage",
    ])

    add_leaves(legal, ["Safety_and_Protection_Solutions", "Digital_and_Identity_Safety"], [
        "CyberSafety_Support_Service",
        "Identity_Protection_Service",
        "Online_Harassment_Response_Support",
        "Privacy_Settings_Optimization",
        "Account_Recovery_And_Security_Hardening_Support",
    ])

    SOCIAL["Legal_Safety_and_Rights_Support"] = legal

    # ============================================================
    # 8) Culture, Identity, Discrimination, and Social Inclusion
    # ============================================================
    inclusion: dict = {}

    add_leaves(inclusion, ["Culture_Identity_and_Inclusion_Solutions", "Affirming_Community_Connection"], [
        "Identity_Affirming_Community_Connection",
        "Culturally_Specific_Community_Program",
        "Religious_or_Spiritual_Community_Connection",
        "Interfaith_Community_Connection",
        "Language_Community_Connection",
        "Diaspora_Community_Connection",
        "LGBTQIA_Affirming_Community_Connection",
        "Disability_Affirming_Community_Connection",
        "Neurodiversity_Affirming_Community_Connection",
        "Immigrant_Integration_Community_Connection",
    ])

    add_leaves(inclusion, ["Culture_Identity_and_Inclusion_Solutions", "Anti_Discrimination_and_Advocacy"], [
        "Anti_Racism_Support_Service",
        "Anti_Discrimination_Advocacy_Support",
        "Workplace_Inclusion_Advocacy",
        "School_Inclusion_Advocacy",
        "Accessible_Service_Standards_Advocacy",
        "Hate_Incident_Support_and_Reporting_Navigation",
        "Community_Accountability_Process_Linkage",
    ])

    add_leaves(inclusion, ["Culture_Identity_and_Inclusion_Solutions", "Belonging_and_Social_Safety_Skills"], [
        "Community_Belonging_Coaching",
        "Stigma_Reduction_Support_Group",
        "Self_Advocacy_For_Identity_Safety_Support",
        "Microaggression_Response_Script_Support",
        "Cultural_Safety_Plan_For_New_Environments",
    ])

    add_leaves(inclusion, ["Culture_Identity_and_Inclusion_Solutions", "Language_and_Communication_Access"], [
        "Language_Interpretation_Service",
        "Translated_Materials_Access",
        "Bilingual_Peer_Connector_Service",
        "Conversation_Partner_Program_Connection",
        "Cultural_Broker_Service",
    ])

    SOCIAL["Culture_Identity_and_Social_Inclusion"] = inclusion

    # ============================================================
    # 9) Digital Social Environment and Boundaries
    # ============================================================
    digital: dict = {}

    add_leaves(digital, ["Digital_Social_Environment_Solutions", "Boundaries_and_Wellbeing"], [
        "Digital_Boundary_Setting_Plan",
        "Device_Free_Social_Rituals_Design",
        "Reduce_Social_Comparison_Triggers_Strategy",
        "Curate_Social_Media_Feed_Strategy",
        "Notification_And_Attention_Management_Plan",
        "Digital_Overload_Recovery_Plan",
    ])

    add_leaves(digital, ["Digital_Social_Environment_Solutions", "Safety_Privacy_and_Conflict"], [
        "Online_Community_Safety_Practices",
        "Privacy_Settings_Optimization",
        "Mute_Block_Toxic_Contacts_Strategy",
        "Online_Conflict_Deescalation_Strategy",
        "Cyberbullying_Response_Support_Service",
        "Online_Harassment_Response_Support",
        "Digital_Communication_Norms_Agreement",
        "Account_Security_Basics_Coaching",
    ])

    add_leaves(digital, ["Digital_Social_Environment_Solutions", "Healthy_Online_Community_Engagement"], [
        "Healthy_Online_Community_Selection",
        "Moderated_Support_Community_Enrollment",
        "Prosocial_Online_Participation_Plan",
        "Online_Role_And_Contribution_Planning",
        "Online_Group_Joining_Scripts_And_Etiquette",
    ])

    SOCIAL["Digital_Social_Environment"] = digital

    # ============================================================
    # 10) Interpersonal Therapeutic Services (explicitly social-facing)
    #     Clinically graded, high-resolution, and non-redundant with domains 1–9.
    #     Scope: therapeutic/coaching/structured services that directly target interpersonal functioning.
    #     Notes:
    #       - No disorder names as category labels (complies with design principle).
    #       - Includes clinically relevant specializations (e.g., social learning/neurodevelopmental profiles)
    #         as "Clinical_Specialization_Focus" rather than disorder-named branches.
    #       - Leaf nodes are actionable solution-variables (services / protocols / structured supports).
    # ============================================================
    interpersonal: dict = {}

    # ----------------------------
    # A) Intake, Assessment, and Formulation (interpersonal mechanisms)
    # ----------------------------
    add_leaves(interpersonal,
               ["Interpersonal_Therapeutic_and_Coaching_Services", "Assessment_and_Formulation", "Intake_and_Goals"], [
                   "Interpersonal_Functioning_Intake_Session",
                   "Interpersonal_Goals_And_Values_Clarification_Session",
                   "Relationship_Context_Mapping_Session",
                   "Interpersonal_Strengths_And_Protective_Factors_Assessment",
                   "Interpersonal_Barriers_And_Triggers_Assessment",
                   "Shared_Treatment_Target_Prioritization_Session",
               ])

    add_leaves(interpersonal,
               ["Interpersonal_Therapeutic_and_Coaching_Services", "Assessment_and_Formulation", "Pattern_Assessment"],
               [
                   "Communication_Pattern_Assessment_Service",
                   "Conflict_Pattern_Assessment_Service",
                   "Interpersonal_Avoidance_And_Safety_Behavior_Assessment",
                   "Boundary_And_Role_Clarity_Assessment",
                   "Attachment_Related_Interaction_Pattern_Assessment",
                   "Social_Cognition_Profile_Assessment_PracticeBased",
                   "Emotion_Recognition_And_Labeling_Assessment",
                   "Conversation_Skills_Baseline_Assessment",
                   "Group_Interaction_Behavior_Assessment",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Assessment_and_Formulation",
                               "Risk_and_Safety_in_Relationships"], [
                   "Interpersonal_Safety_And_Coercion_Risk_Screening",
                   "Harassment_Or_Threat_Response_Planning_Session",
                   "Digital_Interpersonal_Safety_Planning_Session",
                   "Conflict_Escalation_Risk_Assessment",
                   "Protective_Boundary_Plan_CoDesign_Session",
               ])

    # ----------------------------
    # B) Core Interpersonal Skills (universal / low-to-moderate acuity)
    # ----------------------------
    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Core_Interpersonal_Skills",
                               "Foundational_Communication"], [
                   "Communication_Skills_Workshop",
                   "Active_Listening_And_Validation_Coaching",
                   "Clear_Request_Making_Skills_Coaching",
                   "Feedback_Giving_And_Receiving_Skills_Coaching",
                   "Difficult_Conversation_Planning_Coaching",
                   "Conversation_Initiation_And_Maintenance_Coaching",
                   "Repair_Attempt_Skills_Coaching",
                   "Misunderstanding_Clarification_Skills_Coaching",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Core_Interpersonal_Skills",
                               "Emotion_in_Interaction"], [
                   "Emotion_Expression_And_Request_Skills_Coaching",
                   "Emotion_Regulation_In_Conversation_Coaching",
                   "Deescalation_Skills_For_High_Arousal_Interactions",
                   "Co_Regulation_Skills_Practice",
                   "Shame_And_Embarrassment_Tolerance_In_Social_Contexts_Coaching",
                   "Rejection_Sensitivity_Skills_Coaching",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Core_Interpersonal_Skills",
                               "Assertiveness_and_Boundaries"], [
                   "Assertiveness_Workshop",
                   "Boundary_Setting_Workshop",
                   "Saying_No_And_Limit_Setting_Coaching",
                   "Boundary_Repair_After_Violation_Coaching",
                   "Role_Clarity_And_Expectation_Setting_Coaching",
                   "Interpersonal_Prioritization_And_Overcommitment_Boundary_Coaching",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Core_Interpersonal_Skills",
                               "Perspective_Taking_and_Mentalizing"], [
                   "Perspective_Taking_And_Empathy_Skills_Training",
                   "Mental_State_Attribution_Skills_Practice",
                   "Assumption_Checking_And_Curiosity_Skills_Coaching",
                   "Nonverbal_Cue_Interpretation_Skills_Training",
                   "Social_Context_Reading_Skills_Training",
                   "Cognitive_Flexibility_In_Social_Interpretation_Coaching",
               ])

    # ----------------------------
    # C) Social Skills Training and Social Learning (structured, skill-acquisition oriented)
    # ----------------------------
    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Social_Skills_Training_and_Coaching",
                               "Individual_Format"], [
                   "Social_Skills_Coaching_Service",
                   "Role_Play_Based_Social_Skills_Practice",
                   "Behavioral_Rehearsal_With_Feedback_Service",
                   "Video_Feedback_Social_Skills_Coaching",
                   "Conversation_Timing_And_TurnTaking_Coaching",
                   "Friendship_Initiation_And_Deepening_Skills_Coaching",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Social_Skills_Training_and_Coaching",
                               "Group_Format"], [
                   "Interpersonal_Skills_Group_Program",
                   "Structured_Social_Skills_Group_Program",
                   "Group_Role_Play_And_Feedback_Lab",
                   "Social_Confidence_And_Exposure_Group_Program",
                   "In_Vivo_Social_Practice_Group_With_Coaching",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Social_Skills_Training_and_Coaching",
                               "Generalization_and_Maintenance"], [
                   "Between_Session_Social_Practice_Assignment_Coaching",
                   "Skills_Generalization_To_Real_World_Support",
                   "Social_Setback_Review_And_Adjustment_Session",
                   "Maintenance_Plan_For_Interpersonal_Skills",
                   "Relapse_Prevention_For_Interpersonal_Avoidance_Patterns",
               ])

    # ----------------------------
    # D) Conflict, Repair, and Mediation (moderate-to-high acuity / relational rupture)
    # ----------------------------
    add_leaves(interpersonal,
               ["Interpersonal_Therapeutic_and_Coaching_Services", "Conflict_Repair_and_Mediation", "Conflict_Skills"],
               [
                   "Conflict_Resolution_Workshop",
                   "Negotiation_And_Compromise_Skills_Coaching",
                   "Problem_Solving_As_A_Dyad_Coaching",
                   "Fair_Fighting_Rules_Development_Session",
                   "Boundary_Negotiation_In_Conflict_Coaching",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Conflict_Repair_and_Mediation",
                               "Anger_and_Escalation"], [
                   "Anger_And_Conflict_Mediation_Service",
                   "Conflict_Deescalation_Plan_Development",
                   "High_Arousal_Conversation_Protocol_Coaching",
                   "Rage_Trigger_And_Repair_Pathway_Coaching",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Conflict_Repair_and_Mediation",
                               "Repair_and_Restorative_Practice"], [
                   "Apology_And_Repair_Skills_Coaching",
                   "Rupture_Repair_Facilitation_Session",
                   "Restorative_Conversation_Facilitation",
                   "Accountability_And_Amends_Planning_Session",
                   "Trust_Rebuilding_Behavioral_Commitment_Plan",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Conflict_Repair_and_Mediation",
                               "Mediation_Services_by_Context"], [
                   "Community_Mediation_Service",
                   "Workplace_Mediation_Service",
                   "School_Conflict_Mediation_Service",
                   "Neighbor_Dispute_Mediation_Service",
                   "Family_Mediation_Session_Focused_On_Roles_And_Rules",
                   "Family_Group_Conference_Facilitation",
               ])

    # ----------------------------
    # E) Relational Therapy and Dyadic Work (clinical grade; structured therapeutic services)
    # ----------------------------
    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Relational_Therapy_Services",
                               "Interpersonal_Therapy_and_Role_Work"], [
                   "Interpersonal_Therapy_Service",
                   "Role_Dispute_Resolution_Therapy_Service",
                   "Role_Transition_Support_Therapy_Service",
                   "Interpersonal_Grief_And_Loss_Focused_Therapy_Service",
                   "Interpersonal_Deficit_Focused_Therapy_Service",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Relational_Therapy_Services",
                               "Attachment_and_Connection_Focused"], [
                   "Attachment_Focused_Interpersonal_Therapy_Service",
                   "Trust_And_Safety_In_Relationships_Therapy_Service",
                   "Closeness_And_Vulnerability_Skills_Therapy_Service",
                   "Jealousy_And_Insecurity_Interpersonal_Therapy_Service",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Relational_Therapy_Services",
                               "Family_System_Communication_Focused"], [
                   "Family_Communication_Intervention_Service",
                   "Reducing_Expressed_Emotion_Interpersonal_Coaching",
                   "Care_Planning_Family_Meeting_Facilitation",
                   "Parent_Adolescent_Communication_Therapy_Service",
               ])

    # ----------------------------
    # F) Group Therapy and Therapeutic Group Processes (clinical grade)
    # ----------------------------
    add_leaves(interpersonal,
               ["Interpersonal_Therapeutic_and_Coaching_Services", "Group_Therapy_Formats", "Process_Oriented"], [
                   "Group_Therapy_Interpersonal_Process_Format",
                   "Therapeutic_Process_Group_With_Feedback",
                   "Interpersonal_Rupture_And_Repair_Group_Format",
               ])

    add_leaves(interpersonal,
               ["Interpersonal_Therapeutic_and_Coaching_Services", "Group_Therapy_Formats", "Skills_Oriented"], [
                   "Interpersonal_Skills_Group_Program",
                   "Structured_Social_Skills_Group_Program",
                   "Emotion_Regulation_In_Relationships_Skills_Group",
                   "Conflict_Resolution_Skills_Group",
               ])

    add_leaves(interpersonal,
               ["Interpersonal_Therapeutic_and_Coaching_Services", "Group_Therapy_Formats", "Contextual_Applications"],
               [
                   "School_Based_Support_Group_Program",
                   "Workplace_Team_Communication_Workshop",
                   "Caregiver_Communication_Workshop",
               ])

    # ----------------------------
    # G) Facilitation, Support Circles, and Coordinated Dialogue (structured but non-community-domain)
    #     (Focus is on facilitated interpersonal process, not participation in community activities.)
    # ----------------------------
    add_leaves(interpersonal,
               ["Interpersonal_Therapeutic_and_Coaching_Services", "Facilitation_and_Coordinated_Dialogue",
                "Support_Circles"], [
                   "Facilitated_Support_Circle_Setup",
                   "Facilitated_Difficult_Conversation_Session",
                   "Multi_Party_Expectation_Setting_Facilitation",
                   "Shared_Agreements_And_Norms_Facilitation",
               ])

    add_leaves(interpersonal,
               ["Interpersonal_Therapeutic_and_Coaching_Services", "Facilitation_and_Coordinated_Dialogue",
                "Peer_Led_Facilitation"], [
                   "Peer_Led_Support_Group_Facilitation",
                   "Peer_Facilitator_Training_Service",
                   "Group_Safety_And_Norms_Setup_Facilitation",
               ])

    add_leaves(interpersonal,
               ["Interpersonal_Therapeutic_and_Coaching_Services", "Facilitation_and_Coordinated_Dialogue",
                "Integration_Coaching"], [
                   "Community_Integration_Coaching_Service",
                   "Return_To_Social_Roles_Integration_Coaching",
                   "Workplace_Relationship_Reintegration_Coaching",
               ])

    # ----------------------------
    # H) Relationships, Dating, and Social Role Coaching (specialized, non-redundant with intimacy domain)
    #     (This section stays interpersonal-process focused; avoids sexual health content covered elsewhere.)
    # ----------------------------
    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Relationships_and_Dating_Coaching",
                               "Friendship_and_Peer_Relationships"], [
                   "Friendship_Building_Coaching",
                   "Friendship_Maintenance_And_Repair_Coaching",
                   "Social_Signal_Reading_For_Friendship_Coaching",
                   "Handling_Ghosting_And_Rejection_Coaching",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Relationships_and_Dating_Coaching",
                               "Dating_Process_and_Communication"], [
                   "Dating_Coaching_Service",
                   "Dating_Conversation_And_Flirting_Skills_Coaching",
                   "Date_Planning_And_Expectation_Setting_Coaching",
                   "Boundary_Setting_In_Dating_Coaching",
                   "Conflict_And_Misalignment_Navigation_In_Dating_Coaching",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Relationships_and_Dating_Coaching",
                               "CoParenting_and_Family_Roles"], [
                   "Co_Parenting_Skills_Workshop",
                   "Co_Parenting_Communication_Coaching",
                   "Parenting_Role_Negotiation_Coaching",
                   "Caregiver_Role_Communication_Coaching",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Relationships_and_Dating_Coaching",
                               "Relationship_Coaching"], [
                   "Relationship_Coaching_Service",
                   "Shared_Goals_And_Roles_Negotiation_Coaching",
                   "Division_Of_Labor_And_Mental_Load_Negotiation_Coaching",
                   "Trust_Rebuilding_Coaching_Service",
                   "Repair_After_Conflict_Coaching_Service",
               ])

    # ----------------------------
    # I) Clinical Specialization Focus (designated clinical fields; NOT disorder-labeled branches)
    #     Intended for service matching when interpersonal needs stem from specific clinical profiles.
    # ----------------------------
    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Clinical_Specialization_Focus",
                               "Social_Learning_and_Neurodevelopmental_Profile_Support"], [
                   "Social_Communication_Skills_Therapy_Service",
                   "Pragmatic_Language_Interpersonal_Coaching",
                   "Explicit_Social_Rules_And_Context_Coaching",
                   "Nonverbal_Communication_Training_Service",
                   "Theory_Of_Mind_Skills_Practice_Service",
                   "Sensory_Aware_Interaction_Planning_Coaching",
                   "Workplace_Social_Norms_Coaching_Service",
                   "Peer_Relationship_Support_Coaching_Service",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Clinical_Specialization_Focus",
                               "Trauma_Informed_Interpersonal_Work"], [
                   "Trauma_Informed_Boundary_And_Safety_Skills_Therapy_Service",
                   "Trust_And_Safety_Rebuilding_Interpersonal_Therapy_Service",
                   "Triggers_In_Relationships_Recognition_And_Response_Coaching",
                   "Interpersonal_Dissociation_Stabilization_Skills_Coaching",
                   "Shame_And_Self_Protection_Pattern_Interpersonal_Therapy_Service",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Clinical_Specialization_Focus",
                               "Personality_Functioning_And_Relational_Patterns"], [
                   "Chronic_Interpersonal_Pattern_Change_Therapy_Service",
                   "Rupture_And_Repair_Focused_Therapy_Service",
                   "Interpersonal_Impulsivity_And_Reactivity_Coaching",
                   "Interpersonal_BlackAndWhite_Thinking_Flexibility_Coaching",
                   "Abandonment_Fear_And_Closeness_Tolerance_Therapy_Service",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Clinical_Specialization_Focus",
                               "Social_Anxiety_and_Avoidance_Profile_Support"], [
                   "Graded_Interpersonal_Exposure_Coaching_Service",
                   "Social_Evaluation_Fear_Response_Skills_Coaching",
                   "Post_Interaction_Rumination_Reduction_Coaching",
                   "Assertive_Visibility_And_Self_Advocacy_Coaching",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Clinical_Specialization_Focus",
                               "Cognitive_Communication_And_Executive_Profile_Support"], [
                   "Conversation_Planning_With_Executive_Support_Coaching",
                   "Working_Memory_Load_Reduction_In_Conversation_Strategy_Coaching",
                   "Interpersonal_Task_Sequencing_And_FollowThrough_Coaching",
                   "Fatigue_And_Pacing_For_Social_Interaction_Coaching",
               ])

    # ----------------------------
    # J) Stepped-Care Intensity Tiering (clinically graded routing targets)
    #     Leaves represent service-level options (not schedules), enabling triage in your solver.
    # ----------------------------
    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Clinical_Grading_and_Stepped_Care",
                               "Low_Acuity_Skill_Building"], [
                   "Brief_Interpersonal_Skills_Coaching_Service",
                   "Workshop_Based_Interpersonal_Skills_Service",
                   "Structured_Practice_With_Feedback_Service",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Clinical_Grading_and_Stepped_Care",
                               "Moderate_Acuity_Structured_Therapy"], [
                   "Manualized_Interpersonal_Therapy_Service",
                   "Structured_Group_Therapy_Interpersonal_Skills_Service",
                   "Rupture_Repair_And_Conflict_Therapy_Service",
               ])

    add_leaves(interpersonal, ["Interpersonal_Therapeutic_and_Coaching_Services", "Clinical_Grading_and_Stepped_Care",
                               "High_Acuity_Risk_And_Safety_Informed"], [
                   "Interpersonal_Safety_Informed_Therapy_Service",
                   "High_Conflict_Multi_Party_Mediation_Service",
                   "Crisis_Stabilization_For_Interpersonal_Escalation_Service",
               ])

    SOCIAL["Interpersonal_Therapeutic_Solutions"] = interpersonal

    # ============================================================
    # 9) Social Media Ecosystem and Interventions (solution-variable oriented)
    #    Expanded, high-resolution SOCIAL MEDIA sub-ontology
    #    - Leaf nodes = actionable platform features, supports, services, coaching modules
    #    - No disorder-named branches; no schedule/frequency/dose nodes
    # ============================================================
    social_media: dict = {}

    # ------------------------------------------------------------
    # A) Platform Selection, Onboarding, and Accessibility
    # ------------------------------------------------------------
    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Platform_Selection_and_Onboarding", "Platform_Portfolio_Strategy"], [
        "Platform_Fit_Assessment_For_Wellbeing_Goals",
        "Platform_Portfolio_Minimization_Strategy",
        "Platform_Segmentation_By_Purpose_Strategy",
        "High_Risk_Feature_Avoidance_Strategy",
        "Supportive_Community_First_Platform_Selection",
        "Creator_Centric_Versus_Connection_Centric_Platform_Choice",
        "Anonymity_Option_Evaluation_For_Safety",
        "Professional_Versus_Personal_Platform_Separation_Strategy",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Platform_Selection_and_Onboarding", "Onboarding_Safety_Setup"], [
        "Safety_First_Onboarding_Checklist",
        "Privacy_Defaults_Hardening_Setup",
        "Trusted_Contacts_And_Recovery_Setup",
        "Reporting_And_Blocking_Tool_Orientation",
        "Content_Preference_Seeding_For_Feed",
        "Harassment_Filter_Enablement_Setup",
        "Sensitive_Content_Filter_Enablement_Setup",
        "Account_Discovery_Limit_Setup",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Platform_Selection_and_Onboarding", "Accessibility_and_Language_Support"], [
        "Captioning_Default_Enablement_Setup",
        "ScreenReader_Optimized_Interface_Setup",
        "Text_Size_And_Contrast_Optimization_Setup",
        "Alt_Text_Creation_Support_For_Posts",
        "Language_Preference_And_Translation_Setup",
        "Sensory_Load_Reduction_Interface_Setup",
        "Reduced_Animation_And_Motion_Setup",
        "Audio_Control_Defaults_For_Autoplay_Setup",
    ])

    # ------------------------------------------------------------
    # B) Account, Identity, and Profile Governance
    # ------------------------------------------------------------
    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Account_and_Profile_Governance", "Account_Security_and_Recovery"], [
        "Multi_Factor_Authentication_Enablement_Support",
        "Password_Hygiene_And_Reset_Plan",
        "Login_Alert_Enablement_Setup",
        "Suspicious_Login_Response_Playbook",
        "Account_Recovery_Email_And_Phone_Verification_Setup",
        "Compromised_Account_Containment_Playbook",
        "Impersonation_Prevention_Profile_Verification_Strategy",
        "Device_And_Session_Management_Setup",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Account_and_Profile_Governance", "Identity_and_Self_Presentation"], [
        "Pseudonym_And_Handle_Selection_For_Safety",
        "Profile_Disclosure_Level_Planning",
        "Identity_Context_Switching_Strategy",
        "Professional_Profile_Boundary_Setup",
        "Personal_Profile_Boundary_Setup",
        "Profile_Content_Tone_And_Purpose_Declaration",
        "Values_Statement_And_Community_Norms_Profile_Text",
        "Selective_Audience_Self_Presentation_Coaching",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Account_and_Profile_Governance", "Audience_and_Visibility_Controls"], [
        "Private_Account_Enablement_Setup",
        "Follower_Approval_Workflow_Setup",
        "Close_Friends_Audience_List_Setup",
        "Post_Audience_Default_Setup",
        "Tagging_And_Mention_Approval_Setup",
        "Comment_Audience_Restriction_Setup",
        "Direct_Message_Request_Filter_Setup",
        "Search_And_Discoverability_Reduction_Setup",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Account_and_Profile_Governance", "Data_Rights_and_Portability"], [
        "Data_Download_And_Archive_Setup",
        "Data_Deletion_Request_Navigation_Support",
        "Third_Party_App_Permissions_Audit",
        "Contact_Sync_Disablement_Setup",
        "Ad_Personalization_OptOut_Setup",
        "Location_Metadata_Removal_Setup",
        "CrossPlatform_Tracking_Reduction_Setup",
        "Privacy_Policy_Literacy_Toolkit",
    ])

    # ------------------------------------------------------------
    # C) Feed, Recommendations, and Exposure Management
    # ------------------------------------------------------------
    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Feed_and_Recommendation_Management", "Follow_Mute_Block_Controls"], [
        "Intentional_Follow_List_Curation",
        "Unfollow_Cleanup_Assistance",
        "Mute_And_Snooze_Toxic_Accounts_Strategy",
        "Block_And_Restrict_Workflow_Setup",
        "Keyword_Based_Mute_Setup",
        "Hide_Replies_And_Filter_Comments_Setup",
        "Limit_DMs_From_Unknown_Accounts_Setup",
        "Circle_Or_List_Based_Feed_Organization",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Feed_and_Recommendation_Management", "Sensitive_Content_and_Trigger_Controls"], [
        "Sensitive_Content_Blur_Enablement_Setup",
        "Topic_Exclusion_Filter_Setup",
        "Graphic_Content_Reduction_Setup",
        "Self_Harm_Content_Exposure_Reduction_Setup",
        "Eating_And_Body_Image_Content_Exposure_Reduction_Setup",
        "Violence_And_Hate_Content_Exposure_Reduction_Setup",
        "Political_Stress_Content_Exposure_Reduction_Setup",
        "Personal_Trigger_Topic_Blocklist_Setup",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Feed_and_Recommendation_Management", "Algorithm_Tuning_and_Reset"], [
        "Recommendation_Reset_Assistance",
        "Interest_Signal_Seeding_For_Supportive_Content",
        "Explore_Page_Content_Tuning_Strategy",
        "Not_Interested_Feedback_Practice_Coaching",
        "Diverse_Viewpoint_Exposure_Tuning",
        "Local_Community_Content_Discovery_Tuning",
        "Supportive_Creator_Discovery_Tuning",
        "Ad_Category_Blocking_And_Filtering_Setup",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Feed_and_Recommendation_Management", "Attention_and_Friction_Settings"], [
        "Autoplay_Disablement_Setup",
        "Infinite_Scroll_Interruption_Strategy",
        "ShortForm_Video_Feed_Reduction_Strategy",
        "Read_Before_Share_Prompt_Enablement_Setup",
        "Late_Night_Content_Sensitivity_Mode_Setup",
        "Notification_Batching_Strategy",
        "Home_Screen_App_Icon_Deemphasis_Strategy",
        "Remove_InApp_Purchase_Triggers_Strategy",
    ])

    # ------------------------------------------------------------
    # D) Communication, Interaction, and Social Norms
    # ------------------------------------------------------------
    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Interaction_and_Communication_Supports", "Commenting_and_Discourse_Controls"], [
        "Comment_Filtering_By_Keyword_Setup",
        "Comment_Approval_Workflow_Setup",
        "Limit_Comments_To_Followers_Setup",
        "Disable_QuotePosts_Or_Reshares_Setup",
        "Anti_PileOn_Protection_Settings_Setup",
        "Pinned_Comment_For_Norms_And_Boundaries",
        "TurnOff_Reaction_Options_Strategy",
        "Reply_Delay_And_Cooldown_Strategy",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Interaction_and_Communication_Supports", "Direct_Messaging_Boundaries"], [
        "DM_Request_Filter_Setup",
        "Trusted_DM_Only_Mode_Setup",
        "Message_Read_Receipt_Privacy_Setup",
        "Harassment_DM_Screenshot_And_Report_Playbook",
        "Boundary_Scripts_For_DM_Disengagement",
        "Unwanted_Intimacy_Or_Sexual_DM_Response_Scripts",
        "Scam_And_Manipulation_DM_Screening_Toolkit",
        "Supportive_DM_CheckIn_Script_Templates",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Interaction_and_Communication_Supports", "Group_Chats_and_Community_Governance"], [
        "Group_Rules_And_Norms_Template",
        "Moderator_Role_Assignment_Playbook",
        "New_Member_Onboarding_Message_Template",
        "Conflict_Mediation_Pathway_For_Groups",
        "Remove_Member_And_Ban_Workflow_Playbook",
        "Crisis_Escalation_Protocol_For_Groups",
        "Privacy_Safe_Group_Invite_Strategy",
        "Community_Safety_Checklist_For_Group_Admins",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Interaction_and_Communication_Supports", "Prosocial_Communication_Prompts"], [
        "Empathy_Prompt_Before_Reply_Module",
        "Assumption_Checking_Prompt_Module",
        "Deescalation_Language_Suggestion_Module",
        "Repair_Attempt_Message_Templates",
        "Gratitude_And_Appreciation_Comment_Prompts",
        "Supportive_Response_For_Disclosure_Toolkit",
        "Bystander_Support_Comment_Templates",
        "Kindness_And_Nonjudgment_Pledge_Module",
    ])

    # ------------------------------------------------------------
    # E) Community, Belonging, and Peer Support via Social Media
    # ------------------------------------------------------------
    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Community_and_Peer_Support_Online", "Support_Community_Discovery_and_Fit"], [
        "Moderated_Support_Community_Enrollment",
        "Peer_Led_Support_Community_Enrollment",
        "Identity_Affirming_Online_Community_Enrollment",
        "Local_Community_Groups_Discovery_Support",
        "Interest_Based_Community_Discovery_Support",
        "Community_Safety_And_Rules_Fit_Check",
        "Lurking_To_Participation_Onramp_Support",
        "Safe_Intro_Post_Template_For_New_Members",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Community_and_Peer_Support_Online", "Peer_Matching_and_Buddying"], [
        "Peer_Buddy_Matching_Workflow",
        "Accountability_Partner_Matching_Workflow",
        "Mentor_Mentee_Matching_Workflow",
        "Shared_Goals_Peer_Matching_Workflow",
        "Boundaries_And_Expectations_Agreement_For_Peers",
        "Mutual_Support_CheckIn_Template_For_Peers",
        "Peer_Safety_And_Confidentiality_Agreement_Template",
        "Escalation_Pathway_For_Peer_Support_Limits",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Community_and_Peer_Support_Online", "Mutual_Aid_and_Resource_Sharing"], [
        "Resource_Request_Posting_Template",
        "Resource_Offer_Posting_Template",
        "Mutual_Aid_Verification_And_Safety_Checklist",
        "Community_Resource_Directory_Pin_And_Curation",
        "Local_Service_Referral_Posting_Workflow",
        "Crisis_Resource_Pinning_For_Communities",
        "Transportation_And_Logistics_Coordination_Template",
        "Reciprocity_And_Burnout_Protection_Guidelines",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Community_and_Peer_Support_Online", "Online_To_Offline_Bridging"], [
        "Event_Discovery_And_Attendance_Planning_Support",
        "Buddy_Attendance_Pairing_For_Events",
        "Safety_Logistics_For_Meetups_Planning",
        "Public_Venue_First_Meetup_Planning_Toolkit",
        "Post_Event_FollowUp_Message_Templates",
        "Gradual_Trust_Building_Meetup_Onramp",
        "Community_Role_Onramp_For_Newcomers",
        "Exit_And_Disengagement_From_Groups_Plan",
    ])

    # ------------------------------------------------------------
    # F) Social Comparison, Metrics, and Self-Evaluation Protection
    # ------------------------------------------------------------
    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Social_Comparison_and_Metrics", "Metrics_Visibility_Controls"], [
        "Hide_Like_Counts_Setup",
        "Hide_View_Counts_Setup",
        "Hide_Follower_Count_Strategy",
        "Disable_Public_Reaction_Visibility_Strategy",
        "TurnOff_Trending_And_Viral_Metrics_Strategy",
        "Remove_Analytics_Dashboard_Access_Strategy",
        "Creator_Metrics_Buffering_Coaching",
        "Engagement_Interpretation_Reframe_Module",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Social_Comparison_and_Metrics", "Comparison_Trigger_Management"], [
        "Social_Comparison_Trigger_Labeling_Module",
        "Envy_And_Threat_Trigger_Management_Plan",
        "Achievement_Comparison_Buffering_Strategy",
        "Lifestyle_Comparison_Exposure_Reduction_Strategy",
        "Success_Story_Consumption_Boundary_Strategy",
        "FOMO_Trigger_Management_Module",
        "Parasocial_Comparison_Buffering_Strategy",
        "Self_Compassion_Prompt_After_Scroll_Module",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Social_Comparison_and_Metrics", "Body_Image_and_Appearance_Protection"], [
        "Appearance_Focused_Content_Exposure_Reduction_Setup",
        "Photo_Filter_Self_Criticism_Reduction_Module",
        "Diverse_Body_Representation_Feed_Curation_Strategy",
        "Diet_Culture_Content_Filtering_Setup",
        "BeforeAfter_Content_Exposure_Reduction_Setup",
        "Cosmetic_Procedure_Advertising_Filter_Setup",
        "Body_Neutral_Self_Talk_Prompt_Module",
        "Mirror_And_Self_Image_Compassion_Coaching_Module",
    ])

    # ------------------------------------------------------------
    # G) Relationship Boundaries, Consent, and Contextual Sharing
    # ------------------------------------------------------------
    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Boundaries_and_Consent_in_Sharing", "Consentful_Sharing_Practices"], [
        "Consent_To_Post_Others_Checklist",
        "Tagging_Consent_Agreement_Template",
        "Children_And_Family_Privacy_Sharing_Guidelines",
        "Location_Sharing_Consent_And_Safety_Checklist",
        "Screenshot_And_Forwarding_Consent_Norms_Template",
        "Sensitive_Disclosure_Sharing_Safety_Checklist",
        "Relationship_Content_Sharing_Boundary_Agreement",
        "Group_Photo_Posting_Consent_Workflow",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Boundaries_and_Consent_in_Sharing", "Context_Collapse_Protection"], [
        "Audience_Segmentation_Strategy",
        "Close_Friends_And_Trusted_List_Posting_Strategy",
        "Workplace_Audience_Separation_Strategy",
        "Family_Audience_Separation_Strategy",
        "Public_Versus_Private_Account_Strategy",
        "Alt_Account_Risk_Assessment_And_Safety_Plan",
        "Content_Ephemerality_Option_Selection_Strategy",
        "Reply_And_Quote_Visibility_Control_Strategy",
    ])

    # ------------------------------------------------------------
    # H) Safety, Harassment, and Crisis-Adjacent Protections
    # ------------------------------------------------------------
    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Safety_and_Harassment_Response", "Harassment_and_Bullying_Protections"], [
        "Anti_Harassment_Filter_Enablement_Setup",
        "Restrict_Mode_Enablement_Setup",
        "Mass_Blocking_Workflow_Setup",
        "Harassment_Documentation_Toolkit",
        "Report_Submission_Support_Playbook",
        "Bystander_Block_And_Report_Guide",
        "Comment_PileOn_Containment_Playbook",
        "Support_Person_Handover_For_Online_Harassment",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Safety_and_Harassment_Response", "Hate_Targeting_and_Minority_Stress_Protection"], [
        "Hate_Speech_Filtering_Enablement_Setup",
        "Identity_Targeting_Keyword_Filter_Setup",
        "Raid_And_Brigading_Protection_Playbook",
        "Community_Moderation_For_Targeted_Abuse_Playbook",
        "Public_Facing_Account_Risk_Assessment_Toolkit",
        "Selective_Disclosure_For_Identity_Safety_Plan",
        "Trusted_Allies_Amplification_And_Shielding_Strategy",
        "Rapid_Deplatforming_Response_And_Backup_Channel_Plan",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Safety_and_Harassment_Response", "Stalking_Doxxing_and_Impersonation"], [
        "Doxxing_Prevention_Privacy_Hardening_Playbook",
        "Reverse_Image_Search_Risk_Mitigation_Strategy",
        "Impersonation_Report_And_Takedown_Playbook",
        "Location_Metadata_Scrub_Playbook",
        "Harassment_Account_Network_Mapping_Toolkit",
        "Platform_Safety_Escalation_Channel_Navigation",
        "Safe_Contact_Channel_Migration_Plan",
        "Trusted_Contact_Warning_And_Safety_Network_Activation",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Safety_and_Harassment_Response", "Crisis_Resource_Linkage_and_CheckIn"], [
        "InApp_Crisis_Resource_Pinning_Setup",
        "Trusted_Contact_CheckIn_Workflow_Setup",
        "Safety_Plan_Link_Sharing_Workflow",
        "Help_Seeking_Post_Composition_Support",
        "Supportive_Response_Guidance_For_Peers",
        "High_Risk_Keyword_Response_Guide_For_Moderators",
        "Escalation_To_Offline_Support_Playbook",
        "Post_Crisis_Digital_Boundary_Reset_Plan",
    ])

    # ------------------------------------------------------------
    # I) Information Quality, Misinformation Resilience, and News Stress
    # ------------------------------------------------------------
    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Information_Quality_and_Media_Literacy", "Misinformation_Resilience_Skills"], [
        "Source_Credibility_Check_Toolkit",
        "Cross_Checking_Practice_Module",
        "Manipulated_Media_Detection_Toolkit",
        "Ragebait_Recognition_And_Disengagement_Module",
        "Conspiracy_Content_Exposure_Reduction_Strategy",
        "Avoid_Sharing_Unverified_Content_Pledge_Module",
        "Fact_Checker_Reference_Workflow",
        "Community_Norms_For_Accuracy_And_Corrections_Template",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Information_Quality_and_Media_Literacy", "Health_Information_Safety"], [
        "Health_Claim_Verification_Workflow",
        "Medical_Misinformation_Filtering_Strategy",
        "Evidence_And_Uncertainty_Literacy_Module",
        "Influencer_Health_Advice_Risk_Screening_Toolkit",
        "Curate_Evidence_Based_Health_Sources_Feed_Strategy",
        "Avoid_Dangerous_Challenge_Content_Strategy",
        "Myth_Correction_Response_Templates",
        "Clinician_Approved_Resource_Sharing_Workflow",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Information_Quality_and_Media_Literacy", "News_Stress_and_Doomscrolling_Protection"], [
        "News_Feed_Segmentation_Strategy",
        "Crisis_News_Exposure_Reduction_Strategy",
        "Triggering_Headline_Blur_Enablement_Setup",
        "News_Consumption_Intention_Prompt_Module",
        "After_News_Grounding_Prompt_Module",
        "Local_Actionable_News_Curation_Strategy",
        "Replace_Rumination_Loops_With_Action_Steps_Module",
        "Unfollow_Stress_Amplifying_Sources_Strategy",
    ])

    # ------------------------------------------------------------
    # J) Content Creation, Expression, and Prosocial Contribution
    # ------------------------------------------------------------
    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Content_Creation_and_Expression", "Expressive_And_Reflective_Creation"], [
        "Expressive_Writing_Post_Template",
        "Narrative_Reframing_Story_Template",
        "Gratitude_Sharing_Post_Template",
        "Values_Aligned_Identity_Expression_Prompt_Module",
        "Emotion_Labeling_Caption_Prompt_Module",
        "Self_Compassion_Storytelling_Prompt_Module",
        "Strengths_And_Wins_Sharing_Balanced_Template",
        "Private_Draft_Journaling_Mode_Workflow",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Content_Creation_and_Expression", "Creative_Practice_and_Play"], [
        "Creative_Challenge_Participation_Safe_Onramp",
        "Art_And_Maker_Content_Creation_Onramp",
        "Music_And_Performance_Sharing_Onramp",
        "Humor_And_Play_Content_Creation_Guidelines",
        "Collaborative_Creation_Partner_Matching",
        "Creative_Feedback_Request_Boundary_Template",
        "Safe_Critique_And_Comment_Norms_Template",
        "Creator_Comparison_Buffering_For_Creatives_Module",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Content_Creation_and_Expression", "Prosocial_Contribution_and_Advocacy"], [
        "Peer_Supportive_Commenting_Routine_Template",
        "Resource_SignalBoosting_Guidelines",
        "Community_Recognition_And_Appreciation_Post_Template",
        "Mutual_Aid_Share_Safety_Guidelines",
        "Bystander_Intervention_Online_Toolkit",
        "Civic_Engagement_Content_Sharing_Guidelines",
        "Advocacy_Boundary_And_Risk_Assessment_Toolkit",
        "Avoid_Harmful_Callout_Escalation_Guidelines",
    ])

    # ------------------------------------------------------------
    # K) Parasocial Dynamics, Influencers, and Audience Management
    # ------------------------------------------------------------
    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Parasocial_and_Audience_Dynamics", "Parasocial_Boundaries_and_Expectations"], [
        "Parasocial_Attachment_Boundary_Planning_Module",
        "Influencer_Content_Exposure_Balancing_Strategy",
        "Unrealistic_Standard_Debiasing_Module",
        "Creator_Deification_Risk_Reduction_Strategy",
        "Fan_Community_Norms_And_Safety_Checklist",
        "Celebrity_News_Stress_Reduction_Strategy",
        "Replace_Parasocial_Time_With_Reciprocal_Connection_Plan",
        "Recognize_Manipulative_Engagement_Tactics_Toolkit",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Parasocial_and_Audience_Dynamics", "Creator_Audience_Boundaries"], [
        "Creator_DM_Boundary_Setup",
        "Creator_Comment_Moderation_Workflow",
        "Creator_Community_Guidelines_Post_Template",
        "Creator_Emotional_Labor_Boundary_Planning",
        "Creator_Safety_Escalation_And_Support_Person_Setup",
        "Creator_Content_Consent_And_Sharing_Policy_Template",
        "Creator_Collaboration_Risk_Assessment_Toolkit",
        "Creator_Offline_Privacy_Protection_Playbook",
    ])

    # ------------------------------------------------------------
    # L) Digital Wellbeing Supports (intentional use, reflection, recovery)
    # ------------------------------------------------------------
    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Digital_Wellbeing_and_Recovery", "Intentional_Use_and_Goal_Setting"], [
        "Intentional_Use_Goal_Definition_Module",
        "Pre_Open_Intention_Prompt_Module",
        "Post_Use_Reflection_Prompt_Module",
        "Mood_And_Energy_Check_Prompt_Module",
        "If_Then_Plan_For_Triggering_Content_Module",
        "Replace_Scroll_With_Connection_Action_Menu",
        "Values_Aligned_Following_Check_Module",
        "Self_Reward_Without_Engagement_Validation_Module",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Digital_Wellbeing_and_Recovery", "Notification_and_Attention_Management"], [
        "Notification_Privacy_And_Preview_Control_Setup",
        "Disable_Nonessential_Push_Notifications_Setup",
        "Priority_Contacts_Notification_Allowlist_Setup",
        "Quiet_Mode_Enablement_Strategy",
        "Email_Digest_Instead_Of_Push_Strategy",
        "InApp_Badge_Removal_Strategy",
        "Triggering_Notification_Keyword_Filter_Strategy",
        "Attention_Reset_And_Breath_Prompt_On_Open_Module",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Digital_Wellbeing_and_Recovery", "Recovery_and_Decompression_After_Use"], [
        "After_Scroll_Grounding_Exercise_Module",
        "Body_Scan_And_Stretch_Prompt_Module",
        "Decompression_Music_Or_Silence_Prompt_Module",
        "Rumination_Interrupt_And_Reframe_Module",
        "Comparison_Hangover_Recovery_Module",
        "Reengage_Offline_Support_Action_Prompt_Module",
        "Digital_Boundary_Reset_After_Difficult_Interaction_Module",
        "Self_Compassion_After_Online_Conflict_Module",
    ])

    # ------------------------------------------------------------
    # M) Youth/Family Safeguards and Care Context Adaptations
    # ------------------------------------------------------------
    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Youth_Family_and_Caregiver_Safeguards", "Family_Linked_Account_Guardrails"], [
        "Family_Safety_Settings_Review_Workflow",
        "Age_Appropriate_Content_Filter_Enablement_Setup",
        "Stranger_Contact_Limit_Setup",
        "Live_Location_Sharing_Disablement_Setup",
        "DM_From_Unknowns_Block_Setup",
        "Reporting_And_Trusted_Adult_Escalation_Plan",
        "School_Context_Privacy_And_Bullying_Safety_Plan",
        "Family_Dialogue_Guide_About_Online_Social_Life",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Youth_Family_and_Caregiver_Safeguards", "Digital_Consent_and_Civility_Education"], [
        "Consentful_Sharing_Education_Module",
        "Respectful_Disagreement_Education_Module",
        "Bystander_Support_Education_Module",
        "Recognize_And_Report_Manipulation_Education_Module",
        "Privacy_And_Identity_Safety_Education_Module",
        "Group_Chat_Norms_And_Conflict_Education_Module",
        "Scam_And_Exploitation_Recognition_Education_Module",
        "Help_Seeking_Online_Education_Module",
    ])

    # ------------------------------------------------------------
    # N) Clinical / Programmatic Integration (optional routing targets)
    # ------------------------------------------------------------
    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Programmatic_and_Care_Integration", "Clinician_Guided_Plans"], [
        "Clinician_Guided_Feed_Curation_Plan",
        "Clinician_Guided_Online_Community_Engagement_Plan",
        "Clinician_Guided_Digital_Boundary_Plan",
        "Clinician_Guided_Safety_And_Harassment_Response_Plan",
        "Clinician_Guided_Social_Comparison_Protection_Plan",
        "Clinician_Guided_Online_To_Offline_Connection_Plan",
        "Care_Team_Aligned_Resource_List_For_Sharing",
        "Privacy_Preserving_Symptom_And_Stress_CheckIn_Module",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Programmatic_and_Care_Integration", "Digital_Intervention_Delivery_Primitives"], [
        "InApp_Psychoeducation_Card_Library",
        "Micro_Skills_Practice_Prompt_Library",
        "Supportive_Message_Template_Library",
        "Community_Mod_Training_Microcourse",
        "Safety_Checklist_Wizard_For_Settings",
        "Reflection_Journal_With_Sharing_Controls",
        "Goal_Aligned_Following_Audit_Wizard",
        "Crisis_Resources_Quick_Access_Widget",
    ])

    # ------------------------------------------------------------
    # O) Addiction-Related Digital Use Patterns and Supports (SOCIAL MEDIA)
    #    Comprehensive, solution-variable oriented
    #    - Leaf nodes = actionable supports, platform configurations, coaching modules, care linkages
    #    - No disorder-labeled branches; no schedule/frequency/dose nodes
    # ------------------------------------------------------------
    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Addiction_Related_Digital_Use_and_Supports", "Use_Pattern_Assessment_and_Awareness"], [
        "Problematic_Use_Self_Assessment_Toolkit",
        "Trigger_Context_Mapping_For_Compulsive_Use",
        "Urge_Surfing_And_Craving_Labeling_Module",
        "Emotion_State_To_Scrolling_Link_Insight_Module",
        "Cue_Routine_Reward_Loop_Mapping_Exercise",
        "Dopamine_Seeking_Pattern_Insight_Psychoeducation",
        "Automaticity_Disruption_Checklist",
        "Digital_Use_Functional_Analysis_Workflow",
        "High_Risk_Situations_Identification_For_Compulsive_Use",
        "Loss_Of_Control_Early_Warning_Signs_Identification",
        "Negative_Consequence_Tracking_And_Reflection_Module",
        "Values_Conflict_Insight_Module_For_Digital_Use",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Addiction_Related_Digital_Use_and_Supports", "Environmental_Control_and_Friction_Design"], [
        "Remove_OneTap_Access_Shortcuts_Strategy",
        "Home_Screen_Deemphasis_And_App_Hiding_Strategy",
        "Autoplay_Disablement_Setup",
        "Infinite_Scroll_Interruption_Strategy",
        "Disable_Personalized_Recommendations_Strategy",
        "Disable_Trending_And_Viral_Feed_Strategy",
        "Disable_InApp_Gambling_Like_Mechanics_Strategy",
        "Disable_Live_Stream_Discovery_Strategy",
        "Disable_ShortForm_Video_Feed_Strategy",
        "Disable_Rewarding_Streaks_And_Badges_Strategy",
        "Remove_Shopping_And_LiveCommerce_Triggers_Strategy",
        "Block_Triggering_Hashtags_And_Keywords_Setup",
        "Restrict_Account_Suggestions_And_Friend_Recommendations_Setup",
        "Use_Alternative_ReadOnly_Client_Strategy",
        "Device_Level_Content_Blocking_Profile_Setup",
        "Account_Level_Content_Category_Blocking_Setup",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Addiction_Related_Digital_Use_and_Supports", "Notification_and_Interruption_Control"], [
        "Disable_Nonessential_Push_Notifications_Setup",
        "Notification_Privacy_And_Preview_Control_Setup",
        "Notification_Category_Allowlist_Setup",
        "Disable_Email_Reengagement_Promotions_Strategy",
        "Disable_InApp_Badge_Indicators_Strategy",
        "Mute_Reengagement_Pings_And_Mentions_Strategy",
        "Disable_Comment_Reaction_Notifications_Strategy",
        "Limit_DM_Notifications_To_Trusted_Contacts_Setup",
        "Reduce_Algorithmic_Reengagement_Alerts_Strategy",
        "Quiet_Mode_And_Focus_Mode_Enablement_Setup",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Addiction_Related_Digital_Use_and_Supports", "Craving_and_Urge_Management_Skills"], [
        "Craving_Surfing_Guided_Module",
        "Delay_And_Decide_Friction_Prompt_Module",
        "If_Then_Urge_Response_Planning_Module",
        "Replacement_Action_Menu_For_Urges",
        "Grounding_Prompt_On_App_Open_Module",
        "Micro_Recovery_Breathing_Prompt_Module",
        "Emotion_Labeling_Before_Scroll_Prompt_Module",
        "Self_Compassion_After_Slip_Module",
        "Cognitive_Distortion_Check_For_Justification_Module",
        "Shame_Reduction_And_Recommitment_Module",
        "Alternative_Rewards_And_Pleasure_Planning_Module",
        "Stress_To_Scroll_Interruption_Skills_Module",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Addiction_Related_Digital_Use_and_Supports", "Goal_Aligned_Use_and_Behavioral_Contracts"], [
        "Intentional_Use_Goal_Definition_Module",
        "Pre_Open_Intention_Prompt_Module",
        "Goal_Based_App_Access_Gatekeeper_Module",
        "Commitment_Statement_And_Pledge_Module",
        "Personal_Rules_For_Social_Media_Use_Template",
        "Values_Aligned_Use_Boundaries_Coaching_Module",
        "Accountability_Contract_With_Trusted_Person_Template",
        "Public_Versus_Private_Commitment_Strategy",
        "Relapse_Prevention_Plan_For_Compulsive_Use",
        "Slip_Analysis_And_Plan_Update_Workflow",
        "Sober_Scroll_Mode_ReadOnly_Strategy",
        "Scheduled_Content_Consumption_Window_Strategy",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Addiction_Related_Digital_Use_and_Supports", "Substitution_and_Healthy_Replacement_Building"], [
        "Offline_Connection_Action_Prompt_Module",
        "Movement_Break_Substitution_Prompt_Module",
        "Micro_Task_Substitution_Action_Menu",
        "Meaningful_Hobby_Substitution_Planning_Module",
        "Social_Support_Contact_Substitution_Prompt_Module",
        "Sleep_Hygiene_Protection_From_Late_Night_Scrolling_Strategy",
        "Meal_And_Self_Care_Substitution_Prompt_Module",
        "Study_Or_Work_Deep_Focus_Substitution_Toolkit",
        "Mindfulness_Practice_Substitution_Prompt_Module",
        "Environmental_Cue_Swap_Strategy_For_Boredom_Scrolling",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Addiction_Related_Digital_Use_and_Supports", "Social_Accountability_and_Support_Networks"], [
        "Accountability_Partner_Matching_Workflow",
        "Peer_Support_CheckIn_Template_For_Digital_Use_Goals",
        "Family_Or_Household_Boundary_Agreement_For_Device_Use",
        "Co_Use_Reduction_As_Dyad_Coaching_Module",
        "Supportive_Nudge_Message_Template_Library",
        "Trusted_Contact_Intervention_Plan_For_Risk_Moments",
        "Buddy_System_For_Digital_Detox_Transitions",
        "Peer_Group_For_Intentional_Digital_Use_Enrollment",
        "Coach_Led_Digital_Use_Change_Program",
        "Moderated_Recovery_Or_Wellbeing_Online_Community_Enrollment",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Addiction_Related_Digital_Use_and_Supports", "Financial_and_Transactional_Risk_Reduction"], [
        "Disable_InApp_Purchases_Strategy",
        "Remove_Saved_Payment_Methods_Strategy",
        "Ad_And_Shopping_Content_Blocking_Setup",
        "LiveCommerce_And_FlashSale_Exposure_Reduction_Setup",
        "Impulse_Spending_Interruption_Prompt_Module",
        "Creator_Tipping_And_Donations_Boundary_Strategy",
        "Subscription_Audit_And_Cancellation_Workflow",
        "Microtransaction_Trigger_Recognition_Toolkit",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Addiction_Related_Digital_Use_and_Supports", "CoOccurring_Behavior_Triggers_in_Social_Media"], [
        "Alcohol_And_Substance_Cue_Content_Exposure_Reduction_Setup",
        "Gambling_Cue_Content_Exposure_Reduction_Setup",
        "Pornography_Trigger_Content_Exposure_Reduction_Setup",
        "Dating_App_CrossTrigger_Reduction_Strategy",
        "Food_And_Binge_Cue_Content_Exposure_Reduction_Setup",
        "Anger_And_Ragebait_Exposure_Reduction_Strategy",
        "Self_Comparison_And_Shame_Trigger_Reduction_Strategy",
        "Conflict_Seeking_And_Debate_Compulsion_Reduction_Strategy",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Addiction_Related_Digital_Use_and_Supports", "Care_Integration_and_Escalation_Pathways"], [
        "Clinician_Guided_Digital_Use_Change_Plan",
        "Motivational_Interviewing_Style_Digital_Use_Module",
        "Brief_Intervention_For_Compulsive_Digital_Use_Service",
        "Referral_Pathway_To_Behavioral_Health_Support",
        "Peer_Recovery_Coach_Linkage_For_Behavior_Change",
        "Family_Support_And_Collateral_Coaching_For_Digital_Use",
        "Crisis_Escalation_Pathway_When_Use_Becomes_Unsafe",
        "Integrated_Safety_Planning_For_Digital_Risk_Contexts",
    ])

    add_leaves(social_media, ["Social_Media_Ecosystem_and_Interventions", "Addiction_Related_Digital_Use_and_Supports", "Maintenance_Recovery_and_Identity_Shift"], [
        "Identity_Based_Habit_Change_Module_For_Digital_Use",
        "Build_New_Rituals_To_Replace_Scrolling_Module",
        "Reinforce_Self_Efficacy_After_Progress_Module",
        "Sustainability_Check_And_Boundary_Adjustment_Module",
        "Rebuild_Attention_And_Deep_Work_Capacity_Module",
        "Rebuild_Social_Connection_Offline_Integration_Plan",
        "Compassionate_Relapse_Review_And_Recommitment_Workflow",
        "LongTerm_Digital_Wellbeing_Ruleset_Template",
    ])

    # Attach as a top-level SOCIAL section
    SOCIAL["Social_Media_Ecosystem"] = social_media

    return {"SOCIAL": SOCIAL}


# ------------------------ Writer + metadata ------------------------

def write_outputs(ontology: dict, out_json_path: str) -> tuple[str, str, dict]:
    out_json_path = os.path.expanduser(out_json_path)
    out_dir = os.path.dirname(out_json_path)

    # If path is not viable, fall back to a local SOCIAL/ folder next to this script
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        out_dir = os.path.join(os.path.dirname(__file__), "SOCIAL")
        os.makedirs(out_dir, exist_ok=True)
        out_json_path = os.path.join(out_dir, "SOCIAL.json")

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(ontology, f, ensure_ascii=False, indent=2)

    leaf_paths = list(iter_leaf_paths(ontology["SOCIAL"]))
    leaf_count = count_leaves(ontology["SOCIAL"])
    node_count = count_nodes(ontology["SOCIAL"])
    depth = max_depth(ontology["SOCIAL"])
    top_counts = subtree_leaf_counts(ontology["SOCIAL"])
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
    default_out = "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/PREDICTOR/steps/separate/SOCIAL/SOCIAL.json"
    out_json_path = os.environ.get("SOCIAL_OUT_PATH", default_out)

    ontology = build_social_ontology()

    # Guardrail: reject explicit schedule/frequency/dose tokens
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
    leaf_paths = list(iter_leaf_paths(ontology["SOCIAL"]))
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

#TODO: further optimize coverage (both in depth and breadth) and quality
