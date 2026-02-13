#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PERSON PHOENIX_ontology Generator (High-Resolution Person-Attribute + Value-Space)

Design principles:
- Entity-only hierarchical taxonomy (nested dict; leaves are empty dicts).
- INTERNAL NODES = variables/attributes (e.g., Religion, Gender_Identity, Employment).
- LEAF NODES = possible values / states for those variables (e.g., Islam/Shia, Gender/Nonbinary, Employment/Employed).
- Avoid leaf names like "*_Indicator" or "*_Level" or "*_Band" as the *leaf itself*.
  Instead: variable node -> enumerated value leaves.
- Include "Unknown" and "Prefer_Not_To_Say" value leaves where appropriate.
- Some fields remain as raw variables where enumeration is not appropriate (e.g., IDs, numeric scores).
- Adds higher-resolution semantics for:
  - Age (clinical bands + multiple developmental-stage theories/frameworks + puberty/frailty/reproductive stages)
  - Socioeconomic status (objective/subjective, deprivation/strain, class frameworks, neighborhood indices, mobility)
  - Sex/Gender/SOGI (separate identity/attraction/behavior, additional scales, trans/cis status, legal markers)
  - Migration/citizenship detail, household/relationship legality, etc.
- Writes:
  1) PERSON.json
  2) metadata.txt (same folder)

Override output path:
  PERSON_OUT_PATH="/path/to/PERSON/PERSON.json" python generate_person_ontology.py
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

def add_yes_no_unknown(tree: dict, base_path: list[str], include_prefer_not: bool = True) -> None:
    leaves = ["Yes", "No", "Unknown"]
    if include_prefer_not:
        leaves.append("Prefer_Not_To_Say")
    add_leaves(tree, base_path, leaves)

def add_trinary(tree: dict, base_path: list[str], include_prefer_not: bool = True) -> None:
    leaves = ["Low", "Moderate", "High", "Unknown"]
    if include_prefer_not:
        leaves.append("Prefer_Not_To_Say")
    add_leaves(tree, base_path, leaves)

def add_likert_5(tree: dict, base_path: list[str], include_prefer_not: bool = True) -> None:
    leaves = ["Strongly_Disagree", "Disagree", "Neutral", "Agree", "Strongly_Agree", "Unknown"]
    if include_prefer_not:
        leaves.append("Prefer_Not_To_Say")
    add_leaves(tree, base_path, leaves)

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

def build_person_ontology() -> dict:
    PERSON: dict = {}

    UNK_PNTS = ["Unknown", "Prefer_Not_To_Say"]
    NA_UNK_PNTS = ["Not_Applicable", "Unknown", "Prefer_Not_To_Say"]

    # ============================================================
    # 0) Administrative + Data Handling (non-PII oriented taxonomy)
    # ============================================================
    admin: dict = {}

    # Raw variable fields (not enumerated)
    add_leaves(admin, ["Identifiers"], [
        "Person_Record_ID",
        "Study_Participant_ID",
        "Pseudonymous_ID",
        "Record_Linkage_Key",
        "External_System_ID",
        "Family_Unit_ID",
        "Household_ID",
    ])

    add_leaves(admin, ["Data_Source"], [
        "Primary_Data_Source",
        "Secondary_Data_Sources",
        "Collection_Context",
        "Collection_Mode",
        "Data_Collection_Timestamp",
    ])

    add_leaves(admin, ["Collection_Mode"], [
        "Self_Report",
        "Clinician_Reported",
        "Caregiver_Reported",
        "Administrative_Registry",
        "Sensor_or_Device_Derived",
        "Derived_From_Record_Linkage",
        "Other",
        "Unknown",
    ])

    # Enumerated consent + permissions
    add_leaves(admin, ["Consent", "Consent_Decision"], [
        "Granted",
        "Declined",
        "Withdrawn",
        "Unknown",
    ])
    add_leaves(admin, ["Consent", "Consent_Scope"], [
        "Care_Only",
        "Research_Only",
        "Care_And_Research",
        "Future_Contact_Allowed",
        "Future_Contact_Not_Allowed",
        "Data_Sharing_Allowed",
        "Data_Sharing_Not_Allowed",
        "Genomics_Allowed",
        "Biobanking_Allowed",
        "Unknown",
    ])

    add_leaves(admin, ["Communication_Preferences", "Preferred_Channel"], [
        "In_Person",
        "Telephone",
        "Video_Call",
        "Email",
        "Messaging_App",
        "SMS",
        "Postal_Mail",
        "Unknown",
    ])
    add_leaves(admin, ["Communication_Preferences", "Preferred_Language"], [
        "Dutch",
        "French",
        "German",
        "English",
        "Spanish",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(admin, ["Respondent"], [
        "Self",
        "Caregiver",
        "Clinician",
        "Administrative_Staff",
        "Other_Proxy",
        "Unknown",
    ])
    add_leaves(admin, ["Data_Quality", "Verification"], [
        "Unverified",
        "Partially_Verified",
        "Verified",
        "Unknown",
    ])
    add_leaves(admin, ["Data_Quality", "Completeness"], [
        "Complete",
        "Mostly_Complete",
        "Partially_Complete",
        "Sparse",
        "Unknown",
    ])
    add_leaves(admin, ["Data_Quality", "Consistency"], [
        "Consistent",
        "Inconsistent",
        "Needs_Review",
        "Unknown",
    ])

    PERSON["Administrative_and_Data_Context"] = admin

    # ============================================================
    # 1) Demographics, Identity, and Life-Stage Semantics (high-resolution)
    # ============================================================
    demo: dict = {}

    # Vital status
    add_leaves(demo, ["Vital_Status", "Status"], ["Alive", "Deceased"] + UNK_PNTS)
    add_leaves(demo, ["Vital_Status", "Death_Details"], [
        "Date_Of_Death",
        "Year_Of_Death",
        "Place_Of_Death",
    ])

    # ----------------------------
    # Age: raw age variables
    # ----------------------------
    add_leaves(demo, ["Age", "Chronological_Age"], [
        "Age_Years",
        "Age_Months",
        "Age_Days",
        "Date_Of_Birth",
        "Year_Of_Birth",
    ])
    add_leaves(demo, ["Age", "Perinatal_and_Corrected_Age"], [
        "Gestational_Age_At_Birth",
        "Corrected_Age_For_Prematurity",
        "Postmenstrual_Age",
    ])

    # ----------------------------
    # Age: high-resolution bands (clinical + public health + legal-ish)
    # ----------------------------
    add_leaves(demo, ["Age", "Age_Bands", "Clinical_Pediatric"], [
        "Prenatal",
        "Perinatal",
        "Neonate_Early",
        "Neonate_Late",
        "Infant_Early",
        "Infant_Late",
        "Toddler",
        "Preschool",
        "School_Age_Child",
        "Early_Adolescent",
        "Mid_Adolescent",
        "Late_Adolescent",
    ] + UNK_PNTS)

    add_leaves(demo, ["Age", "Age_Bands", "Clinical_Adult"], [
        "Emerging_Adult",
        "Young_Adult",
        "Early_Midlife",
        "Late_Midlife",
        "Older_Adult_Young_Old",
        "Older_Adult_Old_Old",
        "Oldest_Old",
    ] + UNK_PNTS)

    add_leaves(demo, ["Age", "Age_Bands", "Geriatric_Risk_Grouping"], [
        "Pre_Geriatric",
        "Geriatric",
        "Advanced_Geriatric",
        "Extreme_Longevity",
    ] + UNK_PNTS)

    add_leaves(demo, ["Age", "Age_Bands", "Public_Health_Grouping"], [
        "Child",
        "Adolescent",
        "Youth",
        "Adult",
        "Older_Adult",
    ] + UNK_PNTS)

    add_leaves(demo, ["Age", "Age_Bands", "Legal_Social_Stage"], [
        "Minor",
        "Working_Age",
        "Retirement_Age",
    ] + UNK_PNTS)

    # ----------------------------
    # Age: developmental stage frameworks (explicit stage lists)
    # ----------------------------
    add_leaves(demo, ["Age", "Developmental_Stage", "Erikson_Psychosocial"], [
        "Trust_vs_Mistrust",
        "Autonomy_vs_Shame_and_Doubt",
        "Initiative_vs_Guilt",
        "Industry_vs_Inferiority",
        "Identity_vs_Role_Confusion",
        "Intimacy_vs_Isolation",
        "Generativity_vs_Stagnation",
        "Integrity_vs_Despair",
    ] + UNK_PNTS)

    add_leaves(demo, ["Age", "Developmental_Stage", "Piaget_Cognitive"], [
        "Sensorimotor",
        "Preoperational",
        "Concrete_Operational",
        "Formal_Operational",
    ] + UNK_PNTS)

    add_leaves(demo, ["Age", "Developmental_Stage", "Kohlberg_Moral"], [
        "Preconventional_Obedience_and_Punishment",
        "Preconventional_Individualism_and_Exchange",
        "Conventional_Good_Interpersonal_Relationships",
        "Conventional_Maintaining_Social_Order",
        "Postconventional_Social_Contract_and_Individual_Rights",
        "Postconventional_Universal_Ethical_Principles",
    ] + UNK_PNTS)

    add_leaves(demo, ["Age", "Developmental_Stage", "Freud_Psychosexual"], [
        "Oral",
        "Anal",
        "Phallic",
        "Latency",
        "Genital",
    ] + UNK_PNTS)

    add_leaves(demo, ["Age", "Developmental_Stage", "Havighurst_Developmental_Period"], [
        "Infancy_and_Early_Childhood",
        "Middle_Childhood",
        "Adolescence",
        "Early_Adulthood",
        "Middle_Age",
        "Later_Maturity",
    ] + UNK_PNTS)

    # Marcia identity status (identity formation; often used in psych development)
    add_leaves(demo, ["Age", "Developmental_Stage", "Marcia_Identity_Status"], [
        "Identity_Diffusion",
        "Identity_Foreclosure",
        "Identity_Moratorium",
        "Identity_Achievement",
    ] + UNK_PNTS)

    # Schaie adult cognitive development stages (common adult-development framework)
    add_leaves(demo, ["Age", "Developmental_Stage", "Schaie_Adult_Cognitive_Stage"], [
        "Acquisitive",
        "Achieving",
        "Responsible",
        "Executive",
        "Reintegrative",
    ] + UNK_PNTS)

    # Attachment style / pattern (infant + adult)
    add_leaves(demo, ["Age", "Developmental_Stage", "Infant_Attachment_Pattern_Ainsworth"], [
        "Secure",
        "Avoidant",
        "Resistant_Ambivalent",
        "Disorganized",
    ] + UNK_PNTS)

    add_leaves(demo, ["Age", "Developmental_Stage", "Adult_Attachment_Style_Bartholomew"], [
        "Secure",
        "Anxious_Preoccupied",
        "Dismissive_Avoidant",
        "Fearful_Avoidant",
    ] + UNK_PNTS)

    # A pragmatic, theory-agnostic life-course stage used in biomedical contexts
    add_leaves(demo, ["Age", "Developmental_Stage", "Life_Course_Generic"], [
        "Early_Life",
        "School_Years",
        "Transition_to_Adulthood",
        "Reproductive_Years",
        "Midlife",
        "Later_Life",
    ] + UNK_PNTS)

    # ----------------------------
    # Puberty & reproductive maturation (biomedical, non-diagnostic)
    # ----------------------------
    add_leaves(demo, ["Age", "Puberty_and_Maturation", "Pubertal_Status"], [
        "Prepubertal",
        "In_Puberty",
        "Postpubertal",
    ] + UNK_PNTS)

    add_leaves(demo, ["Age", "Puberty_and_Maturation", "Tanner_Stage"], [
        "Stage_I",
        "Stage_II",
        "Stage_III",
        "Stage_IV",
        "Stage_V",
    ] + UNK_PNTS)

    add_leaves(demo, ["Age", "Puberty_and_Maturation", "Puberty_Timing"], [
        "Early_Timing",
        "On_Time",
        "Late_Timing",
    ] + UNK_PNTS)

    add_leaves(demo, ["Age", "Puberty_and_Maturation", "Menarche_Status"], [
        "Not_Started",
        "Started",
    ] + UNK_PNTS)

    add_leaves(demo, ["Age", "Puberty_and_Maturation", "Voice_Change_Status"], [
        "Not_Started",
        "In_Progress",
        "Completed",
    ] + UNK_PNTS)

    # ----------------------------
    # Reproductive life stage (generic + menstrual/menopause)
    # ----------------------------
    add_leaves(demo, ["Reproductive_and_Family_Planning", "Pregnancy_Status"], [
        "Not_Pregnant",
        "Pregnant",
        "Postpartum",
        "Recently_Postpartum",
    ] + UNK_PNTS)

    add_leaves(demo, ["Reproductive_and_Family_Planning", "Lactation_Status"], [
        "Not_Lactating",
        "Lactating",
        "Recently_Stopped_Lactating",
    ] + UNK_PNTS)

    add_leaves(demo, ["Reproductive_and_Family_Planning", "Menstrual_Status"], [
        "Regular_Cycles",
        "Irregular_Cycles",
        "Amenorrhea",
        "Oligomenorrhea",
        "Not_Menstruating",
    ] + UNK_PNTS)

    # STRAW-like reproductive aging stage names (label-based, non-numeric)
    add_leaves(demo, ["Reproductive_and_Family_Planning", "Reproductive_Aging_Stage"], [
        "Pre_Menarche",
        "Early_Reproductive",
        "Peak_Reproductive",
        "Late_Reproductive",
        "Early_Menopausal_Transition",
        "Late_Menopausal_Transition",
        "Early_Postmenopause",
        "Late_Postmenopause",
        "Surgical_Menopause",
    ] + UNK_PNTS)

    add_leaves(demo, ["Reproductive_and_Family_Planning", "Contraception_Status"], [
        "No_Contraception",
        "Barrier_Method",
        "Hormonal_Method",
        "Intrauterine_Device",
        "Sterilization",
        "Fertility_Awareness",
        "Emergency_Contraception_Use",
        "Other",
    ] + UNK_PNTS)

    add_leaves(demo, ["Reproductive_and_Family_Planning", "Fertility_Intentions"], [
        "Wants_Children",
        "Does_Not_Want_Children",
        "Undecided",
        "Currently_Trying_To_Conceive",
        "Fertility_Preservation_Interest",
    ] + UNK_PNTS)

    # Raw reproductive history variables (kept as raw)
    add_leaves(demo, ["Reproductive_and_Family_Planning", "Reproductive_History_Raw"], [
        "Gravida_Count",
        "Para_Count",
        "Live_Births_Count",
        "Pregnancy_Loss_Count",
        "Age_At_Menarche_Raw",
        "Age_At_Menopause_Raw",
    ])

    # ----------------------------
    # Frailty / biological aging (non-diagnostic categories)
    # ----------------------------
    add_leaves(demo, ["Age", "Biological_Age_and_Frailty", "Frailty_Status"], [
        "Robust",
        "Prefrail",
        "Frail",
    ] + UNK_PNTS)

    # Clinical Frailty Scale (names only)
    add_leaves(demo, ["Age", "Biological_Age_and_Frailty", "Clinical_Frailty_Scale_Category"], [
        "Very_Fit",
        "Well",
        "Managing_Well",
        "Vulnerable",
        "Mildly_Frail",
        "Moderately_Frail",
        "Severely_Frail",
        "Very_Severely_Frail",
        "Terminally_Ill",
    ] + UNK_PNTS)

    # Raw biological age metrics placeholders
    add_leaves(demo, ["Age", "Biological_Age_and_Frailty", "Biological_Age_Metrics_Raw"], [
        "Epigenetic_Age",
        "Phenotypic_Age",
        "Telomere_Length",
        "Allostatic_Load_Score",
    ])

    # ----------------------------
    # Sex & gender (SOGI: separate dimensions)
    # ----------------------------
    add_leaves(demo, ["Sex_Assigned_At_Birth"], [
        "Female",
        "Male",
        "Intersex",
        "Another_Sex_Assignment",
    ] + UNK_PNTS)

    add_leaves(demo, ["Legal_Sex_Marker"], [
        "Female",
        "Male",
        "Nonbinary_or_X",
        "Not_Recorded",
    ] + UNK_PNTS)

    # Sex characteristics (non-diagnostic, structural)
    add_leaves(demo, ["Sex_Characteristics", "Chromosomal_Pattern"], [
        "XX",
        "XY",
        "XXY",
        "X0",
        "XXX",
        "XYY",
        "Mosaic_or_Chimeric",
        "Other",
    ] + UNK_PNTS)

    add_leaves(demo, ["Sex_Characteristics", "Gonadal_Tissue"], [
        "Ovarian",
        "Testicular",
        "Ovotesticular",
        "Dysgenetic_or_Undifferentiated",
        "Other",
    ] + UNK_PNTS)

    add_leaves(demo, ["Sex_Characteristics", "Reproductive_Anatomy"], [
        "Typically_Female",
        "Typically_Male",
        "Variation_in_Sex_Characteristics",
        "Other",
    ] + UNK_PNTS)

    # Gender identity (expanded)
    add_leaves(demo, ["Gender_Identity", "Woman_Identities"], [
        "Woman",
        "Cis_Woman",
        "Trans_Woman",
        "Transfeminine",
        "Other_Woman_Identity",
    ])
    add_leaves(demo, ["Gender_Identity", "Man_Identities"], [
        "Man",
        "Cis_Man",
        "Trans_Man",
        "Transmasculine",
        "Other_Man_Identity",
    ])
    add_leaves(demo, ["Gender_Identity", "Nonbinary_and_Gender_Diverse"], [
        "Nonbinary",
        "Genderqueer",
        "Agender",
        "Genderfluid",
        "Two_Spirit",
        "Demigender",
        "Third_Gender",
        "Gender_Nonconforming",
        "Other_Gender_Diverse_Identity",
    ])
    add_leaves(demo, ["Gender_Identity"], [
        "Questioning",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Gender modality / trans status as its own variable (useful analytically)
    add_leaves(demo, ["Gender_Modality"], [
        "Cisgender",
        "Transgender",
        "Nonbinary_or_Gender_Diverse",
        "Questioning",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Gender expression / presentation
    add_leaves(demo, ["Gender_Expression"], [
        "Feminine",
        "Masculine",
        "Androgynous",
        "Varies_By_Context",
        "Gender_Nonconforming",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Pronouns (extendable)
    add_leaves(demo, ["Pronouns"], [
        "He_Him",
        "She_Her",
        "They_Them",
        "He_They",
        "She_They",
        "Any_Pronouns",
        "No_Pronouns",
        "Use_My_Name",
        "Other_Pronouns",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # SOGI disclosure ("outness") as a separate dimension
    add_leaves(demo, ["SOGI_Disclosure_Level"], [
        "Not_Disclosed",
        "Disclosed_To_Close_Others",
        "Disclosed_In_Many_Contexts",
        "Disclosed_In_Most_Contexts",
        "Disclosed_Publicly",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Sexual orientation: separate identity / attraction / behavior + scales
    add_leaves(demo, ["Sexual_Orientation", "Identity"], [
        "Heterosexual",
        "Gay",
        "Lesbian",
        "Bisexual",
        "Pansexual",
        "Asexual",
        "Demisexual",
        "Queer",
        "Questioning",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(demo, ["Sexual_Orientation", "Attraction"], [
        "Attracted_To_Women",
        "Attracted_To_Men",
        "Attracted_To_Multiple_Genders",
        "Attracted_To_None",
        "Attraction_Is_Fluid",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(demo, ["Sexual_Orientation", "Behavior"], [
        "Sex_With_Women",
        "Sex_With_Men",
        "Sex_With_Multiple_Genders",
        "No_Sexual_Behavior",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(demo, ["Sexual_Orientation", "Temporal_Pattern"], [
        "Stable",
        "Some_Fluidity",
        "Highly_Fluid",
        "Questioning",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Kinsey-style categorical scale (labels, not numeric-only)
    add_leaves(demo, ["Sexual_Orientation", "Kinsey_Scale_Category"], [
        "Exclusively_Heterosexual",
        "Predominantly_Heterosexual_Incidental_Homosexual",
        "Predominantly_Heterosexual_More_Than_Incidental_Homosexual",
        "Bisexual_Equal",
        "Predominantly_Homosexual_More_Than_Incidental_Heterosexual",
        "Predominantly_Homosexual_Incidental_Heterosexual",
        "Exclusively_Homosexual",
        "No_Sociosexual_Contacts",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Asexual spectrum resolution
    add_leaves(demo, ["Sexual_Orientation", "Asexual_Spectrum"], [
        "Not_Asexual",
        "Gray_Asexual",
        "Demisexual",
        "Asexual",
        "Questioning",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Romantic orientation
    add_leaves(demo, ["Romantic_Orientation"], [
        "Heteroromantic",
        "Homoromantic",
        "Biromantic",
        "Panromantic",
        "Aromantic",
        "Demiromantic",
        "Questioning",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Relationship / family formation
    add_leaves(demo, ["Relationship_Status"], [
        "Single",
        "Dating",
        "Partnered",
        "Married",
        "Separated",
        "Divorced",
        "Widowed",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(demo, ["Relationship_Structure"], [
        "Monogamous",
        "Non_Monogamous",
        "Polyamorous",
        "Open_Relationship",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(demo, ["Cohabitation"], [
        "Living_Alone",
        "Living_With_Partner",
        "Living_With_Family",
        "Shared_Housing",
        "Institutional_Living",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(demo, ["Relationship_Legal_Recognition"], [
        "Legally_Recognized",
        "Not_Legally_Recognized",
        "Partially_Recognized",
        "Not_Applicable",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Ethnocultural background (generic + measurement context)
    add_leaves(demo, ["Ethnocultural_Background", "Self_Identified"], [
        "African_Descent",
        "East_Asian_Descent",
        "South_Asian_Descent",
        "Southeast_Asian_Descent",
        "Middle_Eastern_Descent",
        "European_Descent",
        "Latinx_Descent",
        "Indigenous_Descent",
        "Pacific_Islander_Descent",
        "Mixed_Background",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(demo, ["Ethnocultural_Background", "Perceived_By_Others"], [
        "Often_Matches_Self_Identification",
        "Sometimes_Matches_Self_Identification",
        "Often_Does_Not_Match_Self_Identification",
        "Not_Applicable",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(demo, ["Ethnocultural_Background", "Ancestry_Raw"], [
        "Ancestry_Text",
        "Grandparental_Origins_Text",
    ])

    # Citizenship / residency status (high-level; detailed in Migration section too)
    add_leaves(demo, ["Citizenship_Status"], [
        "Citizen",
        "Dual_Citizen",
        "Permanent_Resident",
        "Temporary_Resident",
        "Stateless",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(demo, ["Country_Of_Birth"], [
        "Known",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(demo, ["Marital_History"], [
        "Never_Married",
        "Currently_Married",
        "Previously_Married",
        "Multiple_Marriages",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Anthropometrics (raw variables; often useful in biomedical data models)
    add_leaves(demo, ["Anthropometrics_Raw"], [
        "Height",
        "Weight",
        "Body_Mass_Index",
        "Waist_Circumference",
        "Hip_Circumference",
    ])

    PERSON["Demographics_and_Identity"] = demo

    # ============================================================
    # 2) Language and Communication (high-resolution value space)
    # ============================================================
    lang: dict = {}

    add_leaves(lang, ["Primary_Language"], [
        "Dutch",
        "French",
        "German",
        "English",
        "Spanish",
        "Portuguese",
        "Italian",
        "Arabic",
        "Hebrew",
        "Turkish",
        "Persian",
        "Urdu",
        "Hindi",
        "Bengali",
        "Mandarin_Chinese",
        "Cantonese_Chinese",
        "Japanese",
        "Korean",
        "Russian",
        "Ukrainian",
        "Polish",
        "Romanian",
        "Greek",
        "Swedish",
        "Norwegian",
        "Danish",
        "Finnish",
        "Basque",
        "Catalan",
        "Galician",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(lang, ["Additional_Languages", "Count_Category"], [
        "None",
        "One",
        "Two",
        "Three",
        "Four_Or_More",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(lang, ["Proficiency", "Speaking"], [
        "None",
        "Basic",
        "Conversational",
        "Professional",
        "Native",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(lang, ["Proficiency", "Reading"], [
        "None",
        "Basic",
        "Functional",
        "Proficient",
        "Advanced",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(lang, ["Proficiency", "Writing"], [
        "None",
        "Basic",
        "Functional",
        "Proficient",
        "Advanced",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(lang, ["Health_Literacy"], [
        "Limited",
        "Basic",
        "Adequate",
        "Advanced",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(lang, ["Numeracy"], [
        "Limited",
        "Basic",
        "Adequate",
        "Advanced",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(lang, ["Interpreter_Need"], [
        "Needed",
        "Not_Needed",
        "Sometimes_Needed",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(lang, ["Communication_Access", "Hearing"], [
        "No_Restriction",
        "Hard_Of_Hearing",
        "Deaf",
        "Uses_Hearing_Aid",
        "Uses_Cochlear_Implant",
        "Uses_Sign_Language",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(lang, ["Communication_Access", "Vision"], [
        "No_Restriction",
        "Low_Vision",
        "Blind",
        "Uses_Corrective_Lenses",
        "Uses_Screen_Reader",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(lang, ["Preferred_Communication_Mode"], [
        "Spoken_Language",
        "Written_Text",
        "Sign_Language",
        "AAC_Assistive_Communication",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    PERSON["Language_and_Communication"] = lang

    # ============================================================
    # 3) Religion, Spirituality, and Worldview (high-resolution)
    # ============================================================
    belief: dict = {}

    add_leaves(belief, ["Affiliation"], [
        "Christianity",
        "Islam",
        "Judaism",
        "Hinduism",
        "Buddhism",
        "Sikhism",
        "Jainism",
        "Bahai",
        "Shinto",
        "Taoism",
        "Confucianism",
        "Zoroastrianism",
        "Indigenous_Traditional_Religion",
        "New_Religious_Movement",
        "Spiritual_But_Not_Religious",
        "Agnostic",
        "Atheist",
        "No_Religion",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(belief, ["Christianity", "Catholicism"], [
        "Roman_Catholic",
        "Eastern_Catholic",
        "Other_Catholic",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(belief, ["Christianity", "Orthodoxy"], [
        "Eastern_Orthodox",
        "Oriental_Orthodox",
        "Assyrian_Church_of_the_East",
        "Other_Orthodox",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(belief, ["Christianity", "Protestantism", "Mainline"], [
        "Anglican_Episcopal",
        "Lutheran",
        "Reformed",
        "Presbyterian",
        "Methodist",
        "Baptist",
        "Congregational",
        "Other_Mainline_Protestant",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(belief, ["Christianity", "Protestantism", "Evangelical_and_Pentecostal"], [
        "Evangelical",
        "Pentecostal",
        "Charismatic",
        "Non_Denominational",
        "Other_Evangelical_or_Pentecostal",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(belief, ["Christianity", "Restorationist_and_Other"], [
        "Seventh_day_Adventist",
        "Jehovahs_Witnesses",
        "Latter_day_Saints",
        "Other_Christian",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(belief, ["Islam", "Tradition"], [
        "Sunni",
        "Shia",
        "Ibadi",
        "Ahmadiyya",
        "Sufi",
        "Other_Islam",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(belief, ["Judaism", "Tradition"], [
        "Orthodox",
        "Conservative",
        "Reform",
        "Reconstructionist",
        "Renewal",
        "Secular_Cultural_Jewish",
        "Other_Judaism",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(belief, ["Buddhism", "Tradition"], [
        "Theravada",
        "Mahayana",
        "Vajrayana",
        "Zen",
        "Pure_Land",
        "Other_Buddhism",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(belief, ["Hinduism", "Tradition"], [
        "Vaishnavism",
        "Shaivism",
        "Shaktism",
        "Smartism",
        "Other_Hinduism",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(belief, ["Practice_Engagement"], [
        "Not_Practicing",
        "Occasional_Practice",
        "Regular_Practice",
        "Highly_Observant",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(belief, ["Community_Connection"], [
        "Not_Connected",
        "Loosely_Connected",
        "Connected",
        "Central_Community_Role",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    PERSON["Religion_Spirituality_and_Worldview"] = belief

    # ============================================================
    # 4) Household, Family Structure, and Care Roles
    # ============================================================
    household: dict = {}

    add_leaves(household, ["Household_Composition"], [
        "Living_Alone",
        "Living_With_Partner",
        "Living_With_Parents",
        "Living_With_Children",
        "Living_With_Extended_Family",
        "Shared_Housing",
        "Supported_Housing",
        "Institutional_Living",
        "Homeless_or_No_Fixed_Address",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(household, ["Household_Size"], [
        "One",
        "Two",
        "Three",
        "Four",
        "Five_Or_More",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(household, ["Children_Present"], [
        "Yes",
        "No",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(household, ["Children_Age_Groups"], [
        "No_Children",
        "Infant_Child",
        "School_Age_Child",
        "Adolescent_Child",
        "Adult_Child",
        "Multiple_Age_Groups",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(household, ["Parenting_Role"], [
        "Primary_Caregiver",
        "Co_Parent",
        "Shared_Custody",
        "Non_Custodial_Parent",
        "Guardian",
        "Step_Parent",
        "Foster_Parent",
        "Other",
        "Not_Applicable",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(household, ["Caregiving_Role"], [
        "None",
        "Caregiver_For_Child",
        "Caregiver_For_Adult",
        "Caregiver_For_Disabled_Person",
        "Caregiver_For_Elder",
        "Multiple_Caregiving_Roles",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(household, ["Informal_Support_Availability"], [
        "None",
        "Limited",
        "Moderate",
        "Strong",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    PERSON["Household_and_Family"] = household

    # ============================================================
    # 5) Education, Training, and Skills (higher resolution)
    # ============================================================
    edu: dict = {}

    add_leaves(edu, ["Highest_Education"], [
        "No_Formal_Education",
        "Primary",
        "Lower_Secondary",
        "Upper_Secondary",
        "Post_Secondary_Non_Tertiary",
        "Short_Cycle_Tertiary",
        "Bachelors",
        "Masters",
        "Doctoral",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(edu, ["Current_Enrollment"], [
        "Not_Enrolled",
        "School",
        "Vocational_Training",
        "University",
        "Adult_Education",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(edu, ["Field_of_Study", "Broad_Domain"], [
        "Arts_and_Humanities",
        "Social_Sciences",
        "Business_and_Law",
        "Science",
        "Technology",
        "Engineering",
        "Mathematics",
        "Health_and_Medicine",
        "Education",
        "Agriculture_and_Veterinary",
        "Services_and_Hospitality",
        "Other",
        "Not_Applicable",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(edu, ["Training_and_Certification"], [
        "No_Certifications",
        "Trade_Certification",
        "Professional_License",
        "Language_Certification",
        "Technical_Certification",
        "Healthcare_Credential",
        "Other_Certification",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(edu, ["Educational_Trajectory"], [
        "Continuous",
        "Interrupted",
        "Nonlinear",
        "Restarted_After_Break",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(edu, ["Digital_Literacy"], [
        "Low",
        "Basic",
        "Proficient",
        "Advanced",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(edu, ["General_Literacy"], [
        "Low",
        "Basic",
        "Functional",
        "Proficient",
        "Advanced",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    PERSON["Education_and_Skills"] = edu

    # ============================================================
    # 6) Employment, Work Arrangement, and Occupational Context
    # ============================================================
    work: dict = {}

    add_leaves(work, ["Employment_Status"], [
        "Employed",
        "Unemployed",
        "Student",
        "Retired",
        "Homemaker",
        "Unable_To_Work",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(work, ["Employment_Arrangement"], [
        "Full_Time",
        "Part_Time",
        "Temporary_Contract",
        "Permanent_Contract",
        "Self_Employed",
        "Gig_Work",
        "Informal_Work",
        "Multiple_Jobs",
        "Not_Applicable",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(work, ["Work_Mode"], [
        "Onsite",
        "Remote",
        "Hybrid",
        "Not_Applicable",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(work, ["Work_Schedule_Type"], [
        "Standard_Daytime",
        "Shift_Work",
        "Night_Work",
        "Rotating_Shifts",
        "On_Call",
        "Irregular",
        "Not_Applicable",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Occupation: generic major groups (ISCO-inspired)
    add_leaves(work, ["Occupation", "Major_Group"], [
        "Managers",
        "Professionals",
        "Technicians_and_Associate_Professionals",
        "Clerical_Support_Workers",
        "Service_and_Sales_Workers",
        "Skilled_Agricultural_Forestry_and_Fishery_Workers",
        "Craft_and_Related_Trades_Workers",
        "Plant_and_Machine_Operators_and_Assemblers",
        "Elementary_Occupations",
        "Armed_Forces_Occupations",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(work, ["Sector"], [
        "Public",
        "Private",
        "Nonprofit",
        "Informal",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(work, ["Job_Security"], [
        "Secure",
        "Somewhat_Secure",
        "Insecure",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(work, ["Workplace_Support"], [
        "Low",
        "Moderate",
        "High",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(work, ["Workplace_Discrimination_Experience"], [
        "None",
        "Suspected",
        "Confirmed",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    PERSON["Employment_and_Work"] = work

    # ============================================================
    # 7) Socioeconomic and Material Resources (high-resolution)
    # ============================================================
    ses: dict = {}

    # Objective SEP components (categorical + raw placeholders)
    add_leaves(ses, ["Socioeconomic_Position", "Measurement_Timeframe"], [
        "Current",
        "Childhood",
        "Lifetime_or_Typical",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(ses, ["Socioeconomic_Position", "Income", "Household_Income_Category"], [
        "Very_Low",
        "Low",
        "Lower_Middle",
        "Middle",
        "Upper_Middle",
        "High",
        "Very_High",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(ses, ["Socioeconomic_Position", "Income", "Personal_Income_Category"], [
        "Very_Low",
        "Low",
        "Lower_Middle",
        "Middle",
        "Upper_Middle",
        "High",
        "Very_High",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(ses, ["Socioeconomic_Position", "Income", "Income_Raw"], [
        "Household_Income_Amount",
        "Personal_Income_Amount",
        "Income_Currency",
        "Income_Reference_Period",
    ])

    add_leaves(ses, ["Socioeconomic_Position", "Income", "Income_Stability"], [
        "Stable",
        "Variable",
        "Seasonal",
        "Unpredictable",
        "No_Income",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(ses, ["Socioeconomic_Position", "Wealth_and_Assets", "Wealth_Category"], [
        "No_Assets",
        "Limited_Assets",
        "Moderate_Assets",
        "Substantial_Assets",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(ses, ["Socioeconomic_Position", "Wealth_and_Assets", "Savings_Buffer"], [
        "None",
        "Short_Term",
        "Moderate",
        "Robust",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(ses, ["Socioeconomic_Position", "Wealth_and_Assets", "Debt_Burden"], [
        "None",
        "Low",
        "Moderate",
        "High",
        "Severe",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Subjective SES (e.g., MacArthur ladder)
    add_leaves(ses, ["Socioeconomic_Position", "Subjective_Status", "Subjective_SES_Category"], [
        "Very_Low",
        "Low",
        "Middle",
        "High",
        "Very_High",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(ses, ["Socioeconomic_Position", "Subjective_Status", "MacArthur_Ladder_Raw"], [
        "MacArthur_Ladder_Rung",
        "MacArthur_Ladder_Context",
    ])

    # Childhood SEP proxies
    add_leaves(ses, ["Socioeconomic_Position", "Childhood_SES", "Childhood_SES_Category"], [
        "Very_Low",
        "Low",
        "Middle",
        "High",
        "Very_High",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(ses, ["Socioeconomic_Position", "Childhood_SES", "Parental_Education"], [
        "No_Formal_Education",
        "Primary",
        "Secondary",
        "Tertiary",
        "Postgraduate",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Social class frameworks (broad, cross-national)
    add_leaves(ses, ["Socioeconomic_Position", "Social_Class", "Class_Self_Identification"], [
        "Working_Class",
        "Lower_Middle_Class",
        "Middle_Class",
        "Upper_Middle_Class",
        "Upper_Class",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(ses, ["Socioeconomic_Position", "Social_Class", "Bourdieu_Capitals"], [
        "Economic_Capital_Low",
        "Economic_Capital_Moderate",
        "Economic_Capital_High",
        "Cultural_Capital_Low",
        "Cultural_Capital_Moderate",
        "Cultural_Capital_High",
        "Social_Capital_Low",
        "Social_Capital_Moderate",
        "Social_Capital_High",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Intergenerational mobility
    add_leaves(ses, ["Socioeconomic_Position", "Intergenerational_Mobility"], [
        "Upward_Mobility",
        "Stable",
        "Downward_Mobility",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Material deprivation (item-level; can be derived into an index)
    add_leaves(ses, ["Material_Deprivation", "Core_Needs"], [
        "Unable_To_Pay_Bills",
        "Unable_To_Keep_Home_Warm",
        "Unable_To_Afford_Protein_Regularly",
        "Unable_To_Afford_Medications",
        "Unable_To_Afford_Healthcare_Visits",
        "Missed_Rent_or_Mortgage",
        "Missed_Utility_Payments",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(ses, ["Material_Deprivation", "Transportation_Deprivation"], [
        "No_Public_Transport_Access",
        "Cannot_Afford_Transport",
        "Mobility_Constraints",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(ses, ["Material_Deprivation", "Digital_Deprivation"], [
        "No_Internet_Access",
        "No_Device_Access",
        "Insufficient_Data_Plan",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Financial strain (subjective stress)
    add_leaves(ses, ["Financial_Strain", "Difficulty_Making_Ends_Meet"], [
        "Very_Difficult",
        "Difficult",
        "Manageable",
        "Comfortable",
        "Very_Comfortable",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(ses, ["Financial_Strain", "Unexpected_Expense_Capacity"], [
        "Cannot_Cover",
        "Can_Cover_With_Difficulty",
        "Can_Cover",
        "Can_Easily_Cover",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(ses, ["Financial_Strain", "Financial_Inclusion"], [
        "Banked",
        "Underbanked",
        "Unbanked",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Food security (high-resolution)
    add_leaves(ses, ["Food_Access"], [
        "Secure",
        "Occasionally_Insecure",
        "Insecure",
        "Severely_Insecure",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Benefits / social protection
    add_leaves(ses, ["Benefits_And_Social_Protection"], [
        "None",
        "Means_Tested_Benefits",
        "Disability_Benefits",
        "Unemployment_Benefits",
        "Family_Benefits",
        "Housing_Benefits",
        "Old_Age_Pension",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Transportation access (SEP-relevant)
    add_leaves(ses, ["Transportation_Access"], [
        "Reliable",
        "Mostly_Reliable",
        "Limited",
        "No_Access",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Housing cost burden (SEP + housing intersection)
    add_leaves(ses, ["Housing_Cost_Burden"], [
        "Not_Burdened",
        "Moderately_Burdened",
        "Severely_Burdened",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Neighborhood socioeconomic context
    add_leaves(ses, ["Neighborhood_SES", "Area_Deprivation_Level"], [
        "Low_Deprivation",
        "Moderate_Deprivation",
        "High_Deprivation",
        "Very_High_Deprivation",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(ses, ["Neighborhood_SES", "Index_Raw"], [
        "Neighborhood_Deprivation_Index_Value",
        "Neighborhood_Deprivation_Index_Name",
    ])
    add_leaves(ses, ["Neighborhood_SES", "Local_Resource_Access"], [
        "High_Access",
        "Moderate_Access",
        "Low_Access",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    PERSON["Socioeconomic_and_Resources"] = ses

    # ============================================================
    # 8) Housing and Neighborhood (high-resolution value-space)
    # ============================================================
    housing: dict = {}

    add_leaves(housing, ["Housing_Tenure"], [
        "Owner",
        "Private_Renter",
        "Social_Renter",
        "Living_With_Family_No_Rent",
        "Employer_Provided",
        "Temporary_Accommodation",
        "Shelter",
        "Street_Homelessness",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(housing, ["Housing_Stability"], [
        "Stable",
        "Somewhat_Stable",
        "Unstable",
        "At_Risk_Of_Loss",
        "No_Fixed_Address",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(housing, ["Crowding"], [
        "Not_Crowded",
        "Mildly_Crowded",
        "Crowded",
        "Severely_Crowded",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(housing, ["Housing_Quality"], [
        "Good",
        "Adequate",
        "Substandard",
        "Hazardous",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(housing, ["Environmental_Hazards_At_Home"], [
        "None_Noted",
        "Damp_Or_Mold",
        "Pests",
        "Unsafe_Structure",
        "Poor_Ventilation",
        "Extreme_Temperatures",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(housing, ["Neighborhood_Safety"], [
        "Safe",
        "Mostly_Safe",
        "Unsafe",
        "Very_Unsafe",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(housing, ["Neighborhood_Context"], [
        "High_Social_Cohesion",
        "Moderate_Social_Cohesion",
        "Low_Social_Cohesion",
        "High_Disorder",
        "Moderate_Disorder",
        "Low_Disorder",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(housing, ["Local_Service_Access"], [
        "Good_Access",
        "Moderate_Access",
        "Poor_Access",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    PERSON["Housing_and_Neighborhood"] = housing

    # ============================================================
    # 9) Migration, Residency, and Integration (high-resolution)
    # ============================================================
    mig: dict = {}

    add_leaves(mig, ["Migration_Position"], [
        "Non_Migrant",
        "First_Generation_Migrant",
        "Second_Generation",
        "Multiple_Generation_Diaspora",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(mig, ["Residency_Status"], [
        "Citizen",
        "Permanent_Resident",
        "Temporary_Resident",
        "Student_Visa",
        "Work_Visa",
        "Family_Reunification",
        "Refugee",
        "Asylum_Seeker",
        "Undocumented",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(mig, ["Migration_Motivation"], [
        "Work",
        "Study",
        "Family",
        "Safety_Or_Asylum",
        "Economic_Necessity",
        "Health_Related",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(mig, ["Acculturation_Orientation"], [
        "Integration",
        "Assimilation",
        "Separation",
        "Marginalization",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(mig, ["Host_Society_Belonging"], [
        "Low",
        "Moderate",
        "High",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(mig, ["Length_Of_Stay_Category"], [
        "Recently_Arrived",
        "Short_Term",
        "Medium_Term",
        "Long_Term",
        "Lifelong",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(mig, ["Length_Of_Stay_Raw"], [
        "Years_In_Host_Country",
        "Age_At_Migration",
    ])

    add_leaves(mig, ["Credential_Recognition"], [
        "Not_Applicable",
        "Recognized",
        "Partially_Recognized",
        "Not_Recognized",
        "In_Process",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    PERSON["Migration_and_Residency"] = mig

    # ============================================================
    # 10) Geography and Location (logical categories + regions)
    # ============================================================
    geo: dict = {}

    add_leaves(geo, ["Residence_Context"], [
        "Current_Residence",
        "Primary_Residence",
        "Secondary_Residence",
        "Prior_Residence",
        "Work_Location",
        "Study_Location",
    ])

    add_leaves(geo, ["Broad_Region", "Continent"], [
        "Africa",
        "Asia",
        "Europe",
        "North_America",
        "South_America",
        "Oceania",
        "Antarctica",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(geo, ["Broad_Region", "Europe_Subregion"], [
        "Western_Europe",
        "Northern_Europe",
        "Southern_Europe",
        "Eastern_Europe",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(geo, ["Broad_Region", "Africa_Subregion"], [
        "North_Africa",
        "West_Africa",
        "Central_Africa",
        "East_Africa",
        "Southern_Africa",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(geo, ["Broad_Region", "Asia_Subregion"], [
        "West_Asia",
        "Central_Asia",
        "South_Asia",
        "East_Asia",
        "Southeast_Asia",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(geo, ["Urbanicity"], [
        "Urban_Core",
        "Suburban",
        "PeriUrban",
        "Rural",
        "Remote_Rural",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(geo, ["Mobility_Pattern"], [
        "Mostly_Home_Based",
        "Local_Mobility",
        "Regional_Mobility",
        "Frequent_Travel",
        "Highly_Mobile",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Raw variable placeholders for codes/names without enumerating all places
    add_leaves(geo, ["Administrative_Units_Raw"], [
        "Country_Code",
        "Region_Code",
        "Province_Code",
        "City_Code",
        "Neighborhood_Code",
        "Postal_Code",
        "Service_Catchment_Code",
    ])

    PERSON["Geography_and_Location"] = geo

    # ============================================================
    # 11) Social, Community, and Discrimination (person-level)
    # ============================================================
    social: dict = {}

    add_leaves(social, ["Social_Network_Size"], [
        "Very_Small",
        "Small",
        "Moderate",
        "Large",
        "Very_Large",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(social, ["Close_Support_Availability"], [
        "None",
        "Limited",
        "Moderate",
        "Strong",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(social, ["Loneliness"], [
        "Low",
        "Moderate",
        "High",
        "Very_High",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(social, ["Community_Participation"], [
        "None",
        "Occasional",
        "Regular",
        "Highly_Engaged",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(social, ["Civic_Engagement"], [
        "None",
        "Occasional",
        "Active",
        "Leadership",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(social, ["Discrimination_Exposure", "Context"], [
        "None_Noted",
        "Workplace",
        "Education",
        "Healthcare",
        "Housing",
        "Public_Space",
        "Online",
        "Multiple_Contexts",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(social, ["Discrimination_Exposure", "Basis"], [
        "Race_Ethnicity",
        "Nationality",
        "Religion",
        "Gender",
        "Sexual_Orientation",
        "Disability",
        "Socioeconomic_Position",
        "Age",
        "Language",
        "Migration_Status",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(social, ["Social_Capital", "Bridging"], [
        "Low",
        "Moderate",
        "High",
        "Unknown",
        "Prefer_Not_To_Say",
    ])
    add_leaves(social, ["Social_Capital", "Bonding"], [
        "Low",
        "Moderate",
        "High",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    PERSON["Social_and_Community"] = social

    # ============================================================
    # 12) Safety, Legal, and Justice-System Context (HIGH-RESOLUTION)
    # ============================================================
    safety: dict = {}

    # ----------------------------
    # 12.1 Perceived / contextual safety
    # ----------------------------
    add_leaves(safety, ["Perceived_Safety", "Overall"], [
        "Safe",
        "Mostly_Safe",
        "Unsafe",
        "Very_Unsafe",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Perceived_Safety", "Context"], [
        "Home",
        "Neighborhood",
        "Workplace",
        "School_or_Training",
        "Healthcare_Settings",
        "Public_Transport",
        "Public_Spaces",
        "Online",
        "Multiple_Contexts",
        "Not_Applicable",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Perceived_Safety", "Timeframe"], [
        "Current",
        "Past_Year",
        "Earlier_Life",
        "Multiple_Timeframes",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # ----------------------------
    # 12.2 Threat / violence exposure (high-resolution)
    # ----------------------------
    add_leaves(safety, ["Violence_and_Threat_Exposure", "Exposure_Status"], [
        "None_Noted",
        "Possible",
        "Confirmed",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Violence_and_Threat_Exposure", "Role_or_Position"], [
        "Victim_Survivor",
        "Witness",
        "Both_Victim_and_Witness",
        "Not_Applicable",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Violence_and_Threat_Exposure", "Context"], [
        "Domestic_or_Intimate_Partner",
        "Family_NonPartner",
        "Community",
        "Workplace",
        "School_or_Training",
        "Institutional_or_Care_Setting",
        "Hate_Motivated",
        "Law_Enforcement_or_Security",
        "Online",
        "Other",
        "Multiple_Contexts",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Violence_and_Threat_Exposure", "Type"], [
        "Physical_Violence",
        "Sexual_Violence",
        "Psychological_or_Emotional_Abuse",
        "Coercive_Control",
        "Stalking",
        "Harassment_or_Threats",
        "Robbery_or_Assault",
        "Weapon_Related_Threat",
        "Neglect",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Violence_and_Threat_Exposure", "Frequency_Pattern"], [
        "Single_Incident",
        "Occasional",
        "Repeated",
        "Chronic_or_Ongoing",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Violence_and_Threat_Exposure", "Recency"], [
        "Past_Week",
        "Past_Month",
        "Past_Year",
        "More_Than_One_Year_Ago",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Violence_and_Threat_Exposure", "Severity_Impact"], [
        "Low_Impact",
        "Moderate_Impact",
        "High_Impact",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Violence_and_Threat_Exposure", "Medical_Attention_Required"], [
        "No",
        "Yes",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # ----------------------------
    # 12.3 Legal resources, supports, and protection
    # ----------------------------
    add_leaves(safety, ["Legal_Support_and_Resources", "Access_Level"], [
        "No_Access",
        "Limited_Access",
        "Adequate_Access",
        "Strong_Access",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Legal_Support_and_Resources", "Barriers"], [
        "Cost",
        "Language",
        "Transport_or_Distance",
        "Documentation",
        "Fear_of_Retaliation",
        "Fear_of_Deportation_or_Immigration_Consequences",
        "Low_Trust_in_System",
        "Prior_Negative_Experience",
        "Stigma",
        "Complexity_or_Bureaucracy",
        "Time_Constraints",
        "Other",
        "Multiple_Barriers",
        "None_Noted",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Legal_Support_and_Resources", "Support_Types_Available"], [
        "Legal_Aid_or_Public_Defender",
        "Private_Attorney",
        "Advocacy_or_Caseworker",
        "Victim_Services",
        "Shelter_or_Protection_Services",
        "Immigration_Legal_Support",
        "Workplace_or_Union_Support",
        "Other",
        "None",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Legal_Support_and_Resources", "Protection_Orders"], [
        "Not_Applicable",
        "Never_Sought",
        "Sought_Not_Granted",
        "Granted_Active",
        "Granted_Expired",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # ----------------------------
    # 12.4 Justice system contact (structured)
    # ----------------------------
    add_leaves(safety, ["Justice_System_Contact", "Contact_Status"], [
        "None",
        "Any_Contact",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Justice_System_Contact", "Role_or_Context"], [
        "Victim_Contact",
        "Witness_Contact",
        "Accused_or_Suspected",
        "Charged",
        "Convicted",
        "Probation_or_Parole",
        "Prior_Incarceration",
        "Civil_or_Family_Court",
        "Immigration_Proceedings",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Justice_System_Contact", "Timeframe"], [
        "Current_or_Ongoing",
        "Past_Year",
        "Earlier_Life",
        "Multiple_Timeframes",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Justice_System_Contact", "Frequency"], [
        "Single",
        "Occasional",
        "Repeated",
        "Chronic_or_Recurrent",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # ----------------------------
    # 12.5 Criminal record (explicit; higher resolution)
    # ----------------------------
    add_leaves(safety, ["Criminal_Record", "Record_Status"], [
        "No_Known_Record",
        "Record_Present",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Criminal_Record", "Record_Detail_Level"], [
        "Arrest_or_Detention_Only",
        "Charge_Only",
        "Conviction",
        "Diversion_or_Deferred_Prosecution",
        "Expunged_or_Sealed_Record",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Criminal_Record", "Supervision_or_Custody_History"], [
        "None",
        "Probation_History",
        "Parole_History",
        "Pretrial_Supervision",
        "Incarceration_History",
        "Detention_or_Remand",
        "Electronic_Monitoring",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Criminal_Record", "Offense_Category", "Violent"], [
        "Assault_or_Battery",
        "Robbery",
        "Homicide_Related",
        "Weapon_Related",
        "Domestic_Violence_Offense",
        "Sexual_Violence_Offense",
        "Other_Violent",
        "Not_Applicable",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Criminal_Record", "Offense_Category", "Nonviolent_Property"], [
        "Theft_or_Larceny",
        "Burglary",
        "Fraud_or_Financial_Crime",
        "Property_Damage",
        "Other_Nonviolent_Property",
        "Not_Applicable",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Criminal_Record", "Offense_Category", "Substance_Related"], [
        "Possession",
        "Distribution_or_Trafficking",
        "DUI_or_Impaired_Driving",
        "Other_Substance_Related",
        "Not_Applicable",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Criminal_Record", "Offense_Category", "Public_Order_or_Administrative"], [
        "Disorderly_Conduct",
        "Trespass",
        "Protest_or_Assembly_Related",
        "Immigration_or_Documentation_Offense",
        "Other_Public_Order",
        "Not_Applicable",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Criminal_Record", "Offense_Category", "Traffic_or_Motor_Vehicle"], [
        "Traffic_Violation_Serious",
        "DUI_or_Impaired_Driving",
        "Driving_While_Suspended",
        "Other_Traffic_or_Motor_Vehicle",
        "Not_Applicable",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Criminal_Record", "Recency"], [
        "Current_or_Past_Year",
        "One_to_Five_Years_Ago",
        "More_Than_Five_Years_Ago",
        "Multiple_Timeframes",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(safety, ["Criminal_Record", "Legal_Consequences"], [
        "None_or_Unknown",
        "Fines_or_Fees",
        "Community_Service",
        "Mandated_Treatment_or_Programs",
        "Restraining_or_NoContact_Order",
        "License_Suspension",
        "Employment_or_Licensing_Restrictions",
        "Housing_Restrictions",
        "Immigration_Consequences",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    # Raw placeholders (keep un-enumerated to avoid sensitive specifics but allow provenance)
    add_leaves(safety, ["Criminal_Record", "Record_Metadata_Raw"], [
        "Jurisdiction_Text",
        "Record_Source",
        "Record_Last_Updated",
    ])

    PERSON["Safety_and_Legal"] = safety

    #
    # ============================================================
    # 13) Digital and Technology Access (as person resources)
    # ============================================================
    digital: dict = {}

    add_leaves(digital, ["Internet_Access"], [
        "No_Access",
        "Intermittent_Access",
        "Reliable_Access",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(digital, ["Device_Access"], [
        "No_Device",
        "Shared_Device",
        "Owns_Smartphone",
        "Owns_Computer",
        "Owns_Tablet",
        "Multiple_Devices",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(digital, ["Private_Digital_Space"], [
        "No_Private_Space",
        "Limited_Private_Space",
        "Private_Space_Available",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(digital, ["Digital_Confidence"], [
        "Low",
        "Moderate",
        "High",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(digital, ["Digital_Risks", "Online_Harassment_Exposure"], [
        "None",
        "Occasional",
        "Frequent",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    PERSON["Digital_and_Technology"] = digital

    # ============================================================
    # 14) Functional Abilities and Accessibility (non-diagnosis)
    # ============================================================
    function: dict = {}

    add_leaves(function, ["Mobility"], [
        "No_Limitations",
        "Mild_Limitations",
        "Moderate_Limitations",
        "Severe_Limitations",
        "Uses_Mobility_Aid",
        "Wheelchair_User",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(function, ["Self_Care_Function"], [
        "Independent",
        "Needs_Some_Assistance",
        "Needs_Substantial_Assistance",
        "Dependent",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(function, ["Daily_Living_Function"], [
        "Independent",
        "Needs_Some_Assistance",
        "Needs_Substantial_Assistance",
        "Dependent",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(function, ["Cognitive_Communication_Function"], [
        "No_Difficulties",
        "Mild_Difficulties",
        "Moderate_Difficulties",
        "Severe_Difficulties",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(function, ["Accessibility_Accommodations"], [
        "None",
        "Physical_Access",
        "Communication_Access",
        "Sensory_Access",
        "Learning_Access",
        "Workplace_Accommodations",
        "Multiple_Accommodations",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    PERSON["Function_and_Accessibility"] = function

    # ============================================================
    # 15) Healthcare Access and Care Preferences (logistics only)
    # ============================================================
    care: dict = {}

    add_leaves(care, ["Primary_Care_Access"], [
        "No_Regular_Provider",
        "Has_Regular_Provider",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(care, ["Care_Access_Barriers"], [
        "None_Noted",
        "Cost",
        "Transportation",
        "Language",
        "Time_Constraints",
        "Stigma",
        "Documentation",
        "Long_Wait",
        "Other",
        "Multiple_Barriers",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(care, ["Insurance_Coverage"], [
        "None",
        "Public",
        "Private",
        "Mixed",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(care, ["Care_Mode_Preference"], [
        "In_Person",
        "Telehealth",
        "Hybrid",
        "No_Preference",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(care, ["Trust_in_Healthcare_System"], [
        "Low",
        "Moderate",
        "High",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(care, ["Care_Preferences", "Shared_Decision_Making_Preference"], [
        "Clinician_Led",
        "Shared",
        "Patient_Led",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    PERSON["Healthcare_Access_and_Preferences"] = care

    # ============================================================
    # 16) Life Course and Developmental Background (person history, non-clinical)
    # ============================================================
    life: dict = {}

    add_leaves(life, ["Family_of_Origin_Context"], [
        "Raised_By_Both_Parents",
        "Raised_By_Single_Parent",
        "Raised_By_Extended_Family",
        "Foster_Care_History",
        "Institutional_Care_History",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(life, ["Educational_Environment_History"], [
        "Mainstream_Schooling",
        "Special_Education_Support",
        "Home_Schooling",
        "Interrupted_Schooling",
        "Other",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(life, ["Major_Life_Transitions"], [
        "Recent_Relocation",
        "Recent_Job_Change",
        "Recent_Education_Change",
        "Recent_Relationship_Change",
        "Recent_Bereavement",
        "None_Noted",
        "Multiple_Transitions",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    add_leaves(life, ["Adverse_Experiences_Exposure"], [
        "None_Noted",
        "Possible",
        "Confirmed",
        "Unknown",
        "Prefer_Not_To_Say",
    ])

    PERSON["Life_Course_and_Background"] = life

    return {"PERSON": PERSON}


# ------------------------ Writer + metadata ------------------------

def write_outputs(ontology: dict, out_json_path: str) -> tuple[str, str, dict]:
    out_json_path = os.path.expanduser(out_json_path)
    out_dir = os.path.dirname(out_json_path)

    # If path is not viable, fall back to a local PERSON/ folder next to this script
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        out_dir = os.path.join(os.path.dirname(__file__), "PERSON")
        os.makedirs(out_dir, exist_ok=True)
        out_json_path = os.path.join(out_dir, "PERSON.json")

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(ontology, f, ensure_ascii=False, indent=2)

    leaf_paths = list(iter_leaf_paths(ontology["PERSON"]))
    leaf_count = count_leaves(ontology["PERSON"])
    node_count = count_nodes(ontology["PERSON"])
    depth = max_depth(ontology["PERSON"])
    top_counts = subtree_leaf_counts(ontology["PERSON"])
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
    # Required output path per spec
    default_out = "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/01_raw/PERSON/PERSON.json"
    out_json_path = os.environ.get("PERSON_OUT_PATH", default_out)

    ontology = build_person_ontology()

    # Guardrail: reject explicit schedule/frequency tokens in entity names
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
    leaf_paths = list(iter_leaf_paths(ontology["PERSON"]))
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

# TODO: do not run script! personality, political profiling, values and goals components have been manually added INSIDE the json
