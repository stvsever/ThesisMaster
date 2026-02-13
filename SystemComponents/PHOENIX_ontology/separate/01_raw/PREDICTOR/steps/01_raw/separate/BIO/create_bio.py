#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIO Predictor PHOENIX_ontology Generator (Solution-Variable-Oriented)

Design goals
- Leaf nodes are actionable solution variables (interventions, optimizations, choices, tools).
- No disorder-labeled branches (no diagnostic labels as taxonomy anchors).
- No schedule/parameter variables as taxonomy nodes (no frequency, duration, minutes, Hz, etc.).
- Medication nodes are class/mechanism/formulation types (no brand names).
- Broad biopsychosocial coverage: this file generates the BIO layer (sleep, nutrition, movement,
  physiology, environment, medical evaluation, neuromodulation, pharmacology) while keeping all
  nodes solution-variable oriented.

Writes:
  1) BIO.json
  2) metadata.txt (same folder)

Override output path:
  BIO_OUT_PATH="/path/to/BIO/BIO.json" python generate_bio_ontology.py
"""

from __future__ import annotations

import os
import json
import re
import hashlib
import datetime
import statistics
from typing import Dict, List, Tuple, Iterable, Any


# ======================== Tree helpers ========================

def add_path(tree: Dict[str, Any], path: List[str]) -> None:
    node = tree
    for p in path:
        node = node.setdefault(p, {})

def add_leaves(tree: Dict[str, Any], base_path: List[str], leaves: List[str]) -> None:
    for leaf in leaves:
        add_path(tree, base_path + [leaf])

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

def iter_paths(node: Any, prefix: Tuple[str, ...] = ()) -> Iterable[Tuple[str, ...]]:
    """Iterate all node paths (including internal nodes)."""
    if isinstance(node, dict):
        yield prefix
        for k, v in node.items():
            yield from iter_paths(v, prefix + (k,))
    else:
        yield prefix

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

def scan_forbidden_tokens(
    all_paths: List[Tuple[str, ...]],
    forbidden_patterns: List[str],
    limit: int = 50
) -> List[Tuple[Tuple[str, ...], str]]:
    bad = []
    for p in all_paths:
        s = " ".join(p)
        for pat in forbidden_patterns:
            if re.search(pat, s, flags=re.IGNORECASE):
                bad.append((p, pat))
                break
        if len(bad) >= limit:
            break
    return bad

def validate_key_format(all_paths: List[Tuple[str, ...]], limit: int = 50) -> List[Tuple[Tuple[str, ...], str]]:
    """
    Enforce consistent ontology keys:
    - Use underscores instead of spaces
    - Avoid punctuation (except underscore)
    - Alphanumeric + underscore only
    """
    bad: List[Tuple[Tuple[str, ...], str]] = []
    pat = re.compile(r"^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*$")
    for p in all_paths:
        for tok in p:
            if tok == "":
                bad.append((p, "empty_token"))
                break
            if not pat.match(tok):
                bad.append((p, f"bad_token:{tok}"))
                break
        if len(bad) >= limit:
            break
    return bad


# ======================== PHOENIX_ontology builder ========================

def build_bio_ontology() -> Dict[str, Any]:
    """
    BIO ontology:
    - Primary nodes are clear, high-level BIO domains.
    - Subnodes provide stable conceptual organization.
    - Leaf nodes are candidate solution variables or selectable entities.
    """
    BIO: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1) Sleep, Circadian, and Restoration
    # ------------------------------------------------------------------
    sleep: Dict[str, Any] = {}

    # 1A) Sleep environment: light / noise / climate / air / comfort / safety
    add_leaves(sleep, ["Sleep_Environment", "Light_and_Display_Control"], [
        "Blackout_Curtains",
        "Blackout_Shades",
        "Light_Blocking_Eye_Mask",
        "Amber_Night_Light",
        "Red_Spectrum_Bulbs",
        "Warm_Dimmable_Bedside_Lamp",
        "Smart_Circadian_Lighting_System",
        "Screen_Night_Mode",
        "Screen_Brightness_Limiter",
        "Blue_Light_Filter_Glasses",
        "Low_Lux_Bedroom_Lighting",
        "Remove_Standby_LEDs",
        "Outdoor_Daylight_Access_At_Home",
        "Bedroom_Light_Sensor",
        "Sunrise_Alarm_Clock",
        "Window_Film_Light_Control",
        "Curtain_Gap_Sealer",
        "Door_Under_Sweep_Light_Block",
        "Bedside_Lamp_With_Timer",
    ])

    add_leaves(sleep, ["Sleep_Environment", "Noise_and_Vibration_Control"], [
        "White_Noise_Machine",
        "Pink_Noise_Device",
        "Brown_Noise_Device",
        "Fan_Sound_Masking",
        "Earplugs_Foam",
        "Earplugs_Silicone",
        "OverEar_Sleep_Headphones",
        "Door_Seal_Strips",
        "Window_Seal_Strips",
        "Acoustic_Curtains",
        "Acoustic_Panels",
        "Rug_Noise_Dampening",
        "Bed_Frame_Isolation_Pads",
        "Vibration_Dampening_Mat",
        "Roommate_Noise_Agreement",
    ])

    add_leaves(sleep, ["Sleep_Environment", "Thermal_and_Humidity_Control"], [
        "Breathable_Bedding_Cotton",
        "Breathable_Bedding_Linen",
        "Breathable_Bedding_Bamboo",
        "Thermoregulating_Sleepwear",
        "Cooling_Mattress_Pad",
        "Cooling_Pillow",
        "Humidifier",
        "Dehumidifier",
        "Bedroom_Fan",
        "Temperature_Control_Thermostat",
        "Heated_Blanket",
        "Hot_Water_Bottle",
        "Moisture_Wicking_Sheets",
        "Mattress_Protector_Breathable",
        "Cooling_Sleepwear",
        "Wool_Or_Down_Duvet_Choice",
        "Layered_Bedding_System",
    ])

    add_leaves(sleep, ["Sleep_Environment", "Air_Allergen_and_Irritant_Control"], [
        "HEPA_Air_Purifier",
        "Activated_Carbon_Air_Filter",
        "Ventilation_Improvement",
        "Bedroom_CO2_Monitor",
        "Allergen_Proof_Mattress_Encasement",
        "Allergen_Proof_Pillow_Encasement",
        "Dust_Mite_Control",
        "Pet_Dander_Control",
        "Fragrance_Free_Detergent",
        "Fragrance_Free_Bedroom_Policy",
        "Mold_Remediation",
        "Bedding_Hot_Wash",
        "Vacuum_HEPA_Filter",
        "Room_Humidity_Meter",
        "Low_Dust_Bedroom_Declutter",
        "Humidifier_Cleaning_Protocol",
        "Mattress_Vacuuming",
    ])

    add_leaves(sleep, ["Sleep_Environment", "Comfort_and_Body_Support"], [
        "Mattress_Topper",
        "Adjustable_Pillow",
        "Body_Pillow",
        "Weighted_Blanket",
        "Bedding_Texture_Optimization",
        "Sleep_Socks",
        "Knee_Pillow",
        "Neck_Support_Pillow",
        "Pregnancy_Pillow",
        "Sleep_Mask_Contoured",
        "Pillow_Spray_Lavender",
        "Mattress_Firmness_Optimization",
        "Side_Sleeper_Pillow_Alignment",
        "Back_Sleeper_Lumbar_Support",
    ])

    add_leaves(sleep, ["Sleep_Environment", "Bedroom_Safety_and_Security"], [
        "Bedroom_Door_Lock_Function_Check",
        "Night_Light_Pathway_Safety",
        "Trip_Hazard_Removal",
        "Bedside_Water_Access",
        "Phone_Outside_Bed_Reach",
    ])

    # 1B) Sleep routines and practices: decomposed into phases
    add_leaves(sleep, ["Sleep_Routines_and_Practices", "PreSleep_Downshift"], [
        "Wind_Down_Routine",
        "Paper_Book_Reading",
        "Audiobook_Low_Arousal",
        "Guided_Meditation",
        "Yoga_Nidra",
        "Breathwork_Relaxation",
        "Progressive_Muscle_Relaxation",
        "Gentle_Stretching",
        "Warm_Shower_or_Bath",
        "Aromatherapy_Lavender",
        "NonSleep_Deep_Rest_Audio",
        "Cognitive_Unloading_Journaling",
        "ToDo_List_Offloading",
        "Gratitude_Journaling",
        "Body_Scan_Meditation",
        "Soothing_Music_Playlist",
        "Herbal_Tea_Routine",
        "Foot_Bath_Routine",
        "Bedroom_Device_Free_Policy",
        "Clock_Face_Cover",
        "Bedroom_Declutter",
        "Low_Stimulation_Evening_Activities",
        "Evening_Conversation_Boundary",
        "Stimulating_Content_Avoidance",
    ])

    add_leaves(sleep, ["Sleep_Routines_and_Practices", "InBed_Settling"], [
        "Breath_Anchor_Settling",
        "Comfort_Scan_Adjustment",
        "Relaxation_Scan_Head_To_Toe",
        "Imagery_Safe_Place_Practice",
        "Mindful_Labeling_Of_Thoughts",
        "Acceptance_Of_Wakefulness_Practice",
        "Paradoxical_Intention_Sleep_Rationale",
        "Attention_On_Neutral_Sensations",
    ])

    add_leaves(sleep, ["Sleep_Routines_and_Practices", "Night_Awakenings_Support"], [
        "Low_Light_Bathroom_Path",
        "NonEngagement_With_Clock_Checking",
        "Brief_Breath_Reset",
        "Return_To_Bed_Settling_Routine",
        "Neutral_Content_Audio_Option",
        "Warm_Drink_Noncaffeinated_Option",
    ])

    add_leaves(sleep, ["Sleep_Routines_and_Practices", "Morning_Stabilization"], [
        "Morning_Daylight_Exposure",
        "Morning_Breathing_Reset",
        "Hydration_On_Waking",
        "Gentle_Mobility_On_Waking",
        "Morning_Outdoor_Step",
    ])

    # 1C) Circadian entrainment cues (no schedules as nodes)
    add_leaves(sleep, ["Circadian_Entrainment", "Light_Cues"], [
        "Morning_Bright_Light_Exposure",
        "Outdoor_Morning_Walk",
        "Daylight_Exposure_Planning",
        "Evening_Light_Dimming",
        "Dawn_Simulation_Alarm",
        "Dusk_Simulation_Lighting",
        "Late_Evening_Screen_Reduction",
        "Consistent_Morning_Anchor_Routine",
    ])

    add_leaves(sleep, ["Circadian_Entrainment", "Activity_and_Meal_Cues"], [
        "Meal_Timing_Circadian_Cue",
        "Exercise_Timing_Circadian_Cue",
        "Caffeine_Timing_Adjustment",
        "Consistent_Daily_First_Meal_Anchor",
        "Social_Rhythm_Stabilization",
        "Outdoor_Time_Anchor",
    ])

    add_leaves(sleep, ["Circadian_Entrainment", "Context_Shift_Plans"], [
        "Travel_Jetlag_Plan",
        "Shift_Work_Adaptation_Plan",
        "Weekend_Social_Shift_Mitigation",
        "Seasonal_Light_Adjustment",
    ])

    # 1D) Sleep-supporting supplements (grouped by type)
    add_leaves(sleep, ["Sleep_Supplements", "Hormone_and_Amino_Acid_Like"], [
        "Melatonin",
        "Glycine",
        "L_Theanine",
        "Taurine",
        "Inositol",
    ])
    add_leaves(sleep, ["Sleep_Supplements", "Minerals_and_Cofactors"], [
        "Magnesium_Glycinate",
        "Magnesium_Threonate",
        "Magnesium_Citrate",
    ])
    add_leaves(sleep, ["Sleep_Supplements", "Herbal_and_Botanical"], [
        "Chamomile_Extract",
        "Valerian_Extract",
        "Passionflower_Extract",
        "Lavender_Extract",
        "Lemon_Balm",
        "Hops_Extract",
        "Tart_Cherry_Extract",
        "Apigenin",
        "Saffron_Extract",
    ])

    # 1E) Sleep-related pharmacotherapy classes (mechanism/class only)
    add_leaves(sleep, ["Sleep_Pharmacotherapy_Classes"], [
        "Melatonin_Receptor_Agonists",
        "Orexin_Receptor_Antagonists",
        "NonBZD_Hypnotic_Classes",
        "Benzodiazepine_Hypnotic_Classes",
        "Sedating_Antidepressant_Classes",
        "H1_Antihistamine_Sedative_Classes",
        "Alpha2_Adrenergic_Agonist_Sedative_Classes",
        "Antipsychotic_Sedative_Classes",
    ])

    # 1F) Sleep-related devices and screening-support tools
    add_leaves(sleep, ["Sleep_Devices_and_Technologies", "Tracking_and_Feedback"], [
        "Sleep_Tracker_Wearable",
        "Actigraphy_Device",
        "Smart_Alarm",
        "Bedroom_CO2_Monitor",
        "Room_Humidity_Meter",
        "Noise_Meter_Device",
        "Temperature_Sensor_Bedroom",
    ])
    add_leaves(sleep, ["Sleep_Devices_and_Technologies", "Airway_and_Breathing_Support"], [
        "Nasal_Dilator",
        "Nasal_Saline_Irrigation_Device",
        "Positional_Sleep_Support_Device",
        "Mandibular_Advancement_Device",
        "Positive_Airway_Pressure_Device",
        "Humidified_Airflow_Device",
        "Chin_Strap_Snoring_Support",
    ])
    add_leaves(sleep, ["Sleep_Devices_and_Technologies", "Bruxism_and_Muscle_Support"], [
        "Mouthguard_Bruxism_Splint",
        "Jaw_Relaxation_Tool",
        "Warm_Compress_Jaw",
    ])
    add_leaves(sleep, ["Sleep_Devices_and_Technologies", "Thermal_Support"], [
        "Cooling_Pad_Device",
        "Bed_Fan_Device",
        "Heated_Mattress_Pad",
    ])

    BIO["Sleep_Circadian_and_Restoration"] = sleep

    # ------------------------------------------------------------------
    # 2) Nutrition, Hydration, and Metabolic Health
    # ------------------------------------------------------------------
    nut: Dict[str, Any] = {}

    add_leaves(nut, ["Dietary_Patterns_and_Frameworks"], [
        "Mediterranean_Diet",
        "DASH_Diet",
        "Plant_Forward_Diet",
        "Whole_Foods_Diet",
        "Anti_Inflammatory_Diet",
        "Low_Glycemic_Diet",
        "High_Fiber_Diet",
        "Omega3_Rich_Diet",
        "Nordic_Diet",
        "MIND_Diet",
        "Portfolio_Diet",
        "Pescatarian_Diet",
        "Vegetarian_Diet",
        "Vegan_Diet",
        "Flexitarian_Diet",
        "Gluten_Free_Diet",
        "Dairy_Free_Diet",
        "Low_FODMAP_Diet",
        "Elimination_Diet_Clinician_Supervised",
        "Ketogenic_Diet_Medical",
        "Modified_Atkins_Diet",
        "Low_Sodium_Diet",
        "Low_Added_Sugar_Diet",
        "High_Protein_Diet",
        "Protein_Pacing_Diet",
        "Time_Restricted_Eating",
        "Balanced_Macronutrient_Plate",
        "Fermented_Foods_Emphasis",
        "Blood_Lipid_Friendly_Diet",
        "Glycemic_Control_Diet",
        "Low_Histamine_Diet",
        "Autoimmune_Protocol_Diet",
        "Low_Purine_Diet",
        "Traditional_Japanese_Diet",
        "Traditional_Korean_Diet",
        "Traditional_Indian_Diet",
        "Traditional_Mexican_Whole_Foods",
        "Traditional_Levantine_Diet",
    ])

    add_leaves(nut, ["Eating_Behavior_and_Satiety_Skills"], [
        "Mindful_Eating_Practice",
        "Hunger_Fullness_Cue_Tracking",
        "Protein_First_Meal_Construction",
        "Fiber_First_Meal_Construction",
        "Slow_Eating_Pacing",
        "Reduce_Liquid_Calories",
        "Reduce_UltraProcessed_Foods",
        "Increase_Whole_Food_Density",
        "Planned_Snack_Structure",
        "Environment_Design_For_Eating",
        "Grocery_Environment_Design",
        "Portion_Visual_Guides",
    ])

    add_leaves(nut, ["Meal_Planning_and_Structure", "Planning_Tools"], [
        "Grocery_List_Template",
        "Meal_Prep_Batch_Cooking",
        "Meal_Prep_Freezer_Meals",
        "Portion_Containers",
        "Kitchen_Scale",
        "Food_Log_App",
        "Photo_Based_Food_Log",
        "Recipe_Rotation_System",
        "Pantry_Staples_Kit",
        "Healthy_Sauce_and_Dressing_Kit",
        "Spice_Rack_Starter",
        "Dietitian_Coaching",
        "Clinical_Nutrition_Counseling",
        "Cooking_Class_Healthy_Cuisine",
        "Meal_Delivery_Healthy",
        "Budget_Friendly_Meal_Template",
    ])

    add_leaves(nut, ["Meal_Planning_and_Structure", "Cooking_Methods"], [
        "Sheet_Pan_Roasting",
        "Stir_Fry",
        "Slow_Cooker",
        "Pressure_Cooker",
        "Steaming",
        "Grilling",
        "Poaching",
        "Sous_Vide",
        "Air_Fryer",
        "No_Cook_Meals",
        "One_Pot_Meals",
        "Blender_Smoothies",
        "Fermentation_At_Home",
        "Broiling",
        "Baking",
        "Sauteing",
        "Simmering",
        "Blanching",
    ])

    add_leaves(nut, ["Meal_Templates", "Breakfast"], [
        "Oatmeal_Berries_Nuts",
        "Overnight_Oats_Chia",
        "Greek_Yogurt_Fruit_Seeds",
        "Skyr_Bowl_Oats",
        "Egg_Omelet_Vegetables",
        "Eggs_Wholegrain_Toast",
        "Tofu_Scramble_Vegetables",
        "Smoothie_Protein_Fruit",
        "Cottage_Cheese_Fruit",
        "Avocado_Toast_Egg",
        "Chia_Pudding_Fruit",
        "Buckwheat_Porridge_Fruit",
        "Miso_Soup_Breakfast",
        "Savory_Oats_Egg",
        "Breakfast_Burrito_Wholegrain",
        "Protein_Pancakes_Oats",
        "Shakshuka_Vegetables",
        "Tempeh_Breakfast_Bowl",
        "Fruit_Salad_Yogurt",
        "Sardines_Toast_Tomato",
    ])

    add_leaves(nut, ["Meal_Templates", "Lunch"], [
        "Big_Salad_Lean_Protein",
        "Grain_Bowl_Legumes_Veg",
        "Soup_Legume_Veg",
        "Wrap_Wholegrain_Hummus_Veg",
        "Sushi_Bowl_Fish_Veg",
        "Bento_Box_Balanced",
        "Wholegrain_Sandwich_Protein_Veg",
        "Mediterranean_Plate_Hummus_Olives",
        "Lentil_Salad_Veg",
        "Quinoa_Salad_Feta_Veg",
        "Chickpea_Salad_Wrap",
        "Tuna_Bean_Salad",
        "Vegetable_Frittata_Slice",
        "Minestrone_Soup",
        "Greek_Salad_Feta",
        "Poke_Bowl_Fish",
    ])

    add_leaves(nut, ["Meal_Templates", "Dinner"], [
        "Baked_Fish_Roasted_Veg",
        "StirFry_Protein_Veg",
        "Curry_Legumes_Veg",
        "Stew_Lean_Protein_Veg",
        "SheetPan_Chicken_Veg",
        "Wholegrain_Pasta_Veg_Protein",
        "Taco_Bowl_Beans_Veg",
        "Salmon_Quinoa_Veg",
        "Tempeh_Rice_Veg",
        "Eggplant_Tomato_Grain",
        "Chili_Beans_Veg",
        "Miso_Glazed_Fish_Veg",
        "Tofu_Peanut_Sauce_Veg",
        "Mediterranean_Tray_Bake",
        "Stuffed_Peppers",
        "Lentil_Dal_Rice",
        "StirFry_Soba_Veg",
        "Vegetable_Lasagna_Wholegrain",
    ])

    add_leaves(nut, ["Meal_Templates", "Snacks"], [
        "Fruit_Nuts",
        "Greek_Yogurt",
        "Cottage_Cheese",
        "Hummus_Veg_Sticks",
        "Roasted_Chickpeas",
        "Edamame",
        "Protein_Shake",
        "Dark_Chocolate_Nuts",
        "Cheese_Wholegrain_Crackers",
        "Kefir_Drink",
        "Trail_Mix",
        "Nut_Butter_Fruit",
        "Olives_and_Feta",
        "Seaweed_Snacks",
        "Boiled_Egg",
        "Sardines_Crackers",
    ])

    add_leaves(nut, ["Hydration_and_Beverage_Tools"], [
        "Reusable_Water_Bottle",
        "Water_Filter_Pitcher",
        "Water_Filter_UnderSink",
        "Water_Filter_Reverse_Osmosis",
        "Electrolyte_Mix",
        "Oral_Rehydration_Salts",
        "Herbal_Tea_Rotation",
        "Caffeine_Substitution_Decaf",
        "Alcohol_Free_Beverage_Options",
        "Insulated_Travel_Mug",
        "Sparkling_Water_Rotation",
    ])

    add_leaves(nut, ["Caffeine_and_Stimulant_Intake_Modulation"], [
        "Caffeine_Timing_Adjustment",
        "Caffeine_Reduction_Plan",
        "Caffeine_Switch_To_Tea",
        "Caffeine_Switch_To_Decaf",
        "Energy_Drink_Reduction_Plan",
        "PreWorkout_Stimulant_Avoidance",
        "Hydration_First_Beverage_Swap",
    ])

    add_leaves(nut, ["Alcohol_and_Other_Intake_Modulation"], [
        "Alcohol_Intake_Reduction_Framework",
        "Alcohol_Free_Social_Plan",
        "Cannabis_Intake_Reduction_Framework",
        "Nicotine_Intake_Reduction_Framework",
        "Nicotine_Replacement_Therapy_Classes",
        "Cessation_Coaching_Referral",
    ])

    # Micronutrient supplement forms
    vitamins = [
        "Vitamin_A","Vitamin_B1","Vitamin_B2","Vitamin_B3","Vitamin_B5","Vitamin_B6","Vitamin_B7","Vitamin_B9","Vitamin_B12",
        "Vitamin_C","Vitamin_D","Vitamin_E","Vitamin_K"
    ]
    vit_forms = {
        "Vitamin_B9": ["Folic_Acid","L_Methylfolate"],
        "Vitamin_B12": ["Cyanocobalamin","Methylcobalamin","Hydroxocobalamin"],
        "Vitamin_D": ["D2","D3"],
        "Vitamin_K": ["K1","K2_MK4","K2_MK7"],
    }
    for v in vitamins:
        add_leaves(nut, ["Micronutrient_Supplement_Forms", "Vitamins", v], vit_forms.get(v, ["Standard_Form"]))

    minerals = [
        "Calcium","Magnesium","Iron","Zinc","Copper","Selenium","Iodine","Manganese","Chromium","Molybdenum",
        "Potassium","Sodium","Phosphorus","Chloride","Boron","Lithium_Trace"
    ]
    min_forms = {
        "Magnesium": ["Glycinate","Citrate","Threonate","Malate","Oxide","Bisglycinate"],
        "Iron": ["Ferrous_Sulfate","Ferrous_Fumarate","Ferrous_Gluconate","Heme_Iron","Polysaccharide_Iron"],
        "Zinc": ["Picolinate","Gluconate","Citrate","Bisglycinate","Acetate"],
        "Selenium": ["Selenomethionine","Sodium_Selenite"],
        "Iodine": ["Potassium_Iodide","Kelp_Derived"],
        "Calcium": ["Citrate","Carbonate","Hydroxyapatite"],
        "Copper": ["Gluconate","Bisglycinate"],
        "Chromium": ["Picolinate","Nicotinate"],
        "Manganese": ["Gluconate","Bisglycinate"],
        "Molybdenum": ["Glycinate"],
        "Potassium": ["Citrate_Salt","Chloride_Salt"],
    }
    for m in minerals:
        add_leaves(nut, ["Micronutrient_Supplement_Forms", "Minerals", m], min_forms.get(m, ["Standard_Form"]))

    # Supplement and nutraceutical library (broad; still bio-oriented)
    supplement_groups = {
        "Fatty_Acids_and_Lipids": [
            "Omega3_EPA","Omega3_DHA","Omega3_DPA","Krill_Oil","Algal_Oil","GLA","MCT_Oil",
            "Phosphatidylserine","Phosphatidylcholine","Lecithin"
        ],
        "Amino_Acids_and_Precursors": [
            "L_Tryptophan","5_HTP","L_Tyrosine","L_Phenylalanine","Glycine","Taurine","L_Theanine","GABA",
            "L_Glutamine","Arginine","Citrulline","Creatine_Monohydrate","Carnitine","Acetyl_L_Carnitine",
            "Choline_Bitartrate","Alpha_GPC","Citicoline","Serine","Ornithine","Histidine","Methionine","Cysteine"
        ],
        "Polyphenols_and_Phytonutrients": [
            "Curcumin","Resveratrol","Quercetin","EGCG","Anthocyanins","Cocoa_Flavanols","Olive_Leaf_Extract",
            "Grape_Seed_Extract","Pomegranate_Extract","Green_Tea_Extract","Boswellia","Ginger_Extract","Berberine",
            "Silymarin","Hesperidin","Rutin","Luteolin","Apigenin","Sulforaphane_Precursor"
        ],
        "Mitochondrial_and_Energy": [
            "CoQ10","Alpha_Lipoic_Acid","PQQ","NAD_Precursor_Nicotinamide_Riboside","NAD_Precursor_NMN","Ribose","D_Ribose","Malic_Acid"
        ],
        "Herbal_and_Botanical": [
            "Ashwagandha","Rhodiola","Panax_Ginseng","Eleuthero","Bacopa","Saffron_Extract","Lavender_Extract",
            "Passionflower_Extract","Valerian_Extract","Chamomile_Extract","Lemon_Balm","Hops_Extract","Kava_Extract",
            "Holy_Basil","Schisandra","Gotu_Kola","Magnolia_Bark","Ginkgo_Biloba","Lion_s_Mane","Reishi","Cordyceps"
        ],
        "Other_Bioactives": [
            "SAMe","NAC","Inositol","Tart_Cherry_Extract","Theobromine","Betaine","TMG","L_Carnosine",
            "Spirulina","Chlorella","Glucosamine","Chondroitin","Collagen_Peptides","Gelatin","MSM",
            "Lactoferrin","Bromelain","Papain","Propolis"
        ],
        "Fiber_and_Gut_Support": [
            "Psyllium","Inulin","FOS","GOS","PHGG","Resistant_Starch","Pectin","Beta_Glucan","Acacia_Fiber","Guar_Gum",
            "Arabinoxylan","Digestive_Enzymes","Betaine_HCl","Bile_Salts","Zinc_Carnosine","Peppermint_Oil_Enteric","Butyrate_Supplement"
        ],
    }
    for grp, items in supplement_groups.items():
        add_leaves(nut, ["Supplement_and_Nutraceutical_Library", grp], items)

    # Microbiome interventions
    probiotic = {
        "Lactobacillus": ["rhamnosus","reuteri","plantarum","casei","helveticus","acidophilus","gasseri","salivarius","paracasei","fermentum"],
        "Bifidobacterium": ["longum","bifidum","infantis","lactis","adolescentis","breve","animalis"],
        "Saccharomyces": ["boulardii"],
        "Streptococcus": ["thermophilus"],
        "Bacillus": ["coagulans","subtilis"],
        "Lactococcus": ["lactis"],
    }
    for genus, species in probiotic.items():
        add_leaves(nut, ["Microbiome_Interventions", "Probiotics", genus], [f"{genus}_{sp}" for sp in species])

    add_leaves(nut, ["Microbiome_Interventions", "Prebiotics"], [
        "Inulin","FOS","GOS","PHGG","Resistant_Starch","Pectin","Beta_Glucan","Acacia_Fiber","Psyllium","Arabinoxylan","Guar_Gum"
    ])
    add_leaves(nut, ["Microbiome_Interventions", "Fermented_Foods"], [
        "Kefir","Yogurt","Kimchi","Sauerkraut","Miso","Tempeh","Kombucha","Natto","Fermented_Pickles","Kvass","Amazake","Lassi","Fermented_Sourdough"
    ])
    add_leaves(nut, ["Microbiome_Interventions", "Postbiotics"], [
        "Butyrate","Lactate","Propionate","Acetate"
    ])

    add_leaves(nut, ["Medical_Nutrition_Therapy_Tools"], [
        "Registered_Dietitian_Referral",
        "Medical_Nutrition_Therapy_Plan",
        "Enteral_Nutrition_Formula",
        "Oral_Nutrition_Supplement_Shake",
        "Texture_Modified_Diet",
        "Renal_Diet_Framework",
        "Cardiac_Diet_Framework",
        "Glycemic_Control_Medical_Diet",
        "Food_Allergy_Evaluation_Referral",
        "Nutrition_Labs_Review_Referral",
    ])

    BIO["Nutrition_Hydration_and_Metabolic_Health"] = nut

    # ------------------------------------------------------------------
    # 3) Movement, Physical Function, and Somatic Care (BIO-first; high-resolution)
    #     - Focus: modalities, physical capacity domains, tissue care, rehab pathways, recovery tools, monitoring
    #     - Avoid: schedule/frequency/intensity/duration variables as taxonomy nodes
    # ------------------------------------------------------------------
    mov: Dict[str, Any] = {}

    # 3.1 Aerobic / cardiorespiratory modalities (broad coverage)
    add_leaves(mov, ["Aerobic_and_Cardiorespiratory_Movement", "Continuous_Locomotion"], [
        "Walking",
        "Hiking",
        "Trail_Walking",
        "Nordic_Walking",
        "Stair_Climbing",
        "Incline_Treadmill_Walking",
        "Outdoor_Terrain_Rucking_Framework",
        "Active_Commute",
    ])

    add_leaves(mov, ["Aerobic_and_Cardiorespiratory_Movement", "Cycling_and_Wheeled"], [
        "Cycling",
        "Stationary_Bike",
        "E_Bike_Riding",
        "Inline_Skating",
        "Roller_Skating",
    ])

    add_leaves(mov, ["Aerobic_and_Cardiorespiratory_Movement", "Water_Based_Aerobic"], [
        "Swimming",
        "Open_Water_Swimming",
        "Water_Aerobics",
        "Aqua_Jogging",
        "Rowing_On_Water",
        "Kayaking",
        "Canoeing",
        "Paddleboarding",
    ])

    add_leaves(mov, ["Aerobic_and_Cardiorespiratory_Movement", "Machine_and_Studio_Aerobic"], [
        "Rowing_Ergometer",
        "Elliptical",
        "Stair_Climber_Machine",
        "Ski_Ergometer",
        "Aerobics_Class",
        "Spin_Class",
        "Dance_Aerobics",
        "Zumba",
    ])

    add_leaves(mov, ["Aerobic_and_Cardiorespiratory_Movement", "Sports_and_Play_Aerobic"], [
        "Racquet_Sports",
        "Team_Sports",
        "Martial_Arts",
        "Shadowboxing",
        "Dance_Social",
        "Basketball_Recreational",
        "Football_Recreational",
        "Tennis_Recreational",
        "Badminton_Recreational",
    ])

    add_leaves(mov, ["Aerobic_and_Cardiorespiratory_Movement", "Outdoor_Seasonal"], [
        "CrossCountry_Skiing",
        "Snowshoeing",
        "Ice_Skating",
    ])

    add_leaves(mov, ["Aerobic_and_Cardiorespiratory_Movement", "Lifestyle_Energy_Expenditure"], [
        "Gardening",
        "Household_Manual_Tasks",
        "Stair_Use_Preference_Framework",
        "Walking_Meetings_Framework",
    ])

    # 3.2 Strength, power, and resistance modalities
    add_leaves(mov, ["Strength_Power_and_Resistance", "External_Load_Training"], [
        "Free_Weights",
        "Resistance_Machines",
        "Kettlebells",
        "Cable_Training",
        "Medicine_Ball",
        "Sandbag_Training",
        "Sled_Push_Pull",
        "Weighted_Carries_Framework",
    ])

    add_leaves(mov, ["Strength_Power_and_Resistance", "Bodyweight_and_Calisthenics"], [
        "Bodyweight_Training",
        "Calisthenics",
        "Gymnastics_Basics",
        "Pullup_Bar_Training",
        "Dip_Bar_Training",
        "Plyometrics_Framework",
    ])

    add_leaves(mov, ["Strength_Power_and_Resistance", "Elastic_and_Suspension"], [
        "Resistance_Bands",
        "TRX_Suspension",
        "Loop_Band_Hip_Work",
        "Isometric_Band_Training_Framework",
    ])

    add_leaves(mov, ["Strength_Power_and_Resistance", "Technical_Lifting_Families"], [
        "Olympic_Lifting_Technical",
        "Powerlifting_Technical",
        "Strongman_Implements_Framework",
    ])

    add_leaves(mov, ["Strength_Power_and_Resistance", "Climbing_and_Grip_Demanding"], [
        "Climbing_Indoor",
        "Climbing_Outdoor",
        "Bouldering",
        "Hangboard_Training_Framework",
        "Grip_Strength_Tools_Framework",
    ])

    # 3.3 Mobility, flexibility, joint health, and connective tissue care
    add_leaves(mov, ["Mobility_Flexibility_and_Joint_Health", "Mobility_and_Range_of_Motion"], [
        "Mobility_Flow",
        "Dynamic_Stretching",
        "Static_Stretching",
        "PNF_Stretching",
        "Joint_CARS_Routine_Framework",
        "End_Range_Isometrics_Framework",
        "Loaded_Stretching_Framework",
    ])

    add_leaves(mov, ["Mobility_Flexibility_and_Joint_Health", "MindBody_Mobility_Systems"], [
        "Yoga",
        "Pilates",
        "Tai_Chi",
        "Qigong",
        "Feldenkrais",
        "Alexander_Technique",
        "Somatic_Body_Awareness_Practice",
        "Relaxation_Movement_Flow",
    ])

    add_leaves(mov, ["Mobility_Flexibility_and_Joint_Health", "Myofascial_and_Soft_Tissue_Tools"], [
        "Foam_Rolling",
        "Myofascial_Release_Ball",
        "Massage_Gun_Self_Treatment",
        "Stretch_Straps",
        "Trigger_Point_Tool",
        "Compression_Boots_Device",
        "Kinesiology_Tape_Support",
    ])

    # 3.4 Balance, coordination, agility, and proprioception
    add_leaves(mov, ["Balance_Coordination_and_Proprioception", "Static_and_Dynamic_Balance"], [
        "Single_Leg_Stance",
        "Tandem_Walk",
        "Heel_Toe_Walk",
        "Balance_Board",
        "BOSU_Training_Framework",
        "Slackline",
    ])

    add_leaves(mov, ["Balance_Coordination_and_Proprioception", "Agility_and_Motor_Control"], [
        "Agility_Ladder",
        "Cone_Drills",
        "Change_Of_Direction_Drills",
        "Proprioception_Drills",
        "Hand_Eye_Coordination_Drills",
        "Reaction_Time_Drills_Framework",
        "Dual_Task_Motor_Control_Framework",
    ])

    add_leaves(mov, ["Balance_Coordination_and_Proprioception", "Skill_Based_Movement"], [
        "Dance_Balance",
        "Dance_Choreography_Practice",
        "Martial_Arts_Footwork",
        "Ball_Skills_Practice",
    ])

    # 3.5 Rehabilitation, physical therapy pathways, and functional restoration
    add_leaves(mov, ["Rehabilitation_and_Functional_Restoration", "Clinical_Referrals_and_Evaluation"], [
        "Physiotherapy_Referral",
        "Sports_Medicine_Evaluation",
        "Orthopedic_Evaluation_Referral",
        "Rheumatology_Evaluation_Referral",
        "Neurology_Motor_Evaluation_Referral",
        "Podiatry_Gait_Evaluation",
        "Occupational_Therapy_Functional_Training_Referral",
    ])

    add_leaves(mov, ["Rehabilitation_and_Functional_Restoration", "Rehab_Exercise_Families"], [
        "Gait_Retraining_Framework",
        "Return_To_Sport_Framework",
        "Tendon_Load_Management_Framework",
        "Isometric_Pain_Modulation_Framework",
        "Eccentric_Loading_Framework",
        "Scapular_Stabilization_Framework",
        "Core_Stability_Framework",
        "Hip_Stability_Framework",
        "Balance_Rehab_Framework",
        "Neck_And_Upper_Back_Rehab_Framework",
        "Low_Back_Function_Restoration_Framework",
    ])

    add_leaves(mov, ["Rehabilitation_and_Functional_Restoration", "Assistive_and_Support_Devices"], [
        "Orthotics_Evaluation",
        "Ankle_Brace",
        "Knee_Brace",
        "Wrist_Brace",
        "Compression_Garments",
        "Cane_Or_Walking_Pole",
        "Walking_Boot_Immobilizer",
    ])

    # 3.6 Bodywork, manual therapies, and somatic symptom supports
    add_leaves(mov, ["Bodywork_Manual_Therapies_and_Somatic_Supports", "Professional_Bodywork"], [
        "Massage_Therapy",
        "Manual_Therapy",
        "Osteopathic_Manual_Therapy",
        "Chiropractic_Evaluation",
        "Physiotherapy_Manual_Skills",
    ])

    add_leaves(mov, ["Bodywork_Manual_Therapies_and_Somatic_Supports", "Needling_and_Modulation_Techniques"], [
        "Acupuncture",
        "Dry_Needling",
        "Cupping_Therapy",
        "Gua_Sha_Therapy_Framework",
    ])

    add_leaves(mov, ["Bodywork_Manual_Therapies_and_Somatic_Supports", "Electrothermal_and_Topical_Supports"], [
        "TENS_Unit",
        "EMS_Muscle_Stimulation_Device",
        "Heat_Therapy",
        "Cold_Therapy",
        "Contrast_Therapy",
        "Topical_Analgesic_Nonprescription_Class",
        "Topical_Capsaicin_Class",
        "Topical_NSAID_Class",
    ])

    add_leaves(mov, ["Bodywork_Manual_Therapies_and_Somatic_Supports", "Photobiomodulation_and_Light_Local"], [
        "Red_Light_Therapy_Local",
        "Photobiomodulation_Local_Device",
    ])

    add_leaves(mov, ["Bodywork_Manual_Therapies_and_Somatic_Supports", "Thermotherapy_And_Recovery_Environments"], [
        "Sauna_Bathing",
        "Infrared_Sauna",
        "Steam_Room",
        "Cold_Plunge",
        "Cold_Water_Immersion_Framework",
    ])

    # 3.7 Ergonomics, posture, work capacity, and environment-shaping tools
    add_leaves(mov, ["Ergonomics_Posture_and_Work_Capacity", "Workstation_Setup_Tools"], [
        "Standing_Desk",
        "Desk_Ergonomic_Setup",
        "Monitor_Riser",
        "Laptop_Stand",
        "External_Keyboard",
        "Split_Keyboard",
        "Vertical_Mouse",
        "Wrist_Rest_Ergonomic",
        "Document_Holder",
        "Task_Lighting_Lamp",
        "AntiFatigue_Mat",
        "Foot_Rest",
        "Chair_Ergonomic_Adjustment_Framework",
        "Active_Sitting_Chair",
    ])

    add_leaves(mov, ["Ergonomics_Posture_and_Work_Capacity", "Posture_and_Micro_Movement_Aids"], [
        "Lumbar_Support_Cushion",
        "Posture_Reminder_Device",
        "Sitting_Posture_Wedge_Cushion",
        "Standing_Posture_Cue_Tool",
    ])

    add_leaves(mov, ["Ergonomics_Posture_and_Work_Capacity", "Load_Management_and_Carry_Tools"], [
        "Backpack_Ergonomic_Fit",
        "Load_Distribution_Strategy_Framework",
        "Supportive_Footwear_Framework",
    ])

    # 3.8 Respiratory mechanics, autonomic physiology, and cardiorespiratory supports (BIO tools)
    add_leaves(mov, ["Respiratory_and_Autonomic_Physiology_Tools", "Breathing_Mechanics_and_Training"], [
        "Breathing_Retraining",
        "Inspiratory_Muscle_Training_Device",
        "Expiratory_Muscle_Training_Device",
        "Nasal_Breathing_Training",
        "Buteyko_Style_Breathing_Framework",
        "CO2_Tolerance_Training_Framework",
        "Paced_Breathing_Device",
        "Capnography_Feedback_Training_Framework",
    ])

    add_leaves(mov, ["Respiratory_and_Autonomic_Physiology_Tools", "Airway_and_Nasal_Supports"], [
        "Nasal_Dilator",
        "Nasal_Saline_Irrigation",
        "Allergic_Rhinitis_Care_Framework",
        "Mouth_Taping_Risk_Check_Framework",
    ])

    add_leaves(mov, ["Respiratory_and_Autonomic_Physiology_Tools", "Autonomic_Biofeedback"], [
        "HRV_Guided_Breathing_Tool",
        "HRV_Tracker",
        "Respiratory_Biofeedback_Device",
        "Galvanic_Skin_Response_Biofeedback",
        "Thermal_Biofeedback",
    ])

    # 3.9 Injury prevention and tissue integrity (BIO-supportive)
    add_leaves(mov, ["Injury_Risk_Reduction_and_Tissue_Integrity", "Warmup_and_Preparation_Frameworks"], [
        "Movement_Screening_Framework",
        "Warmup_Mobility_Primer_Framework",
        "Prehab_Routine_Framework",
        "Technique_Coaching_Referral",
    ])

    add_leaves(mov, ["Injury_Risk_Reduction_and_Tissue_Integrity", "Foot_Ankle_and_Gait_Health"], [
        "Foot_Strengthening_Framework",
        "Toe_Spacer_Device",
        "Barefoot_Transition_Risk_Check_Framework",
        "Gait_Assessment_Referral",
    ])

    add_leaves(mov, ["Injury_Risk_Reduction_and_Tissue_Integrity", "Bone_and_Connective_Tissue_Support"], [
        "Bone_Density_Scan_Referral",
        "Vitamin_D_Status_Correction_Framework",
        "Calcium_Intake_Optimization_Framework",
        "Collagen_Peptides",
        "Gelatin",
        "Vitamin_C_Collagen_Cofactor_Support",
    ])

    # 3.10 Monitoring, tracking, and performance/health assessment (BIO instrumentation)
    add_leaves(mov, ["Physical_Monitoring_and_Assessment", "Wearables_and_Home_Monitoring"], [
        "Activity_Tracker",
        "Heart_Rate_Monitor",
        "HRV_Tracker",
        "Smart_Scale",
        "Body_Composition_Assessment",
        "Blood_Pressure_Home_Monitor",
        "Pulse_Oximeter",
    ])

    add_leaves(mov, ["Physical_Monitoring_and_Assessment", "Clinical_Tests_and_Imaging_Referrals"], [
        "Body_Composition_Assessment_Referral",
        "Bone_Density_Scan_Referral",
        "Cardiorespiratory_Fitness_Test_Referral",
        "Gait_Analysis_Lab_Referral",
        "Injury_Imaging_Evaluation_Referral",
    ])

    # 3.11 Recovery supports (BIO tools; not schedules)
    add_leaves(mov, ["Recovery_and_Regeneration_Supports", "Compression_and_Circulation_Tools"], [
        "Compression_Garments",
        "Compression_Boots_Device",
        "Leg_Elevation_Device",
    ])

    add_leaves(mov, ["Recovery_and_Regeneration_Supports", "Hydration_and_Electrolytes_For_Recovery"], [
        "Electrolyte_Mix",
        "Oral_Rehydration_Salts",
        "Magnesium_Glycinate",
        "Tart_Cherry_Extract",
    ])

    BIO["Movement_Physical_Function_and_Somatic_Care"] = mov


    # ------------------------------------------------------------------
    # 4) Cognitive Capacity and Brain Health (BIO-first; brain performance, resilience, protection)
    #     - Focus: physiology, substrates, neuroprotection, biomarkers, and evidence-aligned supplements/nootropics
    #     - Avoid: psychotherapy/behavioral skills framing (keep intervention entities BIO-leaning)
    #     - No schedule/frequency/duration parameters as taxonomy nodes
    # ------------------------------------------------------------------
    cog: Dict[str, Any] = {}

    # 4.1 Neuroenergetics, mitochondrial support, and metabolic brain health
    add_leaves(cog, ["Neuroenergetics_and_Metabolic_Support", "Glucose_Insulin_and_Metabolic_Control"], [
        "Glycemic_Variability_Reduction_Framework",
        "Insulin_Sensitivity_Optimization_Framework",
        "Continuous_Glucose_Monitoring_For_Cognition",
        "Metabolic_Syndrome_Risk_Reduction_Pathway",
        "Brain_Fuel_Flexibility_Support_Framework",
    ])

    add_leaves(cog, ["Neuroenergetics_and_Metabolic_Support", "Mitochondrial_and_Cell_Energy_Nutraceuticals"], [
        "Creatine_Monohydrate",
        "CoQ10_Ubiquinone",
        "CoQ10_Ubiquinol",
        "Alpha_Lipoic_Acid",
        "Acetyl_L_Carnitine",
        "L_Carnitine_L_Tartrate",
        "PQQ",
        "Ribose",
        "Malic_Acid",
        "NAD_Precursor_Nicotinamide_Riboside",
        "NAD_Precursor_NMN",
    ])

    add_leaves(cog, ["Neuroenergetics_and_Metabolic_Support", "Ketone_and_Alternative_Substrate_Support"], [
        "MCT_Oil",
        "Exogenous_Ketone_Salts",
        "Exogenous_Ketone_Esters",
        "Ketogenic_Diet_Medical_Framework",
        "Modified_Atkins_Diet_Framework",
    ])

    # 4.2 Structural substrates: lipids, membranes, myelin support
    add_leaves(cog, ["Neural_Structure_Membranes_and_Myelination", "Omega3_and_Lipid_Support"], [
        "Omega3_EPA",
        "Omega3_DHA",
        "Omega3_DPA",
        "Krill_Oil",
        "Algal_Oil",
        "Phosphatidylserine",
        "Phosphatidylcholine",
        "Lecithin",
    ])

    add_leaves(cog, ["Neural_Structure_Membranes_and_Myelination", "Cholinergic_Precursors_and_Membrane_Pools"], [
        "Citicoline_CDP_Choline",
        "Alpha_GPC",
        "Choline_Bitartrate",
        "Uridine_Monophosphate",
        "Acetylcholine_Precursor_Stack_Framework",
    ])

    # 4.3 Neuroinflammation, oxidative stress, and neuroprotection
    add_leaves(cog, ["Neuroprotection_Inflammation_and_Oxidative_Stress", "Polyphenols_and_Antioxidant_Bioactives"], [
        "Curcumin",
        "Curcumin_Piperine_Combination",
        "Resveratrol",
        "Quercetin",
        "EGCG_Green_Tea_Extract",
        "Anthocyanins",
        "Cocoa_Flavanols",
        "Olive_Leaf_Extract",
        "Grape_Seed_Extract",
        "Pomegranate_Extract",
        "Sulforaphane_Precursor",
        "Boswellia",
        "Ginger_Extract",
        "Silymarin",
        "Hesperidin",
        "Rutin",
        "Luteolin",
        "Apigenin",
    ])

    add_leaves(cog, ["Neuroprotection_Inflammation_and_Oxidative_Stress", "Glutathione_and_Redox_Support"], [
        "NAC",
        "Glycine",
        "GlyNAC_Stack_Framework",
        "Glutathione_Reduced",
        "Vitamin_C",
        "Vitamin_E",
        "Selenium",
        "Zinc",
    ])

    add_leaves(cog, ["Neuroprotection_Inflammation_and_Oxidative_Stress", "Immune_and_MastCell_Histamine_Modulation_Support"], [
        "Low_Histamine_Diet_Framework",
        "DAO_Enzyme_Supplement",
        "Quercetin_MastCell_Modulation",
        "Vitamin_C_Histamine_Modulation",
    ])

    # 4.4 Neurotransmitter systems (BIO substrates + cofactors; avoid disorder framing)
    add_leaves(cog, ["Neurotransmitter_and_Neuromodulator_Support", "Catecholamine_Precursors_and_Cofactors"], [
        "L_Tyrosine",
        "L_Phenylalanine",
        "Iron_Status_Optimization_Framework",
        "Vitamin_B6_P5P",
        "Vitamin_C_Cofactor_Support",
        "Copper_Status_Check_Framework",
    ])

    add_leaves(cog, ["Neurotransmitter_and_Neuromodulator_Support", "Serotonin_Melatonin_Precursors_and_Cofactors"], [
        "L_Tryptophan",
        "5_HTP",
        "Vitamin_B6_P5P",
        "Magnesium_Glycinate",
        "Folate_L_Methylfolate",
        "Vitamin_B12_Methylcobalamin",
    ])

    add_leaves(cog, ["Neurotransmitter_and_Neuromodulator_Support", "Glutamate_GABA_Balance_Support"], [
        "L_Theanine",
        "Taurine",
        "Magnesium_Threonate",
        "Magnesium_Glycinate",
        "GABA",
    ])

    # 4.5 Sleep/circadian biology as a cognitive substrate (BIO-leaning components only)
    add_leaves(cog, ["Sleep_Circadian_and_Glymphatic_Support", "Circadian_Photo_Biology_Tools"], [
        "Bright_Light_Therapy_Box",
        "Dawn_Simulation_Device",
        "Dusk_Simulation_Lighting_System",
        "Blue_Light_Filter_Glasses",
        "Smart_Circadian_Lighting_System",
    ])

    add_leaves(cog, ["Sleep_Circadian_and_Glymphatic_Support", "Sleep_Support_Nutraceuticals"], [
        "Melatonin",
        "Glycine",
        "Magnesium_Glycinate",
        "Magnesium_Threonate",
        "L_Theanine",
        "Tart_Cherry_Extract",
        "Apigenin",
        "Lavender_Extract",
        "Chamomile_Extract",
        "Valerian_Extract",
        "Passionflower_Extract",
    ])

    # 4.6 Vascular brain health and oxygenation
    add_leaves(cog, ["Cerebrovascular_and_Oxygenation_Support", "Vascular_Risk_Optimization_Frameworks"], [
        "Blood_Pressure_Optimization_Framework",
        "Lipid_Profile_Optimization_Framework",
        "Homocysteine_Reduction_Framework",
        "Endothelial_Function_Support_Framework",
        "Sleep_Apnea_Risk_Evaluation_Referral",
    ])

    add_leaves(cog, ["Cerebrovascular_and_Oxygenation_Support", "Vascular_and_Oxygenation_Support_Nutraceuticals"], [
        "Omega3_EPA",
        "Omega3_DHA",
        "Beetroot_Nitrate_Extract",
        "L_Citrulline",
        "L_Arginine",
        "CoQ10_Ubiquinol",
        "Magnesium",
        "Potassium_Citrate",
    ])

    # 4.7 Microbiome–gut–brain axis (bio interventions + substrates)
    add_leaves(cog, ["Microbiome_GutBrain_Axis_Support", "Probiotics"], [
        "Lactobacillus_rhamnosus",
        "Lactobacillus_helveticus",
        "Lactobacillus_reuteri",
        "Bifidobacterium_longum",
        "Bifidobacterium_bifidum",
        "Bifidobacterium_lactis",
        "Bifidobacterium_breve",
        "Saccharomyces_boulardii",
    ])

    add_leaves(cog, ["Microbiome_GutBrain_Axis_Support", "Prebiotics_and_Fiber"], [
        "Psyllium",
        "Inulin",
        "FOS",
        "GOS",
        "PHGG",
        "Resistant_Starch",
        "Beta_Glucan",
        "Acacia_Fiber",
        "Pectin",
    ])

    add_leaves(cog, ["Microbiome_GutBrain_Axis_Support", "Postbiotics_and_Metabolites"], [
        "Butyrate_Supplement",
        "Sodium_Butyrate",
        "Tributyrin",
    ])

    # 4.8 Minerals, vitamins, and foundational labs tied to cognition (BIO assessment + correction paths)
    add_leaves(cog, ["Micronutrient_Status_and_Correction", "Core_Cognition_Relevant_Labs"], [
        "Vitamin_B12_Test",
        "Folate_Test",
        "Vitamin_D_Test",
        "Iron_Studies",
        "Ferritin_Test",
        "Thyroid_Panel",
        "HbA1c_Test",
        "Fasting_Glucose_Test",
        "Lipid_Panel",
        "CRP_Test",
        "Homocysteine_Test",
        "Omega3_Index_Test",
        "Magnesium_Status_Assessment",
    ])

    add_leaves(cog, ["Micronutrient_Status_and_Correction", "Core_Cognition_Relevant_Repletion_Entities"], [
        "Vitamin_B12_Methylcobalamin",
        "Vitamin_B12_Hydroxocobalamin",
        "Folate_L_Methylfolate",
        "Vitamin_D3",
        "Iron_Repletion_Framework",
        "Iodine_Status_Correction_Framework",
        "Magnesium_Threonate",
        "Magnesium_Glycinate",
        "Zinc_Picolinate",
        "Selenium_Selenomethionine",
    ])

    # 4.9 Nootropics (structured; split by domain; include stacks as frameworks)
    # NOTE: still “candidate entities”; not endorsements, and many require clinician oversight.
    add_leaves(cog, ["Nootropics_and_Cognitive_Enhancers", "Methylxanthines_and_Wakefulness_Modulators"], [
        "Caffeine",
        "Caffeine_L_Theanine_Stack_Framework",
        "Theobromine",
        "Tea_Catechin_Caffeine_Pattern_Framework",
    ])

    add_leaves(cog, ["Nootropics_and_Cognitive_Enhancers", "Cholinergic_and_Acetylcholine_Modulators"], [
        "Citicoline_CDP_Choline",
        "Alpha_GPC",
        "Choline_Bitartrate",
        "Acetylcholinesterase_Inhibitor_Class_Medical",
        "Cholinergic_Stack_Framework",
    ])

    add_leaves(cog, ["Nootropics_and_Cognitive_Enhancers", "Racetams_and_Related_Classes"], [
        "Racetam_Class_Framework",
        "Ampakine_Class_Research_Framework",
    ])

    add_leaves(cog, ["Nootropics_and_Cognitive_Enhancers", "Adaptogens_and_Herbal_Cognition_Support"], [
        "Bacopa",
        "Rhodiola",
        "Panax_Ginseng",
        "Eleuthero",
        "Gotu_Kola",
        "Ginkgo_Biloba",
        "Saffron_Extract",
        "Ashwagandha",
        "Holy_Basil",
    ])

    add_leaves(cog, ["Nootropics_and_Cognitive_Enhancers", "Mushrooms_and_Neurotrophic_Support"], [
        "Lion_s_Mane",
        "Reishi",
        "Cordyceps",
    ])

    add_leaves(cog, ["Nootropics_and_Cognitive_Enhancers", "Amino_Acids_and_Neurochemical_Precursors"], [
        "L_Tyrosine",
        "L_Phenylalanine",
        "L_Tryptophan",
        "5_HTP",
        "Creatine_Monohydrate",
        "Acetyl_L_Carnitine",
        "Taurine",
        "L_Theanine",
    ])

    add_leaves(cog, ["Nootropics_and_Cognitive_Enhancers", "Peptides_and_Advanced_Bioactives_Research"], [
        "Peptide_Nootropic_Research_Framework",
        "Intranasal_Peptide_Delivery_Research_Framework",
    ])

    add_leaves(cog, ["Nootropics_and_Cognitive_Enhancers", "Medication_Classes_With_Cognitive_Effects_Review"], [
        "Medication_Review_For_Cognitive_Load_Referral",
        "Anticholinergic_Burden_Review_Framework",
        "Sedative_Load_Review_Framework",
        "Polypharmacy_Rationalization_Referral",
    ])

    # 4.10 Neurodegeneration-risk and longevity-oriented neurohealth supports (BIO)
    add_leaves(cog, ["Neurodegeneration_Risk_Modifiers_and_Longevity_Neurohealth", "Screening_and_Referrals"], [
        "Neuropsychological_Assessment_Referral",
        "Cognitive_Neurology_Consultation_Referral",
        "Sleep_Clinic_Evaluation_Referral",
        "Hearing_Evaluation_Referral",
        "Vision_Evaluation_Referral",
        "Cardiometabolic_Risk_Clinic_Referral",
    ])

    add_leaves(cog, ["Neurodegeneration_Risk_Modifiers_and_Longevity_Neurohealth", "Genetics_and_Biomarkers"], [
        "APOE_Genotyping",
        "Pharmacogenomics_Panel",
        "Inflammation_Biomarker_Panel_Framework",
        "Neurodegeneration_Biomarker_Research_Framework",
    ])

    # 4.11 Sensory integrity as a brain-health protector (BIO supports)
    add_leaves(cog, ["Sensory_Integrity_and_Access_Supports", "Vision_and_Oculomotor"], [
        "Vision_Correction",
        "Refraction_Assessment_Referral",
        "Dry_Eye_Treatment_Framework",
        "Blue_Light_Exposure_Management_Tools",
    ])

    add_leaves(cog, ["Sensory_Integrity_and_Access_Supports", "Hearing_and_Auditory"], [
        "Audiology_Assessment_Referral",
        "Hearing_Aid_Evaluation",
        "Cerumen_Management_Framework",
        "Noise_Exposure_Reduction_Tools",
    ])

    # 4.12 Neurotoxin and exposure minimization (brain-health oriented; still BIO)
    add_leaves(cog, ["Neurotoxin_and_Exposure_Risk_Reduction", "Heavy_Metals_and_Environmental_Toxins"], [
        "Heavy_Metal_Screen",
        "Lead_Exposure_Risk_Assessment_Framework",
        "Solvent_Exposure_Reduction",
        "Pesticide_Exposure_Reduction",
        "Air_Quality_Monitor",
        "HEPA_Air_Purification",
    ])

    add_leaves(cog, ["Neurotoxin_and_Exposure_Risk_Reduction", "Sleep_Disordered_Breathing_and_Hypoxia_Risk"], [
        "Home_Sleep_Testing_Referral",
        "Polysomnography_Referral",
        "Pulse_Oximeter",
    ])

    BIO["Cognitive_Capacity_and_Brain_Health"] = cog


    # ------------------------------------------------------------------
    # 5) Neuromodulation, Biofeedback, and Neurostimulation
    # ------------------------------------------------------------------
    neu: Dict[str, Any] = {}

    # ------------------------
    # A) Transcranial Magnetic Stimulation (TMS) family
    # ------------------------
    add_leaves(neu, ["Transcranial_Magnetic_Stimulation", "Approach_Families"], [
        "rTMS_Standard_Framework",
        "rTMS_Patterned_Framework",
        "Theta_Burst_Stimulation_Framework",
        "Deep_TMS_Coil_Family",
        "Synchronized_TMS_EEG_Informed_Framework",
        "Paired_Associative_Stimulation_TMS_Framework",
        "Priming_TMS_Framework",
        "Sequential_Targeting_TMS_Framework",
        "State_Dependent_TMS_Framework",
        "Accelerated_TMS_Framework",
        "Maintenance_TMS_Framework",
        "Single_Pulse_TMS_Diagnostic_Use_Framework",
        "Repetitive_Single_Site_TMS_Framework",
        "Multi_Site_TMS_Framework",
    ])

    add_leaves(neu, ["Transcranial_Magnetic_Stimulation", "Stimulation_Pattern_Options"], [
        "Intermittent_Theta_Burst_Option",
        "Continuous_Theta_Burst_Option",
        "Quadripulse_Stimulation_Option",
        "Paired_Pulse_Facilitation_Option",
        "Paired_Pulse_Inhibition_Option",
        "Bilateral_Alternating_Stimulation_Option",
        "Sequential_Stimulation_Option",
        "Burst_Modulated_Stimulation_Option",
    ])

    add_leaves(neu, ["Transcranial_Magnetic_Stimulation", "Coil_and_Hardware_Types"], [
        "Figure8_Coil_Class",
        "Double_Cone_Coil_Class",
        "H_Coil_Class",
        "Cooled_Coil_System_Class",
        "Small_Focal_Coil_Class",
        "Wide_Field_Coil_Class",
        "Robot_Assisted_Coil_Positioning_Class",
    ])

    add_leaves(neu, ["Transcranial_Magnetic_Stimulation", "Targeting_and_Navigation"], [
        "Scalp_Based_Targeting_Heuristic",
        "MRI_Neuronavigation_Guided_Targeting",
        "Functional_Targeting_fMRI_Informed",
        "Functional_Targeting_EEG_Informed",
        "Functional_Targeting_fNIRS_Informed",
        "Connectivity_Guided_Targeting_Framework",
        "Electric_Field_Modeling_Informed_Targeting",
        "Motor_Threshold_Calibration_Framework",
        "Physiology_Guided_Target_Adjustment",
        "Coil_Position_Tracking_Feedback",
    ])

    add_leaves(neu, ["Transcranial_Magnetic_Stimulation", "Common_Target_Regions"], [
        "DLPFC_Left",
        "DLPFC_Right",
        "DLPFC_Bilateral",
        "Dorsomedial_Prefrontal",
        "Ventromedial_Prefrontal",
        "Frontopolar_Cortex",
        "Orbitofrontal_Cortex",
        "Anterior_Cingulate",
        "Posterior_Cingulate",
        "Supplementary_Motor_Area",
        "Primary_Motor_Cortex",
        "Primary_Somatosensory_Cortex",
        "Inferior_Parietal_Lobule",
        "Temporoparietal_Junction",
        "Superior_Temporal_Gyrus",
        "Auditory_Cortex",
        "Visual_Cortex_Occipital",
        "Cerebellar_Targeting",
        "Insular_Targeting",
    ])

    add_leaves(neu, ["Transcranial_Magnetic_Stimulation", "Adjunct_and_Combination_Frameworks"], [
        "TMS_Plus_Cognitive_Training_Framework",
        "TMS_Plus_Exposure_Based_Learning_Framework",
        "TMS_Plus_Psychotherapy_Session_Framework",
        "TMS_Plus_Mindfulness_Practice_Framework",
        "TMS_Plus_Sleep_Optimization_Framework",
        "TMS_Plus_Motor_Rehabilitation_Framework",
        "TMS_Plus_Pain_Rehabilitation_Framework",
        "TMS_Plus_Neurofeedback_Framework",
    ])

    # ------------------------
    # B) Transcranial Electrical Stimulation (tES) family
    # ------------------------
    add_leaves(neu, ["Transcranial_Electrical_Stimulation", "Approach_Families"], [
        "tDCS_Conventional_Framework",
        "tDCS_HighDefinition_Framework",
        "tDCS_Remote_Supervision_Framework",
        "tACS_Conventional_Framework",
        "tACS_HighDefinition_Framework",
        "tRNS_Framework",
        "tPCS_Pulsed_Current_Stimulation_Framework",
        "Slow_Oscillation_Stimulation_Framework",
        "Cranial_Electrotherapy_Stimulation_Framework",
        "Transcranial_Microcurrent_Stimulation_Framework",
    ])

    add_leaves(neu, ["Transcranial_Electrical_Stimulation", "Electrode_and_Hardware_Types"], [
        "Sponge_Electrode_Class",
        "Gel_Electrode_Class",
        "HD_Ring_Electrode_Array_Class",
        "Multi_Channel_Stimulation_Device_Class",
        "Wearable_tES_Device_Class",
        "Cap_Based_Electrode_System_Class",
    ])

    add_leaves(neu, ["Transcranial_Electrical_Stimulation", "Common_Montage_Frameworks"], [
        "Bifrontal_Montage_Framework",
        "Frontoparietal_Montage_Framework",
        "Frontotemporal_Montage_Framework",
        "Frontopolar_Montage_Framework",
        "Motor_Cortex_Montage_Framework",
        "Premotor_Cortex_Montage_Framework",
        "Temporoparietal_Montage_Framework",
        "Parietal_Midline_Montage_Framework",
        "Occipital_Montage_Framework",
        "Cerebellar_Montage_Framework",
        "Extracephalic_Return_Montage_Framework",
        "Targeted_Return_Electrode_Framework",
    ])

    add_leaves(neu, ["Transcranial_Electrical_Stimulation", "Target_Regions"], [
        "Prefrontal_Left_Targeting",
        "Prefrontal_Right_Targeting",
        "Bilateral_Prefrontal_Targeting",
        "Dorsomedial_Prefrontal_Targeting",
        "Frontopolar_Targeting",
        "Motor_Cortex_Targeting",
        "Temporoparietal_Targeting",
        "Parietal_Targeting",
        "Occipital_Targeting",
        "Cerebellar_Targeting",
    ])

    add_leaves(neu, ["Transcranial_Electrical_Stimulation", "Personalization_and_QA_Tools"], [
        "Impedance_Check_Framework",
        "Skin_Tolerance_Protocol_Framework",
        "Electrode_Placement_Template_System",
        "MRI_Based_Current_Flow_Modeling",
        "Electric_Field_Modeling_Informed_tES",
        "Home_Use_Safety_Checklist",
        "Remote_Adherence_and_CheckIn_Framework",
        "Adverse_Event_Monitoring_Log",
    ])

    # ------------------------
    # C) Ultrasound & Mechanical neuromodulation
    # ------------------------
    add_leaves(neu, ["Ultrasound_and_Mechanical_Neuromodulation", "Approach_Families"], [
        "Focused_Ultrasound_Neuromodulation_Framework",
        "Transcranial_Focused_Ultrasound_Framework",
        "Low_Energy_Pulsed_Ultrasound_Framework",
        "Transcranial_Pulse_Stimulation_Framework",
        "Blood_Brain_Barrier_Modulation_Ultrasound_Framework",
        "Vibrotactile_Neuromodulation_Framework",
    ])

    add_leaves(neu, ["Ultrasound_and_Mechanical_Neuromodulation", "Target_Frameworks"], [
        "Cortical_Targeting_Framework",
        "Subcortical_Targeting_Framework",
        "Thalamic_Targeting_Framework",
        "Basal_Ganglia_Targeting_Framework",
        "Limbic_Targeting_Framework",
        "Cerebellar_Targeting_Framework",
    ])

    add_leaves(neu, ["Ultrasound_and_Mechanical_Neuromodulation", "Guidance_and_Safety"], [
        "Imaging_Guided_Targeting_Framework",
        "Acoustic_Modeling_Informed_Targeting",
        "Skull_Anatomy_Assessment_Framework",
        "Neurophysiology_Monitoring_Framework",
        "Adverse_Event_Monitoring_Log",
    ])

    # ------------------------
    # D) Peripheral neuromodulation (non-implantable)
    # ------------------------
    add_leaves(neu, ["Peripheral_Neuromodulation", "NonImplantable_Device_Families"], [
        "Auricular_Vagus_Nerve_Stimulation_Transcutaneous",
        "Cervical_Vagus_Nerve_Stimulation_Noninvasive",
        "Trigeminal_Nerve_Stimulation_Transcutaneous",
        "Supraorbital_Trigeminal_Stimulation_Transcutaneous",
        "Occipital_Nerve_Stimulation_Transcutaneous",
        "Median_Nerve_Stimulation_Transcutaneous",
        "Tibial_Nerve_Stimulation_Transcutaneous",
        "Transcutaneous_Spinal_Stimulation",
        "Transcutaneous_Peripheral_Nerve_Field_Stimulation",
        "Phrenic_Nerve_External_Stimulation_Framework",
        "Vestibular_Nerve_Stimulation_Transcutaneous",
    ])

    add_leaves(neu, ["Peripheral_Neuromodulation", "Target_Sites"], [
        "Auricular_Concha_Targeting",
        "Auricular_Tragus_Targeting",
        "Cervical_Vagus_Targeting",
        "Supraorbital_Targeting",
        "Infraorbital_Targeting",
        "Occipital_Targeting",
        "Median_Wrist_Targeting",
        "Posterior_Tibial_Targeting",
        "Spinal_Paraspinal_Targeting",
    ])

    add_leaves(neu, ["Peripheral_Neuromodulation", "Combination_Frameworks"], [
        "Peripheral_Stimulation_Plus_Breathwork_Framework",
        "Peripheral_Stimulation_Plus_HRV_Biofeedback_Framework",
        "Peripheral_Stimulation_Plus_Sleep_Optimization_Framework",
        "Peripheral_Stimulation_Plus_Pain_Rehabilitation_Framework",
        "Peripheral_Stimulation_Plus_Mindfulness_Framework",
    ])

    # ------------------------
    # E) Implantable or surgical-evaluation neuromodulation
    # ------------------------
    add_leaves(neu, ["Implantable_or_Surgical_Neuromodulation", "Peripheral_Implantables_and_Evaluations"], [
        "Cervical_Vagus_Nerve_Stimulation_Implant_Evaluation",
        "Occipital_Nerve_Stimulation_Implant_Evaluation",
        "Hypoglossal_Nerve_Stimulation_Evaluation",
        "Sacral_Nerve_Stimulation_Evaluation",
        "Phrenic_Nerve_Stimulation_Evaluation",
        "Tibial_Nerve_Stimulation_Implant_Evaluation",
    ])

    # ============================================================
    # BIO :: Neuromodulation (Implantable / Surgical) — expanded, high-resolution
    # Focus: central implantables, especially DBS, with deeper hierarchical layers.
    # (No disorder names as category labels; leaves are actionable solution variables.)
    # ============================================================

    # --- Central implantables: general pathways (shared prerequisites / governance) ---
    add_leaves(neu,
               ["Implantable_or_Surgical_Neuromodulation", "Central_Implantables_and_Evaluations", "General_Pathways"],
               [
                   "Stereotactic_Neurosurgery_Consultation_Pathway",
                   "Multidisciplinary_Neuromodulation_Case_Conference",
                   "Treatment_History_And_Refractoriness_Review",
                   "Medication_Device_Interaction_Review",
                   "Neuropsychological_Baseline_Assessment_Pathway",
                   "Neuropsychiatric_Risk_Screening_Pathway",
                   "Capacity_And_Informed_Consent_Assessment_Pathway",
                   "Ethics_Review_And_Governance_Pathway",
                   "Shared_Decision_Making_For_Implantable_Neuromodulation",
                   "Caregiver_And_Support_System_Education_For_Implantables",
                   "MRI_And_Device_Safety_Clearance_Pathway",
               ])

    # ============================================================
    # Deep Brain Stimulation (DBS) — comprehensive sub-structure
    # ============================================================

    # --- DBS: candidacy and preoperative evaluation ---
    add_leaves(neu, ["Implantable_or_Surgical_Neuromodulation", "Central_Implantables_and_Evaluations",
                     "Deep_Brain_Stimulation", "Candidacy_and_Preoperative_Evaluation"], [
                   "Deep_Brain_Stimulation_Evaluation",
                   "DBS_Candidacy_Screening_Pathway",
                   "Symptom_Domain_Target_Map_Development",
                   "Functional_Impairment_And_Goal_Setting_For_DBS",
                   "Baseline_Quality_Of_Life_And_Patient_Reported_Outcomes_Capture",
                   "Neuroimaging_Workup_For_Target_Planning",
                   "Structural_MRI_Review_For_Targeting_Constraints",
                   "Vascular_Risk_Assessment_For_Stereotactic_Trajectory",
                   "Cognitive_Risk_Assessment_For_Implantable_Neuromodulation",
                   "Affective_Instability_Risk_Assessment_For_Implantable_Neuromodulation",
                   "Impulse_Control_Risk_Assessment_For_Implantable_Neuromodulation",
                   "Sleep_And_Arousal_Profile_Assessment_For_DBS_Planning",
                   "Substance_Use_Risk_Assessment_For_Implantable_Neuromodulation",
                   "Anesthesia_And_Periprocedural_Medical_Clearance",
               ])

    # --- DBS: targeting and surgical planning ---
    add_leaves(neu, ["Implantable_or_Surgical_Neuromodulation", "Central_Implantables_and_Evaluations",
                     "Deep_Brain_Stimulation", "Targeting_and_Surgical_Planning"], [
                   "DBS_Target_Selection_Conference",
                   "Tractography_Informed_Targeting_Framework",
                   "Connectomic_Targeting_Framework",
                   "Stereotactic_Trajectory_Planning_Service",
                   "Hemorrhage_Risk_Minimization_Trajectory_Planning",
                   "Lead_Type_Selection_Pathway",
                   "Directional_Lead_Planning_Pathway",
                   "Unilateral_Versus_Bilateral_Implant_Planning",
                   "Awake_Versus_Asleep_DBS_Approach_Selection",
                   "Intraoperative_Testing_Plan",
                   "Microelectrode_Recording_Consideration_Pathway",
               ])

    # --- DBS: implantation and perioperative care ---
    add_leaves(neu, ["Implantable_or_Surgical_Neuromodulation", "Central_Implantables_and_Evaluations",
                     "Deep_Brain_Stimulation", "Implantation_and_Periprocedural_Care"], [
                   "DBS_Implantation_Pathway",
                   "Pulse_Generator_Implantation_Pathway",
                   "Perioperative_Infection_Prevention_Protocol_For_Implants",
                   "Perioperative_Delirium_Risk_Mitigation_Plan",
                   "Postoperative_Wound_And_Device_Site_Care_Pathway",
                   "Postoperative_Imaging_Verification_Of_Lead_Position",
                   "Hardware_Integrity_Check_Pathway_PostImplant",
               ])

    # --- DBS: programming and optimization (avoid explicit dose/frequency tokens) ---
    add_leaves(neu, ["Implantable_or_Surgical_Neuromodulation", "Central_Implantables_and_Evaluations",
                     "Deep_Brain_Stimulation", "Programming_and_Optimization"], [
                   "Initial_DBS_Programming_Pathway",
                   "DBS_Parameter_Optimization_Service",
                   "Directional_Programming_Optimization",
                   "Contact_Selection_And_Field_Shaping_Optimization",
                   "Side_Effect_Mitigation_Programming_Strategy",
                   "Stimulation_Induced_Mood_Change_Management_Protocol",
                   "Stimulation_Induced_Cognitive_Effect_Management_Protocol",
                   "Stimulation_Induced_Speech_Or_Motor_Effect_Management_Protocol",
                   "Goal_Attainment_Oriented_Programming_Framework",
                   "Device_Interrogation_And_Telemetry_Review",
                   "Clinician_Patient_Programming_Education_Pathway",
               ])

    # --- DBS: advanced control (closed-loop / adaptive / sensing) ---
    add_leaves(neu, ["Implantable_or_Surgical_Neuromodulation", "Central_Implantables_and_Evaluations",
                     "Deep_Brain_Stimulation", "Advanced_Control_Frameworks"], [
                   "Closed_Loop_DBS_Framework",
                   "Adaptive_DBS_Framework",
                   "Sensing_Informed_Programming_Framework",
                   "Biomarker_Candidate_Selection_For_Adaptive_DBS",
                   "State_Detection_And_Control_Policy_Design_Framework",
                   "Artifact_And_Signal_Quality_Management_For_Sensing_DBS",
                   "Patient_State_Annotation_For_ClosedLoop_Tuning",
               ])

    # --- DBS: follow-up, maintenance, and lifecycle management ---
    add_leaves(neu, ["Implantable_or_Surgical_Neuromodulation", "Central_Implantables_and_Evaluations",
                     "Deep_Brain_Stimulation", "Followup_Maintenance_and_Lifecycle"], [
                   "DBS_Followup_Care_Pathway",
                   "Battery_Status_Monitoring_And_Replacement_Planning",
                   "Rechargeable_Pulse_Generator_Training_Pathway",
                   "Hardware_Integrity_Troubleshooting_Pathway",
                   "Lead_Impedance_And_Connection_Check_Pathway",
                   "MRI_With_Implant_Safety_Protocol",
                   "Medical_Procedure_Compatibility_Clearance_For_Implants",
                   "Device_Replacement_And_Upgrade_Consultation",
               ])

    # --- DBS: complications, revision, explant, and safety management ---
    add_leaves(neu, ["Implantable_or_Surgical_Neuromodulation", "Central_Implantables_and_Evaluations",
                     "Deep_Brain_Stimulation", "Complications_Revision_and_Safety"], [
                   "DBS_Complication_Triage_Pathway",
                   "Suspected_Infection_Evaluation_And_Management_Pathway_For_Implants",
                   "Lead_Migration_Or_Malposition_Evaluation_Pathway",
                   "Hardware_Erosion_Or_Skin_Breakdown_Management_Pathway",
                   "Neurological_Adverse_Event_Evaluation_Pathway_PostDBS",
                   "Neuropsychiatric_Adverse_Effect_Evaluation_Pathway_PostDBS",
                   "Stimulation_Intolerance_Management_Pathway",
                   "DBS_Lead_Revision_Consultation_Pathway",
                   "DBS_Hardware_Revision_Consultation_Pathway",
                   "DBS_Explant_Consultation_Pathway",
                   "DBS_Reimplantation_Evaluation_Pathway",
               ])

    # --- DBS: adjunctive integration supports (kept procedure-adjacent; not duplicating domains 1–9) ---
    add_leaves(neu, ["Implantable_or_Surgical_Neuromodulation", "Central_Implantables_and_Evaluations",
                     "Deep_Brain_Stimulation", "Adjunctive_Integration_Supports"], [
                   "Symptom_Tracking_For_DBS_Response_Attribution",
                   "Functional_Goal_Coaching_PostImplant",
                   "Medication_Adjustment_Coordination_For_DBS_Optimization",
                   "Expectation_Management_And_Response_Timeline_Psychoeducation_For_DBS",
                   "Caregiver_Coaching_For_PostImplant_Changes",
               ])

    # ============================================================
    # Responsive / Closed-loop Neurostimulation (central)
    # ============================================================

    add_leaves(neu, ["Implantable_or_Surgical_Neuromodulation", "Central_Implantables_and_Evaluations",
                     "Responsive_Neurostimulation", "Candidacy_and_Evaluation"], [
                   "Responsive_Neurostimulation_Evaluation",
                   "Closed_Loop_Neurostimulation_Candidacy_Screening",
                   "Signal_Source_And_Sensing_Target_Evaluation",
                   "Symptom_State_And_Biomarker_Assessment_Framework",
                   "Implant_Site_Selection_Conference_For_Responsive_Stimulation",
               ])

    add_leaves(neu, ["Implantable_or_Surgical_Neuromodulation", "Central_Implantables_and_Evaluations",
                     "Responsive_Neurostimulation", "Programming_and_Control"], [
                   "Detection_Parameter_Optimization_Service",
                   "Stimulation_Response_Optimization_Service",
                   "Artifact_Management_For_Sensing_Devices",
                   "Patient_State_Annotation_For_ClosedLoop_Tuning",
               ])

    add_leaves(neu, ["Implantable_or_Surgical_Neuromodulation", "Central_Implantables_and_Evaluations",
                     "Responsive_Neurostimulation", "Maintenance_and_Safety"], [
                   "Device_Interrogation_And_Log_Review",
                   "Hardware_Integrity_Troubleshooting_Pathway",
                   "Revision_Or_Explant_Consultation_Pathway_For_Responsive_Devices",
               ])

    # ============================================================
    # Cortical implant neurostimulation (central)
    # ============================================================

    add_leaves(neu, ["Implantable_or_Surgical_Neuromodulation", "Central_Implantables_and_Evaluations",
                     "Cortical_Implant_Neurostimulation", "Candidacy_and_Evaluation"], [
                   "Cortical_Implant_Neurostimulation_Evaluation",
                   "Cortical_Target_Localization_Workup",
                   "Functional_Mapping_Informed_Implant_Planning",
                   "Cortical_Safety_And_Seizure_Risk_Assessment_For_Implants",
               ])

    add_leaves(neu, ["Implantable_or_Surgical_Neuromodulation", "Central_Implantables_and_Evaluations",
                     "Cortical_Implant_Neurostimulation", "Programming_and_Optimization"], [
                   "Cortical_Stimulation_Parameter_Optimization_Service",
                   "Side_Effect_Mitigation_Programming_Strategy_For_Cortical_Stimulation",
               ])

    add_leaves(neu, ["Implantable_or_Surgical_Neuromodulation", "Central_Implantables_and_Evaluations",
                     "Cortical_Implant_Neurostimulation", "Maintenance_and_Safety"], [
                   "Hardware_Integrity_Troubleshooting_Pathway_For_Cortical_Implants",
                   "Revision_Or_Explant_Consultation_Pathway_For_Cortical_Implants",
               ])

    # ============================================================
    # Subcortical implant neurostimulation (non-DBS, central) — optional category kept explicit
    # ============================================================

    add_leaves(neu, ["Implantable_or_Surgical_Neuromodulation", "Central_Implantables_and_Evaluations",
                     "Subcortical_Implant_Neurostimulation", "Candidacy_and_Evaluation"], [
                   "Subcortical_Implant_Neurostimulation_Evaluation",
                   "Subcortical_Targeting_And_Trajectory_Planning_Service",
               ])

    add_leaves(neu, ["Implantable_or_Surgical_Neuromodulation", "Central_Implantables_and_Evaluations",
                     "Subcortical_Implant_Neurostimulation", "Programming_and_Maintenance"], [
                   "Subcortical_Stimulation_Optimization_Service",
                   "Hardware_Integrity_Troubleshooting_Pathway_For_Subcortical_Implants",
                   "Revision_Or_Explant_Consultation_Pathway_For_Subcortical_Implants",
               ])

    add_leaves(neu, ["Implantable_or_Surgical_Neuromodulation", "Programming_and_FollowUp_Frameworks"], [
        "Device_Programming_Optimization_Framework",
        "Closed_Loop_Sensing_Informed_Programming",
        "Patient_Controller_Use_Training",
        "Remote_Device_Monitoring_Framework",
        "Device_Adverse_Event_Monitoring_Log",
    ])

    # ------------------------
    # F) Convulsive & seizure-based therapies
    # ------------------------
    add_leaves(neu, ["Convulsive_and_Seizure_Based_Therapies", "Approach_Families"], [
        "Electroconvulsive_Therapy",
        "Magnetic_Seizure_Therapy",
        "Focal_Seizure_Therapy_Framework",
        "Anesthesia_Assisted_Neurotherapy_Framework",
    ])

    add_leaves(neu, ["Convulsive_and_Seizure_Based_Therapies", "Supportive_Optimization"], [
        "PeriProcedural_Medical_Clearance_Framework",
        "Cognitive_SideEffect_Monitoring_Framework",
        "Memory_Support_Strategy_Framework",
        "PostProcedure_Recovery_Protocol_Framework",
    ])

    # ------------------------
    # G) Light, photo, and sensory neuromodulation
    # ------------------------
    add_leaves(neu, ["Light_and_Photo_Neuromodulation", "Approach_Families"], [
        "Bright_Light_Therapy_Box",
        "Dawn_Simulation_Device",
        "Dusk_Simulation_Lighting_System",
        "Smart_Circadian_Lighting_System",
        "Blue_Light_Blocking_Glasses",
        "Spectral_Lighting_Optimization_Framework",
        "Photobiomodulation_Transcranial",
        "Photobiomodulation_Intranasal",
        "Photobiomodulation_Systemic",
        "Wearable_Light_Therapy_Device_Class",
    ])

    add_leaves(neu, ["Sensory_Entrainment_and_Modulation", "Auditory_Approaches"], [
        "Auditory_Entrainment_Therapy_Framework",
        "Binaural_Beat_Stimulation",
        "Isochronic_Tone_Stimulation",
        "Music_Therapy_Structured_Framework",
        "Sound_Masking_Therapy_Framework",
        "Pink_Noise_Sleep_Entrainment_Framework",
        "Nature_Soundscape_Therapy_Framework",
    ])

    add_leaves(neu, ["Sensory_Entrainment_and_Modulation", "Visual_Approaches"], [
        "Visual_Pattern_Stimulation",
        "Photic_Stimulation_Framework",
        "Visual_Relaxation_Training_Framework",
        "Light_Flicker_Entrainment_Framework",
    ])

    add_leaves(neu, ["Sensory_Entrainment_and_Modulation", "Somatosensory_and_Vestibular"], [
        "Somatosensory_Stimulation",
        "Vibrotactile_Stimulation_Framework",
        "Vestibular_Stimulation",
        "Balance_Platform_Sensory_Training_Framework",
    ])

    add_leaves(neu, ["Sensory_Entrainment_and_Modulation", "Olfactory_and_Multisensory"], [
        "Olfactory_Stimulation_Therapeutic",
        "Aromatherapy_Structured_Framework",
        "Multisensory_Relaxation_Environment_Framework",
    ])

    # ------------------------
    # H) Biofeedback & neurofeedback (high-resolution)
    # ------------------------
    add_leaves(neu, ["Biofeedback_and_Neurofeedback", "Physiological_Biofeedback"], [
        "HRV_Biofeedback",
        "Respiratory_Biofeedback",
        "Capnography_Informed_Breathing_Training_Framework",
        "EMG_Biofeedback",
        "Thermal_Biofeedback",
        "Galvanic_Skin_Response_Biofeedback",
        "Blood_Pressure_Biofeedback_Framework",
        "Posture_and_Muscle_Tension_Biofeedback_Framework",
        "Wearable_Biofeedback_Device_Class",
        "Mobile_App_Biofeedback_Guidance_Framework",
    ])

    add_leaves(neu, ["Biofeedback_and_Neurofeedback", "Neurofeedback_Modalities"], [
        "EEG_Neurofeedback",
        "QEEG_Guided_Neurofeedback_Framework",
        "InfraLow_Frequency_Neurofeedback_Framework",
        "Alpha_Theta_Training_Framework",
        "SMR_Training_Framework",
        "Sensorimotor_Network_Training_Framework",
        "fNIRS_Neurofeedback",
        "fMRI_Neurofeedback_Framework",
        "HEG_Neurofeedback_Framework",
        "Multimodal_Neurofeedback_Platform",
    ])

    add_leaves(neu, ["Biofeedback_and_Neurofeedback", "Closed_Loop_and_BCI_Frameworks"], [
        "Closed_Loop_Biofeedback_Framework",
        "Closed_Loop_Neurofeedback_Framework",
        "EEG_BCI_Assisted_Training_Framework",
        "Neuroadaptive_Digital_Therapeutic_Framework",
        "Physiology_Adaptive_Stimulation_Framework",
    ])

    add_leaves(neu, ["Biofeedback_and_Neurofeedback", "Implementation_and_Quality_Tools"], [
        "Signal_Quality_Assurance_Framework",
        "Artifact_Handling_Framework",
        "Calibration_and_Baseline_Framework",
        "Progress_Tracking_Dashboard_Framework",
        "Home_Training_Supervision_Framework",
    ])

    # ------------------------
    # I) Digital, immersive, and combined neuromodulation
    # ------------------------
    add_leaves(neu, ["Digital_and_Immersive_Neuromodulation", "Immersive_Formats"], [
        "Virtual_Reality_Therapeutic_Exposure",
        "Virtual_Reality_Relaxation_Training_Framework",
        "Augmented_Reality_Neuromodulation_Framework",
        "Guided_Imagery_Immersive_Framework",
        "Embodied_Motor_Training_Immersive_Framework",
    ])

    add_leaves(neu, ["Digital_and_Immersive_Neuromodulation", "Digital_Therapeutic_Frameworks"], [
        "Closed_Loop_Digital_Neuromodulation",
        "Game_Based_Neural_Training",
        "Cognitive_Training_Digital_Framework",
        "Attention_Training_Digital_Framework",
        "Emotion_Regulation_Skills_Digital_Framework",
        "Digital_Sleep_Training_Framework",
    ])

    add_leaves(neu, ["Digital_and_Immersive_Neuromodulation", "Combination_Stacks"], [
        "Digital_Therapeutic_Plus_tES_Framework",
        "Digital_Therapeutic_Plus_TMS_Framework",
        "Digital_Therapeutic_Plus_Biofeedback_Framework",
        "VR_Plus_Biofeedback_Framework",
    ])

    # ------------------------
    # J) Electromagnetic field and other emerging approaches (without parameter tokens)
    # ------------------------
    add_leaves(neu, ["Emerging_and_Experimental_Approaches", "Electromagnetic_and_Field_Therapies"], [
        "Electromagnetic_Field_Therapy_Low_Energy",
        "Pulsed_Electromagnetic_Field_Framework",
        "Static_Magnetic_Field_Therapy_Framework",
    ])

    add_leaves(neu, ["Emerging_and_Experimental_Approaches", "Molecular_and_Nanoscale_Research_Frameworks"], [
        "Optogenetic_Research_Framework",
        "Chemogenetic_Research_Framework",
        "Nanoparticle_Mediated_Neuromodulation",
    ])

    add_leaves(neu, ["Emerging_and_Experimental_Approaches", "Hybrid_and_Multimodal_Frameworks"], [
        "Hybrid_TMS_tES_Framework",
        "Hybrid_Stimulation_Neurofeedback_Framework",
        "Hybrid_Sensory_Stimulation_Biofeedback_Framework",
        "Hybrid_Ultrasound_Neurofeedback_Framework",
    ])

    # ------------------------
    # K) Safety, candidacy, and care-pathway scaffolding (solution-variable oriented)
    # ------------------------
    add_leaves(neu, ["Clinical_Safety_and_Care_Pathways", "Screening_and_Candidacy_Tools"], [
        "Contraindication_Screening_Checklist",
        "Implant_and_Metal_Safety_Screening",
        "Seizure_Risk_Assessment_Framework",
        "Pregnancy_Safety_Review_Framework",
        "Medication_and_Substance_Safety_Review_Framework",
        "Cardiovascular_Clearance_Framework",
        "Sleep_Apnea_Risk_Screening_Framework",
    ])

    add_leaves(neu, ["Clinical_Safety_and_Care_Pathways", "Monitoring_and_FollowUp_Tools"], [
        "Adverse_Event_Monitoring_Log",
        "Symptom_Tracking_Instrument_Framework",
        "Cognitive_Safety_Monitoring_Framework",
        "Sleep_Tracking_Integration_Framework",
        "Wearable_Data_Integration_Framework",
        "Clinician_CheckIn_Workflow_Framework",
    ])

    add_leaves(neu, ["Clinical_Safety_and_Care_Pathways", "Service_Delivery_Formats"], [
        "Clinic_Based_Delivery_Pathway",
        "Home_Based_Remote_Supervision_Pathway",
        "Hybrid_Clinic_Home_Pathway",
        "Group_Biofeedback_Training_Format",
        "Self_Guided_With_Safety_Gating_Framework",
    ])

    # ------------------------------------------------------------------
    # 6) Pharmacology and Biomedical Treatments (mechanism + class; high-resolution)
    #     - Leaf nodes are actionable candidate solution entities (generic drugs, device/formulation options, tools).
    #     - No brand names. No schedule/frequency/duration/intensity nodes. No disorder-labeled branches.
    #     - IMPORTANT: medication choices require clinician oversight; this is an ontology taxonomy, not prescribing advice.
    # ------------------------------------------------------------------
    pharm: Dict[str, Any] = {}

    # =========================
    # 6.1 Psychiatric + CNS pharmacotherapy (by mechanism families)
    # =========================

    # --- Monoamine modulators ---
    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Monoamine_Modulators", "Serotonergic_Reuptake_Inhibitors_SSRIs"], [
        "Fluoxetine",
        "Sertraline",
        "Citalopram",
        "Escitalopram",
        "Paroxetine",
        "Fluvoxamine",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Monoamine_Modulators", "Serotonin_Norepinephrine_Reuptake_Inhibitors_SNRIs"], [
        "Venlafaxine",
        "Desvenlafaxine",
        "Duloxetine",
        "Levomilnacipran",
        "Milnacipran",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Monoamine_Modulators", "Norepinephrine_Dopamine_Reuptake_Inhibitors_NDRIs"], [
        "Bupropion",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Monoamine_Modulators", "Tricyclic_Monoamine_Modulators_TCAs"], [
        "Amitriptyline",
        "Nortriptyline",
        "Imipramine",
        "Desipramine",
        "Clomipramine",
        "Doxepin",
        "Trimipramine",
        "Protriptyline",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Monoamine_Modulators", "MAO_Inhibitors_MAOIs"], [
        "Phenelzine",
        "Tranylcypromine",
        "Isocarboxazid",
        "Selegiline",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Monoamine_Modulators", "Serotonin_Modulators_and_Antagonists"], [
        "Trazodone",
        "Nefazodone",
        "Vilazodone",
        "Vortioxetine",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Monoamine_Modulators", "Melatonergic_Modulators"], [
        "Agomelatine",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Monoamine_Modulators", "5HT1A_Partial_Agonist_Modulators"], [
        "Buspirone",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Monoamine_Modulators", "Norepinephrine_Reuptake_Inhibitors_NRIs"], [
        "Atomoxetine",
        "Reboxetine",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Monoamine_Modulators", "Alpha2_Adrenergic_Agonist_Modulators"], [
        "Guanfacine",
        "Clonidine",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Monoamine_Modulators", "Antiadrenergic_Modulators"], [
        "Prazosin",
        "Propranolol",
    ])

    # --- Dopamine / antipsychotic mechanism families ---
    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Dopamine_and_Antipsychotic_Modulators", "Second_Generation_Antipsychotics"], [
        "Risperidone",
        "Paliperidone",
        "Olanzapine",
        "Quetiapine",
        "Ziprasidone",
        "Lurasidone",
        "Asenapine",
        "Iloperidone",
        "Clozapine",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Dopamine_and_Antipsychotic_Modulators", "First_Generation_Antipsychotics"], [
        "Haloperidol",
        "Chlorpromazine",
        "Fluphenazine",
        "Perphenazine",
        "Thiothixene",
        "Trifluoperazine",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Dopamine_and_Antipsychotic_Modulators", "Dopamine_Partial_Agonist_Modulators"], [
        "Aripiprazole",
        "Brexpiprazole",
        "Cariprazine",
    ])

    # --- Mood stabilization (non-disorder-labeled) ---
    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Mood_Stabilizing_Classes", "Mood_Stabilizing_Salts"], [
        "Lithium",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Mood_Stabilizing_Classes", "Mood_Stabilizing_Anticonvulsants"], [
        "Valproate",
        "Carbamazepine",
        "Oxcarbazepine",
        "Lamotrigine",
        "Topiramate",
    ])

    # --- Glutamate / GABA / calcium-channel families ---
    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Glutamate_GABA_and_Ion_Channel_Modulators", "Glutamatergic_Modulators_RapidActing"], [
        "Ketamine",
        "Esketamine",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Glutamate_GABA_and_Ion_Channel_Modulators", "Glutamatergic_Modulators_NonRapid"], [
        "Memantine",
        "Amantadine",
        "Riluzole",
        "Lamotrigine_GlutamateRelease_Modulation",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Glutamate_GABA_and_Ion_Channel_Modulators", "GABAergic_Modulators_Benzodiazepines"], [
        "Diazepam",
        "Lorazepam",
        "Clonazepam",
        "Alprazolam",
        "Oxazepam",
        "Temazepam",
        "Chlordiazepoxide",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Glutamate_GABA_and_Ion_Channel_Modulators", "GABAergic_Modulators_Other"], [
        "Phenobarbital",
        "Primidone",
        "Meprobamate",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Glutamate_GABA_and_Ion_Channel_Modulators", "Alpha2Delta_Calcium_Channel_Ligands"], [
        "Gabapentin",
        "Pregabalin",
    ])

    # --- Arousal / attention / wakefulness ---
    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Arousal_and_Attention_Modulators", "Psychostimulants"], [
        "Methylphenidate",
        "Dexmethylphenidate",
        "Dextroamphetamine",
        "Lisdexamfetamine",
        "Mixed_Amphetamine_Salts",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Arousal_and_Attention_Modulators", "Wake_Promoting_Modulators"], [
        "Modafinil",
        "Armodafinil",
        "Solriamfetol",
        "Pitolisant",
    ])

    # --- Cognitive modulators (primarily dementia/cognition pathways; still useful for brain-health ontologies) ---
    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Cognitive_Modulators", "Cholinesterase_Inhibitors"], [
        "Donepezil",
        "Rivastigmine",
        "Galantamine",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Cognitive_Modulators", "NMDA_Receptor_Modulators"], [
        "Memantine",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Cognitive_Modulators", "Nicotinic_Receptor_Modulators"], [
        "Varenicline",
        "Nicotine_Medication_Class",
    ])

    # --- Opioid system / neuropeptide system modulators with mental-health relevance ---
    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Opioid_System_Modulators"], [
        "Naltrexone",
        "Nalmefene",
        "Buprenorphine",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Neuropeptide_and_Novel_Target_Modulators"], [
        "Orexin_Receptor_Antagonists_Class",
        "Kappa_Opioid_Receptor_Modulators_Class",
        "CRF_Receptor_Modulators_Research_Class",
    ])

    # --- Emerging / investigational psychoactive classes (kept mechanism-oriented; no brand names) ---
    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Emerging_and_Investigational_Pharmacology", "Classic_Psychedelic_5HT2A_Agonist_Compounds"], [
        "Psilocybin",
        "Lysergic_Acid_Diethylamide",
        "Dimethyltryptamine",
        "Mescaline",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Emerging_and_Investigational_Pharmacology", "Entactogen_and_Empathogen_Classes"], [
        "MDMA",
    ])

    add_leaves(pharm, ["Psychiatric_and_CNS_Pharmacotherapy", "Emerging_and_Investigational_Pharmacology", "GABA_A_Neurosteroid_Modulators"], [
        "Brexanolone",
        "Zuranolone",
    ])

    # =========================
    # 6.2 Sleep-related pharmacotherapy (mechanism families + exemplar generics)
    # =========================
    add_leaves(pharm, ["Sleep_Related_Pharmacotherapy", "Melatonin_Pathway_Modulators", "Melatonin_Receptor_Agonists"], [
        "Ramelteon",
        "Tasimelteon",
        "Melatonin",
    ])

    add_leaves(pharm, ["Sleep_Related_Pharmacotherapy", "Orexin_Pathway_Modulators", "Orexin_Receptor_Antagonists"], [
        "Suvorexant",
        "Lemborexant",
        "Daridorexant",
    ])

    add_leaves(pharm, ["Sleep_Related_Pharmacotherapy", "GABA_A_Hypnotic_Modulators", "NonBZD_Hypnotics"], [
        "Zolpidem",
        "Zaleplon",
        "Eszopiclone",
    ])

    add_leaves(pharm, ["Sleep_Related_Pharmacotherapy", "GABA_A_Hypnotic_Modulators", "Benzodiazepine_Hypnotics"], [
        "Temazepam",
        "Triazolam",
        "Flurazepam",
        "Estazolam",
        "Quazepam",
    ])

    add_leaves(pharm, ["Sleep_Related_Pharmacotherapy", "Antihistaminergic_Sedatives", "H1_Antihistamines_Sedative"], [
        "Diphenhydramine",
        "Doxylamine",
        "Hydroxyzine",
        "Promethazine",
    ])

    add_leaves(pharm, ["Sleep_Related_Pharmacotherapy", "Sedating_Antidepressant_Classes"], [
        "Trazodone",
        "Doxepin",
        "Mirtazapine",
        "Amitriptyline",
    ])

    add_leaves(pharm, ["Sleep_Related_Pharmacotherapy", "Alpha2_Adrenergic_Agonist_Sedative_Classes"], [
        "Clonidine",
        "Guanfacine",
    ])

    add_leaves(pharm, ["Sleep_Related_Pharmacotherapy", "Antipsychotic_Sedative_Classes"], [
        "Quetiapine",
        "Olanzapine",
        "Chlorpromazine",
    ])

    # =========================
    # 6.3 Endocrine, metabolic, and hormone therapeutics (biomedical levers)
    # =========================
    add_leaves(pharm, ["Endocrine_and_Metabolic_Therapeutics", "Insulin_Sensitizer_Classes"], [
        "Metformin",
        "Pioglitazone",
    ])

    add_leaves(pharm, ["Endocrine_and_Metabolic_Therapeutics", "GLP1_and_Incretin_Modulator_Classes"], [
        "Semaglutide",
        "Liraglutide",
        "Dulaglutide",
        "Exenatide",
        "Tirzepatide",
    ])

    add_leaves(pharm, ["Endocrine_and_Metabolic_Therapeutics", "Lipid_Lowering_Classes", "Statins"], [
        "Atorvastatin",
        "Rosuvastatin",
        "Simvastatin",
        "Pravastatin",
        "Pitavastatin",
    ])

    add_leaves(pharm, ["Endocrine_and_Metabolic_Therapeutics", "Lipid_Lowering_Classes", "NonStatin_Lipid_Lowering"], [
        "Ezetimibe",
        "Fenofibrate",
        "Gemfibrozil",
        "Bempedoic_Acid",
        "Alirocumab",
        "Evolocumab",
    ])

    add_leaves(pharm, ["Endocrine_and_Metabolic_Therapeutics", "Thyroid_Hormone_Modulators", "Thyroid_Hormone_Replacement"], [
        "Levothyroxine",
        "Liothyronine",
    ])

    add_leaves(pharm, ["Endocrine_and_Metabolic_Therapeutics", "Thyroid_Hormone_Modulators", "Antithyroid_Classes"], [
        "Methimazole",
        "Propylthiouracil",
    ])

    add_leaves(pharm, ["Endocrine_and_Metabolic_Therapeutics", "Hormone_Replacement_Therapy_Classes"], [
        "Estradiol",
        "Progesterone",
        "Testosterone",
    ])

    add_leaves(pharm, ["Endocrine_and_Metabolic_Therapeutics", "Contraceptive_Hormone_Classes", "Estrogen_Components"], [
        "Ethinyl_Estradiol",
        "Estradiol_Valerate",
    ])

    add_leaves(pharm, ["Endocrine_and_Metabolic_Therapeutics", "Contraceptive_Hormone_Classes", "Progestin_Components"], [
        "Levonorgestrel",
        "Norethindrone",
        "Drospirenone",
        "Etonogestrel",
        "Medroxyprogesterone_Acetate",
    ])

    add_leaves(pharm, ["Endocrine_and_Metabolic_Therapeutics", "Uric_Acid_Lowering_Classes"], [
        "Allopurinol",
        "Febuxostat",
        "Probenecid",
        "Pegloticase",
    ])

    # =========================
    # 6.4 Anti-inflammatory and immunomodulatory therapeutics (systemic contributors)
    # =========================
    add_leaves(pharm, ["AntiInflammatory_and_Immunomodulatory_Therapeutics", "NSAID_Classes"], [
        "Ibuprofen",
        "Naproxen",
        "Diclofenac",
        "Indomethacin",
        "Meloxicam",
    ])

    add_leaves(pharm, ["AntiInflammatory_and_Immunomodulatory_Therapeutics", "COX2_Selective_Classes"], [
        "Celecoxib",
        "Etoricoxib",
    ])

    add_leaves(pharm, ["AntiInflammatory_and_Immunomodulatory_Therapeutics", "Corticosteroid_Classes"], [
        "Prednisone",
        "Prednisolone",
        "Dexamethasone",
        "Methylprednisolone",
        "Hydrocortisone",
    ])

    add_leaves(pharm, ["AntiInflammatory_and_Immunomodulatory_Therapeutics", "Antihistamine_H1_Classes"], [
        "Cetirizine",
        "Levocetirizine",
        "Loratadine",
        "Fexofenadine",
        "Diphenhydramine",
        "Hydroxyzine",
    ])

    add_leaves(pharm, ["AntiInflammatory_and_Immunomodulatory_Therapeutics", "Mast_Cell_Stabilizer_Classes"], [
        "Cromolyn_Sodium",
        "Ketotifen",
    ])

    add_leaves(pharm, ["AntiInflammatory_and_Immunomodulatory_Therapeutics", "Leukotriene_Modulator_Classes"], [
        "Montelukast",
        "Zafirlukast",
    ])

    add_leaves(pharm, ["AntiInflammatory_and_Immunomodulatory_Therapeutics", "JAK_Inhibitor_Classes"], [
        "Tofacitinib",
        "Baricitinib",
        "Upadacitinib",
    ])

    add_leaves(pharm, ["AntiInflammatory_and_Immunomodulatory_Therapeutics", "Cytokine_Blockade_Biologic_Classes", "TNF_Inhibitors"], [
        "Adalimumab",
        "Infliximab",
        "Etanercept",
        "Certolizumab_Pegol",
        "Golimumab",
    ])

    add_leaves(pharm, ["AntiInflammatory_and_Immunomodulatory_Therapeutics", "Cytokine_Blockade_Biologic_Classes", "IL6_Pathway_Blockade"], [
        "Tocilizumab",
        "Sarilumab",
    ])

    add_leaves(pharm, ["AntiInflammatory_and_Immunomodulatory_Therapeutics", "Cytokine_Blockade_Biologic_Classes", "IL17_Pathway_Blockade"], [
        "Secukinumab",
        "Ixekizumab",
        "Brodalumab",
    ])

    add_leaves(pharm, ["AntiInflammatory_and_Immunomodulatory_Therapeutics", "Cytokine_Blockade_Biologic_Classes", "IL12_23_or_IL23_Pathway_Blockade"], [
        "Ustekinumab",
        "Guselkumab",
        "Risankizumab",
        "Tildrakizumab",
    ])

    # =========================
    # 6.5 Pain, somatic symptom, and neurologic symptom pharmacotherapy
    # =========================
    add_leaves(pharm, ["Pain_and_Somatic_Symptom_Therapeutics", "NonOpioid_Analgesic_Classes"], [
        "Acetaminophen",
        "Aspirin",
    ])

    add_leaves(pharm, ["Pain_and_Somatic_Symptom_Therapeutics", "Neuropathic_Pain_Modulator_Classes", "Gabapentinoids"], [
        "Gabapentin",
        "Pregabalin",
    ])

    add_leaves(pharm, ["Pain_and_Somatic_Symptom_Therapeutics", "Neuropathic_Pain_Modulator_Classes", "Sodium_Channel_Blocking_Classes"], [
        "Carbamazepine",
        "Oxcarbazepine",
        "Lamotrigine",
        "Mexiletine",
    ])

    add_leaves(pharm, ["Pain_and_Somatic_Symptom_Therapeutics", "Neuropathic_Pain_Modulator_Classes", "Serotonin_Norepinephrine_Modulators_For_Pain"], [
        "Duloxetine",
        "Venlafaxine",
        "Amitriptyline",
        "Nortriptyline",
    ])

    add_leaves(pharm, ["Pain_and_Somatic_Symptom_Therapeutics", "Topical_Analgesic_Classes"], [
        "Topical_Lidocaine_Class",
        "Topical_Capsaicin_Class",
        "Topical_NSAID_Class",
        "Topical_Menthol_Counterirritant_Class",
    ])

    add_leaves(pharm, ["Pain_and_Somatic_Symptom_Therapeutics", "Migraine_Modulator_Classes", "Triptans"], [
        "Sumatriptan",
        "Rizatriptan",
        "Zolmitriptan",
        "Eletriptan",
        "Naratriptan",
        "Frovatriptan",
    ])

    add_leaves(pharm, ["Pain_and_Somatic_Symptom_Therapeutics", "Migraine_Modulator_Classes", "CGRP_Pathway_Modulators", "CGRP_Monoclonal_Antibodies"], [
        "Erenumab",
        "Fremanezumab",
        "Galcanezumab",
        "Eptinezumab",
    ])

    add_leaves(pharm, ["Pain_and_Somatic_Symptom_Therapeutics", "Migraine_Modulator_Classes", "CGRP_Pathway_Modulators", "Gepants"], [
        "Ubrogepant",
        "Rimegepant",
        "Atogepant",
        "Zavegepant",
    ])

    add_leaves(pharm, ["Pain_and_Somatic_Symptom_Therapeutics", "Migraine_Modulator_Classes", "Ditans"], [
        "Lasmiditan",
    ])

    add_leaves(pharm, ["Pain_and_Somatic_Symptom_Therapeutics", "Muscle_Relaxant_Classes"], [
        "Cyclobenzaprine",
        "Baclofen",
        "Tizanidine",
        "Methocarbamol",
        "Metaxalone",
        "Carisoprodol",
    ])

    add_leaves(pharm, ["Pain_and_Somatic_Symptom_Therapeutics", "Antispasmodic_Classes"], [
        "Dicyclomine",
        "Hyoscyamine",
    ])

    add_leaves(pharm, ["Pain_and_Somatic_Symptom_Therapeutics", "Antiemetic_Classes", "5HT3_Antagonists"], [
        "Ondansetron",
        "Granisetron",
        "Palonosetron",
    ])

    add_leaves(pharm, ["Pain_and_Somatic_Symptom_Therapeutics", "Antiemetic_Classes", "Dopamine_Antagonist_Antiemetics"], [
        "Metoclopramide",
        "Prochlorperazine",
        "Droperidol",
    ])

    add_leaves(pharm, ["Pain_and_Somatic_Symptom_Therapeutics", "Local_Anesthetic_Classes"], [
        "Lidocaine",
        "Bupivacaine",
        "Ropivacaine",
        "Mepivacaine",
    ])

    # =========================
    # 6.6 Substance-intake support pharmacotherapy (harm reduction + cessation supports)
    # =========================
    add_leaves(pharm, ["Substance_Intake_Support_Pharmacotherapy", "Nicotine_Replacement_Therapy_Classes"], [
        "Nicotine_Transdermal_Patch",
        "Nicotine_Gum",
        "Nicotine_Lozenge",
        "Nicotine_Inhaler",
        "Nicotine_Nasal_Spray",
    ])

    add_leaves(pharm, ["Substance_Intake_Support_Pharmacotherapy", "Nicotinic_Receptor_Partial_Agonist_Cessation_Agent_Classes"], [
        "Varenicline",
        "Cytisine",
    ])

    add_leaves(pharm, ["Substance_Intake_Support_Pharmacotherapy", "Alcohol_Craving_Modulator_Classes"], [
        "Naltrexone",
        "Acamprosate",
        "Nalmefene",
        "Topiramate",
        "Gabapentin",
    ])

    add_leaves(pharm, ["Substance_Intake_Support_Pharmacotherapy", "Alcohol_Aversive_Agent_Classes"], [
        "Disulfiram",
    ])

    add_leaves(pharm, ["Substance_Intake_Support_Pharmacotherapy", "Opioid_Agonist_Therapy_Classes"], [
        "Methadone",
        "Buprenorphine",
    ])

    add_leaves(pharm, ["Substance_Intake_Support_Pharmacotherapy", "Opioid_Antagonist_Therapy_Classes"], [
        "Naltrexone",
    ])

    add_leaves(pharm, ["Substance_Intake_Support_Pharmacotherapy", "Opioid_Overdose_Reversal_Tools"], [
        "Naloxone",
    ])

    # =========================
    # 6.7 Medication formulations and routes (generic)
    # =========================
    add_leaves(pharm, ["Medication_Formulations_and_Routes", "Oral"], [
        "Oral_Tablet",
        "Oral_Capsule",
        "Oral_Liquid",
        "Orally_Disintegrating_Tablet",
        "Modified_Release_Oral",
    ])

    add_leaves(pharm, ["Medication_Formulations_and_Routes", "Mucosal"], [
        "Sublingual_Tablet",
        "Buccal_Film",
        "Intranasal_Spray",
    ])

    add_leaves(pharm, ["Medication_Formulations_and_Routes", "Parenteral"], [
        "Injectable",
        "Depot_Injectable",
        "Infusion_Therapy_Framework",
    ])

    add_leaves(pharm, ["Medication_Formulations_and_Routes", "Transdermal_and_Topical"], [
        "Transdermal_Patch",
        "Topical_Cream_Gel",
    ])

    add_leaves(pharm, ["Medication_Formulations_and_Routes", "Inhaled_and_Airway"], [
        "Inhaled_Formulation",
    ])

    add_leaves(pharm, ["Medication_Formulations_and_Routes", "Rectal"], [
        "Rectal_Suppository",
    ])

    add_leaves(pharm, ["Medication_Formulations_and_Routes", "Ophthalmic_and_Otic"], [
        "Ophthalmic_Drops",
        "Otic_Drops",
    ])

    # =========================
    # 6.8 Medication safety, monitoring, and personalization tools (solution supports)
    # =========================
    add_leaves(pharm, ["Medication_Safety_and_Personalization_Tools", "Interaction_and_Appropriateness_Checking"], [
        "Drug_Drug_Interaction_Checker",
        "Drug_Disease_Interaction_Check_Framework",
        "QT_Risk_Screening_Framework",
        "Serotonergic_Toxicity_Risk_Check_Framework",
        "Sedation_Risk_Check_Framework",
        "Anticholinergic_Burden_Check_Framework",
        "Fall_Risk_Medication_Review_Framework",
        "Pregnancy_Lactation_Medication_Review_Framework",
        "Renal_Dose_Adjustment_Review_Framework",
        "Hepatic_Dose_Adjustment_Review_Framework",
        "Polypharmacy_Review_With_Clinician",
    ])

    add_leaves(pharm, ["Medication_Safety_and_Personalization_Tools", "Personalization_and_Testing"], [
        "Pharmacogenomics_Testing",
        "Therapeutic_Drug_Monitoring",
        "Medication_Reconciliation",
        "Allergy_Adverse_Reaction_History_Review",
    ])

    add_leaves(pharm, ["Medication_Safety_and_Personalization_Tools", "Adherence_and_Tracking"], [
        "Blister_Pack_Dispensing",
        "Medication_Organizer",
        "Adherence_Reminder_Device",
        "Side_Effect_Tracking_Log",
        "Symptom_Tracking_Log_Medication_Linked",
        "Shared_Decision_Making_Medication_Review_With_Clinician",
    ])

    BIO["Pharmacology_and_Biomedical_Treatments"] = pharm

    # ------------------------------------------------------------------
    # 7) Somatic / Interventional Care Pathways (BIO)
    #     Expanded, high-resolution, and clearly structured.
    #     Scope here EXCLUDES: ECT, MST, VNS implant, DBS, responsive neurostimulation,
    #     and other implantable/surgical neuromodulation (handled elsewhere).
    #     Focus: somatic/interventional referrals, procedures, and specialty pathways
    #     that can optimize (non-)clinical mental health via biological/systems routes.
    #     Leaves = actionable services/pathways/evaluations (no schedules/frequency).
    # ------------------------------------------------------------------
    som: Dict[str, Any] = {}

    # ============================================================
    # A) Rapid-Acting / Interventional Psychopharmacology Pathways
    # ============================================================
    add_leaves(som, ["Somatic_and_Interventional_Therapies", "RapidActing_Interventional_Psychopharmacology",
                     "Ketamine_And_Related_Pathways"], [
                   "Ketamine_Clinic_Based_RapidActing_Therapy_Class",
                   "Ketamine_Eligibility_Evaluation_Pathway",
                   "Ketamine_Medical_Clearance_Pathway",
                   "Ketamine_Risk_Screening_And_Contraindication_Assessment",
                   "Ketamine_Informed_Consent_And_Expectation_Setting",
                   "Ketamine_Response_Monitoring_And_Outcomes_Tracking_Pathway",
                   "Ketamine_Adverse_Effect_Triage_Pathway",
                   "Ketamine_Continuation_Or_Transition_Planning_Pathway",
               ])

    add_leaves(som, ["Somatic_and_Interventional_Therapies", "RapidActing_Interventional_Psychopharmacology",
                     "Anesthetic_Assisted_Pathways"], [
                   "Anesthetic_Assisted_Psychopharmacology_Class",
                   "Anesthesia_PreProcedure_Evaluation_Pathway",
                   "Cardiorespiratory_Risk_Assessment_For_Anesthetic_Assisted_Therapy",
                   "Medication_Interaction_And_Periprocedural_Adjustment_Pathway",
                   "Airway_Safety_And_Sedation_Risk_Assessment",
                   "Post_Procedure_Recovery_And_Discharge_Planning_Pathway",
                   "Post_Procedure_Neurocognitive_Status_Check_Pathway",
                   "Anesthetic_Assisted_Therapy_Adverse_Event_Triage_Pathway",
               ])

    add_leaves(som, ["Somatic_and_Interventional_Therapies", "RapidActing_Interventional_Psychopharmacology",
                     "Clinical_Operations_and_Care_Integration"], [
                   "Interventional_Clinic_Referral_Pathway",
                   "Pre_Treatment_Labs_And_Vitals_Assessment_Pathway",
                   "Medical_Comorbidity_Coordination_For_Interventional_Therapy",
                   "Substance_Risk_Screening_For_Interventional_Therapy",
                   "Driving_And_Safety_After_Procedure_Planning",
                   "Care_Coordination_With_Outpatient_Team_For_Interventional_Therapy",
               ])

    # ============================================================
    # B) Hyperbaric / Physiologic Environmental Medicine Pathways
    # ============================================================
    add_leaves(som,
               ["Somatic_and_Interventional_Therapies", "Physiologic_Environmental_Medicine", "Hyperbaric_Pathways"], [
                   "Hyperbaric_Oxygen_Therapy",
                   "Hyperbaric_Medicine_Consultation_Pathway",
                   "Hyperbaric_Eligibility_And_Indication_Review",
                   "Hyperbaric_Contraindication_And_Risk_Screening",
                   "Ear_Sinus_Barotrauma_Risk_Assessment_Pathway",
                   "Pulmonary_Risk_Assessment_For_Hyperbaric_Therapy",
                   "Oxygen_Toxicity_Risk_Assessment_Pathway",
                   "Hyperbaric_Response_And_Outcomes_Tracking_Pathway",
                   "Hyperbaric_Adverse_Event_Triage_Pathway",
               ])

    # ============================================================
    # C) Therapeutic Apheresis / Immuno-Hematologic Evaluation Pathways
    # ============================================================
    add_leaves(som, ["Somatic_and_Interventional_Therapies", "ImmunoHematologic_Interventional_Evaluations",
                     "Therapeutic_Apheresis"], [
                   "Therapeutic_Apheresis_Evaluation",
                   "Apheresis_Specialist_Consultation_Pathway",
                   "Autoimmune_And_Inflammatory_Workup_Pathway_For_Apheresis_Consideration",
                   "Vascular_Access_Assessment_For_Apheresis",
                   "Bleeding_Thrombosis_Risk_Assessment_For_Apheresis",
                   "Electrolyte_And_Hemodynamic_Risk_Assessment_For_Apheresis",
                   "Medication_Compatibility_Review_For_Apheresis",
                   "Apheresis_Adverse_Event_Triage_Pathway",
                   "Apheresis_Response_And_Outcomes_Tracking_Pathway",
               ])

    # ============================================================
    # D) Pain Medicine Pathways (somatic drivers of affect/cognition)
    # ============================================================
    add_leaves(som, ["Somatic_and_Interventional_Therapies", "Pain_Medicine_and_Somatic_Distress",
                     "Referral_and_Evaluation"], [
                   "Pain_Clinic_Referral",
                   "Pain_Medicine_Consultation_Pathway",
                   "Chronic_Pain_Multimodal_Assessment_Pathway",
                   "Central_Sensitization_Evaluation_Pathway",
                   "Headache_And_Migraine_Specialty_Referral",
                   "Orofacial_Pain_Referral_Pathway",
                   "Pelvic_Pain_Specialty_Referral",
                   "Neuropathic_Pain_Evaluation_Pathway",
                   "Pain_Related_Sleep_Disruption_Assessment_Pathway",
               ])

    add_leaves(som, ["Somatic_and_Interventional_Therapies", "Pain_Medicine_and_Somatic_Distress",
                     "Interventional_Pain_Options_and_Coordination"], [
                   "Interventional_Pain_Procedure_Eligibility_Evaluation",
                   "Nerve_Block_Evaluation_Pathway",
                   "Trigger_Point_Injection_Evaluation_Pathway",
                   "Epidural_Or_Spinal_Procedure_Evaluation_Pathway",
                   "Neuromuscular_Rehabilitation_Coordination_For_Pain",
                   "Medication_Optimization_Coordination_For_Pain",
                   "Pain_Function_Goal_Setting_And_Tracking_Pathway",
                   "Pain_Crisis_Triage_And_Safety_Planning_Pathway",
               ])

    # ============================================================
    # E) Sleep Medicine Pathways (sleep physiology as mental health lever)
    # ============================================================
    add_leaves(som, ["Somatic_and_Interventional_Therapies", "Sleep_Medicine_and_Circadian_Care",
                     "Referral_and_Diagnostics"], [
                   "Sleep_Clinic_Referral",
                   "Sleep_Medicine_Consultation_Pathway",
                   "Sleep_Disorder_Diagnostic_Workup_Pathway",
                   "Sleep_Breathing_Risk_Assessment_Pathway",
                   "Circadian_Rhythm_Assessment_Pathway",
                   "Parasomnia_Evaluation_Pathway",
                   "Restless_Legs_And_Movement_In_Sleep_Evaluation_Pathway",
                   "Medication_And_Substance_Sleep_Impact_Review",
               ])

    add_leaves(som, ["Somatic_and_Interventional_Therapies", "Sleep_Medicine_and_Circadian_Care",
                     "Therapy_And_Device_Pathways"], [
                   "Sleep_Device_Therapy_Evaluation_Pathway",
                   "CPAP_Or_PAP_Therapy_Coordination_Pathway",
                   "Oral_Appliance_Therapy_Referral_Pathway",
                   "Insomnia_Specialist_Referral_Pathway",
                   "Circadian_Entrainment_Therapy_Referral_Pathway",
                   "Sleep_Wake_Stabilization_Care_Plan_Coordination",
                   "Nightmare_Targeted_Sleep_Specialty_Referral",
               ])

    # ============================================================
    # F) Rehabilitation Medicine / Functional Restoration Pathways
    # ============================================================
    add_leaves(som, ["Somatic_and_Interventional_Therapies", "Rehabilitation_and_Functional_Restoration",
                     "Physiatry_and_Referral"], [
                   "Rehabilitation_Medicine_Referral",
                   "Physical_Medicine_And_Rehabilitation_Consultation_Pathway",
                   "Functional_Impairment_And_Activity_Limitation_Assessment_Pathway",
                   "Fatigue_And_Pacing_Clinical_Program_Referral",
                   "Neurorehabilitation_Consultation_Pathway",
                   "Cognitive_Communication_Rehabilitation_Referral_Pathway",
                   "Vestibular_Rehabilitation_Referral_Pathway",
                   "Occupational_Therapy_Functional_Skills_Referral",
                   "Speech_Language_Pathology_Cognitive_Communication_Referral",
               ])

    add_leaves(som, ["Somatic_and_Interventional_Therapies", "Rehabilitation_and_Functional_Restoration",
                     "Somatic_Symptom_and_Stress_Physiology_Interfaces"], [
                   "Autonomic_Function_Assessment_Referral_Pathway",
                   "Exercise_Intolerance_Evaluation_Referral_Pathway",
                   "Breathing_Pattern_Disorder_Evaluation_Referral_Pathway",
                   "Post_Illness_Functional_Restoration_Pathway",
                   "Return_To_Activity_Clearance_And_Coordination_Pathway",
               ])

    # ============================================================
    # G) Medical Speciality Linkages (common somatic drivers)
    #     (Stays within somatic referral logic; avoids legal/material domains.)
    # ============================================================
    add_leaves(som, ["Somatic_and_Interventional_Therapies", "Medical_Specialty_Linkage",
                     "Metabolic_Endocrine_and_Nutrition"], [
                   "Endocrinology_Referral_Pathway",
                   "Metabolic_Risk_Assessment_Pathway",
                   "Thyroid_Function_Evaluation_Referral_Pathway",
                   "Nutritional_Medicine_Consultation_Pathway",
               ])

    add_leaves(som,
               ["Somatic_and_Interventional_Therapies", "Medical_Specialty_Linkage", "Cardiopulmonary_and_Autonomic"], [
                   "Cardiology_Referral_Pathway",
                   "Pulmonology_Referral_Pathway",
                   "Autonomic_Specialist_Referral_Pathway",
                   "Syncope_And_Presyncope_Evaluation_Referral_Pathway",
               ])

    add_leaves(som,
               ["Somatic_and_Interventional_Therapies", "Medical_Specialty_Linkage", "Infectious_And_Inflammatory"], [
                   "Infectious_Disease_Referral_Pathway",
                   "Rheumatology_Referral_Pathway",
                   "Inflammatory_Biomarker_Workup_Referral_Pathway",
                   "Autoimmune_Encephalopathy_Evaluation_Referral_Pathway",
               ])

    # ============================================================
    # H) Safety, Monitoring, and Escalation (cross-cutting)
    # ============================================================
    add_leaves(som, ["Somatic_and_Interventional_Therapies", "Safety_Monitoring_and_Escalation"], [
        "Medical_Contraindication_Screening_Pathway",
        "Procedure_Risk_Benefit_Clarification_Session",
        "Informed_Consent_And_Expectation_Setting_Session",
        "Adverse_Event_Triage_Pathway_For_Somatic_Interventions",
        "Care_Transition_Planning_After_Interventional_Therapy",
        "Outcomes_Tracking_And_Response_Attribution_Pathway",
        "Device_And_Procedure_Safety_Clearance_Pathway",
    ])

    BIO["Somatic_and_Interventional_Care_Pathways"] = som

    # ------------------------------------------------------------------
    # 8) Medical Assessment, Physiology, and Testing (solution-variable framing)
    #     Goal: identify modifiable biological contributors to mood, energy, cognition,
    #     sleep, anxiety-arousal, irritability, and stress tolerance (non-/subclinical + clinical).
    #     Notes:
    #       - Leaf nodes are *candidate* tests/referrals/tools; selection depends on history/exam.
    #       - No schedule/frequency/duration tokens. No disorder-labeled branches.
    # ------------------------------------------------------------------
    med: Dict[str, Any] = {}

    # =========================
    # 8.1 Laboratory tests
    # =========================

    add_leaves(med, ["Laboratory_Tests", "Core_Panels_and_Safety_Baseline"], [
        "Complete_Blood_Count",
        "Comprehensive_Metabolic_Panel",
        "Basic_Metabolic_Panel",
        "Liver_Function_Panel",
        "Renal_Function_Panel",
        "Electrolytes_Panel",
        "Urinalysis",
        "Pregnancy_Test_If_Relevant",
    ])

    add_leaves(med, ["Laboratory_Tests", "Cardiometabolic_Risk_and_Energy_Metabolism"], [
        "Lipid_Panel",
        "Apolipoprotein_B_Test",
        "Lipoprotein_a_Test",
        "HbA1c_Test",
        "Fasting_Glucose_Test",
        "Fasting_Insulin_Test",
        "HOMA_IR_Estimation_Framework",
        "Oral_Glucose_Tolerance_Test_Referral",
        "Uric_Acid_Test",
        "High_Sensitivity_CRP_Test",
    ])

    add_leaves(med, ["Laboratory_Tests", "Thyroid_and_Related_Endocrine"], [
        "TSH_Test",
        "Free_T4_Test",
        "Free_T3_Test",
        "Thyroid_Peroxidase_Antibodies_Test",
        "Thyroglobulin_Antibodies_Test",
        "Thyroid_Ultrasound_Referral",
    ])

    add_leaves(med, ["Laboratory_Tests", "Hematologic_Iron_and_Oxygen_Carrying_Capacity"], [
        "Iron_Studies",
        "Ferritin_Test",
        "Transferrin_Saturation_Test",
        "Reticulocyte_Count_Test",
        "Vitamin_B12_Test",
        "Folate_Test",
        "Methylmalonic_Acid_Test",
    ])

    add_leaves(med, ["Laboratory_Tests", "Micronutrients_and_Metabolic_Cofactors"], [
        "Vitamin_D_Test",
        "Magnesium_Test_Serum",
        "Magnesium_RBC_Test",
        "Zinc_Test",
        "Copper_Test",
        "Ceruloplasmin_Test",
        "Selenium_Test",
        "Omega3_Index_Test",
        "Homocysteine_Test",
        "MTHFR_Related_Methylation_Workup_Framework",
    ])

    add_leaves(med, ["Laboratory_Tests", "Inflammation_Immune_and_Autoimmune_Screening"], [
        "CRP_Test",
        "High_Sensitivity_CRP_Test",
        "ESR_Test",
        "ANA_Test",
        "Rheumatoid_Factor_Test",
        "Anti_CCP_Test",
        "Complement_C3_Test",
        "Complement_C4_Test",
        "Immunoglobulins_Quantitative_Test",
        "Celiac_Serology",
        "Food_Allergy_IgE_Testing_Referral",
        "Environmental_Allergy_Testing_Referral",
    ])

    add_leaves(med, ["Laboratory_Tests", "Histamine_and_Mast_Cell_Related_Evaluation"], [
        "Tryptase_Test",
        "Plasma_Histamine_Test",
        "Urinary_N_Methylhistamine_Test",
        "Urinary_Prostaglandin_D2_Metabolite_Test_Referral",
        "Mast_Cell_Mediator_Evaluation_Referral",
    ])

    add_leaves(med, ["Laboratory_Tests", "Sex_Hormones_and_Reproductive_Endocrine"], [
        "Sex_Hormone_Panel",
        "Estradiol_Test",
        "Progesterone_Test",
        "Testosterone_Total_Test",
        "Testosterone_Free_Test",
        "SHBG_Test",
        "LH_Test",
        "FSH_Test",
        "DHEA_S_Test",
    ])

    add_leaves(med, ["Laboratory_Tests", "HPA_Axis_and_Stress_Physiology"], [
        "Morning_Cortisol_Test",
        "ACTH_Test",
        "Dexamethasone_Suppression_Test_Referral",
        "Late_Night_Salivary_Cortisol_Test",
        "Cortisol_Rhythm_Test",
    ])

    add_leaves(med, ["Laboratory_Tests", "Sleep_Related_and_Fatigue_Differential"], [
        "Ferritin_Test",
        "Vitamin_D_Test",
        "TSH_Test",
        "CRP_Test",
        "EBV_Serology_Referral",
        "CMV_Serology_Referral",
        "HIV_Testing_Referral",
        "Hepatitis_Panel_Referral",
    ])

    add_leaves(med, ["Laboratory_Tests", "Infectious_And_PostInfectious_Evaluation"], [
        "Infection_Screening_Referral",
        "Lyme_Serology_Referral",
        "Syphilis_Testing_Referral",
        "Tuberculosis_Screening_Referral",
    ])

    add_leaves(med, ["Laboratory_Tests", "Toxic_Exposure_and_Environmental_Medicine"], [
        "Heavy_Metal_Screen",
        "Lead_Level_Test",
        "Mercury_Level_Test",
        "Arsenic_Level_Test",
        "Occupational_Exposure_Panel_Referral",
        "Pesticide_Exposure_Evaluation_Referral",
    ])

    add_leaves(med, ["Laboratory_Tests", "Nutritional_And_GI_Malabsorption_Workup"], [
        "Celiac_Serology",
        "Stool_Fat_Test_Referral",
        "Pancreatic_Elastase_Test_Referral",
        "Fecal_Calprotectin_Test",
        "Fecal_Lactoferrin_Test_Referral",
    ])

    # =========================
    # 8.2 Microbiome and GI testing
    # =========================
    add_leaves(med, ["Microbiome_and_GI_Testing", "Stool_Testing"], [
        "Stool_Microbiome_Test",
        "Stool_Inflammation_Markers_Test",
        "Stool_Pathogen_PCR_Panel_Referral",
        "Stool_Ova_And_Parasites_Test_Referral",
        "Stool_Helicobacter_Pylori_Antigen_Test_Referral",
        "Fecal_Calprotectin_Test",
    ])

    add_leaves(med, ["Microbiome_and_GI_Testing", "Breath_And_Functional_GI_Evaluation"], [
        "Breath_Test_GI_Evaluation_Referral",
        "Lactose_Intolerance_Breath_Test_Referral",
        "Fructose_Malabsorption_Breath_Test_Referral",
    ])

    # =========================
    # 8.3 Cardiometabolic physiology & autonomic function
    # =========================
    add_leaves(med, ["Cardiometabolic_Physiology_Tests", "Cardiac_Rhythm_and_Structure"], [
        "Electrocardiogram",
        "Holter_Monitor_Referral",
        "Event_Monitor_Referral",
        "Echocardiogram",
    ])

    add_leaves(med, ["Cardiometabolic_Physiology_Tests", "Blood_Pressure_and_Vascular_Assessment"], [
        "Ambulatory_Blood_Pressure_Monitor",
        "Orthostatic_Vitals_Assessment_Framework",
        "Vascular_Risk_Assessment_Referral",
        "Carotid_Ultrasound_Referral",
    ])

    add_leaves(med, ["Cardiometabolic_Physiology_Tests", "Metabolic_Body_Composition_and_Bone"], [
        "Body_Composition_Assessment",
        "Bone_Density_Scan",
        "Resting_Metabolic_Rate_Test_Referral",
    ])

    add_leaves(med, ["Cardiometabolic_Physiology_Tests", "Autonomic_and_POTS_Related_Evaluation"], [
        "Tilt_Table_Test_Referral",
        "Heart_Rate_Variability_Assessment_Framework",
        "QSART_Sudomotor_Test_Referral",
    ])

    # =========================
    # 8.4 Imaging and neurophysiology (brain-body contributors)
    # =========================
    add_leaves(med, ["Imaging_and_Neurophysiology_Tests", "Neuroimaging"], [
        "MRI_Brain",
        "CT_Head",
        "MR_Angiography_Referral",
        "Carotid_Ultrasound_Referral",
    ])

    add_leaves(med, ["Imaging_and_Neurophysiology_Tests", "Electrophysiology_and_Functional_Tests"], [
        "EEG_Clinical_Evaluation",
        "Ambulatory_EEG_Referral",
        "EMG_Nerve_Conduction_Study_Referral",
    ])

    add_leaves(med, ["Imaging_and_Neurophysiology_Tests", "Headache_and_Neurovascular_Workup"], [
        "Migraine_Evaluation_Referral",
        "Secondary_Headache_RedFlag_Evaluation_Referral",
    ])

    add_leaves(med, ["Imaging_and_Neurophysiology_Tests", "Endocrine_Imaging"], [
        "Ultrasound_Thyroid",
        "Pituitary_MRI_Referral",
    ])

    add_leaves(med, ["Imaging_and_Neurophysiology_Tests", "Inflammation_Imaging"], [
        "Inflammation_Imaging_Evaluation",
        "PET_Imaging_Referral",
    ])

    # =========================
    # 8.5 Sleep and respiratory testing (fatigue, cognition, mood stability)
    # =========================
    add_leaves(med, ["Sleep_and_Respiratory_Testing", "Sleep_Disordered_Breathing"], [
        "Polysomnography",
        "Home_Sleep_Testing",
        "Sleep_Apnea_Evaluation_Referral",
        "Pulse_Oximeter",
        "Capnography_Monitoring",
    ])

    add_leaves(med, ["Sleep_and_Respiratory_Testing", "Pulmonary_Function"], [
        "Spirometry",
        "Peak_Flow_Meter",
        "DLCO_Test_Referral",
    ])

    add_leaves(med, ["Sleep_and_Respiratory_Testing", "Circadian_and_Sleep_Tracking"], [
        "Actigraphy_Referral",
        "Sleep_Diary",
    ])

    # =========================
    # 8.6 Medication and iatrogenic contributors (optimization reviews)
    # =========================
    add_leaves(med, ["Medication_and_Iatrogenic_Contributor_Review", "Medication_Review_Frameworks"], [
        "Medication_Review_With_Clinician",
        "Polypharmacy_Review_With_Clinician",
        "Sedation_Risk_Medication_Review_Framework",
        "Anticholinergic_Burden_Review_Framework",
        "QT_Risk_Medication_Review_Framework",
        "CNS_Stimulant_And_Caffeine_Use_Review_Framework",
        "Substance_Intake_Review_With_Clinician",
    ])

    add_leaves(med, ["Medication_and_Iatrogenic_Contributor_Review", "Therapeutic_Drug_Monitoring_Referrals"], [
        "Therapeutic_Drug_Monitoring",
        "Pharmacogenomics_Testing",
    ])

    # =========================
    # 8.7 Physiology and somatic optimization tools (actionable supports)
    # =========================
    add_leaves(med, ["Physiology_and_Somatic_Optimization_Tools", "Respiratory_Airway_and_ENT"], [
        "Breathing_Retraining",
        "Inspiratory_Muscle_Training_Device",
        "Nasal_Saline_Irrigation",
        "Allergic_Rhinitis_Care",
        "Nasal_Obstruction_Evaluation_Referral",
        "Sinus_Disease_Evaluation_Referral",
    ])

    add_leaves(med, ["Physiology_and_Somatic_Optimization_Tools", "Oral_Health_and_Inflammation_Load"], [
        "Oral_Health_Periodontal_Care",
        "Bruxism_Evaluation_Referral",
        "Dental_Sleep_Medicine_Evaluation_Referral",
    ])

    add_leaves(med, ["Physiology_and_Somatic_Optimization_Tools", "Sensory_Function"], [
        "Vision_Correction",
        "Vision_Evaluation_Referral",
        "Hearing_Support",
        "Hearing_Evaluation_Referral",
        "Vestibular_Evaluation_Referral",
    ])

    add_leaves(med, ["Physiology_and_Somatic_Optimization_Tools", "Cardiometabolic_Home_Monitoring"], [
        "Blood_Pressure_Home_Monitor",
        "Glucose_Monitoring_Device",
        "Smart_Scale",
    ])

    add_leaves(med, ["Physiology_and_Somatic_Optimization_Tools", "Infection_and_Prevention"], [
        "Vaccination_Update",
        "Infection_Screening_Referral",
        "Sleep_Apnea_Evaluation_Referral",
    ])

    add_leaves(med, ["Physiology_and_Somatic_Optimization_Tools", "Pain_and_Physical_Function_Pathways"], [
        "Pain_Rehabilitation_Pathway",
        "Physical_Therapy_Referral",
        "Occupational_Therapy_Referral",
        "Neurology_Evaluation_Referral",
        "Rheumatology_Evaluation_Referral",
    ])

    add_leaves(med, ["Physiology_and_Somatic_Optimization_Tools", "Metabolic_And_Nutritional_Pathways"], [
        "Metabolic_Health_Care_Pathway",
        "Registered_Dietitian_Referral",
        "Medical_Nutrition_Therapy_Plan",
        "Endocrinology_Evaluation_Referral",
    ])

    add_leaves(med, ["Physiology_and_Somatic_Optimization_Tools", "Neurocognitive_and_Functional_Referrals"], [
        "Neuropsychological_Assessment_Referral",
        "Cognitive_Clinic_Evaluation_Referral",
        "Sleep_Clinic_Evaluation_Referral",
    ])

    BIO["Medical_Assessment_and_Physiology_Testing"] = med

    # ------------------------------------------------------------------
    # 9) Environmental Exposures & Built-Environment Optimization (BIO)
    # (formerly: Environment_Exposure_and_BuiltWorld_Optimization)
    # ------------------------------------------------------------------
    env: Dict[str, Any] = {}

    # ----------------------------
    # Indoor air: particles, gases, combustion, allergens, moisture
    # ----------------------------
    add_leaves(env, ["Indoor_Air_Quality_and_Ventilation", "Filtration_and_Purification"], [
        "HEPA_Air_Purification",
        "HEPA_H13_Filter_Selection",
        "Activated_Carbon_Air_Filter",
        "Sorbent_Filter_For_VOCs",
        "Corsi_Rosenthal_Box_DIY_Filter",
        "HVAC_Filter_MERV_Upgrade",
        "Air_Purifier_Placement_Optimization",
        "Filter_Replacement_Protocol",
    ])

    add_leaves(env, ["Indoor_Air_Quality_and_Ventilation", "Ventilation_and_Air_Exchange"], [
        "Ventilation_Improvement",
        "Cross_Ventilation_Strategy",
        "Mechanical_Ventilation_ERV_HRV_Evaluation",
        "Window_Ventilation_Schedule_Framework",
        "Kitchen_Exhaust_Improvement",
        "Bathroom_Exhaust_Improvement",
        "Bedroom_Airflow_Optimization",
        "CO2_Monitor_For_Ventilation",
        "Air_Exchange_Rate_Assessment_Referral",
    ])

    add_leaves(env, ["Indoor_Air_Quality_and_Ventilation", "Monitoring_and_Indicators"], [
        "Air_Quality_Monitor",
        "Indoor_Particulate_Monitor",
        "VOC_Monitor",
        "Formaldehyde_Monitor",
        "Radon_Testing_Kit",
        "CO_Monitor",
        "NO2_Monitor",
        "Humidity_Meter",
        "Mold_Spore_Assessment_Referral",
    ])

    add_leaves(env, ["Indoor_Air_Quality_and_Ventilation", "Moisture_Mold_and_Allergens"], [
        "Humidity_Control",
        "Dehumidifier_Use",
        "Humidifier_Use",
        "Mold_Remediation",
        "Leak_Detection_Protocol",
        "Condensation_Reduction_Protocol",
        "Dust_Mite_Control_Protocol",
        "Pet_Dander_Control_Protocol",
        "Allergen_Control_Protocol",
        "HEPA_Vacuum_Use",
        "Wet_Dusting_Protocol",
        "Washable_Bedding_Allergen_Protocol",
    ])

    add_leaves(env, ["Indoor_Air_Quality_and_Ventilation", "Combustion_and_Smoke_Exposure_Reduction"], [
        "Smoke_Exposure_Reduction",
        "Secondhand_Smoke_Avoidance",
        "Wildfire_Smoke_Plan",
        "Indoor_Clean_Air_Room_Setup",
        "Gas_Stove_Exposure_Reduction",
        "Induction_Cooktop_Transition",
        "No_Idling_Car_Exposure_Reduction",
        "Candle_Incense_Reduction",
        "Fireplace_WoodSmoke_Reduction",
    ])

    # ----------------------------
    # Water: contaminants, plumbing, microplastics, disinfection byproducts
    # ----------------------------
    add_leaves(env, ["Drinking_Water_Quality_and_Exposure_Reduction", "Filtration_and_Treatment"], [
        "Water_Filter_Activated_Carbon",
        "Water_Filter_Reverse_Osmosis",
        "Water_Filter_UnderSink_System",
        "Whole_House_Filter_Evaluation",
        "Shower_Filter_Evaluation",
        "Filter_Maintenance_Protocol",
        "Boiling_Water_Risk_Assessment_Framework",
    ])

    add_leaves(env, ["Drinking_Water_Quality_and_Exposure_Reduction", "Testing_and_Source_Assessment"], [
        "Water_Testing_Kit_Use",
        "Municipal_Water_Report_Review",
        "Well_Water_Testing_Protocol",
        "Lead_In_Water_Test",
        "Nitrate_Test",
        "Arsenic_Test",
        "PFAS_Test_Referral",
        "Microbial_Contamination_Test_Referral",
    ])

    add_leaves(env, ["Drinking_Water_Quality_and_Exposure_Reduction", "Plumbing_and_Distribution"], [
        "Lead_Pipe_Remediation",
        "Lead_Service_Line_Replacement_Referral",
        "Faucet_Filter_Certified_Selection",
        "Flush_Stagnant_Water_Protocol",
        "Hot_Water_For_Consumption_Avoidance",
    ])

    # Microplastics: explicitly separated into sources + countermeasures
    add_leaves(env, ["Microplastics_and_Plastics_Exposure_Reduction", "Water_and_Beverage_Contact"], [
        "Microplastic_Exposure_Reduction_Water",
        "Avoid_SingleUse_Plastic_Water_Bottles",
        "Glass_Water_Bottle_Use",
        "Stainless_Steel_Bottle_Use",
        "Avoid_Plastic_Kettle_Reservoirs",
        "Avoid_Plastic_Coffee_Pod_Systems",
    ])

    add_leaves(env, ["Microplastics_and_Plastics_Exposure_Reduction", "Food_Contact_and_Kitchen"], [
        "Food_Storage_Glass_Containers",
        "Stainless_Steel_Food_Containers",
        "Avoid_Microwaving_In_Plastic",
        "Avoid_Plastic_Cutting_Boards",
        "Replace_Worn_Nonstick_Cookware",
        "Cookware_Low_Leach_Risk_Choice",
        "Silicone_Kitchenware_Quality_Selection",
    ])

    add_leaves(env, ["Microplastics_and_Plastics_Exposure_Reduction", "Textiles_and_Indoor_Dust"], [
        "Microfiber_Shedding_Reduction_Laundry",
        "Laundry_Filter_Microfiber_Capture",
        "Laundry_Bag_Microfiber_Capture",
        "Prefer_Natural_Fibers_When_Feasible",
        "HEPA_Vacuum_For_Dust_Microplastics",
        "Entryway_Shoe_Off_Policy",
    ])

    add_leaves(env, ["Microplastics_and_Plastics_Exposure_Reduction", "Personal_Care_and_Packaging"], [
        "Avoid_Microbeads_Personal_Care",
        "Prefer_Low_Packaging_Personal_Care",
        "Fragrance_Free_Personal_Care",
        "Avoid_PVC_Packaging_When_Feasible",
    ])

    # ----------------------------
    # Chemicals: endocrine disruptors, VOCs, pesticides, PFAS, flame retardants
    # ----------------------------
    add_leaves(env, ["Chemical_Exposure_Reduction", "Endocrine_Active_Plastics_and_Additives"], [
        "BPA_Exposure_Reduction",
        "BPS_BPF_Exposure_Reduction",
        "Phthalate_Exposure_Reduction",
        "PVC_Avoidance_When_Feasible",
        "Thermal_Receipt_Handling_Reduction",
        "Food_Contact_Plastic_Reduction",
    ])

    add_leaves(env, ["Chemical_Exposure_Reduction", "PFAS_and_Fluorinated_Compounds"], [
        "PFAS_Exposure_Reduction_FoodPackaging",
        "PFAS_Exposure_Reduction_Cookware",
        "PFAS_Exposure_Reduction_Textiles",
        "PFAS_Water_Filtration_Optimization",
        "Stain_Resistant_Chemical_Avoidance",
    ])

    add_leaves(env, ["Chemical_Exposure_Reduction", "Pesticides_and_Herbicides"], [
        "Pesticide_Exposure_Reduction",
        "Produce_Wash_Protocol",
        "Peel_When_Appropriate_Protocol",
        "Integrated_Pest_Management_Home",
        "Avoid_Indoor_Pesticide_Sprays",
        "Garden_Pesticide_Alternatives",
    ])

    add_leaves(env, ["Chemical_Exposure_Reduction", "VOCs_Solvents_and_Offgassing"], [
        "Solvent_Exposure_Reduction",
        "Low_VOC_Paint_Selection",
        "Low_Emission_Furniture_Selection",
        "Offgassing_Ventilation_New_Items",
        "Avoid_Scented_Candles_AirFresheners",
        "Safer_Cleaning_Products",
        "Fragrance_Free_Home",
    ])

    add_leaves(env, ["Chemical_Exposure_Reduction", "Flame_Retardants_and_Dust_Bound_Chemicals"], [
        "Flame_Retardant_Dust_Reduction",
        "HEPA_Vacuum_Use",
        "Wet_Dusting_Protocol",
        "Foam_Furniture_Cover_Intactness_Check",
        "Replace_Degrading_Foam_Products",
    ])

    add_leaves(env, ["Chemical_Exposure_Reduction", "Heavy_Metals_and_Persistent_Toxins"], [
        "Heavy_Metal_Exposure_Risk_Audit",
        "Lead_Exposure_Reduction_Home",
        "Mercury_Exposure_Reduction_FishChoice",
        "Arsenic_Exposure_Reduction_WaterFood",
        "Cadmium_Exposure_Reduction_SmokeFood",
        "Occupational_HeavyMetal_Protection",
    ])

    # ----------------------------
    # Sensory environment (non-behavioral design levers)
    # ----------------------------
    add_leaves(env, ["Sensory_Environment_Design", "Noise_and_Vibration"], [
        "Noise_Reduction_Measures",
        "Acoustic_Panel_Installation",
        "Door_Window_Sealing_For_Noise",
        "White_Noise_Device_Use",
        "Hearing_Protection",
        "Noise_Meter_Device",
    ])

    add_leaves(env, ["Sensory_Environment_Design", "Light_and_Circadian_Environment"], [
        "Light_Pollution_Reduction",
        "Blackout_Window_Treatments",
        "Warm_Dimmable_Evening_Lighting",
        "Circadian_Lighting_System",
        "Blue_Light_Reduction_Evening",
        "Daylight_Access_Optimization",
    ])

    add_leaves(env, ["Sensory_Environment_Design", "Indoor_Nature_and_Biophilic_Features"], [
        "Indoor_Plants_Low_Allergen_Selection",
        "Air_Humidification_With_Safe_Maintenance",
        "Outdoor_Air_Exposure_Safe_Window_Strategy",
    ])

    # ----------------------------
    # Thermal comfort + shelter integrity
    # ----------------------------
    add_leaves(env, ["Thermal_Comfort_and_Shelter_Integrity", "Temperature_and_Humidity_Stability"], [
        "Indoor_Temperature_Stabilization",
        "Smart_Thermostat_Control",
        "Draft_Sealing_Protocol",
        "Insulation_Upgrade_Referral",
        "Heat_Stress_Risk_Reduction_Home",
        "Cold_Exposure_Risk_Reduction_Home",
    ])

    add_leaves(env, ["Thermal_Comfort_and_Shelter_Integrity", "Indoor_Sleep_Comfort_Environment"], [
        "Bedding_Thermal_Optimization",
        "Bedroom_Thermal_Zoning",
        "Cooling_Sleep_Surface_Device",
        "Humidification_For_Airway_Comfort",
    ])

    # ----------------------------
    # Occupational / hobby exposures
    # ----------------------------
    add_leaves(env, ["Occupational_and_Hobby_Exposure_Controls", "Respiratory_and_Particulate_Protection"], [
        "Respirator_Use_Occupational",
        "Dust_Control_Worksite",
        "Ventilation_Workshop_Improvement",
        "Welding_Fume_Exposure_Reduction",
        "WoodDust_Exposure_Reduction",
    ])

    add_leaves(env, ["Occupational_and_Hobby_Exposure_Controls", "Dermal_and_Chemical_Protection"], [
        "Protective_Glove_Use_Occupational",
        "Skin_Barrier_Protection_Framework",
        "Solvent_Skin_Exposure_Reduction",
        "Safer_Chemical_Substitution_Workplace",
    ])

    add_leaves(env, ["Occupational_and_Hobby_Exposure_Controls", "Ergonomics_and_Load_Exposure"], [
        "Workstation_Ergonomic_Assessment",
        "Repetitive_Strain_Reduction",
        "Manual_Handling_Load_Reduction",
        "AntiFatigue_Mat_Use",
    ])

    # ----------------------------
    # External environment exposures (optional but often relevant)
    # ----------------------------
    add_leaves(env, ["Outdoor_Environment_and_Community_Exposures", "Air_and_Pollution_Avoidance"], [
        "AQI_Aware_Route_Planning",
        "Traffic_Exposure_Reduction_RouteChoice",
        "Wildfire_Smoke_Avoidance_Strategy",
    ])

    add_leaves(env, ["Outdoor_Environment_and_Community_Exposures", "Noise_and_Light_Community_Exposures"], [
        "Urban_Noise_Exposure_Reduction_Strategy",
        "Nighttime_Light_Exposure_Reduction_Strategy",
    ])

    BIO["Environmental_Exposures_and_Built_Environment_Optimization"] = env


    # ------------------------------------------------------------------
    # 10) Personalization, Monitoring, and Data Feedback (BIO)
    #     Subsection: Genetics and Pharmacogenomics Tests
    #     Focus: clinically actionable tests that can support optimization of
    #     (non-)clinical mental health care via safety, tolerability, and response
    #     personalization. Leaves are "test/service" or "interpretation pathway"
    #     entities (no disorder-labeled branches; no schedules).
    # ------------------------------------------------------------------
    pers: Dict[str, Any] = {}

    # ============================================================
    # A) Medication-Focused Pharmacogenomics (core clinical utility)
    # ============================================================
    add_leaves(pers, ["Genetics_and_Pharmacogenomics_Tests", "Medication_Pharmacokinetics_Genes"], [
        "Pharmacogenomics_Panel",
        "CYP2D6_Genotyping",
        "CYP2C19_Genotyping",
        "CYP2C9_Genotyping",
        "CYP1A2_Genotyping",
        "CYP3A4_Genotyping",
        "CYP3A5_Genotyping",
        "UGT1A4_Genotyping",
        "UGT2B15_Genotyping",
        "DPYD_Genotyping_Clinical_Toxicity_Screen",
    ])

    add_leaves(pers, ["Genetics_and_Pharmacogenomics_Tests", "Medication_Transporter_And_Distribution_Genes"], [
        "ABCB1_Genotyping_Pglycoprotein",
        "SLCO1B1_Genotyping_StatinRisk_Screen",
        "SLC6A4_Serotonin_Transporter_Variant_Test",
    ])

    add_leaves(pers, ["Genetics_and_Pharmacogenomics_Tests", "Medication_Pharmacodynamics_And_Target_Genes"], [
        "HTR2A_Genotyping_SerotoninReceptor",
        "HTR2C_Genotyping_SerotoninReceptor",
        "DRD2_Genotyping_DopamineReceptor",
        "COMT_Genotyping_CatecholMetabolism",
        "ADRA2A_Genotyping_AdrenergicTarget",
    ])

    # ============================================================
    # B) Serious Adverse Reaction Risk Screening (high clinical value)
    # ============================================================
    add_leaves(pers, ["Genetics_and_Pharmacogenomics_Tests", "Severe_Adverse_Reaction_Risk_Screening", "HLA_Tests"], [
        "HLA_Risk_Screening",
        "HLA_B_1502_Genotyping_SJS_TEN_Risk_Screen",
        "HLA_A_3101_Genotyping_Cutaneous_Hypersensitivity_Risk_Screen",
        "HLA_B_5701_Genotyping_Hypersensitivity_Risk_Screen",
    ])

    add_leaves(pers, ["Genetics_and_Pharmacogenomics_Tests", "Severe_Adverse_Reaction_Risk_Screening",
                      "Metabolic_And_Other_Toxicity_Screens"], [
                   "TPMT_Genotyping_Drug_Toxicity_Risk_Screen",
                   "NUDT15_Genotyping_Drug_Toxicity_Risk_Screen",
                   "G6PD_Deficiency_Genetic_Screen",
               ])

    # ============================================================
    # C) Nutrient-Related and One-Carbon Metabolism (supportive personalization)
    #     (Kept as optional optimization tools; interpret with biomarkers/clinical context.)
    # ============================================================
    add_leaves(pers, ["Genetics_and_Pharmacogenomics_Tests", "Nutrigenetic_And_Methylation_Related_Tests"], [
        "MTHFR_Genotyping",
        "MTRR_Genotyping_OneCarbonMetabolism",
        "CBS_Genotyping_Transsulfuration",
        "Folate_Pathway_Genetic_Profile_Test",
    ])

    # ============================================================
    # D) Neurodegenerative / Cognitive Risk and Long-Horizon Planning
    #     (Use for risk counseling and prevention strategy personalization.)
    # ============================================================
    add_leaves(pers, ["Genetics_and_Pharmacogenomics_Tests", "Cognition_And_LongHorizon_Risk_Tests"], [
        "APOE_Genotyping",
        "Cognitive_Risk_Genetic_Counseling_Referral_Pathway",
    ])

    # ============================================================
    # E) Broader Psychiatric-Relevant Risk/Response Adjuncts (optional / context-dependent)
    #     (Avoids disorder-labeled branches; focuses on actionable interpretation.)
    # ============================================================
    add_leaves(pers, ["Genetics_and_Pharmacogenomics_Tests", "Adjunct_Response_Modifier_Tests"], [
        "BDNF_Variant_Test_Response_Modifier",
        "FKBP5_Stress_Response_Modifier_Test",
        "Inflammation_Response_Modifier_Genetic_Profile_Test",
    ])

    # ============================================================
    # F) Reporting, Interpretation, and Clinical Integration Pathways
    #     (Makes the tests usable; keeps outcomes tied to decision-support.)
    # ============================================================
    add_leaves(pers, ["Genetics_and_Pharmacogenomics_Tests", "Interpretation_And_Clinical_Integration"], [
        "Pharmacogenomics_Report_Clinical_Interpretation_Session",
        "Medication_Dosing_Adjustment_Recommendation_From_PGx_Report",
        "Medication_Selection_Support_From_PGx_Report",
        "Polypharmacy_Pharmacogenomics_Interaction_Review",
        "SideEffect_Risk_Mitigation_Plan_From_Genetic_Findings",
        "Genetic_Test_Informed_Consent_And_Limitations_Counseling",
        "Genetic_Privacy_And_Data_Governance_Consultation",
        "Genetic_Counseling_Referral_Pathway",
        "Reanalysis_And_Update_Check_For_Pharmacogenomics_Report",
    ])

    # Optional: keep legacy names for backward compatibility with earlier ontology references
    add_leaves(pers, ["Genetics_and_Pharmacogenomics_Tests", "Legacy_Aliases"], [
        "CYP_Genotyping",
    ])

    # (Caller will attach pers to BIO["Personalization_Monitoring_and_Data_Feedback"] elsewhere in the script.)

    # Wearables + home monitoring (physiology-focused; high-resolution, solution-variable framing)
    # Goal: broaden sensing modalities for sleep, autonomic, cardio-respiratory, metabolic, thermoregulation,
    #       activity/biomechanics, and safety—without encoding schedules/durations.

    # ----------------------------
    # 10) Personalization & Monitoring
    # ----------------------------
    pers: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Wearables_and_Home_Monitoring → Physiology (expanded, hierarchical)
    # ------------------------------------------------------------------

    # 1) Sleep, circadian, and recovery physiology
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Sleep_and_Circadian", "Wearables"], [
        "Sleep_Tracker_Ring",
        "Sleep_Tracker_Wrist_Wearable",
        "Sleep_Tracker_Armband",
        "Sleep_Tracker_Chest_Strap",
        "EEG_Sleep_Headband_Consumer",
        "Actigraphy_Wearable",
    ])
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Sleep_and_Circadian", "Bed_And_Room_Sensors"], [
        "Under_Mattress_Ballisto_Sensor",
        "Bedside_Radar_Sleep_Sensor",
        "Smart_Mattress_Sensor_System",
        "Snore_And_Sleep_Sound_Monitor",
        "Bedroom_Light_Sensor",
        "Bedroom_Noise_Meter",
        "Bedroom_Temperature_Humidity_Sensor",
        "Bedroom_CO2_Monitor",
        "Indoor_Air_Quality_Monitor_PM25_VOC",
    ])
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Sleep_and_Circadian", "Circadian_Exposure_Monitoring"], [
        "Light_Exposure_Dosimeter",
        "Blue_Light_Exposure_Meter",
        "Daylight_Exposure_Tracker",
    ])

    # 2) Autonomic and cardiovascular monitoring
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Autonomic_and_Cardiovascular", "Heart_Rate_and_HRV"], [
        "HRV_Tracker_Ring",
        "HRV_Tracker_Wrist_Wearable",
        "Chest_Strap_ECG_HRV_Device",
        "Photoplethysmography_HRV_Wearable",
        "HRV_Biofeedback_Device",
    ])
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Autonomic_and_Cardiovascular", "Blood_Pressure"], [
        "Blood_Pressure_Home_Monitor_UpperArm_Cuff",
        "Blood_Pressure_Home_Monitor_Wrist_Cuff",
        "Cuffless_Blood_Pressure_Estimation_Wearable",
        "Ambulatory_Blood_Pressure_Monitor_Referral",
    ])
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Autonomic_and_Cardiovascular", "ECG_and_Arrhythmia_Screening"], [
        "Single_Lead_ECG_Wearable",
        "Multi_Lead_ECG_Patch_Monitor",
        "ECG_Chest_Strap_Device",
        "Event_Monitor_Referral",
        "Holter_Monitor_Referral",
    ])
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Autonomic_and_Cardiovascular", "Vascular_and_Perfusion"], [
        "Peripheral_Perfusion_Index_Monitor",
        "Digital_Vascular_Assessment_Device_Referral",
        "Arterial_Stiffness_Pulse_Wave_Velocity_Device_Consumer",
    ])

    # 3) Respiratory and oxygenation monitoring
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Respiratory_and_Oxygenation", "Oxygenation"], [
        "Pulse_Oximeter_Fingertip",
        "Pulse_Oximeter_Wrist_Wearable",
        "Overnight_Oximetry_Device",
    ])
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Respiratory_and_Oxygenation", "Breathing_And_Ventilation"], [
        "Respiratory_Rate_Wearable",
        "Capnography_Monitoring_Device_Referral",
        "Breath_Sensor_Chest_Band",
        "Smart_Inhaler_Tracker",
        "Peak_Flow_Meter",
        "Home_Spirometry_Device",
    ])
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Respiratory_and_Oxygenation", "Sleep_Disordered_Breathing_Screening"], [
        "Snore_And_Breathing_Pattern_Monitor",
        "Home_Sleep_Testing_Device_Referral",
        "Positional_Sleep_Monitor",
    ])

    # 4) Metabolic monitoring (glucose, ketones, lipids proxies)
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Metabolic", "Glucose"], [
        "Continuous_Glucose_Monitoring_Device",
        "Flash_Glucose_Monitoring_Device",
        "Fingerstick_Glucometer",
    ])
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Metabolic", "Ketones_And_Metabolic_Flexibility"], [
        "Blood_Ketone_Meter",
        "Breath_Ketone_Sensor",
    ])
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Metabolic", "Body_Composition_And_Weight"], [
        "Smart_Scale",
        "Bioimpedance_Scale_Consumer",
        "Body_Composition_Analyzer_Handheld_BIA",
        "Skinfold_Caliper_Tool",
        "Tape_Measure_Anthropometrics_Tool",
    ])

    # 5) Temperature, thermoregulation, and illness signals
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Thermoregulation_And_Inflammatory_Signals", "Wearables"], [
        "Skin_Temperature_Sensor_Wearable",
        "Core_Temperature_Estimation_Wearable",
        "Thermal_Ring_Sensor",
    ])
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Thermoregulation_And_Inflammatory_Signals", "Home_Tools"], [
        "Digital_Thermometer_Oral",
        "Digital_Thermometer_Tympanic",
        "Digital_Thermometer_Forehead",
    ])

    # 6) Activity, fitness physiology, and biomechanics (still physiology-anchored)
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Activity_Fitness_And_Biomechanics", "Activity_and_Energy"], [
        "Activity_Tracker_Wearable",
        "Step_Counting_Wearable",
        "Energy_Expenditure_Estimation_Wearable",
        "Stair_Climb_And_Elevation_Tracker",
    ])
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Activity_Fitness_And_Biomechanics", "Cardiorespiratory_Fitness_Proxies"], [
        "VO2max_Estimation_Wearable",
        "Heart_Rate_Training_Zone_Device",
        "Recovery_Readiness_Score_Wearable",
    ])
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Activity_Fitness_And_Biomechanics", "Gait_Posture_And_Movement_Quality"], [
        "Gait_Analysis_Insole_Sensors",
        "Pressure_Sensing_Insoles",
        "Running_Pod_Biomechanics_Sensor",
        "Posture_Tracking_Wearable",
        "Tremor_Motion_Sensor_Consumer",
    ])

    # 7) Nervous system, stress physiology, and arousal proxies
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Neurophysiology_And_Stress_Proxies", "Electrodermal_And_Sympathetic"], [
        "Electrodermal_Activity_Wearable",
        "Skin_Conductance_Sensor",
    ])
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Neurophysiology_And_Stress_Proxies", "EEG_And_Cognition_Adjacent"], [
        "EEG_Headband_Consumer",
        "Neurofeedback_Headset_Consumer",
    ])
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Neurophysiology_And_Stress_Proxies", "Muscle_Tension"], [
        "EMG_Biofeedback_Device_Consumer",
        "Jaw_Clench_Bruxism_Sensor",
    ])

    # 8) Home lab / point-of-care style monitoring (physiology)
    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Physiology", "Home_PointOfCare_Measurements"], [
        "Blood_Pressure_Home_Monitor",
        "Pulse_Oximeter",
        "Fingerstick_Glucometer",
        "Blood_Ketone_Meter",
        "At_Home_CRP_Test_Referral",
        "At_Home_Thyroid_Test_Referral",
        "At_Home_Vitamin_D_Test_Referral",
        "At_Home_Iron_Ferritin_Test_Referral",
    ])


    # (You can keep/replace your original key with this expanded subtree.)
    # BIO["Personalization_and_Monitoring"] = pers


    add_leaves(pers, ["Wearables_and_Home_Monitoring", "Environment"], [
        "Air_Quality_Monitor",
        "Indoor_Particulate_Monitor",
        "Room_Humidity_Meter",
        "Bedroom_CO2_Monitor",
        "Noise_Meter_Device",
        "Light_Sensor_Device",
    ])

    # ------------------------------------------------------------------
    # Self-Assessment, Logs, and Ecological Momentary Assessment (EMA)
    # High-resolution, self-report–based monitoring of internal states,
    # behaviors, context, and subjective experience (non-device).
    # ------------------------------------------------------------------

    add_leaves(pers, ["Self_Assessment_and_Logs", "Daily_and_Retrospective_Logs"], [
        "Sleep_Diary",
        "Sleep_Quality_Rating",
        "Sleep_Timing_Log",
        "Food_Diary",
        "Meal_Timing_Log",
        "Hydration_Log",
        "Caffeine_Alcohol_Intake_Log",
        "Activity_Log",
        "Exercise_Type_and_Perceived_Exertion_Log",
        "Medication_Adherence_Log",
        "Medication_SideEffect_Log",
        "Supplement_Intake_Log",
        "Symptom_Checklist_SelfReport",
        "Pain_Intensity_And_Interference_Log",
    ])

    add_leaves(pers, ["Self_Assessment_and_Logs", "Affect_Mood_and_Stress_SelfReports"], [
        "Mood_Diary",
        "Affect_Valence_Arousal_Rating",
        "Emotion_Labeling_Log",
        "Emotion_Intensity_Rating",
        "Stress_Scale_SelfReport",
        "Perceived_Stress_Level_Rating",
        "Burnout_SelfRating",
        "Irritability_Anger_Log",
        "Anxiety_Arousal_Rating",
        "Calmness_Relaxation_Rating",
    ])

    add_leaves(pers, ["Self_Assessment_and_Logs", "Cognition_and_Energy_SelfReports"], [
        "Mental_Energy_Rating",
        "Fatigue_Severity_SelfReport",
        "Brain_Fog_Severity_Log",
        "Attention_Quality_Rating",
        "Cognitive_Effort_Rating",
        "Motivation_Level_SelfReport",
        "Productivity_SelfRating",
        "Decision_Fatigue_SelfReport",
    ])

    add_leaves(pers, ["Self_Assessment_and_Logs", "Somatic_and_Interoceptive_SelfReports"], [
        "Body_Sensation_Log",
        "Arousal_Sensation_Log",
        "Interoceptive_Awareness_Log",
        "GI_Symptom_Log",
        "Headache_Migraine_Log",
        "Sleepiness_Drowsiness_Scale",
        "Appetite_And_Satiety_Log",
    ])

    add_leaves(pers, ["Self_Assessment_and_Logs", "Context_and_Behavioral_Context_Logs"], [
        "Location_Context_Log",
        "Social_Context_Log",
        "Workload_And_Demand_Log",
        "Screen_Time_SelfReport",
        "Posture_And_Sedentary_Time_Log",
        "Environmental_Stressors_Log",
        "Illness_Symptom_Onset_Log",
    ])

    # ----------------------------
    # Ecological Momentary Assessment (EMA)
    # ----------------------------
    add_leaves(pers, ["Self_Assessment_and_Logs", "Ecological_Momentary_Assessment_EMA", "Momentary_Affect_and_Stress"], [
        "EMA_Mood_CheckIn",
        "EMA_Stress_Level",
        "EMA_Anxiety_Arousal",
        "EMA_Calmness_Safety_Signal",
        "EMA_Irritability_Check",
    ])

    add_leaves(pers, ["Self_Assessment_and_Logs", "Ecological_Momentary_Assessment_EMA", "Momentary_Cognition_and_Energy"], [
        "EMA_Mental_Energy",
        "EMA_Fatigue",
        "EMA_Attention_Quality",
        "EMA_Cognitive_Load",
        "EMA_Motivation_Level",
    ])

    add_leaves(pers, ["Self_Assessment_and_Logs", "Ecological_Momentary_Assessment_EMA", "Momentary_Somatic_and_Physiological_Perception"], [
        "EMA_Body_Tension",
        "EMA_Breathing_Ease",
        "EMA_Heart_Pounding_Perception",
        "EMA_Temperature_Sensation",
        "EMA_Pain_Level",
    ])

    add_leaves(pers, ["Self_Assessment_and_Logs", "Ecological_Momentary_Assessment_EMA", "Momentary_Context_and_Triggers"], [
        "EMA_Current_Activity",
        "EMA_Social_Context",
        "EMA_Environmental_Context",
        "EMA_Trigger_Exposure",
        "EMA_Recent_Stress_Event",
    ])

    add_leaves(pers, ["Self_Assessment_and_Logs", "Ecological_Momentary_Assessment_EMA", "Momentary_Coping_and_Response"], [
        "EMA_Coping_Strategy_Used",
        "EMA_Perceived_Coping_Effectiveness",
        "EMA_Urge_Intensity",
        "EMA_Impulse_Control",
        "EMA_Recovery_State",
    ])

    add_leaves(pers, ["Self_Assessment_and_Logs", "Structured_Scales_and_Questionnaires"], [
        "Standardized_Mood_Scale_SelfReport",
        "Standardized_Stress_Scale_SelfReport",
        "Sleep_Questionnaire_SelfReport",
        "Fatigue_Questionnaire_SelfReport",
        "Quality_Of_Life_SelfReport",
        "Functioning_And_Disability_SelfReport",
    ])


    add_leaves(pers, ["Personalization_Workflows"], [
        "Baseline_Assessment_Workup_Framework",
        "Iterative_Test_And_Adjust_Framework",
        "Clinician_Shared_Decision_Making_Framework",
        "Risk_Benefit_Review_Framework",
        "Adherence_Barrier_Review_Framework",
    ])

    BIO["Personalization_Monitoring_and_Data_Feedback"] = pers

    return {"BIO": BIO}


# ======================== Validation ========================

def validate_ontology(ontology: Dict[str, Any]) -> None:
    """
    Guardrails:
    - Reject explicit schedule/frequency/duration encodings as taxonomy nodes.
    - Avoid disorder-labeled anchors.
    - Avoid parameter words that would turn leaf nodes into dosing/protocol nodes.
    - Enforce stable key formatting.
    """
    forbidden_patterns = [
        # frequency/duration tokens
        r"\b\d+\s*x\s*week\b",
        r"\b\d+\s*times\s*per\s*week\b",
        r"\bper[_\s-]*week\b",
        r"\bweekly\b",
        r"\bdaily\b",
        r"\bper[_\s-]*day\b",
        r"\bmonthly\b",
        r"\byearly\b",
        r"\b\d+\s*min\b",
        r"\b\d+\s*minute(s)?\b",
        r"\b\d+\s*hour(s)?\b",
        r"\b\d+\s*day(s)?\b",
        r"\b\d+\s*week(s)?\b",
        r"\b\d+\s*month(s)?\b",
        r"\b\d+\s*year(s)?\b",
        r"\b\d+\s*hz\b",
        r"\bfrequency\b",
        r"\bduration\b",
        r"\bintensity\b",
        # common shorthand tokens
        r"\bhz\b",
        r"\bmin\b",
        r"\bsec\b",
        r"\bsecond(s)?\b",
    ]

    # Guardrail: avoid disorder-labeled branches (non-exhaustive, common tokens)
    disorder_terms = [
        r"\bdepress(ion|ive)\b",
        r"\banxiety\b",
        r"\bbipolar\b",
        r"\bschizo(phrenia)?\b",
        r"\bptsd\b",
        r"\bocd\b",
        r"\badhd\b",
        r"\bautis(m|tic)\b",
        r"\bpanic\b",
        r"\bpsychosis\b",
        r"\beating\s*disorder\b",
        r"\banorex(ia)?\b",
        r"\bbulimi(a)?\b",
        r"\bsubstance\s*use\s*disorder\b",
    ]

    all_paths = list(iter_paths(ontology["BIO"]))

    bad_sched = scan_forbidden_tokens(all_paths, forbidden_patterns, limit=10)
    if bad_sched:
        p, pat = bad_sched[0]
        raise ValueError(
            "Forbidden schedule/parameter token detected. "
            f"pattern={pat!r} in path={' / '.join(p)}"
        )

    bad_dis = scan_forbidden_tokens(all_paths, disorder_terms, limit=10)
    if bad_dis:
        p, pat = bad_dis[0]
        raise ValueError(
            "Disorder-labeled token detected (taxonomy should be diagnosis-agnostic). "
            f"pattern={pat!r} in path={' / '.join(p)}"
        )

    bad_format = validate_key_format(all_paths, limit=10)
    if bad_format:
        p, why = bad_format[0]
        raise ValueError(
            "Key formatting constraint violated (use alphanumerics + underscore only). "
            f"reason={why!r} in path={' / '.join(p)}"
        )


# ======================== Writer + metadata ========================

def write_outputs(ontology: Dict[str, Any], out_json_path: str) -> Tuple[str, str, Dict[str, Any]]:
    out_json_path = os.path.expanduser(out_json_path)
    out_dir = os.path.dirname(out_json_path) if os.path.dirname(out_json_path) else "."

    # If path is not viable, fall back to a local BIO/ folder next to this script
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        out_dir = os.path.join(os.path.dirname(__file__), "BIO")
        os.makedirs(out_dir, exist_ok=True)
        out_json_path = os.path.join(out_dir, "BIO.json")

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(ontology, f, ensure_ascii=False, indent=2)

    leaf_paths = list(iter_leaf_paths(ontology["BIO"]))
    leaf_count = count_leaves(ontology["BIO"])
    node_count = count_nodes(ontology["BIO"])
    depth = max_depth(ontology["BIO"])
    top_counts = subtree_leaf_counts(ontology["BIO"])
    depths = [len(p) for p in leaf_paths]

    # Hash minified JSON for versioning
    minified = json.dumps(ontology, ensure_ascii=False, separators=(",", ":"))
    sha256 = hashlib.sha256(minified.encode("utf-8")).hexdigest()

    metadata: Dict[str, Any] = {
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


def main() -> None:
    # Default is portable; override via BIO_OUT_PATH env var.
    default_out = os.path.join(os.getcwd(), "BIO", "BIO.json")
    out_json_path = os.environ.get("BIO_OUT_PATH", default_out)

    ontology = build_bio_ontology()
    validate_ontology(ontology)

    out_json_path, meta_path, metadata = write_outputs(ontology, out_json_path)

    print("Leaf nodes:", metadata["leaf_count"])
    print("Wrote JSON:", out_json_path)
    print("Wrote metadata:", meta_path)


if __name__ == "__main__":
    main()

# TODO:
# - Expand Food_Library as a separate optional block (kept smaller here for clarity/size).
# - Add more clinician-facing referral pathways (cardiology, endocrinology, sleep medicine) as solution nodes.
# - Add safety cross-check nodes (contraindication review workflows) without encoding dosing parameters.
