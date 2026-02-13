"""
01_hierarchical_predictor_evaluation_modules.py

Hierarchical evaluation schema for ranking ontology leaf-nodes ("predictors" / solution candidates)
on their suitability for use in (1) multivariate time-series observation models and (2) translation
into actionable treatment targets within a proximal zone of responsibility.

This file is intentionally analogous to `00_hierarchical_evaluation_modules.py` (criteria evaluation),
but adapted for predictors/solution candidates.

Key differences vs. the Criterion schema:
- The *LLM-facing evaluation models contain scores only* (no weights embedded in the nested Pydantic output).
- Default weights are defined as class-level constants / helpers and are only used for scoring utilities.
- Top-level model is a "septette": {metadata + 6 dimension modules}.

Design principles
- All leaf-level features use a 9-point Likert scale (1=very low problem likelihood, 9=very high problem likelihood).
- Overall suitability is computed as a weighted average of "suitability" components (1 - normalized risk).
- This file contains schema + scoring utilities only (no LLM logic).

Pydantic version: v2.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, conint, confloat, model_validator


# -----------------------------------------------------------------------------
# Core scalar types
# -----------------------------------------------------------------------------

Likert9 = conint(ge=1, le=9)

EPS = 1e-9


def _normalize_likert_to_unit_interval(score: int) -> float:
    """Map Likert9 (1..9) -> [0,1] where 1->0 risk and 9->1 risk."""
    return (float(score) - 1.0) / 8.0


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Return a copy of weights normalized to sum to 1 (raises if sum is ~0)."""
    total = float(sum(weights.values()))
    if total <= EPS:
        raise ValueError("Cannot normalize weights: sum is 0 (or ~0).")
    return {k: float(v) / total for k, v in weights.items()}


def _weighted_risk_from_scores(scores: Dict[str, int], weights: Dict[str, float]) -> float:
    """Compute a weighted risk in [0,1] from Likert9 scores and a weight dict."""
    if set(scores.keys()) != set(weights.keys()):
        raise ValueError(f"Scores/weights field mismatch: {set(scores.keys()) ^ set(weights.keys())}")

    w = _normalize_weights(weights)
    risk = 0.0
    for k, s in scores.items():
        risk += w[k] * _normalize_likert_to_unit_interval(int(s))
    return max(0.0, min(1.0, float(risk)))


# -----------------------------------------------------------------------------
# Base models
# -----------------------------------------------------------------------------

class StrictBaseModel(BaseModel):
    """Base model with strict schema validation."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ScoresBase(StrictBaseModel):
    """Base for score sets (Likert9)."""

    def as_dict(self) -> Dict[str, int]:
        return {k: int(v) for k, v in self.model_dump().items()}


class ScoredDimension(StrictBaseModel):
    """Dimension wrapper that contains scores only (weights are defined outside the LLM schema)."""

    scores: ScoresBase

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        raise NotImplementedError

    def weighted_risk(self, weights: Optional[Dict[str, float]] = None) -> float:
        w = weights if weights is not None else self.default_weights()
        return _weighted_risk_from_scores(self.scores.as_dict(), w)

    def suitability(self, weights: Optional[Dict[str, float]] = None) -> float:
        return 1.0 - self.weighted_risk(weights=weights)


# -----------------------------------------------------------------------------
# Dimension 1: Mathematical suitability (time-series analysis)
# -----------------------------------------------------------------------------
# Framed as risks/problems for treating a predictor as a time-varying exogenous/input variable
# in dynamic multivariate models (e.g., time-varying gVAR, state-space, dynamic SEM).

class MathematicalSuitabilityScores(ScoresBase):
    system_variability: Likert9 = Field(
        ...,
        description="Insufficient within-person fluctuation range; low variability limits predictive value.",
    )
    temporal_resolution_mismatch: Likert9 = Field(
        ...,
        description="Mismatch between feasible sampling rate and the predictor's dynamics (too fast/slow).",
    )
    latency_unknown_or_variable: Likert9 = Field(
        ...,
        description="Effect latency to target states is unknown/variable, complicating lag selection.",
    )
    non_stationarity: Likert9 = Field(
        ...,
        description="Strong trends/regime shifts likely; harms stability and interpretability.",
    )
    missing_data_pattern: Likert9 = Field(
        ...,
        description="Missingness/gap distribution likely problematic (MNAR, long gaps, bursty missingness).",
    )
    outlier_sensitivity: Likert9 = Field(
        ...,
        description="Expected sensitivity to extreme values; outliers likely to destabilize estimation.",
    )
    measurement_reactivity: Likert9 = Field(
        ...,
        description="Measuring the predictor likely changes it (reactivity), biasing dynamics.",
    )
    multicollinearity_with_other_predictors: Likert9 = Field(
        ...,
        description="High redundancy with other predictors; inflates variance and harms identifiability.",
    )
    temporal_precedence_ambiguity: Likert9 = Field(
        ...,
        description="Unclear temporal precedence vs. criteria; high risk the predictor is a consequence.",
    )
    controllability_confounding: Likert9 = Field(
        ...,
        description="Strong confounding with contextual factors (seasonality, routine, environment) likely.",
    )


class MathematicalSuitability(ScoredDimension):
    scores: MathematicalSuitabilityScores

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        return {
            "system_variability": 0.14,
            "temporal_resolution_mismatch": 0.10,
            "latency_unknown_or_variable": 0.10,
            "non_stationarity": 0.09,
            "missing_data_pattern": 0.10,
            "outlier_sensitivity": 0.06,
            "measurement_reactivity": 0.08,
            "multicollinearity_with_other_predictors": 0.10,
            "temporal_precedence_ambiguity": 0.13,
            "controllability_confounding": 0.10,
        }


# -----------------------------------------------------------------------------
# Dimension 2: Data collection feasibility (operational) across methods
# -----------------------------------------------------------------------------

class CollectionMethod(str, Enum):
    SELF_REPORT_EMA = "self_report_ema"
    THIRD_PARTY_EMA = "third_party_ema"
    WEARABLE = "wearable"
    USER_DEVICE_DATA = "user_device_data"
    ETL_PIPELINE = "etl_pipeline"
    THIRD_PARTY_API = "third_party_api"


class MethodFeasibilityScores(ScoresBase):
    accessibility: Likert9 = Field(
        ...,
        description="Practical accessibility: can the target population realistically provide this data source?",
    )
    participant_burden: Likert9 = Field(
        ...,
        description="Expected burden (time/effort/discomfort) leading to non-adherence or dropout.",
    )
    coverage_granularity: Likert9 = Field(
        ...,
        description="Insufficient temporal coverage or granularity for meaningful time-series modeling.",
    )
    data_quality_noise: Likert9 = Field(
        ...,
        description="Expected noise/artifacts compromising signal quality (measurement error, sensor noise, rater drift).",
    )
    missingness_risk: Likert9 = Field(
        ...,
        description="Risk of missing data due to device failure, non-wear, non-response, API downtime, etc.",
    )
    integration_complexity: Likert9 = Field(
        ...,
        description="Integration/engineering complexity (ETL, harmonization, maintenance) likely to be high.",
    )
    scalability_cost: Likert9 = Field(
        ...,
        description="Scalability and cost barriers (licenses, hardware, compute, annotation) likely to be high.",
    )


class CollectionMethodFeasibility(ScoredDimension):
    scores: MethodFeasibilityScores

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        return {
            "accessibility": 0.16,
            "participant_burden": 0.18,
            "coverage_granularity": 0.17,
            "data_quality_noise": 0.16,
            "missingness_risk": 0.14,
            "integration_complexity": 0.10,
            "scalability_cost": 0.09,
        }


class DataCollectionAggregation(str, Enum):
    BEST_AVAILABLE = "best_available"
    WEIGHTED_MEAN = "weighted_mean"


class DataCollectionFeasibility(StrictBaseModel):
    """Operational feasibility across different collection methods (scores only)."""

    aggregation: DataCollectionAggregation = Field(
        ...,
        description="Aggregation strategy over available methods: best_available or weighted_mean."
    )

    # Optional method-specific feasibility modules.
    self_report_ema: Optional[CollectionMethodFeasibility] = None
    third_party_ema: Optional[CollectionMethodFeasibility] = None
    wearable: Optional[CollectionMethodFeasibility] = None
    user_device_data: Optional[CollectionMethodFeasibility] = None
    etl_pipeline: Optional[CollectionMethodFeasibility] = None
    third_party_api: Optional[CollectionMethodFeasibility] = None

    @classmethod
    def default_method_weights(cls) -> Dict[str, float]:
        # Weights over methods for WEIGHTED_MEAN aggregation (renormalized over available methods).
        return {
            "self_report_ema": 0.30,
            "third_party_ema": 0.05,
            "wearable": 0.25,
            "user_device_data": 0.15,
            "etl_pipeline": 0.10,
            "third_party_api": 0.15,
        }

    def _available_methods(self) -> List[Tuple[str, CollectionMethodFeasibility]]:
        items: List[Tuple[str, CollectionMethodFeasibility]] = []
        for k in [
            "self_report_ema",
            "third_party_ema",
            "wearable",
            "user_device_data",
            "etl_pipeline",
            "third_party_api",
        ]:
            v = getattr(self, k)
            if v is not None:
                items.append((k, v))
        return items

    def weighted_risk(
        self,
        method_weights: Optional[Dict[str, float]] = None,
        per_method_score_weights: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> float:
        available = self._available_methods()
        if not available:
            raise ValueError("No data-collection methods were provided; cannot compute feasibility.")

        # Compute each method risk (allow per-method override of score weights if provided).
        risks: Dict[str, float] = {}
        for name, dim in available:
            override = per_method_score_weights.get(name) if per_method_score_weights else None
            risks[name] = dim.weighted_risk(weights=override)

        if self.aggregation == DataCollectionAggregation.BEST_AVAILABLE:
            return float(min(risks.values()))

        # Weighted mean over the available methods, renormalizing weights to 1.
        w_all = method_weights if method_weights is not None else self.default_method_weights()
        w_all = _normalize_weights(w_all)

        total_w = sum(w_all[name] for name in risks.keys())
        if total_w <= EPS:
            raise ValueError("Sum of weights for available methods is 0; cannot compute weighted mean.")

        risk = 0.0
        for name, r in risks.items():
            risk += (w_all[name] / total_w) * r
        return max(0.0, min(1.0, float(risk)))

    def suitability(self, **kwargs) -> float:
        return 1.0 - self.weighted_risk(**kwargs)


# -----------------------------------------------------------------------------
# Dimension 3: Validity threats (measurement/bias) for predictors
# -----------------------------------------------------------------------------

class ResponseBiasRiskScores(ScoresBase):
    social_desirability_bias: Likert9 = Field(..., description="Risk of socially desirable responding (if self/other report).")
    demand_characteristics: Likert9 = Field(..., description="Participants infer aims and alter reporting/behavior.")
    recall_bias: Likert9 = Field(..., description="Systematic recall errors (even in short windows).")
    anchoring_bias: Likert9 = Field(..., description="Anchoring on prior ratings/expectations.")
    temporal_aggregation_bias: Likert9 = Field(..., description="Bias from summarizing over windows (peak-end, etc.).")
    rater_drift: Likert9 = Field(..., description="Third-party ratings drift over time / context.")


class ResponseBiasRisk(ScoredDimension):
    scores: ResponseBiasRiskScores

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        return {
            "social_desirability_bias": 0.20,
            "demand_characteristics": 0.18,
            "recall_bias": 0.22,
            "anchoring_bias": 0.14,
            "temporal_aggregation_bias": 0.16,
            "rater_drift": 0.10,
        }


class InsightReportingCapacityRiskScores(ScoresBase):
    lack_of_insight: Likert9 = Field(..., description="Limited insight into the predictor (poor self-monitoring).")
    alexithymia_interoceptive_limits: Likert9 = Field(..., description="Difficulty identifying bodily/affective signals.")
    dissociation_confusional_states: Likert9 = Field(..., description="States limiting consistent reporting.")
    resistance_noncooperation: Likert9 = Field(..., description="Resistance/noncooperation distorts measurement.")
    cognitive_load_limits: Likert9 = Field(..., description="Cognitive limitations make frequent/complex reporting unreliable.")


class InsightReportingCapacityRisk(ScoredDimension):
    scores: InsightReportingCapacityRiskScores

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        return {
            "lack_of_insight": 0.28,
            "alexithymia_interoceptive_limits": 0.22,
            "dissociation_confusional_states": 0.18,
            "resistance_noncooperation": 0.17,
            "cognitive_load_limits": 0.15,
        }


class MeasurementValidityRiskScores(ScoresBase):
    construct_ambiguity: Likert9 = Field(..., description="Predictor construct ambiguous/ill-defined; high measurement error.")
    context_dependence: Likert9 = Field(..., description="Strong context dependence reduces comparability across time/contexts.")
    sensor_artifact_risk: Likert9 = Field(..., description="If sensor/ETL-derived: high artifact risk (device, OS, environment).")
    common_method_variance: Likert9 = Field(..., description="Mono-method bias likely inflates associations.")
    confounding_risk: Likert9 = Field(..., description="High confounding by unobserved variables expected.")
    label_drift: Likert9 = Field(..., description="Definition/measurement likely to drift over time (updates, re-interpretation).")


class MeasurementValidityRisk(ScoredDimension):
    scores: MeasurementValidityRiskScores

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        return {
            "construct_ambiguity": 0.18,
            "context_dependence": 0.16,
            "sensor_artifact_risk": 0.18,
            "common_method_variance": 0.12,
            "confounding_risk": 0.22,
            "label_drift": 0.14,
        }


class ValidityThreatWeights(StrictBaseModel):
    """Internal weights for aggregating validity submodules (not part of LLM-facing schema)."""
    response_bias: confloat(ge=0.0, le=1.0) = 0.40
    insight_capacity: confloat(ge=0.0, le=1.0) = 0.30
    measurement_validity: confloat(ge=0.0, le=1.0) = 0.30

    @model_validator(mode="after")
    def _norm(self) -> "ValidityThreatWeights":
        d = _normalize_weights(self.model_dump())
        for k, v in d.items():
            object.__setattr__(self, k, v)
        return self

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.model_dump().items()}


class ValidityThreats(StrictBaseModel):
    response_bias: Optional[ResponseBiasRisk] = None
    insight_capacity: Optional[InsightReportingCapacityRisk] = None
    measurement_validity: Optional[MeasurementValidityRisk] = None

    def weighted_risk(
        self,
        module_weights: Optional[Dict[str, float]] = None,
        per_module_score_weights: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> float:
        available: List[Tuple[str, ScoredDimension]] = []
        for name in ["response_bias", "insight_capacity", "measurement_validity"]:
            dim = getattr(self, name)
            if dim is not None:
                available.append((name, dim))

        if not available:
            raise ValueError("No validity threat modules provided; cannot compute validity risk.")

        w_all = module_weights if module_weights is not None else ValidityThreatWeights().as_dict()
        total_w = sum(w_all[name] for name, _ in available)
        if total_w <= EPS:
            raise ValueError("Sum of weights for available validity modules is 0; cannot compute.")

        risk = 0.0
        for name, dim in available:
            override = per_module_score_weights.get(name) if per_module_score_weights else None
            risk += (w_all[name] / total_w) * dim.weighted_risk(weights=override)
        return max(0.0, min(1.0, float(risk)))

    def suitability(self, **kwargs) -> float:
        return 1.0 - self.weighted_risk(**kwargs)


# -----------------------------------------------------------------------------
# Dimension 4: Treatment Translation (TT) potential within proximal responsibility
# -----------------------------------------------------------------------------
# Assesses whether a predictor is likely to translate into an actionable target for the client/provider,
# considering controllability, safety, and practical constraints.

class TreatmentTranslationScores(ScoresBase):
    proximal_controllability_low: Likert9 = Field(
        ...,
        description="Low controllability within the client's/provider's proximal zone of responsibility.",
    )
    intervention_availability_low: Likert9 = Field(
        ...,
        description="Few/no feasible interventions exist to modify this predictor in real-world settings.",
    )
    intervention_effect_uncertain: Likert9 = Field(
        ...,
        description="High uncertainty about whether changing predictor leads to meaningful state change.",
    )
    time_to_effect_too_long: Likert9 = Field(
        ...,
        description="Likely time-to-effect is too long for adaptive interventions or timely feedback loops.",
    )
    adherence_feasibility_low: Likert9 = Field(
        ...,
        description="Interventions required are hard to adhere to (complex, effortful, high drop-out).",
    )
    acceptability_low: Likert9 = Field(
        ...,
        description="Low acceptability for clients/stakeholders (stigma, discomfort, cultural mismatch).",
    )
    safety_risk_high: Likert9 = Field(
        ...,
        description="Meaningful risk of iatrogenic harm or adverse events if targeted.",
    )
    equity_access_barriers_high: Likert9 = Field(
        ...,
        description="High barriers due to cost, access, disability, language, geography, or digital divide.",
    )


class TreatmentTranslationPotential(ScoredDimension):
    scores: TreatmentTranslationScores

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        return {
            "proximal_controllability_low": 0.20,
            "intervention_availability_low": 0.16,
            "intervention_effect_uncertain": 0.16,
            "time_to_effect_too_long": 0.10,
            "adherence_feasibility_low": 0.12,
            "acceptability_low": 0.10,
            "safety_risk_high": 0.10,
            "equity_access_barriers_high": 0.06,
        }


# -----------------------------------------------------------------------------
# Dimension 5: Juridical / regulatory (EU-oriented) risk
# -----------------------------------------------------------------------------

class GDPRComplianceRiskScores(ScoresBase):
    special_category_data: Likert9 = Field(
        ..., description="Processing likely involves special category data (health/biometrics) increasing burden."
    )
    lawful_basis_complexity: Likert9 = Field(
        ..., description="Obtaining/maintaining appropriate lawful basis (and any Art. 9 condition) is hard."
    )
    consent_withdrawal_impact: Likert9 = Field(
        ..., description="Withdrawal/rights exercise likely to materially disrupt datasets/modeling."
    )
    data_minimization_fit: Likert9 = Field(
        ..., description="Predictor likely requires excessive data vs. necessity (data minimization concerns)."
    )
    purpose_limitation_risk: Likert9 = Field(
        ..., description="High risk of purpose creep/secondary use incompatibility."
    )
    retention_deletion_complexity: Likert9 = Field(
        ..., description="Retention limits and deletion workflows (incl. backups) likely complex."
    )
    cross_border_transfer_risk: Likert9 = Field(
        ..., description="International transfers likely, increasing legal/contractual burden."
    )
    transparency_explainability_burden: Likert9 = Field(
        ..., description="Hard to provide understandable information (Arts. 13/14) about processing."
    )


class GDPRComplianceRisk(ScoredDimension):
    scores: GDPRComplianceRiskScores

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        return {
            "special_category_data": 0.18,
            "lawful_basis_complexity": 0.18,
            "consent_withdrawal_impact": 0.12,
            "data_minimization_fit": 0.12,
            "purpose_limitation_risk": 0.10,
            "retention_deletion_complexity": 0.10,
            "cross_border_transfer_risk": 0.10,
            "transparency_explainability_burden": 0.10,
        }


class EUAIActRiskScores(ScoresBase):
    high_risk_likelihood: Likert9 = Field(
        ..., description="Likelihood intended system use is categorized as high-risk under the EU AI Act."
    )
    data_governance_burden: Likert9 = Field(
        ..., description="Burden to meet data governance/quality requirements for this predictor is high."
    )
    transparency_obligations: Likert9 = Field(
        ..., description="Transparency obligations (user info, labeling, instructions) are hard to meet."
    )
    human_oversight_requirements: Likert9 = Field(
        ..., description="Human oversight requirements hard to implement given predictor's role."
    )
    technical_documentation_burden: Likert9 = Field(
        ..., description="Technical documentation/logging burden high for this predictor's pipeline."
    )


class EUAIActRisk(ScoredDimension):
    scores: EUAIActRiskScores

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        return {
            "high_risk_likelihood": 0.30,
            "data_governance_burden": 0.20,
            "transparency_obligations": 0.15,
            "human_oversight_requirements": 0.15,
            "technical_documentation_burden": 0.20,
        }


class MedicalDeviceRegRiskScores(ScoresBase):
    mdr_classification_risk: Likert9 = Field(
        ..., description="Risk intended use triggers MDR medical device qualification/classification burden."
    )
    clinical_evaluation_burden: Likert9 = Field(
        ..., description="Expected clinical evaluation / evidence burden related to this predictor is high."
    )
    post_market_surveillance_burden: Likert9 = Field(
        ..., description="Post-market surveillance/vigilance complexity related to this predictor is high."
    )
    quality_management_system_burden: Likert9 = Field(
        ..., description="QMS/traceability burden increases substantially due to this predictor's role."
    )


class MedicalDeviceRegRisk(ScoredDimension):
    scores: MedicalDeviceRegRiskScores

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        return {
            "mdr_classification_risk": 0.35,
            "clinical_evaluation_burden": 0.30,
            "post_market_surveillance_burden": 0.20,
            "quality_management_system_burden": 0.15,
        }


class ePrivacyRiskScores(ScoresBase):
    terminal_equipment_access: Likert9 = Field(
        ..., description="Likely requires access to terminal equipment/metadata triggering ePrivacy constraints."
    )
    cookie_consent_dependency: Likert9 = Field(
        ..., description="Collection depends on consent mechanisms that are brittle/variable across contexts."
    )
    communications_confidentiality_risk: Likert9 = Field(
        ..., description="Risks touching communications content/metadata with higher confidentiality constraints."
    )


class ePrivacyRisk(ScoredDimension):
    scores: ePrivacyRiskScores

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        return {
            "terminal_equipment_access": 0.45,
            "cookie_consent_dependency": 0.30,
            "communications_confidentiality_risk": 0.25,
        }


class CybersecurityRiskScores(ScoresBase):
    security_controls_complexity: Likert9 = Field(
        ..., description="Security controls needed (encryption, key mgmt, access) are complex for this predictor."
    )
    attack_surface_increase: Likert9 = Field(
        ..., description="Predictor increases attack surface (new integrations, devices, endpoints)."
    )
    incident_impact: Likert9 = Field(
        ..., description="Potential harm if breached is high (sensitivity, identifiability, downstream effects)."
    )


class CybersecurityRisk(ScoredDimension):
    scores: CybersecurityRiskScores

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        return {
            "security_controls_complexity": 0.35,
            "attack_surface_increase": 0.30,
            "incident_impact": 0.35,
        }


class RegulatoryModuleWeights(StrictBaseModel):
    gdpr: confloat(ge=0.0, le=1.0) = 0.45
    eu_ai_act: confloat(ge=0.0, le=1.0) = 0.20
    medical_device: confloat(ge=0.0, le=1.0) = 0.20
    eprivacy: confloat(ge=0.0, le=1.0) = 0.05
    cybersecurity: confloat(ge=0.0, le=1.0) = 0.10

    @model_validator(mode="after")
    def _norm(self) -> "RegulatoryModuleWeights":
        d = _normalize_weights(self.model_dump())
        for k, v in d.items():
            object.__setattr__(self, k, v)
        return self

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.model_dump().items()}


class EURegulatoryRisk(StrictBaseModel):
    gdpr: Optional[GDPRComplianceRisk] = None
    eu_ai_act: Optional[EUAIActRisk] = None
    medical_device: Optional[MedicalDeviceRegRisk] = None
    eprivacy: Optional[ePrivacyRisk] = None
    cybersecurity: Optional[CybersecurityRisk] = None

    def weighted_risk(
        self,
        module_weights: Optional[Dict[str, float]] = None,
        per_module_score_weights: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> float:
        available: List[Tuple[str, ScoredDimension]] = []
        for k in ["gdpr", "eu_ai_act", "medical_device", "eprivacy", "cybersecurity"]:
            v = getattr(self, k)
            if v is not None:
                available.append((k, v))
        if not available:
            raise ValueError("No regulatory submodules provided; cannot compute regulatory risk.")

        w_all = module_weights if module_weights is not None else RegulatoryModuleWeights().as_dict()
        total_w = sum(w_all[name] for name, _ in available)
        if total_w <= EPS:
            raise ValueError("Sum of weights for available regulatory modules is 0; cannot compute.")

        risk = 0.0
        for name, dim in available:
            override = per_module_score_weights.get(name) if per_module_score_weights else None
            risk += (w_all[name] / total_w) * dim.weighted_risk(weights=override)
        return max(0.0, min(1.0, float(risk)))

    def suitability(self, **kwargs) -> float:
        return 1.0 - self.weighted_risk(**kwargs)


# -----------------------------------------------------------------------------
# Dimension 6: Scientific utility (predictor-specific)
# -----------------------------------------------------------------------------

class ScientificUtilityScores(ScoresBase):
    causal_interpretability_low: Likert9 = Field(
        ..., description="Hard to interpret causally (likely a proxy/marker rather than a causal driver)."
    )
    manipulability_low: Likert9 = Field(
        ..., description="Even if associated, hard to manipulate without affecting many other variables."
    )
    mechanistic_plausibility_low: Likert9 = Field(
        ..., description="Weak mechanistic plausibility linking predictor to target criteria."
    )
    redundancy_with_other_predictors_high: Likert9 = Field(
        ..., description="High redundancy; limited incremental explanatory/predictive value."
    )
    temporal_responsiveness_low: Likert9 = Field(
        ..., description="Changes too slowly or too erratically to support adaptive decisions."
    )
    generalizability_low: Likert9 = Field(
        ..., description="Poor expected generalizability across contexts/populations/devices."
    )
    intervention_evidence_low: Likert9 = Field(
        ..., description="Limited evidence that interventions affecting this predictor improve outcomes."
    )


class ScientificUtility(ScoredDimension):
    scores: ScientificUtilityScores

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        return {
            "causal_interpretability_low": 0.18,
            "manipulability_low": 0.14,
            "mechanistic_plausibility_low": 0.14,
            "redundancy_with_other_predictors_high": 0.14,
            "temporal_responsiveness_low": 0.14,
            "generalizability_low": 0.10,
            "intervention_evidence_low": 0.16,
        }


# -----------------------------------------------------------------------------
# Top-level: Predictor evaluation (LLM-facing: scores only) + overall suitability
# -----------------------------------------------------------------------------

class PredictorMetadata(StrictBaseModel):
    predictor_id: str = Field(..., description="Unique leaf-node identifier in the predictor ontology.")
    label: str = Field(..., description="Human-readable name of the predictor.")
    definition: Optional[str] = Field(None, description="Short definition/description of the predictor.")
    ontology_name: Optional[str] = Field(None, description="Name/version tag of the source ontology.")
    age_group: Optional[str] = Field(None, description="Age-specific ontology identifier, if applicable.")
    biopsychosocial_layer: Optional[str] = Field(
        None,
        description="Optional tag (biological / psychological / social) or custom layer label.",
    )


class PredictorEvaluation(StrictBaseModel):
    """
    Septette nested Pydantic model:
    - 1 metadata module
    - 6 evaluation modules (scores only)
    """

    metadata: PredictorMetadata
    mathematical_suitability: MathematicalSuitability
    data_collection_feasibility: DataCollectionFeasibility
    validity_threats: ValidityThreats
    treatment_translation: TreatmentTranslationPotential
    eu_regulatory_risk: EURegulatoryRisk
    scientific_utility: ScientificUtility


class SuitabilityBreakdown(StrictBaseModel):
    overall_suitability: confloat(ge=0.0, le=1.0) = Field(..., description="Weighted overall suitability in [0,1].")
    by_dimension_suitability: Dict[str, confloat(ge=0.0, le=1.0)] = Field(..., description="Suitability per dimension.")
    by_dimension_risk: Dict[str, confloat(ge=0.0, le=1.0)] = Field(..., description="Risk per dimension.")
    used_dimension_weights: Dict[str, confloat(ge=0.0, le=1.0)] = Field(..., description="Weights used (sum=1).")


def default_dimension_weights() -> Dict[str, float]:
    """Default weights across the 6 top-level dimensions (normalized to 1)."""
    # Emphasize (a) mathematical suitability for dynamic modeling and (b) treatment translation/actionability.
    return _normalize_weights(
        {
            "mathematical_suitability": 0.30,
            "data_collection_feasibility": 0.15,
            "validity_threats": 0.15,
            "treatment_translation": 0.25,
            "eu_regulatory_risk": 0.05,
            "scientific_utility": 0.10,
        }
    )


def compute_overall_suitability(
    e: PredictorEvaluation,
    dimension_weights: Optional[Dict[str, float]] = None,
) -> SuitabilityBreakdown:
    dim_w = _normalize_weights(dimension_weights) if dimension_weights else default_dimension_weights()

    dim_risk: Dict[str, float] = {
        "mathematical_suitability": e.mathematical_suitability.weighted_risk(),
        "data_collection_feasibility": e.data_collection_feasibility.weighted_risk(),
        "validity_threats": e.validity_threats.weighted_risk(),
        "treatment_translation": e.treatment_translation.weighted_risk(),
        "eu_regulatory_risk": e.eu_regulatory_risk.weighted_risk(),
        "scientific_utility": e.scientific_utility.weighted_risk(),
    }
    dim_suit: Dict[str, float] = {k: (1.0 - v) for k, v in dim_risk.items()}

    overall = 0.0
    for k, w in dim_w.items():
        if k not in dim_suit:
            raise ValueError(f"Unknown dimension weight key: {k}")
        overall += float(w) * float(dim_suit[k])
    overall = max(0.0, min(1.0, float(overall)))

    return SuitabilityBreakdown(
        overall_suitability=overall,
        by_dimension_suitability={k: float(v) for k, v in dim_suit.items()},
        by_dimension_risk={k: float(v) for k, v in dim_risk.items()},
        used_dimension_weights={k: float(v) for k, v in dim_w.items()},
    )


__all__ = [
    "Likert9",
    "PredictorMetadata",
    "PredictorEvaluation",
    "SuitabilityBreakdown",
    "compute_overall_suitability",
    "default_dimension_weights",
    "CollectionMethod",
    "DataCollectionAggregation",
    "MathematicalSuitability",
    "DataCollectionFeasibility",
    "ValidityThreats",
    "TreatmentTranslationPotential",
    "EURegulatoryRisk",
    "ScientificUtility",
]
