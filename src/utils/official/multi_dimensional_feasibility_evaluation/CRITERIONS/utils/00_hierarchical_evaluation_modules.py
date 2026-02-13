"""
00_hierarchical_criterion_evaluation_modules.py

Hierarchical evaluation schema for ranking ontology leaf-nodes ("criterions") on:
1) Mathematical suitability for multivariate time-series modeling (e.g., time-varying gVAR),
2) Operational feasibility of collecting valid multivariate time-series data (multiple collection methods),
3) Validity threats (bias, insight/agnosognosia, measurement error),
4) Juridical/regulatory compliance risk in an EU context,
5) General importance (incl. prevalence),
6) Scientific utility.

This file is an update of the earlier `00_hierarchical_evaluation_modules.py` in which:
- The *LLM-facing evaluation models contain scores only* (no weights embedded in the nested Pydantic output).
- Default weights are defined as class-level helpers and are only used for local scoring utilities.
- The top-level evaluation model is a "septette": {metadata + 6 dimension modules}.

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
# Dimension 1: Mathematical restrictions (time-series analysis)
# -----------------------------------------------------------------------------

class MathematicalRestrictionsScores(ScoresBase):
    system_variability: Likert9 = Field(
        ...,
        description="Insufficient within-person/system fluctuation range; low variability harms dynamic modeling.",
    )
    process_memory: Likert9 = Field(
        ...,
        description="Autocorrelation / process memory structure inadequacy (e.g., too short/too long memory).",
    )
    outlier_sensitivity: Likert9 = Field(
        ...,
        description="Expected sensitivity to extreme values; outliers likely to destabilize estimation.",
    )
    non_stationarity: Likert9 = Field(
        ...,
        description="Violation of (local) stationarity assumptions; strong trends/regime shifts likely.",
    )
    missing_data_pattern: Likert9 = Field(
        ...,
        description="Missingness/gap distribution likely problematic (MNAR, long gaps, bursty missingness).",
    )
    sampling_distribution_skewness: Likert9 = Field(
        ...,
        description="Sampling distribution expected to be strongly skewed, affecting estimation/inference.",
    )
    sampling_distribution_kurtosis: Likert9 = Field(
        ...,
        description="Sampling distribution expected to have problematic kurtosis/heavy tails.",
    )
    sampling_consistency: Likert9 = Field(
        ...,
        description="Irregular measurement intervals / inconsistent sampling likely to harm model assumptions.",
    )
    data_length: Likert9 = Field(
        ...,
        description="Expected time-series length too short for stable estimation (power / convergence issues).",
    )
    lag_order_selection: Likert9 = Field(
        ...,
        description="Lag-order identification likely to be unreliable (aliasing, sparse sampling, weak dynamics).",
    )
    multicollinearity: Likert9 = Field(
        ...,
        description="Expected multicollinearity / redundancy with other variables inflating variance and instability.",
    )


class MathematicalRestrictions(ScoredDimension):
    scores: MathematicalRestrictionsScores

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        return {
            "system_variability": 0.17,
            "process_memory": 0.08,
            "outlier_sensitivity": 0.06,
            "non_stationarity": 0.10,
            "missing_data_pattern": 0.11,
            "sampling_distribution_skewness": 0.06,
            "sampling_distribution_kurtosis": 0.04,
            "sampling_consistency": 0.13,
            "data_length": 0.13,
            "lag_order_selection": 0.06,
            "multicollinearity": 0.06,
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
    """
    Operational feasibility of collecting time series data for this criterion across different methods.

    NOTE (important for Structured Outputs):
    - 'aggregation' is REQUIRED (no default here) to prevent JSON Schema `$ref` + `default` conflicts.

    LLM-facing schema:
    - scores only (no method weights embedded)
    """

    aggregation: DataCollectionAggregation = Field(
        ...,
        description="Aggregation strategy over available methods: best_available or weighted_mean."
    )

    self_report_ema: Optional[CollectionMethodFeasibility] = None
    third_party_ema: Optional[CollectionMethodFeasibility] = None
    wearable: Optional[CollectionMethodFeasibility] = None
    user_device_data: Optional[CollectionMethodFeasibility] = None
    etl_pipeline: Optional[CollectionMethodFeasibility] = None
    third_party_api: Optional[CollectionMethodFeasibility] = None

    @classmethod
    def default_method_weights(cls) -> Dict[str, float]:
        """Weights over data-collection methods for WEIGHTED_MEAN aggregation (renormalized over available methods)."""
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

        risks: Dict[str, float] = {}
        for name, dim in available:
            override = per_method_score_weights.get(name) if per_method_score_weights else None
            risks[name] = dim.weighted_risk(weights=override)

        if self.aggregation == DataCollectionAggregation.BEST_AVAILABLE:
            return float(min(risks.values()))

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
# Dimension 3: Validity threats
# -----------------------------------------------------------------------------

class ResponseBiasRiskScores(ScoresBase):
    social_desirability_bias: Likert9 = Field(..., description="Risk of socially desirable responding.")
    incentive_bias: Likert9 = Field(..., description="Bias induced by incentives or perceived rewards.")
    demand_characteristics: Likert9 = Field(..., description="Participants infer study aims and alter responses/behavior.")
    confirmation_bias: Likert9 = Field(..., description="Tendency to interpret experiences to confirm expectations.")
    attentional_bias: Likert9 = Field(..., description="Attentional capture/neglect affecting reporting accuracy.")
    acquiescence_bias: Likert9 = Field(..., description="Agreeing regardless of content; yes-saying.")
    anchoring_bias: Likert9 = Field(..., description="Anchoring on prior ratings/expectations.")
    recall_bias: Likert9 = Field(..., description="Systematic recall errors (even in short windows).")
    temporal_aggregation_bias: Likert9 = Field(..., description="Bias from summarizing over time windows (peak-end, etc.).")


class ResponseBiasRisk(ScoredDimension):
    scores: ResponseBiasRiskScores

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        return {
            "social_desirability_bias": 0.16,
            "incentive_bias": 0.09,
            "demand_characteristics": 0.12,
            "confirmation_bias": 0.09,
            "attentional_bias": 0.10,
            "acquiescence_bias": 0.08,
            "anchoring_bias": 0.09,
            "recall_bias": 0.15,
            "temporal_aggregation_bias": 0.12,
        }


class InsightReportingCapacityRiskScores(ScoresBase):
    memory_impairment: Likert9 = Field(..., description="Memory impairment likely to prevent accurate reporting.")
    distorted_reality_testing: Likert9 = Field(..., description="Distorted reality testing (e.g., psychotic-like) likely.")
    lack_of_insight: Likert9 = Field(..., description="Limited insight into symptoms/states; poor self-monitoring.")
    low_self_consciousness: Likert9 = Field(..., description="Low self-consciousness or meta-awareness impacting reports.")
    alexithymia_interoceptive_limits: Likert9 = Field(
        ..., description="Difficulty identifying/describing feelings or bodily states."
    )
    dissociation_confusional_states: Likert9 = Field(
        ..., description="Dissociation/confusional episodes limiting consistent self-report."
    )
    resistance_noncooperation: Likert9 = Field(
        ..., description="Psychological resistance/noncooperation likely to distort reports."
    )


class InsightReportingCapacityRisk(ScoredDimension):
    scores: InsightReportingCapacityRiskScores

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        return {
            "memory_impairment": 0.17,
            "distorted_reality_testing": 0.16,
            "lack_of_insight": 0.18,
            "low_self_consciousness": 0.10,
            "alexithymia_interoceptive_limits": 0.14,
            "dissociation_confusional_states": 0.12,
            "resistance_noncooperation": 0.13,
        }


class MeasurementValidityRiskScores(ScoresBase):
    construct_ambiguity: Likert9 = Field(
        ..., description="Construct is ambiguous or poorly defined, increasing measurement error/criterion drift."
    )
    context_dependence: Likert9 = Field(
        ..., description="Strong context dependence reduces comparability across time/contexts."
    )
    reactivity: Likert9 = Field(
        ..., description="Measurement likely changes the state itself (reactivity), harming validity."
    )
    common_method_variance: Likert9 = Field(
        ..., description="Common-method variance likely inflates relationships (esp. mono-method self-report)."
    )
    confounding_risk: Likert9 = Field(
        ..., description="High confounding by external factors not captured/controlled in typical deployments."
    )
    algorithmic_label_noise: Likert9 = Field(
        ..., description="If derived via algorithms/ETL, label noise/drift expected to be high."
    )


class MeasurementValidityRisk(ScoredDimension):
    scores: MeasurementValidityRiskScores

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        return {
            "construct_ambiguity": 0.18,
            "context_dependence": 0.16,
            "reactivity": 0.14,
            "common_method_variance": 0.14,
            "confounding_risk": 0.18,
            "algorithmic_label_noise": 0.20,
        }


class ValidityThreatWeights(StrictBaseModel):
    """Internal weights for aggregating validity submodules (not part of LLM-facing schema)."""
    response_bias: confloat(ge=0.0, le=1.0) = 0.40
    insight_capacity: confloat(ge=0.0, le=1.0) = 0.35
    measurement_validity: confloat(ge=0.0, le=1.0) = 0.25

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
# Dimension 4: Juridical / regulatory (EU-oriented)
# -----------------------------------------------------------------------------

class GDPRComplianceRiskScores(ScoresBase):
    special_category_data: Likert9 = Field(
        ..., description="Processing likely involves special category data (health/biometrics) increasing burden."
    )
    lawful_basis_complexity: Likert9 = Field(
        ..., description="Obtaining/maintaining an appropriate lawful basis (and any Art. 9 condition) is hard."
    )
    consent_withdrawal_impact: Likert9 = Field(
        ..., description="Withdrawal/rights exercise likely to materially disrupt the dataset/modeling."
    )
    data_minimization_fit: Likert9 = Field(
        ..., description="Criterion likely requires excessive data vs. necessity (data minimization concerns)."
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
        ..., description="Hard to provide understandable information (Arts. 13/14) about this data processing."
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
        ..., description="Likelihood the intended system use falls under a high-risk category."
    )
    data_governance_burden: Likert9 = Field(
        ..., description="Burden to meet data governance/quality requirements for this criterion is high."
    )
    transparency_obligations: Likert9 = Field(
        ..., description="Transparency obligations (user info, labeling, instructions) are hard to meet."
    )
    human_oversight_requirements: Likert9 = Field(
        ..., description="Human oversight requirements hard to implement given this criterion's role."
    )
    technical_documentation_burden: Likert9 = Field(
        ..., description="Technical documentation/logging burden high for this criterion's pipeline."
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
        ..., description="Risk that intended use triggers MDR medical device qualification/classification burden."
    )
    clinical_evaluation_burden: Likert9 = Field(
        ..., description="Expected clinical evaluation / evidence burden related to this criterion is high."
    )
    post_market_surveillance_burden: Likert9 = Field(
        ..., description="Post-market surveillance/vigilance complexity related to this criterion is high."
    )
    quality_management_system_burden: Likert9 = Field(
        ..., description="QMS/traceability burden increases substantially due to this criterion's role."
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
        ..., description="Security controls needed (encryption, key mgmt, access) are complex for this criterion."
    )
    attack_surface_increase: Likert9 = Field(
        ..., description="Criterion increases attack surface (new integrations, devices, endpoints)."
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


class RegulatoryWeights(StrictBaseModel):
    """Internal weights for aggregating regulatory submodules (not part of LLM-facing schema)."""
    gdpr: confloat(ge=0.0, le=1.0) = 0.45
    eu_ai_act: confloat(ge=0.0, le=1.0) = 0.20
    medical_device: confloat(ge=0.0, le=1.0) = 0.20
    eprivacy: confloat(ge=0.0, le=1.0) = 0.05
    cybersecurity: confloat(ge=0.0, le=1.0) = 0.10

    @model_validator(mode="after")
    def _norm(self) -> "RegulatoryWeights":
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

        w_all = module_weights if module_weights is not None else RegulatoryWeights().as_dict()
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
# Dimension 5: General importance
# -----------------------------------------------------------------------------

class ImportanceScores(ScoresBase):
    absolute_prevalence_low: Likert9 = Field(..., description="Absolute prevalence is very low; limited coverage.")
    clinical_burden_low: Likert9 = Field(..., description="Low burden / minimal impairment; limited relevance.")
    transdiagnostic_value_low: Likert9 = Field(..., description="Low transdiagnostic value.")
    intervention_actionability_low: Likert9 = Field(..., description="Hard to intervene on/use for decisions.")
    stakeholder_value_low: Likert9 = Field(..., description="Low stakeholder value relative to alternatives.")


class GeneralImportance(ScoredDimension):
    scores: ImportanceScores

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        return {
            "absolute_prevalence_low": 0.35,
            "clinical_burden_low": 0.20,
            "transdiagnostic_value_low": 0.15,
            "intervention_actionability_low": 0.15,
            "stakeholder_value_low": 0.15,
        }


# -----------------------------------------------------------------------------
# Dimension 6: Scientific utility
# -----------------------------------------------------------------------------

class ScientificUtilityScores(ScoresBase):
    construct_specificity_low: Likert9 = Field(..., description="Too broad/unspecific; unclear what it captures.")
    interpretability_low: Likert9 = Field(..., description="Hard to interpret clinically/scientifically.")
    redundancy_with_other_criteria_high: Likert9 = Field(..., description="High redundancy; limited incremental value.")
    temporal_responsiveness_low: Likert9 = Field(..., description="Changes too slowly for actionable modeling.")
    mechanistic_plausibility_low: Likert9 = Field(..., description="Weak mechanistic plausibility.")


class ScientificUtility(ScoredDimension):
    scores: ScientificUtilityScores

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        return {
            "construct_specificity_low": 0.20,
            "interpretability_low": 0.20,
            "redundancy_with_other_criteria_high": 0.20,
            "temporal_responsiveness_low": 0.25,
            "mechanistic_plausibility_low": 0.15,
        }


# -----------------------------------------------------------------------------
# Top-level: Criterion evaluation (scores-only) + overall suitability
# -----------------------------------------------------------------------------

class CriterionMetadata(StrictBaseModel):
    criterion_id: str = Field(..., description="Unique leaf-node identifier in the ontology.")
    label: str = Field(..., description="Human-readable name of the criterion.")
    definition: Optional[str] = Field(None, description="Short definition/description of the criterion.")
    ontology_name: Optional[str] = Field(None, description="Name/version tag of the source ontology.")
    age_group: Optional[str] = Field(None, description="Age-specific ontology identifier, if applicable.")


class CriterionEvaluation(StrictBaseModel):
    """
    Septette nested Pydantic model:
    - 1 metadata module
    - 6 evaluation modules (scores only)
    """

    metadata: CriterionMetadata
    mathematical_restrictions: MathematicalRestrictions
    data_collection_feasibility: DataCollectionFeasibility
    validity_threats: ValidityThreats
    eu_regulatory_risk: EURegulatoryRisk
    general_importance: GeneralImportance
    scientific_utility: ScientificUtility


class SuitabilityBreakdown(StrictBaseModel):
    overall_suitability: confloat(ge=0.0, le=1.0) = Field(..., description="Weighted overall suitability in [0,1].")
    by_dimension_suitability: Dict[str, confloat(ge=0.0, le=1.0)] = Field(..., description="Suitability per dimension.")
    by_dimension_risk: Dict[str, confloat(ge=0.0, le=1.0)] = Field(..., description="Risk per dimension.")
    used_dimension_weights: Dict[str, confloat(ge=0.0, le=1.0)] = Field(..., description="Weights used (sum=1).")


def default_dimension_weights() -> Dict[str, float]:
    """Default weights across the 6 top-level dimensions (normalized to 1)."""
    return _normalize_weights(
        {
            "mathematical_restrictions": 0.55,
            "data_collection_feasibility": 0.10,
            "validity_threats": 0.10,
            "eu_regulatory_risk": 0.05,
            "general_importance": 0.15,
            "scientific_utility": 0.05,
        }
    )


def compute_overall_suitability(
    e: CriterionEvaluation,
    dimension_weights: Optional[Dict[str, float]] = None,
) -> SuitabilityBreakdown:
    dim_w = _normalize_weights(dimension_weights) if dimension_weights else default_dimension_weights()

    dim_risk: Dict[str, float] = {
        "mathematical_restrictions": e.mathematical_restrictions.weighted_risk(),
        "data_collection_feasibility": e.data_collection_feasibility.weighted_risk(),
        "validity_threats": e.validity_threats.weighted_risk(),
        "eu_regulatory_risk": e.eu_regulatory_risk.weighted_risk(),
        "general_importance": e.general_importance.weighted_risk(),
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
    "CriterionMetadata",
    "CriterionEvaluation",
    "SuitabilityBreakdown",
    "compute_overall_suitability",
    "default_dimension_weights",
    "MathematicalRestrictions",
    "DataCollectionFeasibility",
    "ValidityThreats",
    "EURegulatoryRisk",
    "GeneralImportance",
    "ScientificUtility",
    "CollectionMethod",
    "DataCollectionAggregation",
]
