"""TaskTemplate abstraction: encapsulates task-type-specific logic.

Each task type (prediction, causal inference, etc.) implements this ABC.
Agents and the orchestrator consume the template to get task-specific
configuration without hardcoding assumptions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.dataset_adapter import DatasetAdapter


class TaskTemplate(ABC):
    """Abstract base for task-type-specific configuration and validation."""

    @abstractmethod
    def get_name(self) -> str:
        """Return the task type identifier (e.g. 'prediction')."""
        ...

    @abstractmethod
    def get_agent_order(self) -> list[str]:
        """Return the ordered list of agent names for the revision cascade."""
        ...

    @abstractmethod
    def get_evaluation_metrics(self, outcome_type: str) -> dict:
        """Return metric configuration for the given outcome type.

        Returns a dict with keys like 'primary', 'suspicion_threshold', etc.
        """
        ...

    @abstractmethod
    def get_critic_checklist_path(self) -> str:
        """Return the path to the methodological checklist YAML for this task type."""
        ...

    @abstractmethod
    def get_paper_template_path(self, config: dict) -> str:
        """Return the path to the LaTeX paper template."""
        ...

    @abstractmethod
    def validate_research_spec(
        self,
        spec: dict,
        registry: dict,
        dataset_adapter: DatasetAdapter,
    ) -> list[str]:
        """Task-specific validation of the research specification.

        Returns a list of warning strings (non-fatal; Critic enforces hard failures).
        """
        ...


class PredictionTemplate(TaskTemplate):
    """Prediction task type — the original EDM-ARS v1 workflow."""

    def get_name(self) -> str:
        return "prediction"

    def get_agent_order(self) -> list[str]:
        return ["ProblemFormulator", "DataEngineer", "Analyst"]

    def get_evaluation_metrics(self, outcome_type: str) -> dict:
        if outcome_type == "binary":
            return {
                "primary": "AUC",
                "suspicion_threshold": 0.95,
                "higher_is_better": True,
            }
        return {
            "primary": "RMSE",
            "suspicion_threshold": None,
            "higher_is_better": False,
        }

    def get_critic_checklist_path(self) -> str:
        return "data_registry/evaluation_rubrics/methodological_checklist.yaml"

    def get_paper_template_path(self, config: dict) -> str:
        return config["paths"]["paper_template"]

    def validate_research_spec(
        self,
        spec: dict,
        registry: dict,
        dataset_adapter: DatasetAdapter,
    ) -> list[str]:
        """Validate a prediction research spec.

        Checks:
          0. Tier-3 exclusion — no weight/ID/flag variable is modeled.
          1. Temporal ordering — all predictor waves precede the outcome
             wave, with each wave resolved FROM THE REGISTRY (the spec's
             own ``wave`` field is a claim, not evidence).
          2. Feasibility — estimated analytic_n >= 10,000.
          3. Novelty score — novelty_score_self_assessment >= 3.
        """
        from src.agents.problem_formulator import _build_registry_var_map
        from src.registry import is_excluded_variable

        warnings: list[str] = []
        temporal_order: list[str] = registry.get(
            "temporal_order", dataset_adapter.get_temporal_order()
        )
        var_map = _build_registry_var_map(registry)
        tier3_rules: dict = registry.get("tier3_exclusion_rules") or {}

        outcome_var: str = spec.get("outcome_variable", "")
        predictor_set: list[dict] = spec.get("predictor_set", [])

        # --- 0. Tier-3 exclusion --------------------------------------
        # Curation wins over the pattern rules: a name the registry
        # curates under variables.* is a substantive variable even when
        # it matches a Tier-3 pattern (X1IEPFLAG matches 'FLAG$').
        # Only names the registry does NOT curate are screened.
        def _tier3_warning(name: str, role: str) -> str | None:
            if not name or name in var_map:
                return None
            if not is_excluded_variable(name, tier3_rules):
                return None
            return (
                f"TIER-3 EXCLUDED: {role} '{name}' matches the registry's "
                f"tier3_exclusion_rules (sampling weights, IDs, imputation "
                f"and processing flags). It carries no substantive signal "
                f"and must not enter the model."
            )

        outcome_tier3 = _tier3_warning(outcome_var, "outcome")
        if outcome_tier3:
            warnings.append(outcome_tier3)

        # Runs independently of the temporal block below, which is
        # skipped whenever the outcome's wave cannot be resolved.
        tier3_predictors: set[str] = set()
        for pred in predictor_set:
            name = pred.get("variable", "")
            message = _tier3_warning(name, "predictor")
            if message:
                warnings.append(message)
                tier3_predictors.add(name)

        # --- 1. Temporal ordering (waves resolved from the registry) ---
        outcome_meta = var_map.get(outcome_var, {})
        outcome_wave: str | None = outcome_meta.get("wave")

        if not outcome_wave:
            warnings.append(
                f"Outcome variable '{outcome_var}' not found in registry or has no "
                "wave metadata; temporal ordering cannot be verified."
            )
        else:
            if outcome_wave not in temporal_order:
                warnings.append(
                    f"Outcome wave '{outcome_wave}' is not in temporal_order {temporal_order}."
                )
            else:
                outcome_idx = temporal_order.index(outcome_wave)
                for pred in predictor_set:
                    pred_var = pred.get("variable", "")
                    declared_wave = pred.get("wave", "")

                    if pred_var in tier3_predictors:
                        continue  # already reported above

                    pred_meta = var_map.get(pred_var)
                    registry_wave: str = (pred_meta or {}).get("wave") or ""
                    if pred_meta is None:
                        # The registry is the authority on which wave a
                        # variable belongs to; an unlisted name cannot be
                        # verified against it at all. The spec's own
                        # claim is still checked below.
                        warnings.append(
                            f"Predictor '{pred_var}' is not in the dataset "
                            f"registry (spec declares wave="
                            f"'{declared_wave}'); its temporal position "
                            f"cannot be verified against registry metadata."
                        )

                    # Check the registry's wave (authoritative) AND the
                    # spec's own claim when they disagree. Registry truth
                    # catches a mis-declared predictor; the claim catches
                    # a spec that openly asserts a post-outcome predictor
                    # for a variable the registry does not cover.
                    to_check: list[tuple[str, str]] = []
                    if registry_wave:
                        to_check.append(("registry", registry_wave))
                    if declared_wave and declared_wave != registry_wave:
                        to_check.append(("spec-declared", declared_wave))

                    violating: list[str] = []
                    for source, wave in to_check:
                        if wave not in temporal_order:
                            suffix = (
                                " in the registry" if source == "registry" else ""
                            )
                            warnings.append(
                                f"Predictor '{pred_var}' has unknown wave "
                                f"'{wave}'{suffix}."
                            )
                            continue
                        idx = temporal_order.index(wave)
                        if idx >= outcome_idx:
                            violating.append(
                                f"{source} wave={wave}, idx={idx}"
                            )

                    if violating:
                        misdeclared = (
                            f"; the spec declared wave='{declared_wave}'"
                            if (
                                declared_wave
                                and registry_wave
                                and declared_wave != registry_wave
                                and not any(
                                    v.startswith("spec-declared")
                                    for v in violating
                                )
                            )
                            else ""
                        )
                        warnings.append(
                            f"TEMPORAL VIOLATION: predictor '{pred_var}' "
                            f"({'; '.join(violating)}{misdeclared}) does not "
                            f"precede outcome '{outcome_var}' "
                            f"(wave={outcome_wave}, "
                            f"idx={outcome_idx}). This predictor should be removed."
                        )

        # --- 2. Feasibility: estimated analytic_n >= 10,000 ---
        n_full: int = registry.get("levels", {}).get(
            "student", dataset_adapter.get_sample_size()
        )
        outcome_pct_missing: float = outcome_meta.get("pct_missing", 0.0) / 100.0

        total_predictor_missing: float = 0.0
        for pred in predictor_set:
            pred_meta = var_map.get(pred.get("variable", ""), {})
            total_predictor_missing += pred_meta.get("pct_missing", 0.0) / 100.0

        retention = max(
            0.0,
            1.0 - outcome_pct_missing - total_predictor_missing,
        )
        estimated_n = n_full * retention
        if estimated_n < 10_000:
            warnings.append(
                f"Estimated analytic_n ({estimated_n:.0f}) may fall below 10,000 "
                "based on registry missingness. Consider swapping high-missingness "
                "predictors or relaxing the predictor set."
            )

        # --- 3. Novelty score ---
        novelty_score = spec.get("novelty_score_self_assessment")
        if isinstance(novelty_score, (int, float)) and novelty_score < 3:
            warnings.append(
                f"novelty_score_self_assessment = {novelty_score} is below the "
                "minimum of 3. The research question lacks sufficient novelty."
            )

        return warnings


class CausalSOOTemplate(TaskTemplate):
    """Causal inference under selection-on-observables — V3.0 Phase 3b.4.

    Routes a locked research_spec (typically supplied via the
    ``--research-spec`` CLI flag) through the existing
    ProblemFormulator -> DataEngineer -> Analyst -> Critic -> Writer
    pipeline. ProblemFormulator runs in "refine" mode against the locked
    spec rather than generating from scratch.

    Validation is deliberately structural-only: the template confirms
    the spec carries the fields required to dispatch downstream stages,
    but does not assess scientific soundness (median-split flagging,
    post-treatment covariate detection, etc.) — those are PF's job under
    the methodology skills (G1, G2, ...).

    Phase 3b.4 ships this template registered but with the dispatch
    methods scaffolded only — they exist with correct signatures so the
    locked-spec CLI fixture validates and so 3b.5's smoke test has the
    API surface to hook into. The orchestrator state machine still
    drives the pipeline directly via ``_run_*`` stage runners; wiring
    the dispatch methods into the orchestrator is out of scope for 3b.4.
    """

    SUPPORTED_METHODS: frozenset[str] = frozenset({"M1", "M2", "M3", "M4", "M5"})

    # Stage-attachment table per the Phase 3b.2 stage-match design.
    # Maps stage name -> ordered list of skill names attached at that
    # stage. Used by ``dispatch_*`` methods to advertise which skills
    # the stage will receive when actually wired.
    _STAGE_SKILLS: dict[str, list[str]] = {
        "ProblemFormulator": [
            "causal-dag-identification",
            "causal-estimand-definition",
            "hsls09-causal-conventions",
        ],
        "DataEngineer": [
            "hsls09-causal-conventions",
        ],
        "Analyst": [
            "causal-dag-identification",
            "causal-estimand-definition",
            "causal-positivity-diagnostics",
            "causal-balance-diagnostics",
            "causal-sensitivity-unmeasured-confounding",
            "hsls09-causal-conventions",
            # Method skills appended at dispatch time per spec.primary +
            # spec.comparator + spec.secondary
        ],
        "Critic": [
            "causal-dag-identification",
            "causal-estimand-definition",
            "causal-positivity-diagnostics",
            "causal-balance-diagnostics",
            "causal-sensitivity-unmeasured-confounding",
            "hsls09-causal-conventions",
        ],
        "Writer": [
            "causal-estimand-definition",
            "causal-balance-diagnostics",
            "causal-sensitivity-unmeasured-confounding",
            "hsls09-causal-conventions",
        ],
    }

    _METHOD_SKILL_NAMES: dict[str, str] = {
        "M1": "causal-regression-adjustment",
        "M2": "causal-propensity-score-matching",
        "M3": "causal-inverse-probability-weighting",
        "M4": "causal-aipw-tmle",
        "M5": "causal-forest-cate",
    }

    def get_name(self) -> str:
        return "causal_soo"

    def get_agent_order(self) -> list[str]:
        return ["ProblemFormulator", "DataEngineer", "Analyst"]

    def get_evaluation_metrics(self, outcome_type: str) -> dict:
        # Causal headline metric is the ATE/ATT point estimate with a
        # cluster-robust 95% CI (per G2 + D1). The 'primary' label here
        # is descriptive — the orchestrator's prediction-shaped
        # suspicion threshold does not apply to causal estimands and is
        # intentionally None.
        return {
            "primary": "ATE",
            "suspicion_threshold": None,
            "higher_is_better": None,
        }

    def get_critic_checklist_path(self) -> str:
        # Phase 3b.4 deliberately routes to the prediction checklist.
        # B5 (causal-critic-checklist skill) is a separate hand-off; until
        # then, the Critic stage in causal mode runs on V1 prompt content
        # plus the methodology skills attached via _STAGE_SKILLS["Critic"].
        return "data_registry/evaluation_rubrics/methodological_checklist.yaml"

    def get_paper_template_path(self, config: dict) -> str:
        return config["paths"]["paper_template"]

    def validate_research_spec(
        self,
        spec: dict,
        registry: dict | None = None,
        dataset_adapter: "DatasetAdapter | None" = None,
    ) -> list[str]:
        """Structural validation of a causal_soo research_spec.

        Returns a list of warning/error strings. Empty list means the
        spec is structurally well-formed for downstream dispatch. This
        does NOT validate scientific soundness — flagging median-split
        as ESC-07, detecting post-treatment covariates, or rejecting an
        empty adjustment-set under no-unmeasured-confounding are all
        PF's job under G1/G2.

        ``registry`` and ``dataset_adapter`` are accepted to preserve
        the ABC signature but are unused by structural validation.
        """
        warnings: list[str] = []

        if not isinstance(spec, dict):
            return [f"research_spec must be a dict (got {type(spec).__name__})"]

        task_type = spec.get("task_type")
        if task_type != "causal_soo":
            warnings.append(
                f"task_type must be 'causal_soo' (got {task_type!r})"
            )

        treatment = spec.get("treatment")
        if not isinstance(treatment, dict):
            warnings.append("treatment block missing or not a dict")
        else:
            if not treatment.get("variable"):
                warnings.append("treatment.variable is required")
            if not treatment.get("operationalization"):
                warnings.append("treatment.operationalization is required")

        outcome = spec.get("outcome")
        if not isinstance(outcome, dict):
            warnings.append("outcome block missing or not a dict")
        else:
            if not outcome.get("variable"):
                warnings.append("outcome.variable is required")
            if not outcome.get("type"):
                warnings.append("outcome.type is required (e.g., 'binary' or 'continuous')")

        if not spec.get("target_estimand_hint"):
            warnings.append(
                "target_estimand_hint is required (PF will refine into "
                "an explicit estimand declaration per G2)"
            )

        primary = spec.get("primary_method")
        if not primary:
            warnings.append("primary_method is required")
        elif primary not in self.SUPPORTED_METHODS:
            warnings.append(
                f"primary_method {primary!r} not in supported methods "
                f"{sorted(self.SUPPORTED_METHODS)}"
            )

        # secondary_methods is optional but if present must be a list of
        # supported method IDs
        secondary = spec.get("secondary_methods", [])
        if secondary and not isinstance(secondary, list):
            warnings.append("secondary_methods must be a list when provided")
        else:
            for m in secondary:
                if m not in self.SUPPORTED_METHODS:
                    warnings.append(
                        f"secondary_methods entry {m!r} not in supported "
                        f"methods {sorted(self.SUPPORTED_METHODS)}"
                    )

        return warnings

    # ------------------------------------------------------------------
    # Dispatch scaffolding (Phase 3b.4 — methods exist for 3b.5 wiring)
    # ------------------------------------------------------------------
    #
    # These do NOT replace the orchestrator's _run_* stage runners in
    # 3b.4. They surface the per-stage configuration the causal pipeline
    # would consume so that:
    #   (a) 3b.5 has a concrete API to wire into without re-deriving the
    #       stage-skill table at the call site, and
    #   (b) tests can assert that, for a locked spec, the right methods
    #       and skills appear at the right stage.

    def dispatch_data_engineering(self, spec: dict) -> dict:
        """Return the DataEngineer stage configuration for a causal spec.

        Pass-through: the existing DataEngineer agent prepares the
        analytic dataset; this template surfaces the treatment +
        outcome + adjustment_set blocks plus the D1 skill name so the
        agent has the locked spec at hand without scraping it back out
        of the research_spec field.
        """
        return {
            "treatment": spec.get("treatment", {}),
            "outcome": spec.get("outcome", {}),
            "adjustment_set": spec.get("adjustment_set", []),
            "skills": list(self._STAGE_SKILLS["DataEngineer"]),
        }

    def dispatch_analysis(self, spec: dict) -> list[dict]:
        """Return one analysis configuration per method in the spec.

        Each configuration carries the method ID, the matched skill
        names (the standing methodology skills + the method-specific
        skill), and the spec-derived inputs the Analyst stage will need
        to render its task-type-causal_soo prompt branch.

        Returns at least one config (the primary method); additional
        configs are appended for the comparator (M1) and any
        secondary_methods. Methods listed in ``exclude_methods`` are
        dropped.
        """
        primary = spec.get("primary_method")
        comparator = spec.get("comparator_method")
        secondary = spec.get("secondary_methods") or []
        excluded = set(spec.get("exclude_methods") or [])

        # Order: primary first, then comparator (if distinct), then
        # secondary (in spec order, deduped). This matches the spec's
        # rationale_for_method_set narrative and makes the headline
        # estimate the first config in the returned list.
        ordered: list[str] = []
        for m in [primary, comparator, *secondary]:
            if m and m not in ordered and m in self.SUPPORTED_METHODS and m not in excluded:
                ordered.append(m)

        base_skills: list[str] = list(self._STAGE_SKILLS["Analyst"])
        configs: list[dict] = []
        for method_id in ordered:
            method_skill = self._METHOD_SKILL_NAMES[method_id]
            skills = list(base_skills)
            if method_skill not in skills:
                skills.append(method_skill)
            configs.append(
                {
                    "method_id": method_id,
                    "method_skill": method_skill,
                    "skills": skills,
                    "treatment": spec.get("treatment", {}),
                    "outcome": spec.get("outcome", {}),
                    "adjustment_set": spec.get("adjustment_set", []),
                    "estimand_hint": spec.get("target_estimand_hint"),
                }
            )
        return configs

    def dispatch_critique(self, spec: dict, analysis_results: dict) -> dict:
        """Return the Critic stage configuration.

        Phase 3b.4 stops short of authoring the causal-critic-checklist
        skill (B5, separate hand-off). This dispatch returns the
        standing methodology skills attached at the Critic stage; the
        Critic prompt itself runs unchanged (V1 monolithic) and gaps
        catalogued in 3b.5 feed B5.
        """
        return {
            "skills": list(self._STAGE_SKILLS["Critic"]),
            "spec": spec,
            "analysis_results": analysis_results,
        }

    def dispatch_writing(
        self,
        spec: dict,
        analysis_results: dict,
        critic_decision: dict,
    ) -> dict:
        """Return the Writer stage configuration."""
        return {
            "skills": list(self._STAGE_SKILLS["Writer"]),
            "spec": spec,
            "analysis_results": analysis_results,
            "critic_decision": critic_decision,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class CausalITRTemplate(CausalSOOTemplate):
    """Optimal treatment regimes / individualized treatment rules —
    V3.1 Arc R.

    ITR is selection-on-observables PLUS policy learning: the
    identification assumptions, DE data contract, G-family diagnostics,
    and pre-flight guards are inherited unchanged from
    :class:`CausalSOOTemplate`. What changes:

    - The headline estimand is the POLICY VALUE of a learned rule
      V(pi) and its GAIN over the best constant policy (treat-all /
      treat-none), not a single ATE/ATT.
    - The method battery adds M6 (doubly-robust policy learning via
      DR pseudo-outcomes + a shallow sklearn policy tree — no econml
      dependency, the sandbox doesn't ship it) and M7 (cross-fitted
      policy-value evaluation with bootstrap CIs + subgroup value
      parity). M5 (causal forest CATE) feeds heterogeneity evidence.
    - The spec requires ``rule_covariates``: the subset of covariates
      the rule may use — must be observable/actionable at decision
      time (the ITR analogue of the temporal-ordering rule).
    """

    SUPPORTED_METHODS: frozenset[str] = frozenset(
        {"M1", "M2", "M3", "M4", "M5", "M6", "M7"}
    )

    _METHOD_SKILL_NAMES: dict[str, str] = {
        **CausalSOOTemplate._METHOD_SKILL_NAMES,
        "M6": "causal-itr-policy-learning",
        "M7": "causal-itr-policy-value-evaluation",
    }

    def get_name(self) -> str:
        return "causal_itr"

    def get_evaluation_metrics(self, outcome_type: str) -> dict:
        return {
            "primary": "POLICY_VALUE",
            "suspicion_threshold": None,
            "higher_is_better": True,
        }

    def validate_research_spec(
        self,
        spec: dict,
        registry: dict | None = None,
        dataset_adapter: "DatasetAdapter | None" = None,
    ) -> list[str]:
        # Reuse the SOO structural checks with the task_type check
        # swapped: run the parent on a shallow copy that satisfies its
        # task_type gate, then apply ITR-specific requirements.
        proxy = dict(spec) if isinstance(spec, dict) else spec
        if isinstance(proxy, dict) and proxy.get("task_type") == "causal_itr":
            proxy["task_type"] = "causal_soo"
        warnings = super().validate_research_spec(
            proxy, registry=registry, dataset_adapter=dataset_adapter
        )
        if not isinstance(spec, dict):
            return warnings

        if spec.get("task_type") != "causal_itr":
            warnings.append(
                f"task_type must be 'causal_itr' (got {spec.get('task_type')!r})"
            )

        rule_covs = spec.get("rule_covariates")
        if not isinstance(rule_covs, list) or not rule_covs:
            warnings.append(
                "rule_covariates is required for causal_itr: the "
                "non-empty list of covariates the learned rule may "
                "condition on (must be observable at decision time)"
            )
        else:
            adjustment = {
                str(v.get("variable") if isinstance(v, dict) else v)
                for v in (spec.get("adjustment_set") or [])
            }
            unknown = [c for c in rule_covs if adjustment and c not in adjustment]
            if unknown:
                warnings.append(
                    f"rule_covariates {unknown} not in adjustment_set — "
                    f"rule covariates must be measured confounders the "
                    f"DataEngineer carves into the analytic CSV"
                )

        if spec.get("primary_method") not in ("M6",):
            warnings.append(
                "primary_method must be 'M6' (policy learning) for "
                "causal_itr; ATE-style methods belong in secondary_methods"
            )
        return warnings


class CausalDIDTemplate(TaskTemplate):
    """Cross-cohort difference-in-differences - Phase B (V3.6 executable).

    Runs on a pre-harmonized panel dataset (scripts/harmonize_els_hsls.py):
    outcome = within-cohort percentile rank (instrument-invariant),
    group x cohort 2x2 with a follow-wave stability probe. The estimand
    is a GROUP-GAP CHANGE between cohorts, never an absolute
    achievement change (tests are not equated), and no single-policy
    identification is claimed - the paper's framing obligation.
    """

    def get_name(self) -> str:
        return "causal_did"

    def get_agent_order(self) -> list[str]:
        return ["ProblemFormulator", "DataEngineer", "Analyst"]

    def get_evaluation_metrics(self, outcome_type: str) -> dict:
        return {"primary": "DID_GAP_CHANGE", "suspicion_threshold": None,
                "higher_is_better": None}

    def get_critic_checklist_path(self) -> str:
        return "data_registry/evaluation_rubrics/methodological_checklist.yaml"

    def get_paper_template_path(self, config: dict) -> str:
        return config["paths"]["paper_template"]

    def validate_research_spec(self, spec, registry=None, dataset_adapter=None):
        warnings = []
        if not isinstance(spec, dict):
            return ["research_spec must be a dict"]
        if spec.get("task_type") != "causal_did":
            warnings.append(f"task_type must be 'causal_did' (got {spec.get('task_type')!r})")
        for field in ("outcome", "group_variable", "post_variable"):
            if not spec.get(field):
                warnings.append(f"{field} is required for causal_did")
        if spec.get("primary_method") not in ("M8", "M9"):
            warnings.append(
                "primary_method must be 'M8' (raw gap-in-gaps) or 'M9' "
                "(composition-adjusted AIPW gap change)"
            )
        return warnings


class PsychometricsTemplate(TaskTemplate):
    """Measurement/psychometrics studies - V4 (locked-spec entry).

    P-battery: P1 CTT, P2 omega, P3 CFA, P4 GRM, P5 DIF, P6 invariance -
    all through certified analysis_helpers.psy_* wrappers backed by the
    R bridge (see the psychometrics scope note (internal)).
    """

    def get_name(self) -> str:
        return "psychometrics"

    def get_agent_order(self) -> list[str]:
        return ["ProblemFormulator", "DataEngineer", "Analyst"]

    def get_evaluation_metrics(self, outcome_type: str) -> dict:
        return {"primary": "MEASUREMENT_CLAIMS", "suspicion_threshold": None,
                "higher_is_better": None}

    def get_critic_checklist_path(self) -> str:
        return "data_registry/evaluation_rubrics/methodological_checklist.yaml"

    def get_paper_template_path(self, config: dict) -> str:
        return config["paths"]["paper_template"]

    def validate_research_spec(self, spec, registry=None, dataset_adapter=None):
        warnings = []
        if not isinstance(spec, dict):
            return ["research_spec must be a dict"]
        if spec.get("task_type") != "psychometrics":
            warnings.append(
                f"task_type must be 'psychometrics' (got {spec.get('task_type')!r})")
        if not spec.get("scale_name") or not (
                spec.get("item_columns") or spec.get("item_construction")):
            warnings.append(
                "scale_name plus item_columns (survey scales) or "
                "item_construction (log datasets) is required")
        if len(spec.get("item_columns") or []) < 3 and "P3" in (
                spec.get("method_battery") or []):
            warnings.append("CFA (P3) needs >= 3 items per factor")
        battery = spec.get("method_battery") or []
        known = {"P1", "P2", "P3", "P4", "P5", "P6", "P7"}
        bad = [m for m in battery if m not in known]
        if bad:
            warnings.append(f"unknown method IDs: {bad} (known: sorted P1-P6)")
        if any(m in battery for m in ("P5", "P6")) and not spec.get("grouping_vars"):
            warnings.append("P5/P6 require grouping_vars")
        return warnings


_TASK_REGISTRY: dict[str, type[TaskTemplate]] = {
    "prediction": PredictionTemplate,
    "causal_soo": CausalSOOTemplate,
    "causal_itr": CausalITRTemplate,
    "causal_did": CausalDIDTemplate,
    "psychometrics": PsychometricsTemplate,
}


def create_task_template(task_type: str) -> TaskTemplate:
    """Create a TaskTemplate instance for the given task type."""
    cls = _TASK_REGISTRY.get(task_type)
    if cls is None:
        raise ValueError(
            f"Unknown task_type: {task_type!r}. "
            f"Available: {sorted(_TASK_REGISTRY.keys())}"
        )
    return cls()
