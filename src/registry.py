import re
from typing import Optional

import yaml


def is_excluded_variable(
    name: str,
    tier3_rules: Optional[dict] = None,
    category_label: Optional[str] = None,
) -> bool:
    """Return True if ``name`` is excluded by a Tier-3 rule block.

    Pure function over a registry's ``tier3_exclusion_rules`` mapping so
    that callers holding a registry *dict* (task templates, the Arc T
    feasibility screen) enforce exactly the same rules as
    :meth:`RegistryLoader.is_excluded` without re-reading YAML from disk.

    WARNING — these are pattern heuristics aimed at the auto-profiled
    Tier-2/Tier-3 name space (``tier2_config.usage_policy``: "Tier 2
    variables are available after Tier 3 exclusions"). They are NOT safe
    to apply blindly to CURATED Tier-1 variables: measured 2026-07-25,
    ``X1IEPFLAG`` -- a substantive IEP indicator curated under
    ``variables.predictors.academic`` in ``hsls09_public.yaml`` -- matches
    the ``FLAG$`` suffix rule. Callers must therefore check registry
    curation FIRST and only fall back to this predicate for names the
    registry does not curate (see
    ``PredictionTemplate.validate_research_spec``).
    """
    rules = tier3_rules or {}
    name_upper = name.upper()

    exact = {v.upper() for v in rules.get("exact_matches", [])}
    if name_upper in exact:
        return True

    for pattern in rules.get("prefix_patterns", []):
        if re.match(pattern, name_upper):
            return True

    for pattern in rules.get("suffix_patterns", []):
        if re.search(pattern, name_upper):
            return True

    if category_label is not None:
        excluded_labels = set(rules.get("category_labels", []))
        if category_label in excluded_labels:
            return True

    return False


class RegistryLoader:
    def __init__(self, registry_path: str) -> None:
        with open(registry_path, encoding="utf-8") as f:
            self._data = yaml.safe_load(f)
        self._temporal_order: list = self._data.get("temporal_order", [])
        self._tier3_rules: dict = self._data.get("tier3_exclusion_rules", {})
        self._protected_names: set = self._build_protected_set()

    @classmethod
    def from_dict(cls, data: dict) -> "RegistryLoader":
        """Build a loader from an already-parsed registry mapping.

        Callers that receive the registry as a dict (agents, task
        templates, the Arc T feasibility screen) can reach the same
        predicates without a second disk read.
        """
        obj = cls.__new__(cls)
        obj._data = data or {}
        obj._temporal_order = obj._data.get("temporal_order", [])
        obj._tier3_rules = obj._data.get("tier3_exclusion_rules", {})
        obj._protected_names = obj._build_protected_set()
        return obj

    def _build_protected_set(self) -> set:
        protected: set = set()
        predictors = self._data.get("variables", {}).get("predictors", {})
        for cat_vars in predictors.values():
            for var in cat_vars:
                if var.get("protected_attribute", False):
                    protected.add(var["name"])
        return protected

    def get_variable(self, name: str) -> dict:
        variables = self._data.get("variables", {})
        for outcome in variables.get("outcomes", []):
            if outcome["name"] == name:
                return outcome
        for cat_vars in variables.get("predictors", {}).values():
            for var in cat_vars:
                if var["name"] == name:
                    return var
        raise KeyError(f"Variable '{name}' not found in registry")

    def get_outcomes(self) -> list:
        return self._data.get("variables", {}).get("outcomes", [])

    def get_predictors(self, category: Optional[str] = None) -> list:
        predictors = self._data.get("variables", {}).get("predictors", {})
        if category is not None:
            return predictors.get(category, [])
        return [var for cat_vars in predictors.values() for var in cat_vars]

    def validate_temporal_order(self, predictor_wave: str, outcome_wave: str) -> bool:
        try:
            pred_idx = self._temporal_order.index(predictor_wave)
            out_idx = self._temporal_order.index(outcome_wave)
            return pred_idx < out_idx
        except ValueError:
            return False

    def is_protected_attribute(self, name: str) -> bool:
        return name in self._protected_names

    def is_excluded(self, name: str, category_label: Optional[str] = None) -> bool:
        """Tier-3 exclusion predicate (see :func:`is_excluded_variable`)."""
        return is_excluded_variable(name, self._tier3_rules, category_label)
