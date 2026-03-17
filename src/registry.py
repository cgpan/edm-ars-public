import re
from typing import Optional

import yaml


class RegistryLoader:
    def __init__(self, registry_path: str) -> None:
        with open(registry_path) as f:
            self._data = yaml.safe_load(f)
        self._temporal_order: list = self._data.get("temporal_order", [])
        self._tier3_rules: dict = self._data.get("tier3_exclusion_rules", {})
        self._protected_names: set = self._build_protected_set()

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
        rules = self._tier3_rules
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
