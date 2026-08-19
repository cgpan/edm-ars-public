"""Tests for the skill registry infrastructure (V2.0 Phase 1)."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.skills import (
    Skill,
    SkillRegistry,
    format_skills_for_prompt,
    load_skill_from_skillmd,
    load_skills_from_directory,
    match_skills,
    resolve_references,
)
from src.skills.composer import _TRUNCATION_NOTICE
from src.skills.matcher import DEFAULT_TOP_K_PER_LAYER

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "skills"
BROKEN_ROOT = Path(__file__).parent / "fixtures" / "broken_skills"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_round_trip_via_dict(self) -> None:
        skill = Skill(
            name="example",
            layer="methodology",
            description="Example skill.",
            body="Body text.",
            trigger_keywords=["a", "b"],
            applicable_task_types=["prediction"],
            applicable_datasets=["hsls09_public"],
            applicable_stages=["Analyst"],
            priority=2,
            references_skills=["other"],
            resources=["data.yaml"],
            version="1.2",
            source_dir=Path("/tmp/example"),
        )
        round_tripped = Skill.from_dict(skill.to_dict())
        assert round_tripped.name == skill.name
        assert round_tripped.layer == skill.layer
        assert round_tripped.priority == skill.priority
        assert round_tripped.references_skills == skill.references_skills
        assert round_tripped.resources == skill.resources
        assert round_tripped.source_dir == skill.source_dir

    def test_required_fields_enforced(self) -> None:
        with pytest.raises(ValueError, match="layer"):
            Skill(name="x", layer="not-a-layer", description="d", body="")
        with pytest.raises(ValueError, match="description"):
            Skill(name="x", layer="methodology", description="", body="")
        with pytest.raises(ValueError, match="name"):
            Skill(name="", layer="methodology", description="d", body="")

    def test_resource_paths_computed(self, tmp_path: Path) -> None:
        (tmp_path / "data.yaml").write_text("x: 1\n", encoding="utf-8")
        skill = Skill(
            name="r",
            layer="dataset",
            description="d",
            body="",
            resources=["data.yaml"],
            source_dir=tmp_path,
        )
        paths = skill.resource_paths
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].name == "data.yaml"

    def test_resource_paths_empty_without_source_dir(self) -> None:
        skill = Skill(
            name="r", layer="dataset", description="d", body="", resources=["x.yaml"]
        )
        assert skill.resource_paths == []


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TestLoader:
    def test_loads_six_valid_fixtures(self) -> None:
        skills = load_skills_from_directory(FIXTURE_ROOT)
        assert len(skills) == 6
        names = {s.name for s in skills}
        assert names == {
            "prediction-fixture",
            "causal-fixture",
            "hsls09-fixture",
            "cluster-recon-fixture",
            "shap-fixture",
            "acm-fixture",
        }

    def test_broken_fixture_returns_none_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = BROKEN_ROOT / "missing-frontmatter" / "SKILL.md"
        with caplog.at_level(logging.WARNING, logger="src.skills.loader"):
            result = load_skill_from_skillmd(path)
        assert result is None
        assert any("frontmatter" in rec.message.lower() for rec in caplog.records)

    def test_layer_inferred_from_directory(self, tmp_path: Path) -> None:
        # SKILL.md without `layer:` in frontmatter; layer comes from path.
        sk_dir = tmp_path / "methodology" / "from-dir"
        sk_dir.mkdir(parents=True)
        (sk_dir / "SKILL.md").write_text(
            "---\nname: from-dir\ndescription: Inferred layer.\n---\n\nBody.\n",
            encoding="utf-8",
        )
        skill = load_skill_from_skillmd(sk_dir / "SKILL.md")
        assert skill is not None
        assert skill.layer == "methodology"

    def test_directory_frontmatter_mismatch_warns_but_loads(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        sk_dir = tmp_path / "methodology" / "mismatched"
        sk_dir.mkdir(parents=True)
        (sk_dir / "SKILL.md").write_text(
            "---\nname: mismatched\nlayer: writing\ndescription: Mismatch.\n---\n\nBody.\n",
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING, logger="src.skills.loader"):
            skill = load_skill_from_skillmd(sk_dir / "SKILL.md")
        assert skill is not None
        # Frontmatter wins per the loader's policy.
        assert skill.layer == "writing"
        assert any("mismatch" in rec.message.lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_skills() -> list[Skill]:
    return load_skills_from_directory(FIXTURE_ROOT)


class TestMatcher:
    def test_stage_filter_excludes_non_matching(
        self, fixture_skills: list[Skill]
    ) -> None:
        # acm-fixture is restricted to Writer; querying as Analyst must omit it.
        result = match_skills(
            fixture_skills,
            stage="Analyst",
            task_type="prediction",
            dataset="hsls09_public",
        )
        assert "acm-fixture" not in {s.name for s in result}

    def test_task_type_filter_excludes_non_matching(
        self, fixture_skills: list[Skill]
    ) -> None:
        # causal-fixture only applies to causal_inference.
        result = match_skills(
            fixture_skills,
            stage="ProblemFormulator",
            task_type="prediction",
            dataset="hsls09_public",
        )
        assert "causal-fixture" not in {s.name for s in result}

    def test_dataset_filter_excludes_non_matching(
        self, fixture_skills: list[Skill]
    ) -> None:
        # hsls09-fixture restricted to hsls09_public.
        result = match_skills(
            fixture_skills,
            stage="Analyst",
            task_type="prediction",
            dataset="other_dataset",
        )
        assert "hsls09-fixture" not in {s.name for s in result}

    def test_empty_applicable_means_all(self, fixture_skills: list[Skill]) -> None:
        # cluster-recon-fixture has no filters; should appear for any context.
        result = match_skills(
            fixture_skills,
            stage="Critic",
            task_type="anything",
            dataset="anywhere",
        )
        assert "cluster-recon-fixture" in {s.name for s in result}

    def test_keyword_scoring_boosts_matches(self, fixture_skills: list[Skill]) -> None:
        # With "shap" in context, shap-fixture should rank above cluster-recon.
        result = match_skills(
            fixture_skills,
            stage="Analyst",
            task_type="prediction",
            dataset="hsls09_public",
            context="discuss SHAP feature importance",
        )
        names = [s.name for s in result if s.layer == "methodology"]
        assert names.index("shap-fixture") < names.index("cluster-recon-fixture")

    def test_priority_boost_when_no_keywords(
        self, fixture_skills: list[Skill]
    ) -> None:
        # No context: shap-fixture (priority=1) outranks cluster-recon (priority=4).
        result = match_skills(
            fixture_skills,
            stage="Analyst",
            task_type="prediction",
            dataset="hsls09_public",
        )
        method_names = [s.name for s in result if s.layer == "methodology"]
        assert method_names.index("shap-fixture") < method_names.index(
            "cluster-recon-fixture"
        )

    def test_per_layer_top_k_cap_respected(self, fixture_skills: list[Skill]) -> None:
        # Cap methodology to 1 — only the highest-ranked methodology survives.
        result = match_skills(
            fixture_skills,
            stage="Analyst",
            task_type="prediction",
            dataset="hsls09_public",
            top_k_per_layer={"methodology": 1},
        )
        method = [s for s in result if s.layer == "methodology"]
        assert len(method) == 1
        assert method[0].name == "shap-fixture"

    def test_mandatory_skill_bypasses_per_layer_cap(self) -> None:
        """A mandatory-tagged skill must reach the agent even if siblings
        outrank it on score and the layer cap would otherwise drop it.

        Regression test for the Phase 2c-continuation pre-flight finding:
        latex-table-discipline (mandatory) was bumped from the Writer's
        match by tied-priority sibling writing skills.
        """
        # Three writing skills all at priority=1 → tied score. Cap is 2.
        # Without the bypass, the mandatory skill at insertion-index 2
        # would be dropped because the first two ranked entries fill the
        # cap. With the bypass it must come through.
        a = Skill(
            name="rec-alpha",
            layer="writing",
            description="d",
            body="A",
            priority=1,
        )
        b = Skill(
            name="rec-beta",
            layer="writing",
            description="d",
            body="B",
            priority=1,
        )
        mand = Skill(
            name="must-gamma",
            layer="writing",
            description="d",
            body="M",
            priority=1,
            rule_severity="mandatory",
        )
        result = match_skills(
            [a, b, mand],
            stage="Writer",
            task_type="prediction",
            dataset="hsls09_public",
            top_k_per_layer={"writing": 2},
        )
        names = [s.name for s in result]
        assert "must-gamma" in names, (
            f"mandatory skill must bypass cap; got {names}"
        )
        # Cap still applies to the recommended tail; we still get 3 total
        # (2 recommended + 1 mandatory bypassed) but never more.
        assert len(result) == 3

    def test_layer_ordering_in_output(self, fixture_skills: list[Skill]) -> None:
        result = match_skills(
            fixture_skills,
            stage="Writer",
            task_type="prediction",
            dataset="hsls09_public",
        )
        seen_layers = []
        for s in result:
            if s.layer not in seen_layers:
                seen_layers.append(s.layer)
        # Output order must follow task-type → dataset → methodology → writing.
        canonical = ["task-type", "dataset", "methodology", "writing"]
        assert seen_layers == [layer for layer in canonical if layer in seen_layers]


# ---------------------------------------------------------------------------
# Stemmer (Phase 2c plural-aware tokenization)
# ---------------------------------------------------------------------------


class TestStemmer:
    def test_singular_keyword_matches_plural_context(self) -> None:
        """Phase 2a regression: `table` keyword should fire on "tables" context."""
        from src.skills.matcher import _score, _tokenize

        skill = Skill(
            name="latex-tables-fixture",
            layer="writing",
            description="d",
            body="",
            trigger_keywords=["table"],
        )
        ctx_tokens = _tokenize("Discuss the comparison tables in the results section")
        assert _score(skill, ctx_tokens) > _priority_only_score(skill)

    def test_gerund_matches_base_via_shared_stem(self) -> None:
        """`testing` (context) stems to `test` and matches the `test` keyword."""
        from src.skills.matcher import _score, _tokenize

        skill = Skill(
            name="test-fixture",
            layer="methodology",
            description="d",
            body="",
            trigger_keywords=["test"],
        )
        ctx_tokens = _tokenize("We are testing model fairness across subgroups")
        # Both `test` (keyword, no suffix to strip) and `testing` (context,
        # `-ing` stripped) collapse to `test`.
        assert _score(skill, ctx_tokens) > _priority_only_score(skill)

    def test_short_word_guard_does_not_overstrip(self) -> None:
        """Stem floor protects two-letter words like `is` from collapsing to `i`."""
        from src.skills.matcher import _score, _stem, _tokenize

        # Direct stemmer check.
        assert _stem("is") == "is"
        assert _stem("as") == "as"
        assert _stem("use") == "use"  # stripping 'es' would leave 'u' (< 4)

        # And via tokenization: `is` in context must NOT match `i` keyword.
        skill = Skill(
            name="false-positive-fixture",
            layer="methodology",
            description="d",
            body="",
            trigger_keywords=["i"],
        )
        ctx_tokens = _tokenize("the model is fair")
        # No overlap because 'i' (len 1) is not stemmed and 'is' is not stemmed,
        # so the two stay distinct.
        assert _score(skill, ctx_tokens) == _priority_only_score(skill)


def _priority_only_score(skill: Skill) -> float:
    """The score a skill gets with no keyword overlap (priority boost only)."""
    return max(0.0, (10 - skill.priority) / 20.0)


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------


class TestComposer:
    def test_references_pull_in_dependencies(
        self, fixture_skills: list[Skill]
    ) -> None:
        by_name = {s.name: s for s in fixture_skills}
        prediction = by_name["prediction-fixture"]
        composed = resolve_references([prediction], by_name)
        names = [s.name for s in composed]
        assert names == ["prediction-fixture", "shap-fixture"]

    def test_transitive_references(self, fixture_skills: list[Skill]) -> None:
        # Construct a chain: A → B → cluster-recon-fixture.
        by_name = {s.name: s for s in fixture_skills}
        a = Skill(
            name="chain-a",
            layer="methodology",
            description="Chain head.",
            body="A.",
            references_skills=["chain-b"],
        )
        b = Skill(
            name="chain-b",
            layer="methodology",
            description="Chain middle.",
            body="B.",
            references_skills=["cluster-recon-fixture"],
        )
        by_name["chain-a"] = a
        by_name["chain-b"] = b
        composed = resolve_references([a], by_name)
        names = [s.name for s in composed]
        assert names == ["chain-a", "chain-b", "cluster-recon-fixture"]

    def test_cycle_detection_breaks_safely(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        a = Skill(
            name="cyc-a",
            layer="methodology",
            description="d",
            body="A",
            references_skills=["cyc-b"],
        )
        b = Skill(
            name="cyc-b",
            layer="methodology",
            description="d",
            body="B",
            references_skills=["cyc-a"],
        )
        with caplog.at_level(logging.WARNING, logger="src.skills.composer"):
            composed = resolve_references([a], {"cyc-a": a, "cyc-b": b})
        assert {s.name for s in composed} == {"cyc-a", "cyc-b"}
        assert any("cycle" in rec.message.lower() for rec in caplog.records)

    def test_missing_reference_warns_no_crash(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        a = Skill(
            name="lonely",
            layer="methodology",
            description="d",
            body="L",
            references_skills=["does-not-exist"],
        )
        with caplog.at_level(logging.WARNING, logger="src.skills.composer"):
            composed = resolve_references([a], {"lonely": a})
        assert [s.name for s in composed] == ["lonely"]
        assert any("does-not-exist" in rec.message for rec in caplog.records)

    def test_mandatory_skill_renders_with_strong_header(self) -> None:
        sk = Skill(
            name="hsls09-csv-format-quirks",
            layer="dataset",
            description="d",
            body="NEVER call pd.to_numeric on object dtype.",
            rule_severity="mandatory",
        )
        out = format_skills_for_prompt([sk])
        assert "## MANDATORY RULE: hsls09-csv-format-quirks" in out
        assert "binding" in out.lower()  # the banner sentence
        # Standard "Guidance:" header should NOT appear for a mandatory skill.
        assert "## Guidance: hsls09-csv-format-quirks" not in out

    def test_mandatory_skills_sort_before_recommended(self) -> None:
        rec = Skill(
            name="rec-a",
            layer="methodology",
            description="d",
            body="recommended",
            rule_severity="recommended",
        )
        mand = Skill(
            name="mand-b",
            layer="writing",
            description="d",
            body="mandatory",
            rule_severity="mandatory",
        )
        ref = Skill(
            name="ref-c",
            layer="task-type",
            description="d",
            body="reference",
            rule_severity="reference",
        )
        # Input order is recommended, mandatory, reference; output should be
        # mandatory, recommended, reference regardless of the input order.
        out = format_skills_for_prompt([rec, mand, ref])
        idx_mand = out.index("## MANDATORY RULE: mand-b")
        idx_rec = out.index("## Guidance: rec-a")
        idx_ref = out.index("### Reference: ref-c")
        assert idx_mand < idx_rec < idx_ref

    def test_missing_rule_severity_defaults_to_recommended(self) -> None:
        # No `rule_severity` specified at construction → default applies.
        sk = Skill(name="sk", layer="methodology", description="d", body="body")
        assert sk.rule_severity == "recommended"
        # And from_dict with no key should also default.
        sk2 = Skill.from_dict({"name": "sk2", "layer": "methodology", "description": "d"})
        assert sk2.rule_severity == "recommended"
        out = format_skills_for_prompt([sk])
        assert "## Guidance: sk" in out

    def test_format_for_prompt_truncates_lowest_priority_first(self) -> None:
        # Priority 9 (low) should be dropped first when over the cap.
        # Functional behavior unchanged in 3b.8; only the diagnostic
        # comment text differs between the new per-tier path (default)
        # and the legacy uniform-cap path.
        small = Skill(
            name="keep",
            layer="methodology",
            description="d",
            body="X" * 50,
            priority=1,
        )
        big = Skill(
            name="drop",
            layer="methodology",
            description="d",
            body="Y" * 500,
            priority=9,
        )
        out = format_skills_for_prompt([small, big], max_chars=200)
        assert "## Guidance: keep" in out
        # The "drop" skill's H2 header is gone — only its name remains
        # inside the drop diagnostic. Distinguish via the H2 form.
        assert "## Guidance: drop" not in out
        # New 3b.8 / §6.1 diagnostic format. The legacy notice from
        # _TRUNCATION_NOTICE is only emitted under
        # mandatory_chars_unlimited=False (legacy path).
        assert "Dropped from prompt due to budget:" in out
        assert "drop" in out  # name appears in the drop clause

    def test_format_for_prompt_legacy_uniform_cap_path(self) -> None:
        """Cover the legacy uniform-cap path that the
        mandatory_chars_unlimited=False knob preserves. The 3b.7
        notice text is emitted only under this path."""
        small = Skill(
            name="keep", layer="methodology", description="d",
            body="X" * 50, priority=1,
        )
        big = Skill(
            name="drop", layer="methodology", description="d",
            body="Y" * 500, priority=9,
        )
        out = format_skills_for_prompt(
            [small, big], max_chars=200, mandatory_chars_unlimited=False
        )
        assert "## Guidance: keep" in out
        assert "## Guidance: drop" not in out
        assert _TRUNCATION_NOTICE in out


# ---------------------------------------------------------------------------
# Registry facade
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_match_and_compose_combines_filtering_and_composition(self) -> None:
        reg = SkillRegistry(FIXTURE_ROOT)
        composed = reg.match_and_compose(
            stage="Analyst",
            task_type="prediction",
            dataset="hsls09_public",
        )
        names = [s.name for s in composed]
        # prediction-fixture matches and pulls in shap-fixture; hsls09-fixture
        # matches and pulls in cluster-recon-fixture.
        assert "prediction-fixture" in names
        assert "shap-fixture" in names
        assert "hsls09-fixture" in names
        assert "cluster-recon-fixture" in names
        # Each appears once.
        assert len(names) == len(set(names))

    def test_format_for_prompt_produces_expected_headers(self) -> None:
        reg = SkillRegistry(FIXTURE_ROOT)
        text = reg.format_for_prompt(
            stage="Writer",
            task_type="prediction",
            dataset="hsls09_public",
        )
        assert "## Guidance: acm-fixture" in text
        assert "---" in text  # separator between skills

    def test_count_by_layer_matches_loaded_fixtures(self) -> None:
        reg = SkillRegistry(FIXTURE_ROOT)
        assert reg.count() == 6
        counts = reg.count_by_layer()
        assert counts == {
            "task-type": 2,
            "dataset": 1,
            "methodology": 2,
            "writing": 1,
        }


# ---------------------------------------------------------------------------
# Resource resolution
# ---------------------------------------------------------------------------


class TestResources:
    def test_hsls09_fixture_resource_exists(self) -> None:
        reg = SkillRegistry(FIXTURE_ROOT)
        skill = reg.get("hsls09-fixture")
        assert skill is not None
        paths = skill.resource_paths
        assert len(paths) == 1
        assert paths[0].is_absolute()
        assert paths[0].exists()
        assert paths[0].name == "variable_registry_fixture.yaml"

    def test_skill_without_resources_is_empty(self) -> None:
        reg = SkillRegistry(FIXTURE_ROOT)
        skill = reg.get("shap-fixture")
        assert skill is not None
        assert skill.resource_paths == []


# ---------------------------------------------------------------------------
# Sanity: defaults are sane
# ---------------------------------------------------------------------------


def test_default_top_k_per_layer_has_all_layers() -> None:
    assert set(DEFAULT_TOP_K_PER_LAYER.keys()) == {
        "task-type",
        "dataset",
        "methodology",
        "writing",
    }
