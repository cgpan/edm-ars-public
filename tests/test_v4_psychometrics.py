"""V4 psychometrics — R bridge + certified helper gates.

The certification test runs the real gate against the real R
installation (standing Arc-R rule: no downscaling). Skips cleanly on
machines without R.
"""
from __future__ import annotations

import pytest

try:
    from src.r_bridge import RBridgeError, find_rscript

    find_rscript()
    _HAS_R = True
except Exception:
    _HAS_R = False

needs_r = pytest.mark.skipif(not _HAS_R, reason="Rscript not available")


class TestRBridge:
    def test_script_name_traversal_rejected(self) -> None:
        from src.r_bridge import RBridgeError, run_r_script

        for bad in ("../evil.R", "sub/dir.R", "..\\evil.R"):
            with pytest.raises(RBridgeError):
                run_r_script(bad, {})

    def test_unknown_script_lists_available(self) -> None:
        from src.r_bridge import RBridgeError, run_r_script

        with pytest.raises(RBridgeError, match="cfa_fit.R"):
            run_r_script("nope.R", {})

    @needs_r
    def test_round_trip_cfa(self) -> None:
        from scripts.psychometric_gates import sim_congeneric
        from src.r_bridge import run_r_script

        items = sim_congeneric(600, [0.8, 0.7, 0.6], seed=1)
        out = run_r_script(
            "cfa_fit.R",
            {"items": items, "model": "F =~ v1 + v2 + v3"},
        )
        assert out["converged"] is True
        # FIML keeps partially-missing rows but lavaan drops all-missing
        # ones (5% missing x 3 items -> occasionally a fully-NA row)
        assert 590 <= out["n"] <= 600
        assert len(out["loadings"]) == 3


@needs_r
class TestPsychometricCertification:
    def test_full_gate_at_certified_defaults(self) -> None:
        import warnings

        from scripts.psychometric_gates import run_psychometric_gate

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = run_psychometric_gate()
        assert report["cfa"]["passed"], report["cfa"]
        assert report["grm"]["passed"], report["grm"]
        assert report["dif"]["passed"], report["dif"]
        assert report["invariance"]["passed"], report["invariance"]
        assert report["passed"]


class TestPsychometricsWiring:
    def test_template_registered_and_validates(self) -> None:
        from src.task_template import create_task_template

        t = create_task_template("psychometrics")
        spec = {
            "task_type": "psychometrics",
            "scale_name": "eff",
            "item_columns": ["a", "b", "c", "d"],
            "method_battery": ["P1", "P2", "P3", "P6"],
            "grouping_vars": ["X1SEX"],
        }
        assert t.validate_research_spec(spec, {}, None) == []
        bad = dict(spec, method_battery=["P9"])
        assert any("unknown method" in w for w in
                   t.validate_research_spec(bad, {}, None))
        nogroup = dict(spec, grouping_vars=[])
        assert any("grouping_vars" in w for w in
                   t.validate_research_spec(nogroup, {}, None))

    def test_fixtures_load_through_cli_path(self) -> None:
        from src.main import load_locked_research_spec

        for f in ("runs/fixtures/spec_psy_hsls_matheff_invariance.json",
                  "runs/fixtures/spec_psy_els_mathse_calibration.json"):
            spec = load_locked_research_spec(f)
            assert spec["task_type"] == "psychometrics"

    def test_prompt_variants_resolve(self) -> None:
        from pathlib import Path

        from src.agents.base import load_prompt

        root = Path(__file__).resolve().parent.parent
        config = {"paths": {"agent_prompts": str(root / "agent_prompts")}}
        for agent in ("problem_formulator", "data_engineer", "analyst",
                      "writer"):
            body = load_prompt(agent, config,
                               task_type="psychometrics")["system_prompt"]
            assert "PSYCHOMETRICS" in body, agent
            assert "{{SKILLS}}" in body, agent

    @pytest.mark.parametrize(
        "stage", ["ProblemFormulator", "DataEngineer", "Analyst", "Critic",
                  "Writer"]
    )
    def test_protocol_skill_reaches_stage(self, stage: str) -> None:
        from pathlib import Path

        from src.orchestrator import _resolve_skill_caps
        from src.skills import SkillRegistry
        from src.skills.composer import format_skills_for_prompt

        root = Path(__file__).resolve().parent.parent
        registry = SkillRegistry(str(root / "skills"))
        skills = registry.match_and_compose(
            task_type="psychometrics", dataset="hsls09_public",
            stage=stage, context="measurement invariance DIF reliability",
            top_k_per_layer=_resolve_skill_caps("psychometrics"),
        )
        names = [s.name for s in skills]
        assert "psychometrics-measurement-protocol" in names
        rendered = format_skills_for_prompt(skills)
        assert "psy_01" in rendered
        assert "scalar" in rendered
        if stage in ("DataEngineer", "Analyst", "Critic"):
            assert "r-bridge-execution" in names

    def test_items_validator(self, tmp_path) -> None:
        import numpy as np
        import pandas as pd

        from src.agents.data_engineer import DataEngineer

        de = object.__new__(DataEngineer)

        class Ctx:
            pass

        class Adapter:
            def get_multilevel_warning(self):
                return None

        de.ctx = Ctx()
        de.ctx.output_dir = str(tmp_path)
        de.ctx.research_spec = {
            "task_type": "psychometrics",
            "item_columns": ["i1", "i2", "i3"],
            "grouping_vars": ["g"],
        }
        de.dataset_adapter = Adapter()
        # healthy matrix
        n = 1200
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "i1": rng.integers(1, 5, n).astype(float),
            "i2": rng.integers(1, 5, n).astype(float),
            "i3": rng.integers(1, 5, n).astype(float),
            "g": rng.choice(["A", "B"], n),
        })
        df.loc[:100, "i1"] = np.nan  # missingness allowed
        df.to_csv(tmp_path / "items_analytic.csv", index=False)
        rep = de._validate_outputs({"validation_passed": True})
        assert rep["validation_passed"] is True
        assert rep["analytic_n"] == n and rep["n_test"] == 0
        # non-categorical item fails
        df["i2"] = rng.normal(0, 1, n)
        df.to_csv(tmp_path / "items_analytic.csv", index=False)
        rep2 = de._validate_outputs({"validation_passed": True})
        assert rep2["validation_passed"] is False


class TestSectionwiseJournalWriter:
    def _writer_stub(self):
        from src.agents.writer import Writer

        w = object.__new__(Writer)

        class Ctx:
            log = []

        w.ctx = Ctx()
        w.agent_name = "Writer"
        return w

    def test_assembles_synthetic_document(self) -> None:
        w = self._writer_stub()
        calls = {"n": 0}

        def fake_llm(msg, **kw):
            calls["n"] += 1
            if "front matter only" in msg:
                return ("```latex\n" + chr(92) + "title{T}\n"
                        + chr(92) + "begin{abstract}A"
                        + " word" * 60 + chr(92) + "end{abstract}\n"
                        + chr(92) + "keywords{a, b, c, d}\n```")
            if "bibliography only" in msg:
                return "```bibtex\n@article{k1, title={X}, year={2024}}\n```"
            import re
            m = re.search(r"\\section\{\{?([^}]+)", msg)
            name = None
            for nm, _ in w.JOURNAL_SECTIONS:
                if nm in msg:
                    name = nm
            return ("```latex\n" + chr(92) + f"section{{{name}}}\n"
                    "Body text here.\n```\n"
                    f"SUMMARY: covered {name}.")

        w.call_llm = fake_llm
        tex, bib = w._run_journal_sectionwise("CTX", "@misc{fb}")
        assert calls["n"] == 2 + len(w.JOURNAL_SECTIONS)
        assert chr(92) + "title{T}" in tex
        assert chr(92) + "maketitle" in tex
        for nm, _ in w.JOURNAL_SECTIONS:
            assert f"\section{{{nm}}}" in tex
        assert tex.rstrip().endswith("\end{document}")
        assert "@article{k1" in bib

    def test_reassembly_accepts_synthetic_doc(self) -> None:
        from pathlib import Path

        from src.agents.writer import Writer

        w = self._writer_stub()
        root = Path(__file__).resolve().parent.parent
        template = (root / "templates" / "paper_template_journal.tex"
                    ).read_text(encoding="utf-8")
        synthetic = ("\title{A Long Title About Cognitive Diagnosis in "
                     "Tutoring Systems}\n"
                     "\begin{abstract}" + "word " * 60 + "\end{abstract}\n"
                     "\keywords{a, b, c}\n\maketitle\n\n"
                     "\section{Introduction}\nText.\n\n\end{document}\n")
        out = Writer._reassemble_from_template(w, synthetic, template)
        assert "%%PLACEHOLDER" not in out
        assert "\section{Introduction}" in out
        assert "\printbibliography" in out
        assert "..." in out or "Cognitive" in out  # shorttitle filled


class TestHumanizerDefault:
    @pytest.mark.parametrize("tt,ds", [
        ("prediction", "hsls09_public"),
        ("causal_did", "did_els_hsls_panel"),
        ("psychometrics", "assistments_0910"),
    ])
    def test_humanizer_injected_on_every_writer_call(self, tt, ds) -> None:
        # User decision 2026-07-10: the humanizer is a DEFAULT writer
        # skill. mandatory + all-task-types guarantees inclusion even
        # with a keyword-free context; this test pins that behavior.
        from pathlib import Path

        from src.orchestrator import _resolve_skill_caps
        from src.skills import SkillRegistry

        root = Path(__file__).resolve().parent.parent
        reg = SkillRegistry(str(root / "skills"))
        skills = reg.match_and_compose(
            task_type=tt, dataset=ds, stage="Writer",
            context="zzz nothing relevant qqq",
            top_k_per_layer=_resolve_skill_caps(tt),
        )
        assert "natural-academic-prose" in [s.name for s in skills]
