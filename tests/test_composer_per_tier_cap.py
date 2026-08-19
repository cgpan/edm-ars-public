"""V3.0 Phase 3b.8 / §6.1 — per-tier cap in format_skills_for_prompt.

The 3b.7 report surfaced F-3b7-FORMATTER-TRUNCATES-METHOD-SKILLS: under
the legacy uniform-cap rule, mandatory skills exceeding the budget
caused every recommended skill to be silently dropped. The 3b.8 fix
adds tier discipline: mandatory always renders in full; recommended
competes for ``max_chars`` of room independently of mandatory cost.

These tests anchor the new behavior and lock down the legacy path
behind ``mandatory_chars_unlimited=False`` for callers that want
byte-uniform constraints.
"""
from __future__ import annotations

import pytest

from src.skills.composer import format_skills_for_prompt
from src.skills.schema import Skill


def make_fake_skill(
    name: str,
    *,
    tier: str = "recommended",
    body_chars: int = 1000,
    priority: int = 5,
    layer: str = "methodology",
) -> Skill:
    """Build a Skill with body sized to approximately ``body_chars``.

    The actual rendered length will be slightly larger because of the
    header banner the composer adds — we don't try to be exact, just
    to make budget arithmetic predictable in tests.
    """
    return Skill(
        name=name,
        layer=layer,
        description=f"Fake skill {name} for composer tests.",
        body="x" * body_chars,
        rule_severity=tier,  # type: ignore[arg-type]
        priority=priority,
    )


# ---------------------------------------------------------------------------
# §6.1 acceptance tests
# ---------------------------------------------------------------------------


class TestPerTierCap:
    def test_mandatory_skills_render_in_full_above_max_chars(self) -> None:
        """Five mandatory skills with combined size 20K chars must all
        render even if max_chars=12000. This is the F-3b7 fix."""
        mandatory_skills = [
            make_fake_skill(name=f"m{i}", tier="mandatory", body_chars=4000)
            for i in range(5)
        ]
        output = format_skills_for_prompt(mandatory_skills, max_chars=12000)
        for s in mandatory_skills:
            assert s.name in output, f"Mandatory skill {s.name} was dropped"
        # Total rendered output exceeds max_chars — that is the point.
        assert len(output) > 12000

    def test_recommended_skills_dropped_when_budget_exceeded(self) -> None:
        """When non-mandatory total exceeds max_chars, recommended drops.

        Note: per the 3b.8 design, mandatory cost does NOT count against
        recommended budget; recommended competes for ``max_chars`` of its
        own. To trigger drops, recommended total alone must exceed it.
        """
        # 6 recommended skills @ ~1500 chars each = ~9000 budget pressure
        # against max_chars=2000 — at least 4 should drop.
        skills = [
            make_fake_skill(name=f"rec{i}", tier="recommended", body_chars=1500)
            for i in range(6)
        ]
        output = format_skills_for_prompt(skills, max_chars=2000)
        # At least one drop must have occurred.
        assert "Dropped from prompt due to budget:" in output
        # Some recommended skills survived; some were dropped.
        kept = [s.name for s in skills if s.name in output.split("Dropped from prompt due to budget:")[0]]
        assert 0 < len(kept) < 6

    def test_recommended_renders_when_budget_permits(self) -> None:
        """When mandatory leaves room and recommended fits, recommended
        renders normally. No drop diagnostic."""
        mandatory = [make_fake_skill(name="man", tier="mandatory", body_chars=4000)]
        recommended = [make_fake_skill(name="rec", tier="recommended", body_chars=2000)]
        output = format_skills_for_prompt(
            mandatory + recommended, max_chars=12000
        )
        assert "man" in output
        assert "rec" in output
        assert "Dropped from prompt due to budget" not in output

    def test_legacy_uniform_cap_path(self) -> None:
        """mandatory_chars_unlimited=False restores 3b.7-and-earlier
        behavior: shared budget across all tiers."""
        # Build a workload that exercises legacy truncation: mix of
        # mandatory + recommended exceeding max_chars together.
        mandatory = [make_fake_skill(name="man", tier="mandatory", body_chars=4000)]
        recommended = [
            make_fake_skill(name=f"rec{i}", tier="recommended", body_chars=2000, priority=2)
            for i in range(6)
        ]
        output = format_skills_for_prompt(
            mandatory + recommended,
            max_chars=8000,
            mandatory_chars_unlimited=False,
        )
        # Mandatory survives.
        assert "man" in output
        # Legacy truncation notice present (not the new diagnostic).
        assert "skills truncated:" in output
        assert "Dropped from prompt due to budget" not in output

    def test_drop_order_lowest_priority_first(self) -> None:
        """When non-mandatory exceeds budget, lowest priority drops first.
        Highest priority survives (priority=1 is highest in this convention).
        """
        # Both recommended skills are ~3000 chars; max_chars=4000 fits
        # exactly one. The high-priority one (priority=1) must survive.
        recommended_high = make_fake_skill(
            name="rec_hi", tier="recommended", priority=1, body_chars=3000
        )
        recommended_low = make_fake_skill(
            name="rec_lo", tier="recommended", priority=2, body_chars=3000
        )
        mandatory = make_fake_skill(name="man", tier="mandatory", body_chars=10000)

        output = format_skills_for_prompt(
            # Input order is deliberately scrambled so the test only
            # passes if drop order respects priority, not input order.
            [recommended_low, mandatory, recommended_high],
            max_chars=4000,
        )
        assert "man" in output
        # Header for the high-priority skill should appear; low-priority
        # skill should appear in the drop diagnostic.
        assert "rec_hi" in output.split("Dropped from prompt due to budget:")[0]
        assert "rec_lo" in output  # name appears in the drop diagnostic
        # And explicitly: rec_lo is dropped, rec_hi is not.
        drop_clause = output.split("Dropped from prompt due to budget:")[-1]
        assert "rec_lo" in drop_clause
        assert "rec_hi" not in drop_clause


# ---------------------------------------------------------------------------
# Drop-diagnostic format
# ---------------------------------------------------------------------------


class TestDropDiagnosticFormat:
    def test_diagnostic_lists_dropped_skills_and_char_counts(self) -> None:
        """The diagnostic comment line must list each dropped skill with
        its char count, comma-separated."""
        mandatory = make_fake_skill(name="man", tier="mandatory", body_chars=1000)
        # Two recommended skills that won't fit under max_chars=1500.
        recs = [
            make_fake_skill(name="big_a", tier="recommended", priority=2, body_chars=2000),
            make_fake_skill(name="big_b", tier="recommended", priority=3, body_chars=2000),
        ]
        output = format_skills_for_prompt([mandatory, *recs], max_chars=1500)
        # At least one skill dropped; format contains "(N chars)".
        assert "Dropped from prompt due to budget:" in output
        # Char counts are positive integers.
        import re

        matches = re.findall(r"(\w[\w-]*) \((\d+) chars\)", output)
        assert len(matches) >= 1
        for _name, n_str in matches:
            assert int(n_str) > 0

    def test_no_diagnostic_when_no_drops(self) -> None:
        """No diagnostic comment when nothing was dropped."""
        mandatory = make_fake_skill(name="man", tier="mandatory", body_chars=4000)
        rec = make_fake_skill(name="rec", tier="recommended", body_chars=1000)
        output = format_skills_for_prompt([mandatory, rec], max_chars=12000)
        assert "Dropped from prompt due to budget" not in output


# ---------------------------------------------------------------------------
# Backward compatibility — empty list, single skill, etc.
# ---------------------------------------------------------------------------


class TestComposerBackwardCompat:
    def test_empty_skill_list_returns_empty_string(self) -> None:
        assert format_skills_for_prompt([], max_chars=12000) == ""

    def test_single_mandatory_skill_renders_with_banner(self) -> None:
        s = make_fake_skill(name="solo", tier="mandatory", body_chars=500)
        output = format_skills_for_prompt([s], max_chars=12000)
        assert "## MANDATORY RULE: solo" in output

    def test_single_recommended_skill_renders_with_guidance_header(self) -> None:
        s = make_fake_skill(name="solo", tier="recommended", body_chars=500)
        output = format_skills_for_prompt([s], max_chars=12000)
        assert "## Guidance: solo" in output

    def test_severity_ordering_preserved(self) -> None:
        """Mandatory must appear before recommended in the rendered output."""
        rec = make_fake_skill(name="rec", tier="recommended", body_chars=500)
        man = make_fake_skill(name="man", tier="mandatory", body_chars=500)
        output = format_skills_for_prompt([rec, man], max_chars=12000)
        assert output.index("MANDATORY RULE: man") < output.index("Guidance: rec")
