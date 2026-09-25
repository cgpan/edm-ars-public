"""The setup wizard (edmars/wizard.py), driven through scripted fake prompts.

Every collaborator is a fake from ``wizard_fakes``: no keyring, no network,
nothing outside ``tmp_path``. Each test scripts the answers a user would
give and checks what was saved, what was said, and what was never shown.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.cli.wizard_fakes import DEFAULT, Fakes, KeyCheck, NonInteractiveError, install_fakes

GOOD_KEY = "sk-fake-deepseek-0123456789abcdef"
BAD_KEY = "sk-fake-rejected-0123456789abcdef"
OTHER_KEY = "sk-fake-other-0123456789abcdefgh"
ACK = "2026-09-25"


@pytest.fixture
def fx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Fakes:
    return install_fakes(monkeypatch, tmp_path)


def run(section: str | None = None, **kw: object) -> int:
    from edmars.wizard import run_setup

    return run_setup(section, **kw)  # type: ignore[arg-type]


def acknowledged(**extra: object) -> dict[str, object]:
    out: dict[str, object] = {"acknowledged": {"version": ACK, "at": "2026-09-25T10:00:00Z"}}
    out.update(extra)
    return out


def prompt_messages(fx: Fakes) -> list[str]:
    return [message for _, message, _ in fx.ui.prompts]


# ---------------------------------------------------------------------------
# The full interactive flow
# ---------------------------------------------------------------------------

FULL_FLOW = [
    "continue",        # S0 welcome
    "accept",          # S1 notice
    "continue",        # S2 computer check
    DEFAULT,           # S3 studies folder (recommended)
    "deepseek",        # S4 service
    "paste",           # S4 key menu
    GOOD_KEY,          # S4 hidden paste
    "skip",            # S5 Semantic Scholar
    "skip",            # S6 datasets
    "skip",            # S7 PDFs
    "later",           # S8 R
    "skip",            # S9 reviewer
    "Ada Lovelace",    # S10 name
    "",                # S10 affiliation
    "skip",            # S10 advanced
]


def test_full_flow_saves_every_answer_and_never_shows_the_key(fx: Fakes) -> None:
    fx.ui.script = FULL_FLOW + ["later"]  # S11: start a study later

    assert run() == 0

    saved = fx.saved()
    assert saved["acknowledged"]["version"] == ACK
    assert saved["provider"] == "deepseek"
    assert saved["author"]["name"] == "Ada Lovelace"
    assert saved["author"]["affiliation"] is None
    assert saved["latex"]["mode"] == "none"
    assert saved["setup_progress"]["last_completed_screen"] == "S11"
    assert saved["setup_progress"]["completed_at"]
    studies = Path(saved["studies_dir"])
    assert studies == fx.paths.default_studies_dir() and studies.is_dir()
    assert fx.secrets.store == {"DEEPSEEK_API_KEY": GOOD_KEY}
    assert "Saved in" in fx.ui.output
    assert GOOD_KEY not in fx.ui.output
    assert GOOD_KEY not in fx.paths.settings_path().read_text(encoding="utf-8")
    assert fx.ui.script == []
    assert fx.cli.calls == []


def test_every_screen_saves_progress_and_links_are_printed_in_full(fx: Fakes) -> None:
    fx.ui.script = FULL_FLOW + ["later"]
    run()
    out = fx.ui.output
    assert "https://platform.deepseek.com/api_keys" in out
    assert "https://www.semanticscholar.org/product/api#api-key-form" in out
    # one save per screen at least (12 screens)
    assert fx.settings.saves >= 12
    # the recommended provider is marked, and ChatGPT Plus is explained
    s4 = next(choices for kind, msg, choices in fx.ui.prompts if msg == "Which AI service do you want to use?")
    assert s4 is not None and "recommended" in dict(s4)["deepseek"]
    assert "ChatGPT Plus or Claude Pro subscription is not an API key" in out


def test_start_first_study_hands_over_to_edmars_new(fx: Fakes) -> None:
    fx.ui.script = FULL_FLOW + ["new"]
    assert run() == 0
    assert fx.cli.calls == [["new"]]


def test_quit_keeps_progress_and_the_next_run_resumes(fx: Fakes) -> None:
    fx.ui.script = ["continue", "accept", "continue", DEFAULT, "__quit__"]
    assert run() == 130
    assert fx.saved()["setup_progress"]["last_completed_screen"] == "S3"
    assert "Setup paused" in fx.ui.output

    fx.ui.prompts.clear()
    fx.ui.script = ["resume", "__quit__"]
    assert run() == 130
    messages = prompt_messages(fx)
    assert messages[0].startswith("You stopped setup after step 3")
    assert messages[1] == "Which AI service do you want to use?"


def test_go_back_returns_to_the_previous_screen(fx: Fakes) -> None:
    fx.ui.script = ["continue", "accept", "continue", "__back__", "continue", DEFAULT, "__quit__"]
    assert run() == 130
    assert prompt_messages(fx).count("Continue?") == 2
    assert fx.saved()["setup_progress"]["last_completed_screen"] == "S3"


def test_ctrl_c_pauses_setup(fx: Fakes) -> None:
    fx.ui.script = [KeyboardInterrupt()]
    assert run() == 130
    assert "Setup paused" in fx.ui.output


def test_notice_must_be_accepted_to_continue(fx: Fakes) -> None:
    fx.ui.script = ["continue", "disclaimer", "privacy", "__quit__"]
    assert run() == 130
    saved = fx.saved()
    assert not saved.get("acknowledged")
    assert "Disclaimer" in fx.ui.output and "No telemetry" in fx.ui.output
    # markdown links in the full texts are printed literally
    assert "[PRIVACY.md](PRIVACY.md)" in fx.ui.output


def test_accepting_the_notice_is_never_what_enter_does(fx: Fakes) -> None:
    # Pressing Enter through setup used to record consent: "accept" was the
    # default and the first option. Now there is no default and the arrow-key
    # menu starts on "Read the full disclaimer first".
    fx.ui.script = ["continue", DEFAULT]
    with pytest.raises(AssertionError, match="offers no default"):
        run()
    assert not fx.saved().get("acknowledged")
    message, choices = next((m, c) for kind, m, c in fx.ui.prompts if m.startswith("Do you understand"))
    assert fx.ui.defaults[message] is None
    assert choices is not None and choices[0][0] != "accept"
    assert "Enter alone does not accept" in message


def test_dataset_terms_need_an_explicit_agreement(fx: Fakes) -> None:
    fx.ui.script = ["download", DEFAULT]
    with pytest.raises(AssertionError, match="offers no default"):
        run("datasets")
    assert "terms_accepted_at" not in (fx.saved().get("datasets", {}).get("hsls09_public") or {})
    message, choices = next((m, c) for kind, m, c in fx.ui.prompts if m.startswith("Do you agree"))
    assert fx.ui.defaults[message] is None
    assert choices is not None and choices[0][0] == "no"


def test_non_tty_prompt_failure_is_a_clear_message(fx: Fakes) -> None:
    fx.ui.script = [NonInteractiveError("EDM-ARS needs an answer to \"Ready\" but cannot ask (it reached the end "
                                        "of the input, so no one is there to answer).")]
    assert run() == 1
    # Shown once, as written: it used to be wrapped in "Setup needs an answer,
    # but it is running without a terminal (...)", which is false at end of input.
    assert "[x] EDM-ARS needs an answer to \"Ready\" but cannot ask" in fx.ui.output
    assert "without a terminal" not in fx.ui.output
    assert "Run `edmars setup` in a terminal window" in fx.ui.output


# ---------------------------------------------------------------------------
# Re-running setup
# ---------------------------------------------------------------------------

def test_completed_setup_asks_what_to_change(fx: Fakes) -> None:
    fx.write_settings(**acknowledged(setup_progress={"last_completed_screen": "S11"}))
    fx.ui.script = ["literature", "skip", "done"]
    assert run() == 0
    messages = prompt_messages(fx)
    assert messages[0] == "EDM-ARS is already set up. What would you like to change?"
    assert messages[1] == "Semantic Scholar key"


def test_a_changed_notice_is_asked_again_before_the_menu(fx: Fakes) -> None:
    fx.write_settings(acknowledged={"version": "2025-01-01", "at": "2025-01-01T00:00:00Z"},
                      setup_progress={"last_completed_screen": "S11"})
    fx.ui.script = ["accept", "done"]
    assert run() == 0
    assert fx.saved()["acknowledged"]["version"] == ACK
    assert "has changed" in fx.ui.output


def test_unknown_section_lists_the_sections(fx: Fakes) -> None:
    assert run("frobnicate") == 2
    assert "reviewer" in fx.ui.output and "literature" in fx.ui.output


@pytest.mark.parametrize("typed, expected", [
    ("ai", "ai"), ("LSAR", "reviewer"), ("latex", "pdf"), ("data", "datasets"),
    ("semantic_scholar", "literature"), ("start-over", "__all__"), (None, None), ("  ", None),
])
def test_section_names_and_aliases(typed: str | None, expected: str | None) -> None:
    from edmars.wizard import resolve_section

    assert resolve_section(typed) == expected


def test_a_damaged_settings_file_can_be_replaced(fx: Fakes) -> None:
    fx.paths.settings_path().write_text("- not: [a mapping\n", encoding="utf-8")
    fx.ui.script = [True, "skip"]  # start fresh, then S5 skip
    assert run("literature") == 0
    assert (fx.home / "settings.yaml.bak").is_file()


# ---------------------------------------------------------------------------
# S4: the AI service and its key
# ---------------------------------------------------------------------------

def test_rejected_key_says_so_and_a_new_paste_goes_straight_to_the_prompt(fx: Fakes) -> None:
    fx.providers.results[BAD_KEY] = KeyCheck("REJECTED", "401")
    fx.ui.script = ["deepseek", "paste", BAD_KEY, "paste", GOOD_KEY]
    assert run("ai") == 0
    assert "didn't accept this key" in fx.ui.output
    assert fx.secrets.store["DEEPSEEK_API_KEY"] == GOOD_KEY
    assert BAD_KEY not in fx.ui.output and GOOD_KEY not in fx.ui.output
    kinds = [kind for kind, _, _ in fx.ui.prompts]
    assert kinds == ["select", "select", "secret", "select", "secret"]


def test_no_credit_points_to_the_top_up_page_and_can_keep_the_key(fx: Fakes) -> None:
    fx.providers.results[GOOD_KEY] = KeyCheck("NO_CREDIT", "402")
    fx.ui.script = ["deepseek", "paste", GOOD_KEY, "save"]
    assert run("ai") == 0
    assert "https://platform.deepseek.com/top_up" in fx.ui.output
    assert fx.secrets.store["DEEPSEEK_API_KEY"] == GOOD_KEY


def test_network_failure_can_be_rechecked(fx: Fakes) -> None:
    fx.providers.results[GOOD_KEY] = [KeyCheck("NETWORK", "timeout"), KeyCheck("OK", "fine")]
    fx.ui.script = ["deepseek", "paste", GOOD_KEY, "recheck"]
    assert run("ai") == 0
    assert "Couldn't reach DeepSeek" in fx.ui.output
    assert "University networks sometimes block AI services" in fx.ui.output
    assert fx.secrets.store["DEEPSEEK_API_KEY"] == GOOD_KEY


def test_skipping_the_key_warns_that_studies_cannot_start(fx: Fakes) -> None:
    fx.ui.script = ["openai", "skip"]
    assert run("ai") == 0
    assert fx.saved()["provider"] == "openai"
    assert "no working key is saved yet" in fx.ui.output
    assert fx.secrets.store == {}


def test_key_already_in_the_environment_can_be_copied_to_secure_storage(
        fx: Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", GOOD_KEY)
    fx.ui.script = ["deepseek", "keep", True]
    assert run("ai") == 0
    assert fx.secrets.store["DEEPSEEK_API_KEY"] == GOOD_KEY
    key_menu = dict(fx.ui.prompts[1][2] or [])
    assert "environment variable DEEPSEEK_API_KEY" in key_menu["keep"]


def test_key_from_an_old_dotenv_is_offered_and_moved(fx: Fakes) -> None:
    (fx.app / ".env").write_text(f"DEEPSEEK_API_KEY={GOOD_KEY}\nOPENAI_API_KEY=your-key-here\n",
                                 encoding="utf-8")
    fx.ui.script = ["deepseek", "dotenv"]
    assert run("ai") == 0
    assert fx.secrets.store["DEEPSEEK_API_KEY"] == GOOD_KEY
    assert "delete the old .env" in fx.ui.output


def test_key_storage_failure_is_reported_without_the_key(fx: Fakes) -> None:
    fx.secrets.fail_on_set = RuntimeError(f"keyring broke while saving {GOOD_KEY}")
    fx.ui.script = ["deepseek", "paste", GOOD_KEY]
    assert run("ai") == 0
    assert "Couldn't save the key" in fx.ui.output
    assert GOOD_KEY not in fx.ui.output


def test_retired_models_are_flagged_after_a_good_key(fx: Fakes) -> None:
    fx.providers.retired = {"deepseek-flash"}
    fx.ui.script = ["deepseek", "paste", GOOD_KEY]
    assert run("ai") == 0
    assert "can't use these models: deepseek-flash" in fx.ui.output


def test_switching_service_resets_model_overrides(fx: Fakes) -> None:
    fx.write_settings(provider="deepseek", models={"writer": "deepseek-v4-pro"})
    fx.ui.script = ["anthropic", "paste", OTHER_KEY]
    assert run("ai") == 0
    saved = fx.saved()
    assert saved["provider"] == "anthropic" and saved["models"] == {}
    assert fx.secrets.store["ANTHROPIC_API_KEY"] == OTHER_KEY
    assert "less tested" in fx.ui.output


def test_local_model_server(fx: Fakes) -> None:
    fx.ui.script = ["local", "http://localhost:11434/v1", "no", "qwen2.5:72b"]
    assert run("ai") == 0
    saved = fx.saved()
    assert saved["provider"] == "local"
    assert saved["provider_base_url"] == "http://localhost:11434/v1"
    assert set(saved["models"].values()) == {"qwen2.5:72b"}
    assert {"analyst", "critic", "writer", "revision_writer"} <= set(saved["models"])
    assert fx.secrets.store["OPENAI_API_KEY"] == "local"
    assert "128k" in fx.ui.output


def test_local_server_offers_to_replace_a_real_openai_key(fx: Fakes) -> None:
    fx.secrets.store["OPENAI_API_KEY"] = OTHER_KEY
    fx.ui.script = ["local", "http://localhost:1234/v1", "no", "llama3.1:70b", True]
    assert run("ai") == 0
    assert fx.secrets.store["OPENAI_API_KEY"] == "local"


# ---------------------------------------------------------------------------
# S3, S5 to S10
# ---------------------------------------------------------------------------

def test_synced_studies_folder_warns_and_offers_the_local_default(fx: Fakes, tmp_path: Path) -> None:
    synced = tmp_path / "OneDrive" / "studies"
    fx.ui.script = ["other", str(synced), "default"]
    assert run("folders") == 0
    assert "synced to OneDrive" in fx.ui.output
    assert Path(fx.saved()["studies_dir"]) == fx.paths.default_studies_dir()


def test_quoted_pasted_folder_path_is_accepted(fx: Fakes, tmp_path: Path) -> None:
    target = tmp_path / "My Studies"
    fx.ui.script = ["other", f'"{target}"']
    assert run("folders") == 0
    assert Path(fx.saved()["studies_dir"]) == target and target.is_dir()


def test_semantic_scholar_key_is_checked_and_saved(fx: Fakes) -> None:
    fx.ui.script = ["add", "s2-fake-key-0123456789"]
    assert run("literature") == 0
    assert fx.secrets.store["SEMANTIC_SCHOLAR_API_KEY"] == "s2-fake-key-0123456789"
    assert fx.saved()["literature"]["semantic_scholar_key_set"] is True
    assert "arXiv and Crossref need no key" in fx.ui.output


def test_dataset_download_records_terms_and_shows_progress(fx: Fakes) -> None:
    fx.ui.script = ["download", "agree", "continue"]
    assert run("datasets") == 0
    saved = fx.saved()["datasets"]["hsls09_public"]
    assert saved["terms_accepted_at"] and saved["verified_at"] and saved["path"]
    assert fx.datasets.progress_calls
    assert saved["sha256"]  # the fingerprint install() records on first download
    assert "Cite NCES" in fx.ui.output
    # Plain mode: the unzip after the download gets its own lines.
    assert "Unpacking:" in fx.ui.output


def test_dataset_download_failure_is_explained(fx: Fakes) -> None:
    fx.datasets.download_error = ConnectionError("connection reset by peer")
    fx.ui.script = ["download", "agree", "skip"]
    assert run("datasets") == 0
    assert "The download stopped" in fx.ui.output
    assert "continues where it stopped" in fx.ui.output


def test_importing_the_numeric_file_explains_the_labeled_one(fx: Fakes, tmp_path: Path) -> None:
    numeric = tmp_path / "numeric.csv"
    numeric.write_text("STU_ID,X1SEX,X1RACE,X3TGPAACAD,X4EVRATNDCLG\n1,1,8,3.1,1\n", encoding="utf-8")
    labeled = tmp_path / "labeled.csv"
    labeled.write_text("STU_ID,X1SEX,X1RACE,X3TGPAACAD,X4EVRATNDCLG\n1,Male,White,3.1,Yes\n", encoding="utf-8")
    fx.ui.script = ["import", str(numeric), f"& '{labeled}'", "continue"]
    assert run("datasets") == 0
    assert "numeric-coded" in fx.ui.output
    assert fx.datasets.imported == [labeled]
    assert "hsls09_public" in fx.datasets.ready


def test_existing_latex_is_test_compiled(fx: Fakes) -> None:
    fx.proc.tools["pdflatex"] = "/usr/bin/pdflatex"
    fx.ui.script = ["system"]
    assert run("pdf") == 0
    assert fx.saved()["latex"] == {"mode": "system", "pdflatex": "/usr/bin/pdflatex"}
    assert fx.toolchain.compile_calls == 1


def test_tinytex_install_needs_consent(fx: Fakes) -> None:
    fx.ui.script = ["tinytex", False, "skip"]
    assert run("pdf") == 0
    assert not fx.toolchain.tinytex_installed
    assert fx.saved()["latex"]["mode"] == "none"

    fx.ui.script = ["tinytex", DEFAULT]
    assert run("pdf") == 0
    assert fx.toolchain.tinytex_installed
    assert fx.saved()["latex"]["mode"] == "tinytex"
    # TinyTeX's pdflatex is not on PATH in this process; setup still saves
    # it, so the runner can put its folder on the study's PATH.
    assert fx.saved()["latex"]["pdflatex"] == str(fx.toolchain.tinytex_dir / "pdflatex")


def test_r_folder_path_is_normalized_and_packages_installed(fx: Fakes, tmp_path: Path) -> None:
    exe = "Rscript.exe" if os.name == "nt" else "Rscript"
    r_home = tmp_path / "R-4.5.1"
    (r_home / "bin").mkdir(parents=True)
    (r_home / "bin" / exe).write_text("", encoding="utf-8")
    fx.toolchain.packages_missing = True
    fx.ui.script = ["find", str(r_home), DEFAULT]
    assert run("r") == 0
    saved = fx.saved()["r"]
    assert Path(saved["rscript"]) == r_home / "bin" / exe
    assert saved["packages_ok"] is True


def test_reviewer_asks_for_a_deepseek_key_even_with_another_service(fx: Fakes) -> None:
    fx.write_settings(provider="openai")
    fx.secrets.store["OPENAI_API_KEY"] = OTHER_KEY
    fx.ui.script = ["auto", "paste", GOOD_KEY, DEFAULT]
    assert run("reviewer") == 0
    saved = fx.saved()["lsar"]
    assert saved["enabled"] is True and saved["auto_review"] is True
    # The exact commit install() recorded, not the LSAR_REF branch name.
    assert saved["home"] == str(fx.lsar.home) and saved["ref"] == fx.lsar.commit
    assert fx.secrets.store["DEEPSEEK_API_KEY"] == GOOD_KEY
    assert "Your studies still use OpenAI" in fx.ui.output
    assert "differ by about 2 points" in fx.ui.output


def test_reviewer_stays_off_without_a_deepseek_key(fx: Fakes) -> None:
    fx.write_settings(provider="openai")
    fx.ui.script = ["manual", "skip"]
    assert run("reviewer") == 0
    assert fx.saved()["lsar"]["enabled"] is False
    assert fx.lsar.install_calls == 0
    assert "stays off" in fx.ui.output


def test_reviewer_back_from_the_key_returns_to_the_question(fx: Fakes) -> None:
    fx.ui.script = ["auto", "__back__", "skip"]
    assert run("reviewer") == 0
    assert fx.saved()["lsar"]["enabled"] is False


def test_advanced_options(fx: Fakes) -> None:
    fx.ui.script = ["Grace Hopper", "Example University", "show", "budget", "US$2.50", "venue", "JEDM",
                    "format", "journal", "done"]
    assert run("advanced") == 0
    saved = fx.saved()
    assert saved["author"] == {"name": "Grace Hopper", "affiliation": "Example University"}
    assert saved["defaults"]["budget_usd"] == 2.5
    assert saved["defaults"]["venue"] == "JEDM"
    assert saved["defaults"]["paper_format"] == "journal"
    assert "does not stop the study" in fx.ui.output
    # Study-finished notifications were never built; the menu must not offer them.
    menu = next(c for kind, m, c in fx.ui.prompts if m == "Advanced options")
    assert not any("otif" in label for _, label in menu or [])


def test_spending_warning_says_where_it_appears_and_when_it_cannot(fx: Fakes) -> None:
    fx.ui.script = ["", "", "show", "budget", "3", "done"]
    assert run("advanced") == 0
    assert "a warning appears in the study's progress messages" in fx.ui.output
    assert "prices only for DeepSeek" not in fx.ui.output

    fx.write_settings(provider="openai")
    fx.ui.lines.clear()
    fx.ui.script = ["", "", "show", "budget", "3", "done"]
    assert run("advanced") == 0
    # Only DeepSeek's models have prices, so the warning can never fire here.
    assert "prices only for DeepSeek's models" in fx.ui.output and "will not appear" in fx.ui.output


def _venue_labels(fx: Fakes) -> dict[str, str]:
    menu = [choices for kind, message, choices in fx.ui.prompts if message == "Default venue"]
    assert menu, "the Default venue menu was not shown"
    return dict(menu[-1] or [])


def test_venue_labels_come_from_the_installed_reviewers_calibration(fx: Fakes) -> None:
    # JEDM and AERA Open carry a benchmark in the installed calibration, so
    # the review gate holds papers to it; setup once said "no benchmark".
    calibration = fx.lsar.home / "calibration"
    calibration.mkdir(parents=True)
    (calibration / "anchors_edm.yaml").write_text(
        "overall_p25_full: 6.3\n"
        "venues:\n"
        "  JEDM: {p25: 5.15}\n"
        "  AERA_OPEN: {p25: 6.6}\n"
        "  JLA: {}\n",
        encoding="utf-8")
    fx.write_settings(lsar={"enabled": True, "auto_review": True, "home": str(fx.lsar.home)})
    fx.ui.script = ["", "", "show", "venue", "AERA_OPEN", "done"]
    assert run("advanced") == 0
    labels = _venue_labels(fx)
    assert "benchmark (6.6)" in labels["AERA_OPEN"] and "no benchmark" not in labels["AERA_OPEN"]
    assert "benchmark (5.15)" in labels["JEDM"]
    assert "benchmark (6.3)" in labels["EDM"]
    assert "score only, no benchmark" in labels["JLA"]
    assert fx.saved()["defaults"]["venue"] == "AERA_OPEN"


def test_venue_labels_say_nothing_about_reviews_without_the_reviewer(fx: Fakes) -> None:
    fx.ui.script = ["", "", "show", "venue", DEFAULT, "done"]
    assert run("advanced") == 0
    labels = _venue_labels(fx)
    assert not any("benchmark" in label or "reviewer" in label for label in labels.values())


# ---------------------------------------------------------------------------
# Non-interactive mode
# ---------------------------------------------------------------------------

def test_noninteractive_requires_accepting_the_notice(fx: Fakes) -> None:
    assert run(non_interactive=True, options={}) == 1
    assert "--accept-disclosure" in fx.ui.output
    assert fx.ui.prompts == []


def test_noninteractive_full_setup_with_the_standard_key_variable(
        fx: Fakes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", GOOD_KEY)
    studies = tmp_path / "ci-studies"
    code = run(non_interactive=True, options={"accept_disclosure": True, "studies_dir": str(studies),
                                              "author_name": "CI Runner", "venue": "edm"})
    assert code == 0, fx.ui.output
    saved = fx.saved()
    assert saved["acknowledged"]["version"] == ACK
    assert saved["setup_progress"]["last_completed_screen"] == "S11"
    assert saved["studies_dir"] == str(studies) and studies.is_dir()
    assert saved["author"]["name"] == "CI Runner" and saved["defaults"]["venue"] == "EDM"
    assert fx.secrets.set_calls == []  # already in the standard variable: nothing copied
    assert fx.ui.prompts == []
    assert GOOD_KEY not in fx.ui.output


def test_noninteractive_reads_options_from_the_environment(fx: Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDMARS_ACCEPT_DISCLOSURE", "yes")
    monkeypatch.setenv("EDMARS_PROVIDER", "openai")
    monkeypatch.setenv("EDMARS_MODEL", "gpt-test-large")
    monkeypatch.setenv("OPENAI_API_KEY", OTHER_KEY)
    assert run(non_interactive=True) == 0
    assert fx.saved()["provider"] == "openai"
    assert set(fx.saved()["models"].values()) == {"gpt-test-large"}


def test_openai_needs_a_model_because_none_is_shipped(fx: Fakes) -> None:
    fx.providers.results[OTHER_KEY] = KeyCheck("OK", "key accepted", models=["gpt-b", "gpt-a"])
    fx.ui.script = ["openai", "paste", OTHER_KEY, "gpt-a"]
    assert run("ai") == 0
    models = fx.saved()["models"]
    # One model for every step the pipeline calls, including the reviser.
    assert set(models.values()) == {"gpt-a"}
    assert {"critic", "writer", "revision_writer", "outline_agent"} <= set(models)
    assert "EDM-ARS will use OpenAI" in fx.ui.output


def test_openai_model_list_offers_text_models_newest_first_with_no_default(fx: Fakes) -> None:
    # GET /v1/models lists every model the account can reach. Sorted A-Z and
    # cut at 40, the old menu pre-selected babbage-002 for every step and cut
    # off the gpt-5 models the prompt told people to pick.
    listed = ["babbage-002", "chatgpt-4o-latest", "dall-e-2", "dall-e-3", "davinci-002", "gpt-3.5-turbo",
              "gpt-3.5-turbo-0125", "gpt-4", "gpt-4-0613", "gpt-4.1", "gpt-4o", "gpt-4o-2024-08-06",
              "gpt-4o-audio-preview", "gpt-4o-mini", "gpt-4o-mini-tts", "gpt-4o-realtime-preview",
              "gpt-4o-search-preview", "gpt-4o-transcribe", "gpt-5", "gpt-5-mini", "gpt-5.1", "gpt-image-1",
              "o1", "o3", "o3-mini", "o4-mini", "omni-moderation-latest", "text-embedding-3-large",
              "text-embedding-3-small", "tts-1", "whisper-1"]
    listed += [f"ft:gpt-3.5-turbo:example-org:tuned-{i}" for i in range(40)]
    fx.providers.results[OTHER_KEY] = KeyCheck("OK", "key accepted", models=listed)
    fx.ui.script = ["openai", "paste", OTHER_KEY, "gpt-5"]
    assert run("ai") == 0
    assert set(fx.saved()["models"].values()) == {"gpt-5"}
    message, choices = next((m, c) for kind, m, c in fx.ui.prompts if m.startswith("Which OpenAI model"))
    assert fx.ui.defaults[message] is None
    offered = [value for value, _ in choices or []]
    assert offered[0] == "gpt-5.1" and {"gpt-5", "gpt-5-mini", "gpt-4o", "o3"} <= set(offered)
    for unusable in ("babbage-002", "davinci-002", "dall-e-3", "tts-1", "whisper-1", "gpt-image-1",
                     "text-embedding-3-large", "omni-moderation-latest", "gpt-4o-realtime-preview",
                     "gpt-4o-transcribe", "gpt-4o-search-preview", "gpt-4o-audio-preview"):
        assert unusable not in offered
    assert "__type__" in offered


def test_noninteractive_openai_without_a_model_fails(fx: Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", OTHER_KEY)
    code = run(non_interactive=True, options={"accept_disclosure": True, "provider": "openai"})
    assert code == 1
    assert "needs a model name" in fx.ui.output


def test_noninteractive_custom_key_variable_is_copied_to_secure_storage(
        fx: Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI_DEEPSEEK", GOOD_KEY)
    assert run(non_interactive=True, options={"accept_disclosure": True, "key_env": "CI_DEEPSEEK"}) == 0
    assert fx.secrets.store["DEEPSEEK_API_KEY"] == GOOD_KEY


def test_noninteractive_missing_key_fails(fx: Fakes) -> None:
    assert run(non_interactive=True, options={"accept_disclosure": True}) == 1
    assert "No DeepSeek key found" in fx.ui.output


def test_noninteractive_rejected_key_is_not_saved(fx: Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI_DEEPSEEK", BAD_KEY)
    fx.providers.results[BAD_KEY] = KeyCheck("REJECTED", "401")
    assert run(non_interactive=True, options={"accept_disclosure": True, "key_env": "CI_DEEPSEEK"}) == 1
    assert "DEEPSEEK_API_KEY" not in fx.secrets.store


def test_noninteractive_can_skip_the_live_key_check(fx: Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", GOOD_KEY)
    assert run("ai", non_interactive=True, options={"check_keys": "false"}) == 0
    assert fx.providers.calls == []


def test_noninteractive_reviewer_needs_a_deepseek_key(fx: Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", OTHER_KEY)
    code = run(non_interactive=True, options={"accept_disclosure": True, "provider": "openai",
                                              "model": "gpt-test-large", "lsar_action": "auto"})
    assert code == 1
    assert fx.saved()["lsar"]["enabled"] is False
    assert "needs a DeepSeek key" in fx.ui.output
    assert fx.lsar.install_calls == 0


def test_noninteractive_reviewer_installs_with_a_deepseek_key(fx: Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", GOOD_KEY)
    assert run(non_interactive=True, options={"accept_disclosure": True, "lsar_action": "auto"}) == 0
    saved = fx.saved()["lsar"]
    assert saved["enabled"] is True and saved["auto_review"] is True
    assert fx.lsar.install_calls == 1


def test_noninteractive_dataset_download(fx: Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", GOOD_KEY)
    assert run(non_interactive=True, options={"accept_disclosure": True, "dataset_action": "download"}) == 0
    assert "hsls09_public" in fx.datasets.ready
    assert fx.saved()["datasets"]["hsls09_public"]["terms_accepted_at"]


def test_noninteractive_import_of_a_numeric_file_fails_clearly(fx: Fakes, tmp_path: Path) -> None:
    numeric = tmp_path / "numeric.csv"
    numeric.write_text("STU_ID,X1SEX\n1,1\n", encoding="utf-8")
    code = run("datasets", non_interactive=True, options={"dataset_action": "import",
                                                          "dataset_path": str(numeric)})
    assert code == 1
    assert "numeric-coded" in fx.ui.output


def test_noninteractive_local_server(fx: Fakes) -> None:
    code = run("ai", non_interactive=True, options={"provider": "local", "base_url": "http://localhost:8000/v1"})
    assert code == 0, fx.ui.output
    saved = fx.saved()
    assert saved["provider"] == "local" and set(saved["models"].values()) == {"llama3.1:70b"}


def test_noninteractive_unknown_values_are_errors(fx: Fakes) -> None:
    code = run("advanced", non_interactive=True, options={"venue": "NeurIPS", "budget_usd": "lots",
                                                          "bogus_option": 1})
    assert code == 1
    out = fx.ui.output
    assert "Unknown venue" in out and "budget_usd must be a number" in out and "bogus_option" in out


# ---------------------------------------------------------------------------
# House rules
# ---------------------------------------------------------------------------

def test_wizard_and_doctor_never_spawn_processes_themselves() -> None:
    root = Path(__file__).resolve().parents[2] / "edmars"
    for name in ("wizard.py", "doctor.py"):
        source = (root / name).read_text(encoding="utf-8")
        assert "import subprocess" not in source and "subprocess." not in source, name
        assert "shell=True" not in source, name
        assert "os.system" not in source, name


def test_final_check_does_not_call_the_finished_setup_unfinished(fx: Fakes) -> None:
    fx.ui.script = FULL_FLOW + ["later"]
    run()
    assert "Setup was not finished" not in fx.ui.output


def test_a_key_file_is_used_only_with_consent(fx: Fakes) -> None:
    from tests.cli.wizard_fakes import SecretStoreError

    fx.secrets.fail_on_set = SecretStoreError("Could not save DEEPSEEK_API_KEY in the credential store")
    fx.ui.script = ["deepseek", "paste", GOOD_KEY, False, "skip"]
    assert run("ai") == 0
    assert "DEEPSEEK_API_KEY" not in fx.secrets.store
    assert "only your user account can read" in fx.ui.output

    fx.ui.script = ["deepseek", "paste", GOOD_KEY, True]
    assert run("ai") == 0
    assert fx.secrets.store["DEEPSEEK_API_KEY"] == GOOD_KEY


def test_noninteractive_key_file_needs_the_option(fx: Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.cli.wizard_fakes import SecretStoreError

    monkeypatch.setenv("CI_DEEPSEEK", GOOD_KEY)
    fx.secrets.fail_on_set = SecretStoreError("no credential store")
    options = {"accept_disclosure": True, "key_env": "CI_DEEPSEEK"}
    assert run(non_interactive=True, options=options) == 1
    assert "allow_key_file=yes" in fx.ui.output
    assert "DEEPSEEK_API_KEY" not in fx.secrets.store
    assert run(non_interactive=True, options={**options, "allow_key_file": True}) == 0
    assert fx.secrets.store["DEEPSEEK_API_KEY"] == GOOD_KEY
