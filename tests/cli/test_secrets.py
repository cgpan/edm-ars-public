"""Keys: lookup precedence, storage, the consent-gated file fallback, redaction.

Every value below contains "fake" so the public-mirror audit recognises it
as a fixture rather than a leaked credential.
"""

from __future__ import annotations

import os
import stat
import sys
from typing import Any

import pytest

from edmars import secrets
from tests.cli.conftest import FakeKeyring

NAME = "DEEPSEEK_API_KEY"
KEYRING_VALUE = "sk-fake-keyring-0123456789abcdef"
FILE_VALUE = "sk-fake-file-0123456789abcdefgh"
ENV_VALUE = "sk-fake-env-0123456789abcdefghi"


class BrokenKeyring(FakeKeyring):
    def set_password(self, service: str, name: str, value: str) -> None:
        raise RuntimeError("no backend available")

    def get_password(self, service: str, name: str) -> str | None:
        raise RuntimeError("no backend available")


class ForgetfulKeyring(FakeKeyring):
    """Accepts writes and keeps nothing, like keyring's null backend."""

    def set_password(self, service: str, name: str, value: str) -> None:
        return None


def test_nothing_stored_means_none() -> None:
    assert secrets.get_secret(NAME) is None
    assert secrets.secret_source(NAME) is None


def test_file_is_the_last_resort() -> None:
    secrets._file_write({NAME: FILE_VALUE})
    assert secrets.get_secret(NAME) == FILE_VALUE
    assert secrets.secret_source(NAME) == "file"


def test_keyring_beats_file(fake_keyring: FakeKeyring) -> None:
    secrets._file_write({NAME: FILE_VALUE})
    fake_keyring.set_password(secrets.SERVICE, NAME, KEYRING_VALUE)
    assert secrets.get_secret(NAME) == KEYRING_VALUE
    assert secrets.secret_source(NAME) == "keyring"
    assert secrets.stored_location(NAME) == "keyring"


def test_environment_beats_keyring_and_warns_once(
    fake_keyring: FakeKeyring, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_keyring.set_password(secrets.SERVICE, NAME, KEYRING_VALUE)
    monkeypatch.setenv(NAME, ENV_VALUE)
    assert secrets.get_secret(NAME) == ENV_VALUE
    assert secrets.get_secret(NAME) == ENV_VALUE
    assert secrets.secret_source(NAME) == "env"
    err = capsys.readouterr().err
    assert err.count("differs from the key saved") == 1
    assert KEYRING_VALUE not in err and ENV_VALUE not in err


def test_environment_equal_to_stored_key_does_not_warn(
    fake_keyring: FakeKeyring, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_keyring.set_password(secrets.SERVICE, NAME, KEYRING_VALUE)
    monkeypatch.setenv(NAME, KEYRING_VALUE)
    assert secrets.get_secret(NAME) == KEYRING_VALUE
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("placeholder", ["", "   ", "your-key-here", "<your key>", "changeme", "xxxxxxxx"])
def test_placeholder_environment_values_are_ignored(
    placeholder: str, fake_keyring: FakeKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_keyring.set_password(secrets.SERVICE, NAME, KEYRING_VALUE)
    monkeypatch.setenv(NAME, placeholder)
    assert secrets.get_secret(NAME) == KEYRING_VALUE


def test_local_provider_placeholder_is_a_real_value(monkeypatch: pytest.MonkeyPatch) -> None:
    # The "local" provider uses OPENAI_API_KEY=local on purpose.
    monkeypatch.setenv("OPENAI_API_KEY", "local")
    assert secrets.get_secret("OPENAI_API_KEY") == "local"


def test_set_secret_uses_keyring_and_removes_file_copy(fake_keyring: FakeKeyring) -> None:
    secrets._file_write({NAME: FILE_VALUE, "TAVILY_API_KEY": "tvly-fake-0123456789abcdef"})
    assert secrets.set_secret(NAME, "  " + KEYRING_VALUE + "\n") == "keyring"
    assert fake_keyring.store[(secrets.SERVICE, NAME)] == KEYRING_VALUE
    assert NAME not in secrets._file_read()
    assert "TAVILY_API_KEY" in secrets._file_read()


def test_set_secret_without_consent_refuses_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(secrets, "_keyring", lambda: BrokenKeyring())
    with pytest.raises(secrets.SecretStoreError) as info:
        secrets.set_secret(NAME, KEYRING_VALUE)
    assert KEYRING_VALUE not in str(info.value)
    assert info.value.reason and info.value.reason in str(info.value)
    assert not secrets.secrets_file().exists()


def test_set_secret_with_consent_uses_a_private_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(secrets, "_keyring", lambda: BrokenKeyring())
    assert secrets.set_secret(NAME, FILE_VALUE, allow_file=True) == "file"
    path = secrets.secrets_file()
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# EDM-ARS keys")
    assert f"{NAME}={FILE_VALUE}" in text
    assert secrets.get_secret(NAME) == FILE_VALUE
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    leftovers = [p for p in path.parent.iterdir() if p.name.startswith(".secrets-")]
    assert leftovers == []


def test_a_keyring_that_drops_writes_counts_as_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(secrets, "_keyring", lambda: ForgetfulKeyring())
    with pytest.raises(secrets.SecretStoreError):
        secrets.set_secret(NAME, KEYRING_VALUE)
    assert secrets.set_secret(NAME, KEYRING_VALUE, allow_file=True) == "file"


@pytest.mark.parametrize("bad", ["", "   ", "sk-fake-one\nsk-fake-two"])
def test_set_secret_rejects_empty_or_multiline(bad: str) -> None:
    with pytest.raises(ValueError):
        secrets.set_secret(NAME, bad)


@pytest.mark.parametrize("bad_name", ["deepseek_api_key", "1KEY", "KEY-NAME", ""])
def test_names_must_look_like_environment_variables(bad_name: str) -> None:
    with pytest.raises(ValueError):
        secrets.get_secret(bad_name)


def test_delete_secret_clears_keyring_and_file(fake_keyring: FakeKeyring) -> None:
    fake_keyring.set_password(secrets.SERVICE, NAME, KEYRING_VALUE)
    secrets._file_write({NAME: FILE_VALUE})
    secrets.delete_secret(NAME)
    assert secrets.get_secret(NAME) is None
    assert not secrets.secrets_file().exists()
    secrets.delete_secret(NAME)  # deleting twice is fine


def test_child_secrets_only_includes_present_keys(
    fake_keyring: FakeKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_keyring.set_password(secrets.SERVICE, NAME, KEYRING_VALUE)
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "fake-s2-0123456789abcdef")
    found = secrets.child_secrets([NAME, "SEMANTIC_SCHOLAR_API_KEY", "OPENAI_API_KEY"])
    assert found == {NAME: KEYRING_VALUE, "SEMANTIC_SCHOLAR_API_KEY": "fake-s2-0123456789abcdef"}


def test_redact_removes_values_seen_this_session(fake_keyring: FakeKeyring) -> None:
    unusual = "fake0123456789ABCDEFnoprefix"
    fake_keyring.set_password(secrets.SERVICE, "TAVILY_API_KEY", unusual)
    secrets.get_secret("TAVILY_API_KEY")
    text = f"request failed for key {unusual} at step 3"
    assert secrets.redact(text) == "request failed for key [REDACTED] at step 3"


def test_redact_removes_key_shaped_environment_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "fakeTokenValue0123456789")
    assert "fakeTokenValue" not in secrets.redact("token=fakeTokenValue0123456789 used")


@pytest.mark.parametrize(
    ("raw", "gone"),
    [
        ("Error: invalid key sk-fake-abcdefghijklmnop", "sk-fake-abcdefghijklmnop"),
        ("Authorization: Bearer fakeBearer0123456789xyz", "fakeBearer0123456789xyz"),
        ("DEEPSEEK_API_KEY=fake-value-0123456789abc", "fake-value-0123456789abc"),
        ("url?api_key=fake-query-0123456789&x=1", "fake-query-0123456789"),
        ("x-api-key: fake-header-0123456789", "fake-header-0123456789"),
        ("tavily tvly-fake-0123456789abcd", "tvly-fake-0123456789abcd"),
    ],
)
def test_redact_patterns(raw: str, gone: str) -> None:
    out = secrets.redact(raw)
    assert gone not in out
    assert "REDACTED" in out


def test_redact_leaves_ordinary_text_alone() -> None:
    text = "Saved to desk-top-research-folder; AUC 0.78 [0.76, 0.80]; model=xgboost"
    assert secrets.redact(text) == text
    assert secrets.redact("") == ""


def test_register_secret_adds_exact_value() -> None:
    secrets.register_secret("fakeTypedKey0123456789")
    assert secrets.redact("pasted fakeTypedKey0123456789") == "pasted [REDACTED]"


def test_keyring_backend_reports_unusable_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    fail_backend: Any = type("Keyring", (), {"priority": 0})()
    type(fail_backend).__module__ = "keyring.backends.fail"

    class Module:
        @staticmethod
        def get_keyring() -> Any:
            return fail_backend

    monkeypatch.setattr(secrets, "_keyring", lambda: Module())
    assert secrets.keyring_backend() is None


def test_keyring_backend_names_a_working_backend() -> None:
    assert secrets.keyring_backend() is not None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_rewriting_the_file_keeps_it_private() -> None:
    secrets._file_write({NAME: FILE_VALUE})
    secrets._file_write({NAME: FILE_VALUE, "TAVILY_API_KEY": "tvly-fake-0123456789abcd"})
    assert stat.S_IMODE(secrets.secrets_file().stat().st_mode) == 0o600
