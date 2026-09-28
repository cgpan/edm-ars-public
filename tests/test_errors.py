from src.errors import ABORT_CODES, ProviderError, code_for_exception, is_resumable


def test_provider_error_code_is_normalised() -> None:
    assert ProviderError("NO_CREDIT", "balance").code == "NO_CREDIT"
    assert ProviderError("SOMETHING_ELSE", "x").code == "PROVIDER_ERROR"


def test_code_for_exception() -> None:
    assert code_for_exception(ProviderError("KEY_REJECTED", "401")) == "KEY_REJECTED"
    assert code_for_exception(ValueError("Expecting value: line 1 column 1")) == "LLM_OUTPUT_UNPARSEABLE"
    assert code_for_exception(EnvironmentError("DEEPSEEK_API_KEY is not set")) == "KEY_MISSING"
    assert code_for_exception(RuntimeError("boom")) == "UNKNOWN"
    assert code_for_exception(KeyboardInterrupt()) == "INTERRUPTED"


def test_resumability_table() -> None:
    assert is_resumable("NO_CREDIT") and not is_resumable("CRITIC_ABORT")
    assert all(isinstance(v["resumable"], bool) for v in ABORT_CODES.values())
