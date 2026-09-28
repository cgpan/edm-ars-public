from __future__ import annotations

import json
import os
import re
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import yaml

from src.agents.llm_client import (
    LLMSettings,
    build_client,
    call_with_retries,
    llm_settings,
)
from src.cost import (
    TokenUsage,
    cost_usd,
    extract_usage,
    load_pricing,
    rate_is_unverified,
    record_usage,
)
from src.errors import ProviderError
from src.events import emit
from src.skills import Skill, format_skills_for_prompt

_SKILLS_PLACEHOLDER = "{{SKILLS}}"

# Provider identifier kept on the agent so call_llm can branch without
# isinstance checks (which break when openai SDK isn't installed).
_PROVIDER_ANTHROPIC = "anthropic"
_PROVIDER_MINIMAX = "minimax"
_PROVIDER_OPENAI = "openai"
_PROVIDER_DEEPSEEK = "deepseek"


def _finish_reason(raw: Any) -> str | None:
    """A provider's stop reason, with "hit the token limit" as "length".

    OpenAI-compatible APIs (DeepSeek, OpenAI) say ``length``; Anthropic
    and MiniMax say ``max_tokens``.
    """
    if not isinstance(raw, str) or not raw:
        return None
    return "length" if raw in ("length", "max_tokens") else raw


def parse_llm_json(text: str) -> dict:
    """Strip markdown code fences and parse JSON."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(text)


def literature_for_prompt(literature_context: Any) -> Any:
    """The literature context as a model should see it.

    ``retrieval_status`` (CONTRACT section 6) is run bookkeeping that the
    orchestrator reads for pipeline.log, the event stream and
    run_status.json. It is not literature, so every agent prompt that
    pastes the context leaves it out and reads exactly what it read
    before the status was recorded.
    """
    if not isinstance(literature_context, dict):
        return literature_context
    return {k: v for k, v in literature_context.items() if k != "retrieval_status"}


def load_prompt(
    agent_name: str,
    config: dict,
    task_type: str | None = None,
) -> dict:
    """Load agent prompt YAML, optionally selecting a task-type-keyed file.

    File selection (Phase 3b.4 / B2-B4):
      - When ``task_type`` is provided, prefer
        ``{agent_prompts}/{agent_name}_{task_type}.yaml`` if it exists.
        Validate ``task_type`` against the registered TaskTemplates
        first, so an unknown task type fails loudly here rather than
        silently falling through to the default file.
      - Fall back to ``{agent_name}.yaml`` (the V1 prediction default).
      - Returns empty dict if neither file exists.

    The fallback intentionally lets task types without a dedicated
    override file (today: prediction) continue to use the unmodified
    V1 prompt. Causal_soo gets ``problem_formulator_causal_soo.yaml``,
    ``analyst_causal_soo.yaml``, and ``writer_causal_soo.yaml`` —
    additive, not destructive, per the Option-A unblock contract.
    """
    prompts_dir = config["paths"]["agent_prompts"]

    if task_type is not None:
        from src.task_template import _TASK_REGISTRY  # local import to avoid cycle

        if task_type not in _TASK_REGISTRY:
            raise ValueError(
                f"load_prompt: unknown task_type {task_type!r}. "
                f"Registered: {sorted(_TASK_REGISTRY.keys())}"
            )
        override_path = os.path.join(
            prompts_dir, f"{agent_name}_{task_type}.yaml"
        )
        if os.path.exists(override_path):
            with open(override_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

    path = os.path.join(prompts_dir, f"{agent_name}.yaml")
    try:
        # encoding="utf-8" is load-bearing: prompt YAMLs contain em dashes
        # and typographic quotes; without it, Windows (cp1252) silently
        # mojibakes every non-ASCII character in every rendered prompt
        # (found in V2.1 Phase 3b.23 rendered-prompt verification).
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


#: Media types DeepSeek and OpenAI accept as inline image parts.
_IMAGE_MEDIA = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

#: Roughly 5 MB of base64 per image is the practical ceiling; an
#: analysis figure at 150 dpi is far below it.
_MAX_IMAGE_BYTES = 4_000_000


def _image_content_part(path: str) -> dict | None:
    """Encode a local image as an OpenAI-style ``image_url`` part.

    Returns ``None`` -- never a silent empty part -- when the file is
    missing, too large, or of a type the API does not take, and warns,
    so a caller that believed it was checking a figure finds out it was
    not.
    """
    ext = os.path.splitext(path)[1].lower()
    media = _IMAGE_MEDIA.get(ext)
    if media is None:
        warnings.warn(f"not an image type the API accepts: {path}", RuntimeWarning, stacklevel=2)
        return None
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        warnings.warn(f"could not read image {path}: {exc}", RuntimeWarning, stacklevel=2)
        return None
    if len(raw) > _MAX_IMAGE_BYTES:
        warnings.warn(
            f"image {path} is {len(raw)} bytes, above the {_MAX_IMAGE_BYTES} "
            "ceiling; not sent",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    import base64

    b64 = base64.b64encode(raw).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{b64}"}}


#: ``execute_code``'s own default, reported in attempt events when a
#: caller does not pass a timeout.
_DEFAULT_EXEC_TIMEOUT_S = 300

_EXC_LINE_RE = re.compile(
    r"^\s*([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning|Timeout))\b"
)


def _error_class_from_result(result: Any) -> str | None:
    """Best guess at what a failed script died of, from its stderr.

    Python prints the exception class at the start of the last traceback
    line (``KeyError: 'X1SES'``). The executors report a kill on timeout
    as ``Timeout after Ns`` / ``Timed out after Ns``.
    """
    if not isinstance(result, dict):
        return None
    stderr = str(result.get("stderr") or "")
    lowered = stderr.lower()
    if "timeout after" in lowered or "timed out after" in lowered:
        return "Timeout"
    for line in reversed(stderr.strip().splitlines()):
        match = _EXC_LINE_RE.match(line)
        if match:
            return match.group(1).rsplit(".", 1)[-1]
    return "NonZeroExit" if result.get("returncode") not in (0, None) else None


class BaseAgent(ABC):
    def __init__(
        self,
        context: Any,
        agent_name: str,
        config: dict,
        executor: Any = None,
        task_template: Any = None,
        dataset_adapter: Any = None,
        skills: list[Skill] | None = None,
    ) -> None:
        self.ctx = context
        self.agent_name = agent_name
        self.config = config
        self.model: str = ""  # set below after provider is determined
        # V2.0 skill injection: orchestrator overwrites this attribute per
        # stage with the result of SkillRegistry.match_and_compose(...).
        # When None or [], the {{SKILLS}} placeholder (if present) is
        # removed; when the placeholder is absent the prompt is unchanged.
        self.skills: list[Skill] | None = skills

        # Task template and dataset adapter (auto-create from context if not provided)
        if task_template is None:
            from src.task_template import create_task_template
            task_template = create_task_template(
                getattr(context, "task_type", "prediction")
            )
        if dataset_adapter is None:
            from src.dataset_adapter import create_dataset_adapter
            dataset_adapter = create_dataset_adapter(context.dataset_name)
        self.task_template = task_template
        self.dataset_adapter = dataset_adapter

        prompt_data = load_prompt(
            agent_name.lower().replace(" ", "_"),
            config,
            task_type=self.task_template.get_name(),
        )
        self.system_prompt: str = prompt_data.get(
            "system_prompt",
            f"You are the {agent_name} agent for EDM-ARS.",
        )
        self.temperature: float = prompt_data.get(
            "temperature", self._default_temperature()
        )
        self.max_tokens: int = prompt_data.get("max_tokens", 8192)

        # Phase 3b.10 / §10.1.3: per-stage provider resolution.
        # Falls back to the legacy single-provider schema when no
        # per_stage_providers override exists for this agent_key, so
        # existing configs (3b.5 / 3b.7 / 3b.9) keep working unchanged.
        from src.agents.provider_resolver import resolve_provider_for_stage

        agent_key = agent_name.lower().replace(" ", "_")
        provider_cfg = resolve_provider_for_stage(agent_key, config)
        provider = provider_cfg.name
        self._provider: str = provider
        self._provider_cfg = provider_cfg
        # Transport limits (D5): explicit timeout, bounded network retries.
        self._llm_settings: LLMSettings = llm_settings(config)
        # Annotated as Any because the concrete type varies by provider
        # (anthropic.Anthropic for anthropic+minimax; openai.OpenAI for
        # openai+deepseek). Construction -- API-key check, base URL
        # precedence, timeout -- is shared with the review gate in
        # src/agents/llm_client.py so the two cannot drift apart (E3).
        # DeepSeek's API is OpenAI-compatible (Phase 3b.10.5); the
        # MiniMax branch is kept for 3b.5 / 3b.7 / 3b.9 artifacts.
        # The model is resolved first: a missing openai.models entry is a
        # config.yaml problem and should be reported as one even when the
        # key is missing too.
        self.model = self._resolve_model(provider_cfg, agent_key, config)
        self.client: Any = build_client(provider_cfg, self._llm_settings)
        self._pricing: dict | None = None
        #: Why the provider stopped its last answer, normalised: "length"
        #: when it hit max_tokens, else the provider's own word ("stop",
        #: "end_turn") or None when it did not say. A cut-off JSON answer
        #: otherwise surfaces only as "Unterminated string" (round 3).
        self.last_finish_reason: str | None = None

        # Phase 3b.10 / §10.2: per-stage max_tokens resolution.
        # Stash the resolved value as a default; per-call max_tokens
        # passed into call_llm() still wins. Fallback chain:
        #   1. per_stage_max_tokens[agent_key] (3b.10 schema)
        #   2. config.default_max_tokens (3b.10 schema)
        #   3. prompt_data.max_tokens (existing per-prompt YAML default)
        #   4. 8192 (BaseAgent legacy default)
        from src.agents.provider_resolver import resolve_max_tokens_for_stage

        prompt_max = prompt_data.get("max_tokens", 8192)
        resolved_max = resolve_max_tokens_for_stage(
            agent_key, config, fallback=prompt_max
        )
        self.max_tokens = resolved_max

        if executor is not None:
            self._executor = executor
        else:
            from src.sandbox import create_executor
            self._executor = create_executor(config)

    @staticmethod
    def _resolve_model(provider_cfg: Any, agent_key: str, config: dict) -> str:
        """The model id this stage calls, or a configuration error.

        deepseek and minimax fall back to their long-standing defaults.
        openai does NOT: config.yaml ships no ``openai`` block, so a
        fallback meant every stage -- the Critic included, which the SPEC
        puts on the strongest tier -- quietly ran on gpt-4o, and nothing
        said so until the tokens were spent (defect E1).
        """
        from src.agents.provider_resolver import effective_model

        return effective_model(provider_cfg, agent_key)

    # ------------------------------------------------------------------
    # Live progress: pipeline.log lines and structured events
    # ------------------------------------------------------------------

    def _note(self, message: str) -> None:
        """Record *message* in ctx.log AND pipeline.log, right now.

        Agent notes otherwise reach disk only inside checkpoint.json at the
        end of a stage, so a paused run looked frozen (D6). The line format
        is the orchestrator's ``_log`` format exactly, so pipeline.log
        stays one uniform stream. Never raises.
        """
        timestamp = datetime.utcnow().isoformat()
        try:
            self.ctx.log.append(
                {"timestamp": timestamp, "agent": self.agent_name, "message": message}
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            output_dir = getattr(self.ctx, "output_dir", None)
            if isinstance(output_dir, str) and output_dir and os.path.isdir(output_dir):
                with open(
                    os.path.join(output_dir, "pipeline.log"), "a", encoding="utf-8"
                ) as fh:
                    fh.write(f"{timestamp} [{self.agent_name}] {message}\n")
        except Exception:  # noqa: BLE001 -- a log line must not fail a stage
            pass

    def _emit(self, event_type: str, plain: str | None = None, **data: Any) -> None:
        """Emit a structured event tagged with this agent's stage and cycle."""
        try:
            cycle_raw = getattr(self.ctx, "revision_cycle", None)
            cycle = int(cycle_raw) if isinstance(cycle_raw, int) else None
            emit(
                self.ctx,
                event_type,
                stage=getattr(self.ctx, "current_state", None),
                cycle=cycle,
                agent=self.agent_name,
                plain=plain,
                **data,
            )
        except Exception:  # noqa: BLE001
            pass

    def _on_llm_wait(self, seconds: float, attempt: int, reason: str, message: str) -> None:
        """Announce a retry wait before sleeping through it (D6)."""
        self._note(message)
        self._emit(
            "llm.wait",
            plain=f"Waiting {seconds:.0f} s for {self._provider} ({reason.replace('_', ' ')})",
            seconds=seconds,
            attempt=attempt,
            reason=reason,
            model=self.model,
            provider=self._provider,
        )

    def _price(self, usage: TokenUsage | None) -> tuple[float | None, bool]:
        """(USD, is_estimate) for one call; (None, False) when unpriced."""
        if usage is None:
            return None, False
        try:
            if self._pricing is None:
                self._pricing = load_pricing(self.config)
            cost = cost_usd(usage, self._pricing)
            if cost is None:
                return None, False
            return cost, rate_is_unverified(self._pricing.get(usage.model))
        except Exception:  # noqa: BLE001 -- pricing is informational
            return None, False

    def render_system_prompt(self) -> str:
        """Return the system prompt with the {{SKILLS}} placeholder resolved.

        Behavior:
          - If the prompt has no `{{SKILLS}}` placeholder: return the prompt
            unchanged. This is the backward-compat path during the V2.0
            rollout — pre-Phase-2c agent prompts that have not been slimmed
            still work.
          - If the placeholder is present and ``self.skills`` is None or
            empty: replace the placeholder with an empty string so it
            never leaks into the LLM input.
          - If the placeholder is present and ``self.skills`` is non-empty:
            splice in the formatted skill content via
            ``format_skills_for_prompt``.
        """
        if _SKILLS_PLACEHOLDER not in self.system_prompt:
            return self.system_prompt
        if not self.skills:
            return self.system_prompt.replace(_SKILLS_PLACEHOLDER, "")
        skills_block = format_skills_for_prompt(self.skills).rstrip()
        return self.system_prompt.replace(_SKILLS_PLACEHOLDER, skills_block)

    def _default_temperature(self) -> float:
        temps = {
            "problem_formulator": 0.7,
            "data_engineer": 0.0,
            "analyst": 0.0,
            "critic": 0.0,
            "writer": 0.3,
        }
        return temps.get(self.agent_name.lower().replace(" ", "_"), 0.0)

    def _capture_prompt_dir(self) -> str | None:
        """Return the directory to dump rendered_prompt + response_raw for
        the current agent + revision cycle, creating it if needed.

        Returns ``None`` when no output_dir is available (e.g., unit tests
        that don't construct a real run directory) — capture is silently
        skipped in that case to keep tests fast.

        Phase 3b.7 / sub-phase A.2 instrumentation. Layout:

            {output_dir}/prompts/{agent_name}/cycle_{N}/

        The agent_name is normalized to lowercase + underscored. The cycle
        number is taken from ``ctx.revision_cycle`` when present; defaults
        to 0 for the initial pass.
        """
        output_dir = getattr(self.ctx, "output_dir", None)
        if not output_dir:
            return None
        agent_slug = self.agent_name.lower().replace(" ", "_")
        cycle = int(getattr(self.ctx, "revision_cycle", 0) or 0)
        capture_dir = os.path.join(
            output_dir, "prompts", agent_slug, f"cycle_{cycle}"
        )
        try:
            os.makedirs(capture_dir, exist_ok=True)
        except OSError:
            return None
        return capture_dir

    def _write_prompt_capture(
        self, capture_dir: str, rendered_system_prompt: str, user_message: str
    ) -> None:
        """Dump the rendered prompt before the LLM call (additive; failures
        are non-fatal so the LLM call still proceeds)."""
        try:
            path = os.path.join(capture_dir, "rendered_prompt.txt")
            # If the file already exists for this (agent, cycle) — i.e., the
            # agent is making a SECOND call within the same cycle (e.g.,
            # multi-branch PF, retry, etc.) — append rather than clobber.
            mode = "a" if os.path.exists(path) else "w"
            with open(path, mode, encoding="utf-8") as f:
                if mode == "a":
                    f.write("\n\n--- additional call within same cycle ---\n\n")
                f.write("=== SYSTEM PROMPT ===\n")
                f.write(rendered_system_prompt)
                f.write("\n\n=== USER MESSAGE ===\n")
                f.write(user_message)
                f.write("\n")
        except OSError:
            # Capture is best-effort. Never break the LLM call on disk-IO.
            pass

    def _write_response_capture(self, capture_dir: str, response_text: str) -> None:
        try:
            path = os.path.join(capture_dir, "response_raw.txt")
            mode = "a" if os.path.exists(path) else "w"
            with open(path, mode, encoding="utf-8") as f:
                if mode == "a":
                    f.write("\n\n--- additional response within same cycle ---\n\n")
                f.write(response_text)
                f.write("\n")
        except OSError:
            pass

    def _meter(self, response: Any) -> TokenUsage | None:
        """Record measured token usage for one LLM call (K1).

        Every provider path funnels through here so the run's
        ``token_usage.jsonl`` has one row per call with prompt,
        completion and cached-input counts kept SEPARATE — they are
        priced differently, and the old summed ``tokens_used`` could not
        be turned into a defensible dollar figure.

        Best-effort by contract: a provider that omits usage, or a disk
        that refuses the write, must never fail the call.
        """
        try:
            usage = extract_usage(
                response, self.agent_name, self.model, self._provider
            )
            usage.timestamp = datetime.utcnow().isoformat()
            usage.stage = getattr(self.ctx, "current_state", None) and str(
                getattr(self.ctx, "current_state")
            )
            record_usage(getattr(self.ctx, "output_dir", None), usage)
            self.ctx.log.append(
                {
                    "timestamp": usage.timestamp,
                    "agent": self.agent_name,
                    # Legacy key: orchestrator sums it for its total.
                    "tokens_used": usage.total_tokens,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "cached_prompt_tokens": usage.cached_prompt_tokens,
                    "model": self.model,
                }
            )
            return usage
        except Exception:  # noqa: BLE001 — metering is never fatal
            return None

    def call_llm(
        self,
        user_message: str,
        max_tokens: int | None = None,
        temperature_override: float | None = None,
        image_paths: list[str] | None = None,
    ) -> str:
        """Call the configured model.

        ``image_paths`` attaches local images as content parts. Only the
        OpenAI-compatible providers (deepseek, openai) carry them;
        anywhere else they are dropped with a warning rather than
        silently, because a vision check that quietly became a text check
        would report on figures it never saw.
        """
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        self.last_finish_reason = None
        temperature = temperature_override if temperature_override is not None else self.temperature
        # Resolve {{SKILLS}} placeholder against the orchestrator-supplied
        # skill list once per call. This is a no-op for prompts without
        # the placeholder.
        rendered_system_prompt = self.render_system_prompt()
        # Phase 3b.7 / A.2: dump the rendered prompt to disk so 3b.7's
        # report can cite exact prompt content per stage. Best-effort —
        # any failure (disk full, helper raised, etc.) is swallowed so
        # the LLM call continues unimpeded.
        capture_dir = None
        try:
            capture_dir = self._capture_prompt_dir()
            if capture_dir is not None:
                self._write_prompt_capture(
                    capture_dir, rendered_system_prompt, user_message
                )
        except Exception:
            capture_dir = None
        # One content payload, built once. A list of parts for the
        # OpenAI-compatible providers when images are attached; the bare
        # string otherwise, so every existing call is byte-identical.
        user_content: Any = user_message
        if image_paths:
            if self._provider in (_PROVIDER_OPENAI, _PROVIDER_DEEPSEEK):
                parts: list[dict] = [{"type": "text", "text": user_message}]
                for path in image_paths:
                    part = _image_content_part(path)
                    if part is not None:
                        parts.append(part)
                if len(parts) > 1:
                    user_content = parts
            else:
                warnings.warn(
                    f"{len(image_paths)} image(s) were passed to call_llm but "
                    f"provider {self._provider!r} has no image path here; they "
                    "were NOT sent. Anything the caller concludes about those "
                    "figures is about text alone.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # One attempt against the provider. Retrying -- 429 waits, bounded
        # network/server retries -- and turning an unrecoverable failure
        # into a classified ProviderError happen in call_with_retries
        # (src/agents/llm_client.py), which announces every wait before
        # sleeping through it.
        metered: dict[str, Any] = {}

        def _attempt() -> str:
            if self._provider == _PROVIDER_OPENAI:
                # OpenAI Chat Completions path. We don't use streaming
                # here: the client carries an explicit read timeout
                # (llm.request_timeout_s, default 600 s) and the
                # response sizes are typical of EDM agent outputs.
                #
                # Use `max_completion_tokens` (not `max_tokens`) — the
                # GPT-5 family rejects `max_tokens` outright, while
                # gpt-4o accepts both. So `max_completion_tokens` is
                # the cross-model-compatible spelling.
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": rendered_system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                )
                full_text = response.choices[0].message.content or ""
                self.last_finish_reason = _finish_reason(
                    getattr(response.choices[0], "finish_reason", None)
                )
                metered["usage"] = self._meter(response)
                if capture_dir is not None:
                    try:
                        self._write_response_capture(capture_dir, full_text)
                    except Exception:
                        pass
                return full_text

            if self._provider == _PROVIDER_DEEPSEEK:
                # Phase 3b.10.5: DeepSeek-V4-Pro path. Uses the openai
                # SDK against DeepSeek's OpenAI-compatible endpoint.
                #
                # Thinking mode is DISABLED by default. DeepSeek-V4-Pro
                # ships with thinking enabled; the same thinking-block
                # overhead pattern that caused F-3b9-ANALYST-CODEGEN-
                # CRASH and F-3b9-WRITER-ONLY-BIBTEX under MiniMax-M2.7
                # would recur. Thinking can be re-enabled per-stage in
                # the future via a config.per_stage_providers.<stage>
                # .extra block; not implemented in 3b.10.5 (premature
                # until 3b.11 surfaces evidence that thinking helps).
                #
                # DeepSeek's OpenAI compat accepts max_tokens (the
                # standard parameter shown in their docs). Using
                # max_tokens here rather than max_completion_tokens
                # because DeepSeek isn't a GPT-5-family model.
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": rendered_system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                full_text = response.choices[0].message.content or ""
                self.last_finish_reason = _finish_reason(
                    getattr(response.choices[0], "finish_reason", None)
                )
                metered["usage"] = self._meter(response)
                if capture_dir is not None:
                    try:
                        self._write_response_capture(capture_dir, full_text)
                    except Exception:
                        pass
                return full_text

            # Anthropic / MiniMax (Anthropic-SDK-compatible) path.
            # Use streaming to avoid SDK timeout on large responses
            # (> 10 min non-streaming limit).
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=rendered_system_prompt,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                final_message = stream.get_final_message()
            self.last_finish_reason = _finish_reason(
                getattr(final_message, "stop_reason", None)
            )
            # Phase 3b.7 / sub-phase A.1: MiniMax-M2.7 emits "thinking"
            # content blocks alongside text (similar to Anthropic
            # extended thinking). The SDK's get_final_text() raises
            # RuntimeError when thinking-only responses come back.
            # Extract the text content manually from content blocks
            # so both pure-text and thinking+text responses work.
            full_text = "".join(
                getattr(block, "text", "") or ""
                for block in final_message.content
                if getattr(block, "type", None) == "text"
            )
            metered["usage"] = self._meter(final_message)
            if capture_dir is not None:
                try:
                    self._write_response_capture(capture_dir, full_text)
                except Exception:
                    pass
            return full_text

        started = time.monotonic()
        self._emit(
            "llm.start",
            plain=f"Waiting for {self.model}",
            model=self.model,
            provider=self._provider,
        )
        try:
            full_text = call_with_retries(
                _attempt,
                provider_cfg=self._provider_cfg,
                model=self.model,
                settings=self._llm_settings,
                on_wait=self._on_llm_wait,
            )
        except BaseException as exc:
            self._emit(
                "llm.end",
                model=self.model,
                provider=self._provider,
                ok=False,
                error_code=exc.code if isinstance(exc, ProviderError) else None,
                error_class=type(exc).__name__,
                prompt_tokens=0,
                completion_tokens=0,
                cached_tokens=0,
                cost_usd=None,
                duration_s=round(time.monotonic() - started, 3),
            )
            raise
        usage = metered.get("usage")
        cost, estimated = self._price(usage)
        self._emit(
            "llm.end",
            model=self.model,
            provider=self._provider,
            ok=True,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            cached_tokens=getattr(usage, "cached_prompt_tokens", 0),
            cost_usd=cost,
            cost_estimated=estimated,
            duration_s=round(time.monotonic() - started, 3),
        )
        return full_text

    def execute_code(self, code: str, timeout_s: int = 300) -> dict:
        """Execute generated Python code via configured executor (Docker sandbox or subprocess)."""
        return self._executor.run(
            code=code,
            output_dir=self.ctx.output_dir,
            raw_data_path=getattr(self.ctx, "raw_data_path", None),
            timeout_s=timeout_s,
        )

    def execute_code_attempt(
        self,
        code: str,
        *,
        attempt: int,
        max_attempts: int,
        timeout_s: int | None = None,
    ) -> dict:
        """``execute_code`` bracketed by ``attempt.start`` / ``attempt.end``.

        The DataEngineer and Analyst retry loops call this with their own
        attempt numbers, so a live view can say "attempt 2 of 3" while the
        generated script runs, and what the last one died of. It forwards
        to ``execute_code`` with the same arguments the loops always
        passed (``timeout_s`` only when the loop sets one), so a test or
        subclass that replaces ``execute_code`` keeps working.
        """
        effective_timeout = timeout_s if timeout_s is not None else _DEFAULT_EXEC_TIMEOUT_S
        self._emit(
            "attempt.start",
            plain=f"Running the generated code (attempt {attempt} of {max_attempts})",
            attempt=attempt,
            max_attempts=max_attempts,
            timeout_s=effective_timeout,
        )
        started = time.monotonic()
        try:
            if timeout_s is None:
                result = self.execute_code(code)
            else:
                result = self.execute_code(code, timeout_s=timeout_s)
        except BaseException as exc:
            self._emit(
                "attempt.end",
                attempt=attempt,
                max_attempts=max_attempts,
                returncode=None,
                duration_s=round(time.monotonic() - started, 3),
                timeout_s=effective_timeout,
                error_class=type(exc).__name__,
            )
            raise
        returncode = result.get("returncode") if isinstance(result, dict) else None
        self._emit(
            "attempt.end",
            attempt=attempt,
            max_attempts=max_attempts,
            returncode=returncode,
            duration_s=round(time.monotonic() - started, 3),
            timeout_s=effective_timeout,
            error_class=(
                None if returncode == 0 else _error_class_from_result(result)
            ),
        )
        return result

    def load_registry(self) -> dict:
        path = os.path.join(
            self.config["paths"]["data_registry"],
            "datasets",
            f"{self.ctx.dataset_name}.yaml",
        )
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_task_template(self) -> dict:
        """Load the task-template YAML for this task type.

        Returns an empty dict if no task-template YAML exists for the
        current task type (Phase 3b.5 / narrow-exception #1 unblock).
        Skill bodies carry the per-task methodology via the {{SKILLS}}
        injection layer; the task-template YAML is supplementary
        guidance, not load-bearing for correctness. A missing file is
        a documented Bucket C finding, not a crash condition.
        """
        task_name = self.task_template.get_name()
        path = os.path.join(
            self.config["paths"]["data_registry"],
            "task_templates",
            f"{task_name}.yaml",
        )
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    @abstractmethod
    def run(self, **kwargs) -> Any:
        ...
