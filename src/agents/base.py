from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import anthropic  # type: ignore[import-not-found]
import yaml

from src.cost import extract_usage, record_usage
from src.skills import Skill, format_skills_for_prompt

_SKILLS_PLACEHOLDER = "{{SKILLS}}"

# Provider identifier kept on the agent so call_llm can branch without
# isinstance checks (which break when openai SDK isn't installed).
_PROVIDER_ANTHROPIC = "anthropic"
_PROVIDER_MINIMAX = "minimax"
_PROVIDER_OPENAI = "openai"
_PROVIDER_DEEPSEEK = "deepseek"


def parse_llm_json(text: str) -> dict:
    """Strip markdown code fences and parse JSON."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(text)


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
        # Annotated as Any because the concrete type varies by provider
        # (anthropic.Anthropic for anthropic+minimax; openai.OpenAI for openai).
        self.client: Any
        if provider == "minimax":
            api_key = os.environ.get("MINIMAX_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "provider 'minimax' selected but MINIMAX_API_KEY is not set. "
                    "Add it to a .env file or export it in your shell."
                )
            base_url = provider_cfg.base_url or os.environ.get(
                "MINIMAX_BASE_URL", "https://api.minimax.io/anthropic"
            )
            self.model = provider_cfg.model or "MiniMax-M2.5"
            self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "provider 'openai' selected but OPENAI_API_KEY is not set. "
                    "Add it to a .env file or export it in your shell."
                )
            try:
                import openai  # type: ignore[import-not-found]
            except ImportError as exc:
                raise EnvironmentError(
                    "provider 'openai' selected but the openai SDK is not "
                    "installed. Install it with: pip install openai"
                ) from exc
            base_url = provider_cfg.base_url or os.environ.get("OPENAI_BASE_URL")
            self.model = provider_cfg.model or "gpt-4o"
            client_kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = openai.OpenAI(**client_kwargs)
        elif provider == "deepseek":
            # Phase 3b.10.5: DeepSeek-V4-Pro provider integration.
            # API is OpenAI-compatible at https://api.deepseek.com.
            # Replaces MiniMax as the project default per the project
            # instruction. The MiniMax branch (above) is retained for
            # backward-compat with 3b.5 / 3b.7 / 3b.9 run artifacts and
            # any pinned config that still references it.
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "provider 'deepseek' selected but DEEPSEEK_API_KEY is "
                    "not set. Add it to a .env file or export it in your shell."
                )
            try:
                import openai  # type: ignore[import-not-found]
            except ImportError as exc:
                raise EnvironmentError(
                    "provider 'deepseek' selected but the openai SDK is "
                    "not installed (DeepSeek uses the OpenAI-compatible "
                    "endpoint). Install it with: pip install openai"
                ) from exc
            base_url = provider_cfg.base_url or os.environ.get(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            )
            self.model = provider_cfg.model or "deepseek-v4-pro"
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Export it before running the pipeline: "
                    "export ANTHROPIC_API_KEY=sk-ant-..."
                )
            self.model = provider_cfg.model or config.get("models", {}).get(agent_key, "")
            self.client = anthropic.Anthropic(api_key=api_key)

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

    def _meter(self, response: Any) -> None:
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
        except Exception:  # noqa: BLE001 — metering is never fatal
            pass

    def call_llm(
        self,
        user_message: str,
        max_tokens: int | None = None,
        temperature_override: float | None = None,
    ) -> str:
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
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
        # Retry on rate-limit (429): exponential backoff up to 3 attempts
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if self._provider == _PROVIDER_OPENAI:
                    # OpenAI Chat Completions path. We don't use streaming
                    # here because OpenAI's SDK has a generous default
                    # request timeout and the response sizes are typical
                    # of EDM agent outputs.
                    #
                    # Use `max_completion_tokens` (not `max_tokens`) — the
                    # GPT-5 family rejects `max_tokens` outright, while
                    # gpt-4o accepts both. So `max_completion_tokens` is
                    # the cross-model-compatible spelling.
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": rendered_system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        max_completion_tokens=max_tokens,
                        temperature=temperature,
                    )
                    full_text = response.choices[0].message.content or ""
                    self._meter(response)
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
                            {"role": "user", "content": user_message},
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        extra_body={"thinking": {"type": "disabled"}},
                    )
                    full_text = response.choices[0].message.content or ""
                    self._meter(response)
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
                self._meter(final_message)
                if capture_dir is not None:
                    try:
                        self._write_response_capture(capture_dir, full_text)
                    except Exception:
                        pass
                return full_text
            except Exception as exc:
                # Catch rate-limit errors from either SDK and back off.
                # We can't reference openai.RateLimitError unconditionally
                # because the openai SDK may not be installed; rely on
                # class-name + module-name introspection instead.
                exc_name = type(exc).__name__
                exc_module = type(exc).__module__ or ""
                is_rate_limit = (
                    isinstance(exc, anthropic.RateLimitError)
                    or (exc_name == "RateLimitError" and exc_module.startswith("openai"))
                )
                if not is_rate_limit:
                    raise
                if attempt == max_attempts - 1:
                    raise
                wait_s = 60 * (attempt + 1)  # 60s, 120s
                self.ctx.log.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "agent": self.agent_name,
                        "message": (
                            f"Rate limit hit (attempt {attempt + 1}/{max_attempts}); "
                            f"waiting {wait_s}s before retry. ({exc})"
                        ),
                    }
                )
                time.sleep(wait_s)
        raise RuntimeError("call_llm: unreachable")

    def execute_code(self, code: str, timeout_s: int = 300) -> dict:
        """Execute generated Python code via configured executor (Docker sandbox or subprocess)."""
        return self._executor.run(
            code=code,
            output_dir=self.ctx.output_dir,
            raw_data_path=getattr(self.ctx, "raw_data_path", None),
            timeout_s=timeout_s,
        )

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
