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


def parse_llm_json(text: str) -> dict:
    """Strip markdown code fences and parse JSON."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(text)


def load_prompt(agent_name: str, config: dict) -> dict:
    """Load agent prompt YAML. Returns empty dict if file does not exist."""
    prompts_dir = config["paths"]["agent_prompts"]
    path = os.path.join(prompts_dir, f"{agent_name}.yaml")
    try:
        with open(path) as f:
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
    ) -> None:
        self.ctx = context
        self.agent_name = agent_name
        self.config = config
        self.model: str = ""  # set below after provider is determined

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

        prompt_data = load_prompt(agent_name.lower().replace(" ", "_"), config)
        self.system_prompt: str = prompt_data.get(
            "system_prompt",
            f"You are the {agent_name} agent for EDM-ARS.",
        )
        self.temperature: float = prompt_data.get(
            "temperature", self._default_temperature()
        )
        self.max_tokens: int = prompt_data.get("max_tokens", 8192)

        provider = config.get("llm_provider", "anthropic")
        agent_key = agent_name.lower().replace(" ", "_")
        if provider == "minimax":
            api_key = os.environ.get("MINIMAX_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "llm_provider is 'minimax' but MINIMAX_API_KEY is not set. "
                    "Add it to your .env file or set_env.ps1."
                )
            base_url = config.get("minimax", {}).get(
                "base_url", os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")
            )
            self.model = config.get("minimax", {}).get("models", {}).get(
                agent_key, "MiniMax-M2.5"
            )
            self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Export it before running the pipeline: "
                    "export ANTHROPIC_API_KEY=sk-ant-..."
                )
            self.model = config["models"][agent_key]
            self.client = anthropic.Anthropic(api_key=api_key)

        if executor is not None:
            self._executor = executor
        else:
            from src.sandbox import create_executor
            self._executor = create_executor(config)

    def _default_temperature(self) -> float:
        temps = {
            "problem_formulator": 0.7,
            "data_engineer": 0.0,
            "analyst": 0.0,
            "critic": 0.0,
            "writer": 0.3,
        }
        return temps.get(self.agent_name.lower().replace(" ", "_"), 0.0)

    def call_llm(
        self,
        user_message: str,
        max_tokens: int | None = None,
        temperature_override: float | None = None,
    ) -> str:
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        temperature = temperature_override if temperature_override is not None else self.temperature
        # Retry on rate-limit (429): exponential backoff up to 3 attempts
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Use streaming to avoid SDK timeout on large responses (> 10 min non-streaming limit)
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                ) as stream:
                    full_text = stream.get_final_text()
                    final_message = stream.get_final_message()
                self.ctx.log.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "agent": self.agent_name,
                        "tokens_used": (
                            final_message.usage.input_tokens + final_message.usage.output_tokens
                        ),
                    }
                )
                return full_text
            except anthropic.RateLimitError as exc:
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
        with open(path) as f:
            return yaml.safe_load(f)

    def load_task_template(self) -> dict:
        task_name = self.task_template.get_name()
        path = os.path.join(
            self.config["paths"]["data_registry"],
            "task_templates",
            f"{task_name}.yaml",
        )
        with open(path) as f:
            return yaml.safe_load(f)

    @abstractmethod
    def run(self, **kwargs) -> Any:
        ...
