"""Create LLM providers from config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanobot.config.schema import Config
from nanobot.providers.base import GenerationSettings, LLMProvider
from nanobot.providers.registry import find_by_name


@dataclass(frozen=True)
class ProviderSnapshot:
    provider: LLMProvider
    model: str
    context_window_tokens: int
    signature: tuple[object, ...]


def build_provider_for_model(
    config: Config,
    model: str,
    *,
    gen_src: Any | None = None,
) -> LLMProvider:
    """Create an LLM provider for a specific *model* string.

    *gen_src* provides generation settings (temperature, max_tokens,
    reasoning_effort). When omitted, ``config.resolve_preset()`` is used.
    """
    from nanobot.config.schema import ModelPresetConfig

    if gen_src is None:
        gen_src = config.resolve_preset()
    elif not isinstance(gen_src, ModelPresetConfig):
        # Accept a plain object with the three generation attributes
        gen_src = ModelPresetConfig(
            model=model,
            temperature=getattr(gen_src, "temperature", None),
            max_tokens=getattr(gen_src, "max_tokens", None),
            reasoning_effort=getattr(gen_src, "reasoning_effort", None),
        )

    # When a preset explicitly specifies a provider, use it directly instead of
    # inferring from config.defaults (which may point to a different active preset).
    if isinstance(gen_src, ModelPresetConfig) and gen_src.provider != "auto":
        provider_name = gen_src.provider
        p = getattr(config.providers, provider_name, None)
        spec = find_by_name(provider_name)
        api_base = (
            p.api_base
            if p and p.api_base
            else (spec.default_api_base if spec and spec.default_api_base else None)
        )
    else:
        provider_name = config.get_provider_name(model)
        p = config.get_provider(model)
        spec = find_by_name(provider_name) if provider_name else None
        api_base = config.get_api_base(model)
    backend = spec.backend if spec else "openai_compat"

    if backend == "azure_openai":
        if not p or not p.api_key or not p.api_base:
            raise ValueError("Azure OpenAI requires api_key and api_base in config.")
    elif backend == "openai_compat" and not model.startswith("bedrock/"):
        needs_key = not (p and p.api_key)
        exempt = spec and (spec.is_oauth or spec.is_local or spec.is_direct)
        if needs_key and not exempt:
            raise ValueError(f"No API key configured for provider '{provider_name}'.")

    if backend == "openai_codex":
        from nanobot.providers.openai_codex_provider import OpenAICodexProvider

        provider = OpenAICodexProvider(default_model=model)
    elif backend == "github_copilot":
        from nanobot.providers.github_copilot_provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(default_model=model)
    elif backend == "azure_openai":
        from nanobot.providers.azure_openai_provider import AzureOpenAIProvider

        provider = AzureOpenAIProvider(
            api_key=p.api_key,
            api_base=p.api_base,
            default_model=model,
        )
    elif backend == "anthropic":
        from nanobot.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(
            api_key=p.api_key if p else None,
            api_base=api_base,
            default_model=model,
            extra_headers=p.extra_headers if p else None,
        )
    else:
        from nanobot.providers.openai_compat_provider import OpenAICompatProvider

        provider = OpenAICompatProvider(
            api_key=p.api_key if p else None,
            api_base=api_base,
            default_model=model,
            extra_headers=p.extra_headers if p else None,
            spec=spec,
            extra_body=p.extra_body if p else None,
        )

    provider.generation = GenerationSettings(
        temperature=gen_src.temperature,
        max_tokens=gen_src.max_tokens,
        reasoning_effort=gen_src.reasoning_effort,
    )
    return provider


def make_provider(config: Config) -> LLMProvider:
    """Create the LLM provider implied by config (legacy entrypoint)."""
    resolved = config.resolve_preset()
    return build_provider_for_model(config, resolved.model, gen_src=resolved)


def make_provider_factory(config: Config):
    """Build a cached factory that creates providers for arbitrary model strings.

    If a model string matches a preset name in ``config.model_presets``, the
    preset's full config is used.
    """
    cache: dict[str, LLMProvider] = {}
    presets = getattr(config, "model_presets", {}) or {}

    def factory(model_or_preset: str) -> LLMProvider:
        preset = presets.get(model_or_preset)
        actual_model = preset.model if preset else model_or_preset
        key = model_or_preset
        if key not in cache:
            cache[key] = build_provider_for_model(
                config, actual_model, gen_src=preset
            )
        return cache[key]

    return factory


def provider_signature(config: Config) -> tuple[object, ...]:
    """Return the config fields that affect the primary LLM provider."""
    resolved = config.resolve_preset()
    return (
        resolved.model,
        resolved.provider,
        config.get_provider_name(resolved.model),
        config.get_api_key(resolved.model),
        config.get_api_base(resolved.model),
        resolved.max_tokens,
        resolved.temperature,
        resolved.reasoning_effort,
        resolved.context_window_tokens,
    )


def build_provider_snapshot(config: Config) -> ProviderSnapshot:
    resolved = config.resolve_preset()
    return ProviderSnapshot(
        provider=make_provider(config),
        model=resolved.model,
        context_window_tokens=resolved.context_window_tokens,
        signature=provider_signature(config),
    )


def load_provider_snapshot(config_path: Path | None = None) -> ProviderSnapshot:
    from nanobot.config.loader import load_config, resolve_config_env_vars

    return build_provider_snapshot(resolve_config_env_vars(load_config(config_path)))
