from __future__ import annotations

import pytest

from llm_gateway.gateway.config_loader import load_gateway_config
import utils.llm.model_config as model_config_module


@pytest.fixture
def model_config(monkeypatch):
    def _load(openrouter_key: str | None):
        if openrouter_key is None:
            monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
        else:
            monkeypatch.setenv('OPENROUTER_API_KEY', openrouter_key)
        model_config_module._openrouter_fallbacks_recorded.clear()
        return model_config_module

    return _load


def test_without_a_key_preferred_features_keep_their_direct_route(model_config):
    mc = model_config(None)

    assert mc.openrouter_configured() is False
    expected = {
        'memories': ('gpt-5.6-luna', 'openai'),
        'knowledge_graph': ('gpt-5.6-luna', 'openai'),
        'conv_structure': ('gpt-5.6-luna', 'openai'),
        'chat_extraction': ('gpt-5.6-luna', 'openai'),
        'chat_agent': ('claude-sonnet-4-6', 'anthropic'),
        'wrapped_analysis': ('gemini-3-flash-preview', 'gemini'),
        'session_titles': ('gemini-2.5-flash-lite', 'gemini'),
    }
    for feature, route in expected.items():
        assert mc.get_model_config(feature) == route, feature


def test_with_a_key_preferred_features_route_to_openrouter_luna(model_config):
    mc = model_config('sk-openrouter-test')

    assert mc.openrouter_configured() is True
    for feature in ('memories', 'knowledge_graph', 'conv_structure', 'chat_extraction', 'chat_agent', 'session_titles'):
        assert mc.get_model_config(feature) == ('gpt-5.6-luna', 'openrouter'), feature
    assert mc.get_model_config('wrapped_analysis') == ('gemini-3-flash-preview', 'openrouter')


def test_a_blank_key_counts_as_absent(model_config):
    mc = model_config('   ')

    assert mc.openrouter_configured() is False
    assert mc.get_model_config('memories')[1] != 'openrouter'


def test_gateway_routes_follow_the_same_key_gate(model_config):
    model_config(None)
    direct = load_gateway_config(prod_mode=False)

    assert direct.route_artifacts['route.memories.model_config.001'].primary.provider == 'openai'
    assert direct.route_artifacts['route.chat_structured.2026_06_27.001'].primary.provider == 'openai'
    assert direct.route_artifacts['route.public_shared_conversation_chat.2026_07_19.001'].primary.provider == 'openai'

    model_config('sk-openrouter-test')
    managed = load_gateway_config(prod_mode=False)

    assert managed.route_artifacts['route.memories.model_config.001'].primary.model_dump() == {
        'provider': 'openrouter',
        'model': 'openai/gpt-5.6-luna',
    }
    assert managed.route_artifacts['route.chat_agent.model_config.001'].primary.model_dump() == {
        'provider': 'openrouter',
        'model': 'openai/gpt-5.6-luna',
    }
    assert managed.route_artifacts['route.chat_agent.model_config.001'].provider_options == {
        'reasoning_effort': 'none',
    }
    assert managed.route_artifacts['route.wrapped_analysis.model_config.001'].primary.model_dump() == {
        'provider': 'openrouter',
        'model': 'google/gemini-3-flash-preview',
    }
    assert managed.route_artifacts['route.session_titles.model_config.001'].output_budget.max_completion_tokens == 128
    assert managed.route_artifacts['route.chat_structured.2026_06_27.001'].primary.model_dump() == {
        'provider': 'openrouter',
        'model': 'openai/gpt-5.6-luna',
    }
    assert managed.route_artifacts['route.public_shared_conversation_chat.2026_07_19.001'].primary.model_dump() == {
        'provider': 'openrouter',
        'model': 'openai/gpt-5.6-luna',
    }


def test_features_outside_the_preference_set_are_untouched_either_way(model_config):
    without = model_config(None)
    baseline = {f: without.get_model_config(f) for f in ('web_search', 'fair_use')}

    with_key = model_config('sk-openrouter-test')
    for feature, expected in baseline.items():
        assert with_key.get_model_config(feature) == expected, feature


def test_the_fallback_is_recorded_once_per_feature(model_config, monkeypatch):
    mc = model_config(None)
    calls: list[dict] = []
    monkeypatch.setattr(mc, '_openrouter_fallbacks_recorded', set())

    import utils.observability.fallback as fallback_mod

    monkeypatch.setattr(fallback_mod, 'record_fallback', lambda **kwargs: calls.append(kwargs))

    for _ in range(3):
        mc.get_model_config('memories')
    mc.get_model_config('knowledge_graph')

    assert len(calls) == 2
    assert {c['from_mode'] for c in calls} == {'openrouter'}
    assert {c['outcome'] for c in calls} == {'degraded'}
