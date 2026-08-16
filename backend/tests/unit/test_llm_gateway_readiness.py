from __future__ import annotations

from fastapi.testclient import TestClient

from llm_gateway.gateway.config_loader import ConfigValidationError
from llm_gateway.gateway.schemas import LaneConfig
from llm_gateway.main import app
from llm_gateway.routers import health
from utils.llm.model_config import get_all_configured_features


def test_ready_requires_service_auth(monkeypatch):
    monkeypatch.setenv('LLM_GATEWAY_SERVICE_TOKEN', 'shared-secret')

    response = TestClient(app).get('/ready')

    assert response.status_code == 401
    assert response.json()['detail'] == 'invalid service authentication'


def test_ready_validates_gateway_config(monkeypatch):
    monkeypatch.setenv('LLM_GATEWAY_SERVICE_TOKEN', 'shared-secret')

    response = TestClient(app).get('/ready', headers=auth_headers())

    assert response.status_code == 200
    assert response.json()['status'] == 'ready'
    assert 'omi:auto:chat-structured' in response.json()['lanes']
    assert len(response.json()['lanes']) >= len(get_all_configured_features())
    assert response.json()['route_artifact_count'] >= len(get_all_configured_features()) + 2
    assert response.json()['managed_messages_provider'] == 'none'
    assert response.json()['managed_chat_provider'] == 'openai'


def test_ready_does_not_require_anthropic_key_after_chat_agent_migration(monkeypatch):
    monkeypatch.setenv('LLM_GATEWAY_SERVICE_TOKEN', 'shared-secret')
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)

    response = TestClient(app).get('/ready', headers=auth_headers())

    assert response.status_code == 200
    assert response.json()['managed_messages_provider'] == 'none'
    assert response.json()['managed_chat_provider'] == 'openai'


def test_ready_fails_closed_when_managed_openai_key_is_missing(monkeypatch):
    monkeypatch.setenv('LLM_GATEWAY_SERVICE_TOKEN', 'shared-secret')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    response = TestClient(app).get('/ready', headers=auth_headers())

    assert response.status_code == 503
    assert response.json()['detail'] == 'llm gateway managed chat provider is not configured'


def test_ready_fails_closed_when_an_explicit_anthropic_lane_is_present(monkeypatch):
    monkeypatch.setenv('LLM_GATEWAY_SERVICE_TOKEN', 'shared-secret')
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.setattr(health, '_managed_anthropic_messages_enabled', lambda _config: True)
    monkeypatch.setattr(health, '_managed_openai_chat_enabled', lambda _config: False)

    response = TestClient(app).get('/ready', headers=auth_headers())

    assert response.status_code == 503
    assert response.json()['detail'] == 'llm gateway managed messages provider is not configured'


def test_ready_fails_closed_on_invalid_gateway_config(monkeypatch):
    monkeypatch.setenv('LLM_GATEWAY_SERVICE_TOKEN', 'shared-secret')

    def invalid_config():
        raise ConfigValidationError('bad test config')

    monkeypatch.setattr(health, 'get_gateway_config', invalid_config)

    response = TestClient(app).get('/ready', headers=auth_headers())

    assert response.status_code == 503
    assert response.json()['detail'] == 'llm gateway config is invalid'


def test_ready_fails_closed_on_schema_validation_error(monkeypatch):
    monkeypatch.setenv('LLM_GATEWAY_SERVICE_TOKEN', 'shared-secret')

    def invalid_config():
        LaneConfig.model_validate({})

    monkeypatch.setattr(health, 'get_gateway_config', invalid_config)

    response = TestClient(app).get('/ready', headers=auth_headers())

    assert response.status_code == 503
    assert response.json()['detail'] == 'llm gateway config is invalid'


def test_ready_never_leaks_key_material_or_fingerprints(monkeypatch):
    monkeypatch.setenv('LLM_GATEWAY_SERVICE_TOKEN', 'shared-secret')
    secret = 'sk-or-never-log-this-raw-key-9173'
    monkeypatch.setenv('OPENROUTER_API_KEY', secret)
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-openai-never-log-this-too')
    monkeypatch.setattr(health, '_managed_openrouter_chat_enabled', lambda _config: True)
    monkeypatch.setattr(health, '_managed_openai_chat_enabled', lambda _config: False)

    response = TestClient(app).get('/ready', headers=auth_headers())
    body = str(response.json())

    assert response.status_code == 200
    assert secret not in body
    assert secret[:12] not in body
    assert 'OPENROUTER_API_KEY' not in body
    assert 'OPENAI_API_KEY' not in body
    assert 'forwarded_provider_keys' not in body
    # Readiness reports provider routing state, never credentials.
    assert response.json()['managed_chat_provider'] == 'openrouter'


def auth_headers() -> dict[str, str]:
    return {
        'authorization': 'Bearer shared-secret',
        'x-omi-service-caller': 'backend',
    }
