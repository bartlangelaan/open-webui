import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def anthropic_module(monkeypatch):
    open_webui_pkg = types.ModuleType('open_webui')
    open_webui_pkg.__path__ = []
    env_module = types.ModuleType('open_webui.env')
    env_module.AIOHTTP_CLIENT_SESSION_SSL = True
    env_module.AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST = 30
    env_module.ENABLE_FORWARD_USER_INFO_HEADERS = False

    models_pkg = types.ModuleType('open_webui.models')
    users_module = types.ModuleType('open_webui.models.users')
    users_module.UserModel = type('UserModel', (), {})

    utils_pkg = types.ModuleType('open_webui.utils')
    utils_pkg.__path__ = []
    headers_module = types.ModuleType('open_webui.utils.headers')
    headers_module.include_user_info_headers = lambda headers, user: headers

    monkeypatch.setitem(sys.modules, 'open_webui', open_webui_pkg)
    monkeypatch.setitem(sys.modules, 'open_webui.env', env_module)
    monkeypatch.setitem(sys.modules, 'open_webui.models', models_pkg)
    monkeypatch.setitem(sys.modules, 'open_webui.models.users', users_module)
    monkeypatch.setitem(sys.modules, 'open_webui.utils', utils_pkg)
    monkeypatch.setitem(sys.modules, 'open_webui.utils.headers', headers_module)

    module_path = Path(__file__).resolve().parents[2] / 'utils' / 'anthropic.py'
    spec = importlib.util.spec_from_file_location('anthropic_under_test', module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_convert_anthropic_to_openai_payload_uses_output_config_format(anthropic_module):
    expected_format = {
        'type': 'json_schema',
        'json_schema': {'name': 'result', 'schema': {'type': 'object'}},
    }
    anthropic_payload = {
        'model': 'claude-test',
        'messages': [{'role': 'user', 'content': 'hello'}],
        'output_config': {'format': expected_format},
        'output_format': {'type': 'json_object'},
    }

    result = anthropic_module.convert_anthropic_to_openai_payload(anthropic_payload)

    assert result['response_format'] == expected_format


def test_convert_anthropic_to_openai_payload_falls_back_to_output_format(anthropic_module):
    expected_format = {'type': 'json_object'}
    anthropic_payload = {
        'model': 'claude-test',
        'messages': [{'role': 'user', 'content': 'hello'}],
        'output_format': expected_format,
    }

    result = anthropic_module.convert_anthropic_to_openai_payload(anthropic_payload)

    assert result['response_format'] == expected_format
