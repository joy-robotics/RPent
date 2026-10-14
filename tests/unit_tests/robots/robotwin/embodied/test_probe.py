import argparse
import json
from types import SimpleNamespace

import pytest

from robots.robotwin.embodied import probe, suite


@pytest.fixture
def planner_environment(monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "RPENT_MODEL",
        "RPENT_BASE_URL",
        "TEST_PLANNER_KEY",
        "TEST_PLANNER_MODEL",
        "TEST_PLANNER_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def planner_arguments(*arguments):
    parser = argparse.ArgumentParser()
    probe.add_planner_args(parser)
    return parser.parse_args(arguments)


@pytest.mark.parametrize(
    ("model", "auth_name"),
    [
        ("openai:test-model", "OPENAI_API_KEY"),
        ("openai-chat:test-model", "OPENAI_API_KEY"),
        ("anthropic:test-model", "ANTHROPIC_API_KEY"),
    ],
)
def test_probe_uses_provider_environment_without_config(
    planner_environment, model, auth_name
):
    planner_environment.setenv(auth_name, "offline-test-key")
    settings, environment = probe._planner_settings(planner_arguments("--model", model))
    assert settings == {
        "model": model,
        "base_url": None,
        "reasoning_effort": "none",
    }
    assert environment == {auth_name: "offline-test-key"}
    assert "offline-test-key" not in json.dumps(settings)


def test_probe_resolves_config_environment_and_cli_precedence(
    tmp_path, planner_environment
):
    config = tmp_path / "planner.json"
    config.write_text(
        json.dumps(
            {
                "planner": {
                    "model": "openai:configured",
                    "model_env": "TEST_PLANNER_MODEL",
                    "base_url": "https://configured.example/v1",
                    "base_url_env": "TEST_PLANNER_URL",
                    "api_key_env": "TEST_PLANNER_KEY",
                    "reasoning_effort": "low",
                }
            }
        )
    )
    planner_environment.setenv("TEST_PLANNER_KEY", "offline-test-key")
    planner_environment.setenv("TEST_PLANNER_MODEL", "openai:environment")
    planner_environment.setenv("TEST_PLANNER_URL", "https://environment.example/v1")
    arguments = planner_arguments("--planner-config", str(config))
    settings, environment = probe._planner_settings(arguments)
    assert settings == {
        "model": "openai:environment",
        "base_url": "https://environment.example/v1",
        "reasoning_effort": "low",
    }
    assert environment == {"OPENAI_API_KEY": "offline-test-key"}
    arguments.model = "openai-chat:explicit"
    arguments.base_url = "https://explicit.example/v1"
    arguments.reasoning_effort = "medium"
    settings, _ = probe._planner_settings(arguments)
    assert settings == {
        "model": "openai-chat:explicit",
        "base_url": "https://explicit.example/v1",
        "reasoning_effort": "medium",
    }


@pytest.mark.parametrize("field", ["api_key", "token", "secret", "authorization"])
def test_probe_rejects_inline_credentials(tmp_path, planner_environment, field):
    config = tmp_path / "planner.json"
    config.write_text(json.dumps({"planner": {field: "offline-test-key"}}))
    with pytest.raises(ValueError, match="Inline credentials are not allowed"):
        probe._planner_settings(planner_arguments("--planner-config", str(config)))


@pytest.mark.parametrize(
    "base_url",
    [
        "https://user:password@planner.example/v1",
        "https://planner.example/v1?key=secret",
        "https://planner.example/v1#secret",
        "file:///tmp/planner",
    ],
)
def test_probe_rejects_credential_bearing_or_non_http_urls(
    planner_environment, base_url
):
    with pytest.raises(ValueError, match="Planner base URL"):
        probe._planner_settings(
            planner_arguments("--model", "openai:test", "--base-url", base_url)
        )


def test_probe_requires_explicit_provider_credentials(planner_environment):
    with pytest.raises(ValueError, match="Export TEST_PLANNER_KEY"):
        probe._planner_settings(
            planner_arguments(
                "--model", "openai:test", "--api-key-env", "TEST_PLANNER_KEY"
            )
        )


def test_suite_forwards_native_planner_options(tmp_path, monkeypatch):
    calls = []

    def run_probe(arguments):
        calls.append(arguments)
        arguments.output.mkdir()
        return {"passed": True, "real_api_planner": True}

    monkeypatch.setattr(probe, "run_probe", run_probe)
    sample = {
        "task": "video_grounding_3d",
        "images": ["first.png", "last.png"],
        "camera": {"width": 320, "height": 240},
        "target": "chair",
    }
    options = {
        "planner_config": tmp_path / "planner.json",
        "model": "openai:test",
        "base_url": "https://planner.example/v1",
        "reasoning_effort": "low",
        "api_key_env": "TEST_PLANNER_KEY",
    }
    args = SimpleNamespace(
        output=tmp_path,
        endpoint="ws://localhost:19194",
        checkpoint="/checkpoint",
        timeout=180,
        **options,
    )
    report = suite.run_planner_case(sample, args)
    assert report["passed"] is True
    assert report["task"] == "video_grounding_3d"
    assert len(calls) == 1
    forwarded = calls[0]
    assert {name: getattr(forwarded, name) for name in options} == options
    assert forwarded.command == "planner"
    assert forwarded.image.name == "first.png"
    assert [image.name for image in forwarded.additional_image] == ["last.png"]
    assert json.loads(forwarded.camera_meta.read_text()) == sample["camera"]
    assert (
        json.loads((forwarded.output / "dataset_reference.json").read_text()) == sample
    )
