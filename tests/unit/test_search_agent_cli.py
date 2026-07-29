from types import SimpleNamespace
from unittest.mock import patch

import pytest

from search_agent_cli import main, parse_args, print_result
from tools.models import AgentAnswer, Citation


def test_print_result_handles_optional_snippet(capsys):
    answer = AgentAnswer(
        answer="Answer",
        citations=[
            Citation(
                label="[1]",
                title="Source",
                url="https://a.test",
                source_type="web_page",
                snippet=None,
            )
        ],
    )
    print_result(answer)
    output = capsys.readouterr().out
    assert "Answer" in output
    assert "[1] Source" in output


def test_print_result_masks_private_metadata(capsys):
    answer = AgentAnswer(
        answer="See file:///private/secret and https://alice:password@example.test.",
        citations=[
            Citation(
                label="[1]",
                title="Report at /private/report.txt",
                url="https://alice:password@example.test?api_key=secret",
                source_type="web_page",
                snippet="file:///private/evidence.txt",
            )
        ],
    )

    print_result(answer)

    output = capsys.readouterr().out
    assert "/private" not in output
    assert "password" not in output
    assert "api_key=secret" not in output


def test_print_result_rejects_invalid_result():
    with pytest.raises(ValueError, match="invalid result"):
        print_result(object())


def test_cloud_mode_requires_provider_configuration(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    assert main(["--query", "q"]) == 1

    error = capsys.readouterr().err
    assert "could not be initialized" in error
    assert "OPENAI_API_KEY" not in error


def test_local_mode_builds_ollama_compatible_agent(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with patch("search_agent_cli.SearchAgent") as agent_class, patch(
        "search_agent_cli.print_result"
    ):
        agent_class.return_value.run.return_value = AgentAnswer(answer="done")
        assert main(["--local", "--query", "q", "--owner-id", "alice"]) == 0

    kwargs = agent_class.call_args.kwargs
    assert kwargs["base_url"] == "http://localhost:11434/v1"
    assert kwargs["model"] == "llama3.1"
    assert kwargs["owner_id"] == "alice"


def test_demo_and_local_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit) as captured:
        parse_args(["--local", "--demo"])
    assert captured.value.code == 2


def test_query_is_validated_before_agent_run(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    with patch("search_agent_cli.SearchAgent") as agent_class:
        assert main(["--query", "q" * 20_001]) == 2
        agent_class.return_value.run.assert_not_called()

    assert "20,000" in capsys.readouterr().err


def test_model_name_and_query_are_not_echoed_in_status_output(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    secret_model = "private-model-name"
    secret_query = "private research question"
    with patch("search_agent_cli.SearchAgent") as agent_class, patch(
        "search_agent_cli.print_result"
    ):
        agent_class.return_value.run.return_value = AgentAnswer(answer="done")
        assert main(["--model", secret_model, "--query", secret_query]) == 0

    output = capsys.readouterr().out
    assert secret_model not in output
    assert secret_query not in output


def test_agent_initialization_error_is_generic(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    with patch(
        "search_agent_cli.SearchAgent",
        side_effect=RuntimeError("provider failed at /private/config"),
    ):
        assert main(["--query", "q"]) == 1

    error = capsys.readouterr().err
    assert "/private" not in error
    assert "could not be initialized" in error


def test_interactive_request_failure_does_not_exit_or_leak(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    agent = SimpleNamespace(
        run=lambda _query: (_ for _ in ()).throw(
            RuntimeError("secret at /private/provider")
        )
    )
    inputs = iter(["question", "quit"])
    with patch("search_agent_cli.SearchAgent", return_value=agent), patch(
        "builtins.input", side_effect=lambda _prompt: next(inputs)
    ):
        assert main([]) == 0

    output = capsys.readouterr().out
    assert "research request failed" in output
    assert "/private" not in output
