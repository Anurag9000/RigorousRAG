import itertools
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
                url="https://example.test/report?api_key=secret",
                source_type="web_page",
                snippet="file:///private/evidence.txt",
            )
        ],
        warnings=["Check /private/state and api_key=secret."],
    )

    print_result(answer)

    output = capsys.readouterr().out
    assert "/private" not in output
    assert "password" not in output
    assert "api_key=secret" not in output


def test_print_result_removes_terminal_controls(capsys):
    answer = AgentAnswer(
        answer="safe\x1b[2J\x7fanswer\rreplacement",
        warnings=["warn\x1b[31mred\rreset"],
        citations=[
            Citation(
                label="[1]",
                title="title\x1b[2J",
                url="https://example.test/source",
                source_type="web_page",
                snippet="snippet\x1b[31mred\rreset",
            )
        ],
    )

    print_result(answer)

    output = capsys.readouterr().out
    assert "\x1b" not in output
    assert "\x7f" not in output
    assert "\r" not in output
    assert "safe [2J answer replacement" in output


def test_print_result_surfaces_bounded_warnings(capsys):
    answer = AgentAnswer(
        answer="Answer",
        warnings=["Evidence coverage is incomplete.", "Second warning."],
    )

    print_result(answer)

    output = capsys.readouterr().out
    assert "Warnings:" in output
    assert "Evidence coverage is incomplete." in output
    assert "Second warning." in output


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


def test_cli_argument_stream_is_strict_and_bounded(monkeypatch):
    with pytest.raises(ValueError, match="iterable"):
        parse_args("--query q")
    with pytest.raises(ValueError, match="bounded valid strings"):
        parse_args(["--query", object()])
    for value in ("bad\x00arg", "bad\narg", "bad\x1barg", "bad\x7farg"):
        with pytest.raises(ValueError, match="bounded valid strings"):
            parse_args(["--query", value])
    with pytest.raises(ValueError, match="bounded valid strings"):
        parse_args(["--query", "q" * 20_001])

    monkeypatch.setattr("search_agent_cli._MAX_CLI_ARGUMENTS", 3)
    with pytest.raises(ValueError, match="At most 3"):
        parse_args((str(index) for index in itertools.count()))


def test_query_is_validated_before_agent_initialization(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    with patch("search_agent_cli.SearchAgent") as agent_class:
        assert main(["--query", "q" * 20_001]) == 2
        agent_class.assert_not_called()

    assert "valid strings" in capsys.readouterr().err


def test_query_controls_fail_before_agent_initialization(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    for query in ("bad\x00query", "bad\nquery", "bad\x1bquery", "bad\x7fquery"):
        with patch("search_agent_cli.SearchAgent") as agent_class:
            assert main(["--query", query]) == 2
        agent_class.assert_not_called()


def test_model_and_owner_are_validated_before_agent_initialization(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    for arguments in (
        ["--model", "m" * 201, "--query", "q"],
        ["--model", "bad\x1bmodel", "--query", "q"],
        ["--owner-id", "../other", "--query", "q"],
    ):
        with patch("search_agent_cli.SearchAgent") as agent_class:
            assert main(arguments) == 2
        agent_class.assert_not_called()


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
