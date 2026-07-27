from unittest.mock import patch

import pytest

from search_agent_cli import main, print_result
from tools.models import AgentAnswer, Citation


def test_print_result_handles_optional_snippet(capsys):
    answer = AgentAnswer(
        answer="Answer",
        citations=[Citation(
            label="[1]", title="Source", url="https://a.test",
            source_type="web_page", snippet=None,
        )],
    )
    print_result(answer)
    output = capsys.readouterr().out
    assert "Answer" in output
    assert "[1] Source" in output


def test_cloud_mode_requires_provider_configuration(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with patch("sys.argv", ["search_agent_cli.py", "--query", "q"]), pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_local_mode_builds_ollama_compatible_agent(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with patch("sys.argv", ["search_agent_cli.py", "--local", "--query", "q"]), \
         patch("search_agent_cli.SearchAgent") as agent_class, \
         patch("search_agent_cli.print_result"):
        agent_class.return_value.run.return_value = AgentAnswer(answer="done")
        main()
    kwargs = agent_class.call_args.kwargs
    assert kwargs["base_url"] == "http://localhost:11434/v1"
    assert kwargs["model"] == "llama3.1"
