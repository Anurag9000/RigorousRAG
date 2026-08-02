from __future__ import annotations

from types import ModuleType

from tools import evidence_graph_agent_import_hook as import_hook


def test_partial_legacy_module_installs_after_final_required_assignment(
    monkeypatch,
):
    module = ModuleType("partial_search_agent_legacy")
    installed = []

    def install(value):
        installed.append(value)
        value._evidence_graph_agent_tool_installed = True

    monkeypatch.setattr(import_hook, "_install", install)
    import_hook._arm_deferred_install(module)

    assert isinstance(module, import_hook._DeferredEvidenceGraphModule)
    module.TOOLS_SCHEMA = []
    module._TOOL_PARAMETER_SCHEMAS = {}
    assert installed == []

    module.SearchAgent = type("SearchAgent", (), {})

    assert installed == [module]
    assert module.__class__ is ModuleType
    assert module._evidence_graph_agent_tool_installed is True


def test_deferred_install_is_not_armed_twice(monkeypatch):
    module = ModuleType("partial_search_agent_legacy")
    monkeypatch.setattr(import_hook, "_install", lambda value: None)

    import_hook._arm_deferred_install(module)
    watched_class = module.__class__
    import_hook._arm_deferred_install(module)

    assert module.__class__ is watched_class
