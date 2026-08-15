from types import ModuleType

from tools.agent_tool_registry import AgentToolSpec, ToolResult
from tools.agent_tool_registry_integration import (
    install_agent_tool_registry_bridge,
    register_governed_agent_tool,
)


def _module(dispatcher_name: str) -> ModuleType:
    module = ModuleType(f"fake_search_agent_{dispatcher_name}")
    module.TOOLS_SCHEMA = []
    module._TOOL_PARAMETER_SCHEMAS = {}

    class SearchAgent:
        def __init__(self) -> None:
            self.owner_id = "owner-a"
            self.permissions = ()

    def legacy_dispatch(self, name, args):
        return f"legacy:{name}:{args.get('value', '')}", []

    setattr(SearchAgent, dispatcher_name, legacy_dispatch)
    module.SearchAgent = SearchAgent
    return module


def _spec() -> AgentToolSpec:
    def handler(owner_id, arguments, budget):
        assert owner_id == "owner-a"
        assert budget is not None
        return ToolResult(
            content={"owner": owner_id, "echo": arguments["value"]},
            warnings=("governed",),
        )

    return AgentToolSpec(
        name="governed_echo",
        description="Echo a bounded value through the governed registry.",
        schema={
            "type": "object",
            "properties": {"value": {"type": "string", "minLength": 1, "maxLength": 100}},
            "required": ["value"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def test_bridge_prefers_live_dispatch_boundary_and_preserves_fallback():
    module = _module("_dispatch")
    install_agent_tool_registry_bridge(module)
    register_governed_agent_tool(module, _spec())

    assert module._agent_tool_registry_dispatcher_name == "_dispatch"
    assert [item["function"]["name"] for item in module.TOOLS_SCHEMA] == ["governed_echo"]
    assert "governed_echo" in module._TOOL_PARAMETER_SCHEMAS

    agent = module.SearchAgent()
    content, citations = agent._dispatch("governed_echo", {"value": "hello"})
    assert citations == []
    assert '"echo":"hello"' in content
    assert '"calls":1' in content
    assert agent._dispatch("legacy", {"value": "x"}) == ("legacy:legacy:x", [])


def test_bridge_retains_dispatch_tool_compatibility():
    module = _module("_dispatch_tool")
    install_agent_tool_registry_bridge(module)
    register_governed_agent_tool(module, _spec())

    assert module._agent_tool_registry_dispatcher_name == "_dispatch_tool"
    agent = module.SearchAgent()
    content, citations = agent._dispatch_tool("governed_echo", {"value": "hello"})
    assert citations == []
    assert '"echo":"hello"' in content
    assert agent._dispatch_tool("legacy", {"value": "x"}) == ("legacy:legacy:x", [])


def test_bridge_is_idempotent_without_rewrapping_dispatcher():
    module = _module("_dispatch")
    install_agent_tool_registry_bridge(module)
    first = module.SearchAgent._dispatch
    install_agent_tool_registry_bridge(module)
    assert module.SearchAgent._dispatch is first
