import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class StubLLM:
    """
    Modelo de mentira: devuelve, en orden, las llamadas a herramientas que se
    le indiquen. Cuenta cuántas veces se le invoca.
    """

    def __init__(self, tool_calls_per_turn):
        self.turns = list(tool_calls_per_turn)
        self.calls = 0
        self.bound_tool_choice = None

    def bind_tools(self, tools, tool_choice=None):
        self.bound_tool_choice = tool_choice
        return self

    def invoke(self, messages):
        self.calls += 1
        if not self.turns:
            return AIMessage(content="listo")
        calls = self.turns.pop(0)
        if isinstance(calls, Exception):
            raise calls
        return AIMessage(
            content="",
            tool_calls=[{"id": f"call_{self.calls}_{n}", "name": name, "args": args, "type": "tool_call"}
                        for n, (name, args) in enumerate(calls)],
        )


class FailingLLM(StubLLM):
    def __init__(self):
        super().__init__([])

    def invoke(self, messages):
        raise AssertionError("El agente no debía llamarse")


@pytest.fixture
def stub_llm():
    return StubLLM
