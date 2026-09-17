"""
Tools/tool.py

Base dataclass for all tools exposed to the LLM via the OpenAI-compatible
tool-calling interface. Each Tool has a name, description, parameter schema,
and a Python callable that executes the tool's logic.

The LLM never calls these directly — it emits tool_call JSON (either native
OpenAI-style or via <tool_call> tags for local models), which
ToolCalling/executor.py dispatches to the matching Tool's func via the
ToolCalling/register.py registry.
"""

from dataclasses import dataclass
from typing import Callable

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable