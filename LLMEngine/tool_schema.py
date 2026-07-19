"""
LLMEngine/tool_schema.py

Converts the flat {param_name: description} dict used by Tool objects
(see Tools/tool.py, ToolCalling/register.py) into a valid JSON Schema
for the OpenAI-compatible /v1/chat/completions "tools" field.

Split out of llm_client.py, where it was one of five unrelated jobs
living in the same file.
"""


def build_tool_schema(parameters: dict) -> dict:
    """Infers a JSON Schema type per parameter from the leading word of
    its description string (e.g. "bool - ..." -> boolean). All tool
    parameters in this codebase are declared this way — see register.py."""
    if not parameters:
        return {"type": "object", "properties": {}}

    properties = {}
    for name, desc in parameters.items():
        desc_str = str(desc)
        if desc_str.startswith("bool"):
            prop_type = "boolean"
        elif desc_str.startswith("int"):
            prop_type = "integer"
        elif desc_str.startswith("float"):
            prop_type = "number"
        else:
            prop_type = "string"
        properties[name] = {"type": prop_type, "description": desc_str}

    return {
        "type":       "object",
        "properties": properties,
        "required":   list(parameters.keys()),
    }
