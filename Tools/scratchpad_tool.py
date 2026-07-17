import re

import MemoryManagement.shortterm_memory.scratchpad as scratchpad
import MemoryManagement.shortterm_memory.summarizer as summarizer
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

def _coerce_subtask_list(value):
    """
    BUG FIX: update_scratchpad_state's tool schema forces `value` to type
    "string" (see _build_tool_schema in llm_client.py, which has no concept
    of array types), so the LLM can only ever send subtasks as one long
    string like "1. Do X. 2. Do Y. 3. Do Z." — never a real list.
 
    summarizer.py does "\n".join(subtasks) expecting a list. Joining a
    plain string puts a newline between every CHARACTER, not every item —
    that's the letter-per-line output seen in production logs.
 
    This normalizes whatever comes in (already a list, a numbered string,
    a newline-separated string, or a single bare string) into a real list
    of individual subtask strings.
    """
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if not isinstance(value, str):
        return []
 
    text = value.strip()
    if not text:
        return []
 
    # Try numbered list markers first: "1. ", "2. ", etc.
    items = [i.strip().rstrip(".") for i in re.split(r"\d+\.\s*", text) if i.strip()]
    if len(items) > 1:
        return items
 
    # Fallback: newline-separated
    items = [i.strip() for i in text.split("\n") if i.strip()]
    if len(items) > 1:
        return items
 
    # Last resort: treat the whole string as a single subtask
    return [text]


_ALLOWED_KEYS = {
    "planning":   {"current_goal", "subtasks", "completed_subtasks", "current_step", "next_action"},
    "execution":  {"active_tool", "retry_count", "last_error"},
    "reflection": {"seven_notes"},
    # tool_outputs accepts any string key (tool names are dynamic)
    "tool_outputs": None,
}


def get_scratchpad_memory():
    return scratchpad.get_compiled_memory()


def update_scratchpad_summary(summary_text):
    scratchpad.scratchpad.add_conversation_summary(summary_text)
    scratchpad.scratchpad.add_tool_output("update_scratchpad_summary", "Updated scratchpad summary")


def update_scratchpad_seven_notes(notes):
    scratchpad.scratchpad.set_seven_notes(notes)
    scratchpad.scratchpad.add_tool_output("update_scratchpad_seven_notes", "Updated seven notes section")


def update_scratchpad_current_goal(goal):
    scratchpad.scratchpad.set_current_goal(goal)
    scratchpad.scratchpad.add_tool_output("update_scratchpad_current_goal", f"Updated current goal to: {goal}")


def add_scratchpad_subtask(subtask):
    scratchpad.scratchpad.add_subtask(subtask)
    scratchpad.scratchpad.add_tool_output("add_scratchpad_subtask", f"Added subtask: {subtask}")


def mark_scratchpad_subtask_completed(subtask):
    scratchpad.scratchpad.mark_subtask_completed(subtask)
    scratchpad.scratchpad.add_tool_output("mark_scratchpad_subtask_completed", f"Marked completed: {subtask}")


def set_current_step(step):
    scratchpad.scratchpad.set_current_step(step)
    scratchpad.scratchpad.add_tool_output("set_current_step", f"Set current step to: {step}")


def set_next_action(action):
    scratchpad.scratchpad.set_next_action(action)
    scratchpad.scratchpad.add_tool_output("set_next_action", f"Set next action to: {action}")


def set_active_tool(tool_name):
    scratchpad.scratchpad.set_active_tool(tool_name)
    scratchpad.scratchpad.add_tool_output("set_active_tool", f"Set active tool to: {tool_name}")


def add_scratchpad_retrieved_context(context):
    scratchpad.scratchpad.add_retrieved_context(context)
    scratchpad.scratchpad.add_tool_output("add_scratchpad_retrieved_context", "Added retrieved context")


def add_scratchpad_tool_output(tool_name, output):
    scratchpad.scratchpad.add_tool_output(tool_name, output)


def get_scratchpad_memory_updates():
    return scratchpad.scratchpad.get_memory_updates()


def increment_retry_count():
    scratchpad.scratchpad.increment_retry_count()


def set_last_error(error):
    scratchpad.scratchpad.set_last_error(error)
    scratchpad.scratchpad.add_tool_output("set_last_error", f"Error recorded: {error}")


def clear_scratchpad_data():
    scratchpad.scratchpad.reset()
    log.info("Scratchpad cleared")


def update_scratchpad_state(section, key, value):
    """Update the scratchpad state dict at the specified section and key.

    Args:
        section: One of 'planning', 'execution', 'reflection', 'tool_outputs'
        key:     The key within the section to update
        value:   The value to set
    """
    valid_sections = list(_ALLOWED_KEYS.keys())
    if section not in valid_sections:
        raise ValueError(f"Invalid section '{section}'. Must be one of: {valid_sections}")

    allowed = _ALLOWED_KEYS[section]
    if allowed is not None and key not in allowed:
        raise ValueError(
            f"Invalid key '{key}' for section '{section}'. "
            f"Allowed keys: {sorted(allowed)}"
        )
    
    # BUG FIX: subtasks/completed_subtasks must always be a list — the LLM
    # can only send strings through this tool's schema, so normalize here
    # rather than trusting the caller. See _coerce_subtask_list docstring.
    if section == "planning" and key in ("subtasks", "completed_subtasks"):
        value = _coerce_subtask_list(value)

    scratchpad.scratchpad.state[section][key] = value
    scratchpad.scratchpad.add_tool_output(
        "update_scratchpad_state", f"Updated {section}.{key} = {value}"
    )


def get_scratchpad_state():
    return scratchpad.scratchpad.state


def get_scratchpad_retrieved_context(working_memory_only=False,
                                     include_tool_outputs=True,
                                     include_all_working_memory=False):
    import Tools.working_memory_tool as working_memory_tool

    retrieved_context = ""

    if working_memory_only:
        retrieved_context += str(working_memory_tool.get_working_memory())

    if include_tool_outputs:
        retrieved_context += str(scratchpad.scratchpad.get_tool_outputs())

    if include_all_working_memory:
        retrieved_context += str(working_memory_tool.get_all_working_memory_current_session())

    return retrieved_context
