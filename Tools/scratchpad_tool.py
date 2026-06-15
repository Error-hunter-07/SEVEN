import MemoryManagement.shortterm_memory.scratchpad as scratchpad
import MemoryManagement.shortterm_memory.summarizer as summarizer


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
    print("Scratchpad cleared")


def update_scratchpad_state(section, key, value):
    """Update the scratchpad state dict at the specified section and key.

    Args:
        section: One of 'planning', 'execution', 'reflection', 'tool_outputs'
        key: The key within the section to update
        value: The value to set
    """
    valid_sections = ['planning', 'execution', 'reflection', 'tool_outputs']
    if section not in valid_sections:
        raise ValueError(f"Invalid section '{section}'. Must be one of: {valid_sections}")

    scratchpad.scratchpad.state[section][key] = value
    scratchpad.scratchpad.add_tool_output("update_scratchpad_state", f"Updated {section}.{key} = {value}")


def get_scratchpad_state():
    return scratchpad.scratchpad.state


def get_scratchpad_retrieved_context(working_memory_only=False, include_tool_outputs=True, include_all_working_memory=False):
    import Tools.working_memory_tool as working_memory_tool

    retrieved_context = ""

    if working_memory_only:
        retrieved_context += str(working_memory_tool.get_working_memory())

    if include_tool_outputs:

        retrieved_context += str(scratchpad.scratchpad.get_tool_outputs())

    if include_all_working_memory:
        retrieved_context += str(working_memory_tool.get_all_working_memory_current_session())

    return retrieved_context