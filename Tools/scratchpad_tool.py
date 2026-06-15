import MemoryManagement.shortterm_memory.scratchpad as scratchpad
import MemoryManagement.shortterm_memory.summarizer as summarizer

def get_scratchpad_memory():
    return scratchpad.get_compiled_memory()


def update_scratchpad_summary(summary_text):
    scratchpad.scratchpad.add_conversation_summary(summary_text)
    scratchpad.scratchpad.add_tool_output("update_scratchpad_summary", f"Updated scratchpad summary")


def update_scratchpad_seven_notes(notes):
    scratchpad.scratchpad.set_seven_notes(notes)
    scratchpad.scratchpad.add_tool_output("update_scratchpad_seven_notes", f"Updated seven notes section")


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
    """Set the current step in execution"""
    scratchpad.scratchpad.set_current_step(step)
    scratchpad.scratchpad.add_tool_output("set_current_step", f"Set current step to: {step}")


def set_next_action(action):
    """Set the next action to execute"""
    scratchpad.scratchpad.set_next_action(action)
    scratchpad.scratchpad.add_tool_output("set_next_action", f"Set next action to: {action}")


def set_active_tool(tool_name):
    """Set the currently active tool"""
    scratchpad.scratchpad.set_active_tool(tool_name)
    scratchpad.scratchpad.add_tool_output("set_active_tool", f"Set active tool to: {tool_name}")


def add_scratchpad_retrieved_context(context):
    """Add retrieved context to the scratchpad"""
    scratchpad.scratchpad.add_retrieved_context(context)
    scratchpad.scratchpad.add_tool_output("add_scratchpad_retrieved_context", f"Added retrieved context")


def add_scratchpad_tool_output(tool_name, output):
    """Add tool output to the scratchpad"""
    scratchpad.scratchpad.add_tool_output(tool_name, output)


def add_scratchpad_memory_update(type, data, update):
    if update:
        scratchpad.scratchpad.add_memory_update(type, data, True)
    else:
        scratchpad.scratchpad.add_memory_update(type, data, False)
    scratchpad.scratchpad.add_tool_output("add_scratchpad_memory_update", f"Memory update added")


def get_scratchpad_memory_updates():
    return scratchpad.scratchpad.get_memory_updates()


def increment_retry_count():
    """Increment the retry counter"""
    scratchpad.scratchpad.increment_retry_count()


def set_last_error(error):
    """Set the last error message"""
    scratchpad.scratchpad.set_last_error(error)
    scratchpad.scratchpad.add_tool_output("set_last_error", f"Error recorded: {error}")


def clear_scratchpad_data():
    """Clear all scratchpad data"""
    scratchpad.clear_scratchpad()
    print("Scratchpad cleared")


def update_scratchpad_state(section, key, value):
    """Update the scratchpad state dict at the specified section and key.
    
    Args:
        section: One of 'planning', 'execution', 'reflection', 'tool_outputs'
        key: The key within the section to update
        value: The value to set
    
    Examples:
        update_scratchpad_state('planning', 'current_goal', 'Build AI system')
        update_scratchpad_state('execution', 'active_tool', 'search_tool')
        update_scratchpad_state('execution', 'retry_count', 3)
        update_scratchpad_state('reflection', 'seven_notes', 'Important notes...')
    """
    valid_sections = ['planning', 'execution', 'reflection', 'tool_outputs']
    if section not in valid_sections:
        raise ValueError(f"Invalid section '{section}'. Must be one of: {valid_sections}")
    
    scratchpad.scratchpad.state[section][key] = value
    scratchpad.scratchpad.add_tool_output("update_scratchpad_state", f"Updated {section}.{key} = {value}")


def get_scratchpad_state():
    return scratchpad.scratchpad.state


def get_scratchpad_retrieved_context(working_memory_only, include_tool_outputs, include_all_working_memory):
    return scratchpad.scratchpad.get_retrieved_context(working_memory_only, include_tool_outputs, include_all_working_memory)



