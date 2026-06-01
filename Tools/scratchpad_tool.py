import MemoryManagement.shortterm_memory.scratchpad as scratchpad
import MemoryManagement.shortterm_memory.summarizer as summarizer
# import MemoryManagement.shortterm_memory.conversation_history as conversation

def get_scratchpad_memory():
    return summarizer.compiled_scratchpad_memory(
        scratchpad.scratchpad.seven_notes,
        scratchpad.scratchpad.summary,
        scratchpad.scratchpad.current_goal,
        scratchpad.scratchpad.subtasks,
        scratchpad.scratchpad.retrieved_context,    
        scratchpad.scratchpad.tool_outputs,
        scratchpad.scratchpad.memory_updates
    )

def update_scratchpad_summary():
    scratchpad.scratchpad.add_conversation_summary("")

def update_scratchpad_seven_notes(notes):
    scratchpad.scratchpad.seven_notes = notes

def update_scratchpad_current_goal(goal):
    scratchpad.scratchpad.set_current_goal(goal)

def add_scratchpad_subtask(subtask):
    scratchpad.scratchpad.add_subtask(subtask)

def add_scratchpad_retrieved_context(context):
    scratchpad.scratchpad.add_retrieved_context(context)

def add_scratchpad_tool_output(output):
    scratchpad.scratchpad.add_tool_output(output)

def add_scratchpad_memory_update(update):
    scratchpad.scratchpad.add_memory_update(update)

