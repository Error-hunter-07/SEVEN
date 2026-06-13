from .summarizer import compiled_scratchpad_memory

class Scratchpad:
    def __init__(self):
        self.summary = ""
        self.retrieved_memories = []
        self.memory_updates = []
        
        self.state = {
            "planning": {
                "current_goal": "",
                "subtasks": [],
                "completed_subtasks": [],
                "current_step": "",
                "next_action": ""
            },
            "execution": {
                "active_tool": "",
                "retry_count": 0,
                "last_error": None
            },
            "reflection": {
                "seven_notes": ""
            },
            "tool_outputs": {}
        }

    # Planning methods
    def set_current_goal(self, goal):
        self.state["planning"]["current_goal"] = goal

    def get_current_goal(self):
        return self.state["planning"]["current_goal"]

    def add_subtask(self, subtask):
        self.state["planning"]["subtasks"].append(subtask)

    def get_subtasks(self):
        return self.state["planning"]["subtasks"]

    def mark_subtask_completed(self, subtask):
        if subtask in self.state["planning"]["subtasks"]:
            self.state["planning"]["subtasks"].remove(subtask)
            self.state["planning"]["completed_subtasks"].append(subtask)

    def get_completed_subtasks(self):
        return self.state["planning"]["completed_subtasks"]

    def set_current_step(self, step):
        self.state["planning"]["current_step"] = step

    def get_current_step(self):
        return self.state["planning"]["current_step"]

    def set_next_action(self, action):
        self.state["planning"]["next_action"] = action

    def get_next_action(self):
        return self.state["planning"]["next_action"]

    # Reasoning methods
    def add_seven_note(self, text):
        self.state["reflection"]["seven_notes"] += text + "\n"

    def set_seven_notes(self, notes):
        self.state["reflection"]["seven_notes"] = notes

    def clear_seven_notes(self):
        self.state["reflection"]["seven_notes"] = ""

    def get_seven_notes(self):
        return self.state["reflection"]["seven_notes"]

    def add_conversation_summary(self, summary):
        self.summary += summary + "\n"

    def get_conversation_summary(self):
        return self.summary

    # Execution methods
    def set_active_tool(self, tool):
        self.state["execution"]["active_tool"] = tool

    def get_active_tool(self):
        return self.state["execution"]["active_tool"]

    def add_tool_output(self, tool_name, output):
        self.state["tool_outputs"][tool_name] = output

    def get_tool_outputs(self):
        return self.state["tool_outputs"]

    def get_tool_output(self, tool_name):
        return self.state["tool_outputs"].get(tool_name)

    # Context methods
    def add_retrieved_context(self, context):
        self.retrieved_memories.append(context)

    def get_retrieved_context(self):
        return self.retrieved_memories

    def clear_retrieved_context(self):
        self.retrieved_memories = []

    # Memory methods
    def add_memory_update(self, update):
        """Add a memory update with strict format validation.
        
        Required format:
        {
            "action": "add",              # Required: str (e.g., "add", "update", "delete")
            "memory_type": "goal",         # Required: str (e.g., "goal", "fact", "context")
            "key": "current_goal",         # Required: str (unique identifier)
            "value": "Build planner",      # Required: any type
            "priority": 0.95,              # Required: float/int (0-1)
            "confidence": 0.9,             # Required: float/int (0-1)
            # Optional fields:
            # "source": "LLM",
            # "tags": ["planning", "goal"]
        }
        """
        required_fields = {"action", "memory_type", "key", "value", "priority", "confidence"}
        
        if not isinstance(update, dict):
            raise TypeError(f"Memory update must be a dictionary, got {type(update).__name__}")
        
        provided_fields = set(update.keys())
        missing_fields = required_fields - provided_fields
        
        if missing_fields:
            raise ValueError(
                f"Memory update is missing required fields: {missing_fields}. "
                f"Required: action, memory_type, key, value, priority (0-1), confidence (0-1)"
            )
        
        if not isinstance(update["action"], str):
            raise TypeError(f"'action' must be string, got {type(update['action']).__name__}")
        if not isinstance(update["memory_type"], str):
            raise TypeError(f"'memory_type' must be string, got {type(update['memory_type']).__name__}")
        if not isinstance(update["key"], str):
            raise TypeError(f"'key' must be string, got {type(update['key']).__name__}")
        if not isinstance(update["priority"], (int, float)):
            raise TypeError(f"'priority' must be number, got {type(update['priority']).__name__}")
        if not isinstance(update["confidence"], (int, float)):
            raise TypeError(f"'confidence' must be number, got {type(update['confidence']).__name__}")
        if not (0 <= update["priority"] <= 1):
            raise ValueError(f"'priority' must be between 0 and 1, got {update['priority']}")
        if not (0 <= update["confidence"] <= 1):
            raise ValueError(f"'confidence' must be between 0 and 1, got {update['confidence']}")
        
        self.memory_updates.append(update)

    def get_memory_updates(self):
        return self.memory_updates

    def clear_memory_updates(self):
        self.memory_updates = []

    # Robustness methods
    def increment_retry_count(self):
        self.state["execution"]["retry_count"] += 1

    def get_retry_count(self):
        return self.state["execution"]["retry_count"]

    def reset_retry_count(self):
        self.state["execution"]["retry_count"] = 0

    def set_last_error(self, error):
        self.state["execution"]["last_error"] = error

    def get_last_error(self):
        return self.state["execution"]["last_error"]

    def clear_last_error(self):
        self.state["execution"]["last_error"] = None
    


scratchpad = Scratchpad()


def get_compiled_memory():
    return compiled_scratchpad_memory(
        # Planning
        current_goal=scratchpad.get_current_goal(),
        subtasks=scratchpad.get_subtasks(),
        completed_subtasks=scratchpad.get_completed_subtasks(),
        current_step=scratchpad.get_current_step(),
        next_action=scratchpad.get_next_action(),
        # Reasoning
        summary=scratchpad.get_conversation_summary(),
        seven_notes=scratchpad.get_seven_notes(),
        # Execution
        active_tool=scratchpad.get_active_tool(),
        tool_outputs=scratchpad.get_tool_outputs(),
        # Context
        retrieved_context=scratchpad.get_retrieved_context(),
        # Memory
        memory_updates=scratchpad.get_memory_updates(),
        # Robustness
        retry_count=scratchpad.get_retry_count(),
        last_error=scratchpad.get_last_error()
    )


def clear_scratchpad():
    """Clear all scratchpad data"""
    global scratchpad
    scratchpad = Scratchpad()