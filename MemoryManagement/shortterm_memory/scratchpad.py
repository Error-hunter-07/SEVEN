import json
from unittest import case

from .summarizer import compiled_scratchpad_memory
import Tools.working_memory_tool as working_memory_tool

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

    def get_retrieved_context(self, working_memory_only, include_tool_outputs, include_all_working_memory):
        retrieved_context = ""
        if(working_memory_only):
            retrieved_context += working_memory_tool.get_working_memory()
        
        if(include_tool_outputs):
            retrieved_context += self.get_tool_outputs()

        if(include_all_working_memory):
            retrieved_context += working_memory_tool.get_all_working_memory_current_session()

        

    # def add_working_memory_to_scratchpad(self, memory):
    #     self.retrieved_memories.append(memory)

    def clear_retrieved_context(self):
        self.retrieved_memories = []

    # Memory methods
    def add_memory_update(self, type, data, update):
        data_parsed = json.loads(data)
        if type == "working_memory" and not update:
            return working_memory_tool.insert_working_memory(
                memory_type=data_parsed.get("memory_type"),
                key=data_parsed.get("key"),
                value=data_parsed.get("value"),
                priority=data_parsed.get("priority", 0.5),
                relevance=data_parsed.get("relevance", 0.5),
                source=data_parsed.get("source"),
                tags=data_parsed.get("tags")
            )
        elif type == "working_memory" and update:
            return working_memory_tool.update_working_memory(
                memory_id=data_parsed.get("memory_id"),
                key=data_parsed.get("key"),
                value=data_parsed.get("value"),
                priority=data_parsed.get("priority"),
                relevance=data_parsed.get("relevance"),
                expires_at=data_parsed.get("expires_at"),
                source=data_parsed.get("source"),
                tags=data_parsed.get("tags")
            )
        if(update):
            self.memory_updates.append(type+"<update>" +  ":" + data)
        else:            
            self.memory_updates.append(type+"<insert>" + ":" + data)

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
    global scratchpad
    scratchpad = Scratchpad()