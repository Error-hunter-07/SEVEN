class Scratchpad:
    def __init__(self):
        self.seven_notes = ""
        self.summary = ""
        self.current_goal = ""
        self.subtasks = []
        self.retrieved_context = []
        self.tool_outputs = []
        self.memory_updates = []
        self.response_drafts = []


    def add_seven_note(self, text):
        self.seven_notes += text + "\n"

    def clear_seven_notes(self):
        self.seven_notes = ""

    def get_seven_notes(self):
        return self.seven_notes
    
    def add_conversation_summary(self, summary):
        self.summary += summary + "\n"
    
    def get_conversation_summary(self):
        return self.summary
    
    def set_current_goal(self, goal):
        self.current_goal = goal

    def get_current_goal(self):
        return self.current_goal
    
    def add_subtask(self, subtask):
        self.subtasks.append(subtask)

    def get_subtasks(self):
        return self.subtasks
    
    def add_retrieved_context(self, context):
        self.retrieved_context.append(context)

    def get_retrieved_context(self):
        return self.retrieved_context
    
    def add_tool_output(self, output):
        self.tool_outputs.append(output)
    
    def get_tool_outputs(self):
        return self.tool_outputs
    
    def add_memory_update(self, update):
        self.memory_updates.append(update)
    
    def get_memory_updates(self):
        return self.memory_updates
    
    def add_response_draft(self, draft):
        self.response_drafts.append(draft)

    def get_response_drafts(self):
        return self.response_drafts
    
    
