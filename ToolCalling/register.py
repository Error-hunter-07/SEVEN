from Tools.tool import Tool
import Tools.scratchpad_tool as scratchpad_tool

class ToolRegistry:

    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def register(self, tool: Tool):
        self.register_tool(tool)

    def get_tool(self, name):
        return self.tools.get(name)
    
    def list_tools(self):
        return list(self.tools.values())
    
registry = ToolRegistry()

registry.register(
    Tool(
        name="update_scratchpad_summary",
        description="Updates the scratchpad summary with the latest conversation summary.",
        parameters={},
        func=scratchpad_tool.update_scratchpad_summary
    )
)

registry.register(
    Tool(
        name="update_scratchpad_seven_notes",
        description="Updates the seven notes section of the scratchpad.",
        parameters={"notes": "The new seven notes content."},
        func=scratchpad_tool.update_scratchpad_seven_notes
    )
)

registry.register(
    Tool(
        name="update_scratchpad_current_goal",      
        description="Updates the current goal section of the scratchpad.",
        parameters={"goal": "The new current goal."},
        func=scratchpad_tool.update_scratchpad_current_goal
    )
)

registry.register(
    Tool(
        name="add_scratchpad_subtask",
        description="Adds a new subtask to the scratchpad.",
        parameters={"subtask": "The subtask to add."},
        func=scratchpad_tool.add_scratchpad_subtask
    )
)

registry.register(
    Tool(
        name="add_scratchpad_retrieved_context",
        description="Adds retrieved context to the scratchpad.",
        parameters={"context": "The context to add."},
        func=scratchpad_tool.add_scratchpad_retrieved_context
    )
)   

registry.register(
    Tool(
        name="add_scratchpad_tool_output",
        description="Adds tool output to the scratchpad.",
        parameters={"output": "The tool output to add."},
        func=scratchpad_tool.add_scratchpad_tool_output
    )
)

registry.register(
    Tool(
        name="add_scratchpad_memory_update",
        description="Adds a memory update to the scratchpad.",
        parameters={"update": "The memory update to add."},
        func=scratchpad_tool.add_scratchpad_memory_update
    )
)

registry.register(
    Tool(       
        name="add_scratchpad_response_draft",
        description="Adds a response draft to the scratchpad.",
        parameters={"draft": "The response draft to add."},
        func=scratchpad_tool.add_scratchpad_response_draft
    )
)