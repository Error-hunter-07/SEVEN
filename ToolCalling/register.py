from Tools.tool import Tool
import Tools.scratchpad_tool as scratchpad_tool

import Tools.working_memory_tool as working_memory_tool

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
        name="insert_working_memory",
        description="Inserts a new piece of working memory for the current session.",
        parameters={
            "memory_type": "The type/category of the memory.",
            "key": "A key or title for the memory.",
            "value": "The content of the memory.",
            "priority": "A number between 0 and 1 indicating the priority of the memory (default is 0.5).",
            "relevance": "A number between 0 and 1 indicating the relevance of the memory (default is 0.5).",
            "expires_at": "An optional timestamp indicating when the memory expires.",
            "source": "An optional string indicating the source of the memory.",
            "tags": "An optional list of tags associated with the memory."
        },
        func=working_memory_tool.insert_working_memory
    )
)

registry.register(
    Tool(
        name="get_working_memory",
        description="Retrieves a piece of working memory by its ID.",
        parameters={"memory_id": "The ID of the memory to retrieve."},
        func=working_memory_tool.get_working_memory
    )
)

registry.register(
    Tool(
        name="get_all_working_memory_current_session",
        description="Retrieves all working memory for the current session.",
        parameters={},
        func=working_memory_tool.get_all_working_memory_current_session
    )
)

registry.register(
    Tool(
        name="delete_working_memory",
        description="Deletes a piece of working memory by its ID.",
        parameters={"memory_id": "The ID of the memory to delete."},
        func=working_memory_tool.delete_working_memory
    )
)

registry.register(
    Tool(
        name="update_working_memory",
        description="Updates a piece of working memory by its ID.",
        parameters={
            "memory_id": "The ID of the memory to update.",
            "key": "An optional new key or title for the memory.",
            "value": "An optional new content for the memory.",
            "priority": "An optional new priority for the memory (number between 0 and 1).",
            "relevance": "An optional new relevance for the memory (number between 0 and 1).",
            "expires_at": "An optional new expiration timestamp for the memory.",
            "source": "An optional new source string for the memory.",
            "tags": "An optional new list of tags for the memory."
        },
        func=working_memory_tool.update_working_memory
    )
)


