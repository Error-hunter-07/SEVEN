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
        description="Appends text to the scratchpad conversation summary.",
        parameters={"summary_text": "str - The summary text to append."},
        func=scratchpad_tool.update_scratchpad_summary
    )
)


registry.register(
    Tool(
        name="update_scratchpad_state",
        description=(
            "Updates a single key inside the scratchpad state. "
            "Use this to track goals, steps, notes, errors, and active tools. "
            "Valid sections and their keys — "
            "planning: current_goal, subtasks, completed_subtasks, current_step, next_action. "
            "execution: active_tool, retry_count, last_error. "
            "reflection: seven_notes. "
            "tool_outputs: any tool name as key."
        ),
        parameters={
            "section": "str - one of: 'planning', 'execution', 'reflection', 'tool_outputs'",
            "key": "str - the key within the section (e.g. 'current_goal', 'last_error')",
            "value": "any - the value to set"
        },
        func=scratchpad_tool.update_scratchpad_state
    )
)


registry.register(
    Tool(
        name="get_scratchpad_state",
        description="Returns the full current scratchpad state including planning, execution, reflection, and tool_outputs sections.",
        parameters={},
        func=scratchpad_tool.get_scratchpad_state
    )
)


registry.register(
    Tool(
        name="add_scratchpad_memory_update",
        description=(
            "Inserts or updates a working memory entry via the scratchpad bridge. "
            "Currently supports memory_type='working_memory' only. "
            "Set update=False to insert a new record, update=True to update by memory_id."
        ),
        parameters={
            "type": "str - memory category, currently only 'working_memory' is supported",
            "data": (
                "str - JSON-encoded object. "
                "For insert (update=False): {memory_type, key, value, priority (0-1), relevance (0-1), source (optional), tags (optional list)}. "
                "For update (update=True): {memory_id, key (optional), value (optional), priority (optional), relevance (optional), expires_at (optional), source (optional), tags (optional)}"
            ),
            "update": "bool - False to insert a new record, True to update an existing record by memory_id"
        },
        func=working_memory_tool.add_scratchpad_memory_update
    )
)


registry.register(
    Tool(
        name="get_scratchpad_retrieved_context",
        description="Fetches memory context into the scratchpad. Use to pull working memory before reasoning over it.",
        parameters={
            "working_memory_only": "bool - True to fetch the latest working memory row",
            "include_tool_outputs": "bool - True to include the current tool outputs in context",
            "include_all_working_memory": "bool - True to fetch all working memory for the current session"
        },
        func=scratchpad_tool.get_scratchpad_retrieved_context
    )
)


# Direct DB tools remain commented out — LLM accesses memory only through
# the scratchpad bridge. Future memory types (episodic, semantic, procedural)
# will follow the same pattern: their own bridge tool registered here,
# no direct DB tool exposed to the LLM.

# registry.register(Tool(name="insert_working_memory", ...))
# registry.register(Tool(name="update_working_memory", ...))
# registry.register(Tool(name="get_working_memory", ...))
# registry.register(Tool(name="get_all_working_memory_current_session", ...))
# registry.register(Tool(name="delete_working_memory", ...))
# registry.register(Tool(name="add_scratchpad_retrieved_context", ...))

# registry.register(
#     Tool(
#         name="insert_working_memory",
#         description="Inserts a new piece of working memory for the current session.",
#         parameters={
#             "memory_type": "The type/category of the memory.",
#             "key": "A key or title for the memory.",
#             "value": "The content of the memory.",
#             "priority": "A number between 0 and 1 indicating the priority of the memory (default is 0.5).",
#             "relevance": "A number between 0 and 1 indicating the relevance of the memory (default is 0.5).",
#             "expires_at": "An optional timestamp indicating when the memory expires.",
#             "source": "An optional string indicating the source of the memory.",
#             "tags": "An optional list of tags associated with the memory."
#         },
#         func=working_memory_tool.insert_working_memory
#     )
# )

# registry.register(
#     Tool(
#         name="update_working_memory",
#         description="Updates a piece of working memory by its ID.",
#         parameters={
#             "memory_id": "The ID of the memory to update.",
#             "key": "An optional new key or title for the memory.",
#             "value": "An optional new content for the memory.",
#             "priority": "An optional new priority for the memory (number between 0 and 1).",
#             "relevance": "An optional new relevance for the memory (number between 0 and 1).",
#             "expires_at": "An optional new expiration timestamp for the memory.",
#             "source": "An optional new source string for the memory.",
#             "tags": "An optional new list of tags for the memory."
#         },
#         func=working_memory_tool.update_working_memory
#     )
# )



# registry.register(
#     Tool(
#         name="get_working_memory",
#         description="Retrieves a piece of working memory by its ID.",
#         parameters={"memory_id": "The ID of the memory to retrieve."},
#         func=working_memory_tool.get_working_memory
#     )
# )

# registry.register(
#     Tool(
#         name="get_all_working_memory_current_session",
#         description="Retrieves all working memory for the current session.",
#         parameters={},
#         func=working_memory_tool.get_all_working_memory_current_session
#     )
# )

# registry.register(
#     Tool(
#         name="add_scratchpad_retrieved_context",
#         description="Adds retrieved memory or context to the scratchpad for later reference.",
#         parameters={"context": "The context or memory to add."},
#         func=scratchpad_tool.add_scratchpad_retrieved_context
#     )
# )

# registry.register(
#     Tool(
#         name="delete_working_memory",
#         description="Deletes a piece of working memory by its ID.",
#         parameters={"memory_id": "The ID of the memory to delete."},
#         func=working_memory_tool.delete_working_memory
#     )
# )

