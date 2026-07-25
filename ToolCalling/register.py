from Tools.tool import Tool
import Tools.scratchpad_tool as scratchpad_tool
import Tools.working_memory_tool as working_memory_tool
import Tools.semantic_memory_tool as semantic_memory_tool
import Tools.episodic_memory_tool as episodic_memory_tool

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
                "Inserts or updates a working memory entry. "
                "Default to update=False (insert) for anything new. "
                "Only set update=True if you already hold a real memory_id returned "
                "from a previous get_working_memory or get_all_working_memory call — "
                "never guess or invent a memory_id."
            ),
        parameters={
            "memory_type": "str - always pass 'working_memory'",
            "key": "str - the key name for this memory entry",
            "value": "str - the content to store",
            "priority": "float - 0.0 to 1.0, default 0.5",
            "relevance": "float - 0.0 to 1.0, default 0.5",
            "source": "str - optional, where this memory comes from",
            "tags": "str - optional, comma-separated tags",
            "memory_id": "str - only required when update=True",
            "update": "bool - False to insert, True to update"
        },
        func=working_memory_tool.add_scratchpad_memory_update_flat
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


# ── Semantic memory tools ────────────────────────────────────────────────────
# LLM accesses long-term memory only through these bridge tools.
# It never touches ChromaDB or the embedding model directly.

registry.register(
    Tool(
        name="store_semantic_memory",
        description=(
            "Save an important long-term memory fact about the user. "
            "Use this when the user shares something worth remembering across sessions: "
            "their background, goals, skills, preferences, or experiences. "
            "Write the memory as a single self-contained sentence about the user. "
            "Do NOT store one-off commands, greetings, or temporary task context."
        ),
        parameters={
            "text":       "str - A self-contained fact sentence about the user.",
            "importance": "float - 0.0 to 1.0. Use 0.8+ for identity/goals, 0.5 for general info.",
            "category":   "str - One of: identity, education, interests, goals, preferences, experience, relationships, other.",
            "polarity":   "str - One of: positive (user likes/wants), negative (user dislikes/avoids), neutral (factual)."
        },
        func=semantic_memory_tool.store_semantic_memory
    )
)


registry.register(
    Tool(
        name="search_semantic_memory",
        description=(
            "Search long-term memory for a SINGLE STANDING FACT about the user — "
            "their background, skills, preferences, or a specific piece of information. "
            "Use this for fact-lookup questions (\"what's my budget\", \"what do I do for work\"). "
            "Do NOT use this to recall what happened in a past conversation, what was decided, "
            "or what alternatives were considered — use search_episodic_memory for that instead. "
            "Returns the most relevant stored facts."
        ),
        parameters={
            "query": "str - The topic or question to search memory for. E.g. 'user programming skills'.",
            "k":     "int - Number of memories to return (default 5, max 10)."
        },
        func=semantic_memory_tool.search_semantic_memory
    )
)


# ── Episodic memory tool ─────────────────────────────────────────────────────
# LLM accesses episodic memory only through this bridge tool. It never
# touches the episodic Chroma collection directly. A deterministic
# trigger (LLMEngine/episodic_trigger.py) also calls the same underlying
# search-and-compile logic automatically for obvious recall-shaped
# phrasing — this tool is the fallback for everything that trigger
# doesn't catch (e.g. a natural question that doesn't use "last time"/
# "we discussed" phrasing but still needs episodic recall).

registry.register(
    Tool(
        name="search_episodic_memory",
        description=(
            "Recall what happened in a PAST CONVERSATION — a process, decision, sequence of "
            "events, or the reasoning behind a choice. Use this when the user asks what was "
            "discussed, what was decided and why, what alternatives were considered, or wants "
            "you to continue/recall a specific past session (e.g. \"what design were we "
            "considering for the website\", \"what did we decide about the trip budget\"). "
            "Do NOT use this for a single standing fact about the user (e.g. their name, a "
            "preference, a number) — use search_semantic_memory for that instead. "
            "Returns matched session summaries along with any specific facts tied to them."
        ),
        parameters={
            "query": "str - What to recall, e.g. 'website design for Rema's tutorial'.",
            "k":     "int - Max number of past sessions to return (default 3)."
        },
        func=episodic_memory_tool.search_episodic_memory
    )
)


# Direct DB tools remain commented out — LLM accesses memory only through
# the scratchpad bridge. Procedural memory will follow the same pattern:
# its own bridge tool registered here, no direct DB tool exposed to the LLM.

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