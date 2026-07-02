# SessionManager/session_lifecycle.py

from MemoryManagement.shortterm_memory.scratchpad import scratchpad
from MemoryManagement.semantic_memory.semantic_memory import semantic_memory

def on_session_end(session_id: str) -> None:
    """
    Called when user types /stop or the session exits cleanly.
    Promotes important scratchpad state into semantic memory before wiping.
    """
    state = scratchpad.get_all()  # get the full scratchpad dict
    
    # 1. Promote current goal if it exists
    goal = state.get("planning", {}).get("current_goal")
    if goal and len(goal) > 10:
        semantic_memory.store(
            text=f"User was working on: {goal}",
            importance=0.75,
            category="goals",
            source="session_end"
        )
    
    # 2. Promote completed subtasks as experience
    subtasks = state.get("planning", {}).get("subtasks", [])
    completed = [t for t in subtasks if t.get("status") == "done"]
    for task in completed[:3]:  # cap at 3 to avoid bloat
        semantic_memory.store(
            text=f"User completed task: {task.get('description', '')}",
            importance=0.6,
            category="experience",
            source="session_end"
        )
    
    # 3. Promote any memory_updates the LLM explicitly set
    memory_updates = state.get("memory_updates", [])
    for update in memory_updates:
        if isinstance(update, dict) and update.get("text"):
            semantic_memory.store(
                text=update["text"],
                importance=update.get("importance", 0.65),
                category=update.get("category", "other"),
                source="session_end"
            )
    
    # 4. Promote last error if it exists (useful for debugging context next session)
    last_error = state.get("execution", {}).get("last_error")
    if last_error:
        semantic_memory.store(
            text=f"User encountered an issue last session: {last_error[:200]}",
            importance=0.5,
            category="experience",
            source="session_end"
        )
    
    print(f"[SessionLifecycle] Session {session_id} — scratchpad promoted to semantic memory.")



def on_session_start(session_id: str, user_query: str = "") -> None:
    """
    Called at the beginning of every session.
    Seeds the scratchpad with relevant semantic context so the LLM
    doesn't start cold.
    """
    # Pull relevant memories for context seeding
    if user_query:
        context = semantic_memory.retrieve_as_text(query=user_query, k=3)
    else:
        # No query yet — pull goals and recent experience
        goals = semantic_memory.retrieve_as_text(
            query="what is the user working on", k=2, category="goals"
        )
        experience = semantic_memory.retrieve_as_text(
            query="recent user activity", k=2, category="experience"
        )
        context = "\n".join(filter(None, [goals, experience]))

    if context:
        scratchpad.set("session_context", {"semantic_seed": context})

    print(f"[SessionLifecycle] Session {session_id} started with semantic context.")