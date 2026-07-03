# SessionManager/session_lifecycle.py

from MemoryManagement.shortterm_memory.scratchpad import scratchpad
from MemoryManagement.semantic_memory.semantic_memory import semantic_memory
from GlobalHelpers.logger import get_logger, set_session_id, attach_session_file_handler

log = get_logger(__name__)

def on_session_end(session_id: str) -> None:
    """
    Called when user types /stop or the session exits cleanly.
    Promotes important scratchpad state into semantic memory before wiping.
    """

    # 1. Promote current goal if it exists
    goal = scratchpad.get_current_goal()
    if goal and len(goal) > 10:
        semantic_memory.store(
            text=f"User was working on: {goal}",
            importance=0.75,
            category="goals",
            source="session_end"
        )
    
    # 2. Promote completed subtasks as experience
    for task in scratchpad.get_completed_subtasks()[:3]:
        if task and len(str(task)) > 5:
            semantic_memory.store(
                text=f"User completed task: {task}",
                importance=0.6,
                category="experience",
                source="session_end"
            )
    
    # 3. Promote any memory_updates the LLM explicitly set
    for update in scratchpad.get_memory_updates():
        # format is "type<insert>:data" or "type<update>:data"
        if isinstance(update, str) and ":" in update:
            _, data = update.split(":", 1)
            if data.strip():
                semantic_memory.store(
                    text=data.strip(),
                    importance=0.65,
                    category="other",
                    source="session_end"
                )
    
    # 4. Promote last error if it exists (useful for debugging context next session)
    last_error = scratchpad.get_last_error()
    if last_error:
        semantic_memory.store(
            text=f"User encountered an issue last session: {str(last_error)[:200]}",
            importance=0.5,
            category="experience",
            source="session_end"
        )

    scratchpad.reset()
    log.info("Session %s ended — scratchpad promoted and cleared.", session_id)




def on_session_start(session_id: str, user_query: str = "") -> None:
    """
    Called at the beginning of every session.
    Seeds the scratchpad with relevant semantic context so the LLM
    doesn't start cold.
    """
     # Pull goals and recent experience to seed context
    goals = semantic_memory.retrieve_as_text(
        query="what is the user working on", k=2, category="goals"
    )
    experience = semantic_memory.retrieve_as_text(
        query="recent user activity", k=2, category="experience"
    )
    context = "\n".join(filter(None, [goals, experience]))

    # Set the active session id for log lines and create a per-session file
    set_session_id(session_id)
    try:
        attach_session_file_handler(session_id)
    except Exception:
        log.exception("Failed to attach session file handler for %s", session_id)

    if context:
        # Store as a retrieved memory so it appears in the scratchpad
        scratchpad.add_retrieved_context(context)
    log.info("Session %s started with semantic context.", session_id)