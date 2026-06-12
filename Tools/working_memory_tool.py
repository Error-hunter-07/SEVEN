import Database.working_memory_db_client as working_memory_db_client
import Runtime.process_manager as process_manager
import Tools.scratchpad_tool as scratchpad_tool

def insert_working_memory(memory_type, key, value, priority=0.5, relevance=0.5, expires_at=None, source=None, tags=None):
    return working_memory_db_client.insert_working_memory(
        process_manager.ProcessManager.get_instance().get_session_id(),
        memory_type,
        key,
        value,
        priority,
        relevance,
        expires_at,
        source,
        tags
    )

def get_working_memory(memory_id):
    curr_working_memory = working_memory_db_client.get_working_memory(memory_id)
    scratchpad_tool.add_scratchpad_retrieved_context(f"Retrieved working memory: {curr_working_memory}")
    scratchpad_tool.add_scratchpad_tool_output(f"Retrieved working memory with ID {memory_id}")

def get_all_working_memory_current_session():
    all_working_memory = working_memory_db_client.get_all_current_session_working_memory(
        process_manager.ProcessManager.get_instance().get_session_id()
    )
    scratchpad_tool.add_scratchpad_retrieved_context(f"Retrieved all working memory for current session: {all_working_memory}")
    scratchpad_tool.add_scratchpad_tool_output(f"Retrieved all working memory for current session")

def delete_working_memory(memory_id):
    working_memory_db_client.delete_working_memory(memory_id)
    scratchpad_tool.add_scratchpad_tool_output(f"Deleted working memory with ID: {memory_id}")

def update_working_memory(memory_id, key=None, value=None, priority=None, relevance=None, expires_at=None, source=None, tags=None):
    working_memory_db_client.update_working_memory(
        memory_id,
        key,
        value,
        priority,
        relevance,
        expires_at,
        source,
        tags
    )
    scratchpad_tool.add_scratchpad_tool_output(f"Updated working memory with ID: {memory_id}")
