import json
import Database.working_memory_db_client as working_memory_db_client
import Runtime.process_manager as process_manager
import Tools.scratchpad_tool as scratchpad_tool


def insert_working_memory(memory_type, key, value, priority=0.5, relevance=0.5, source=None, tags=None):
    result = working_memory_db_client.insert_working_memory(
        process_manager.ProcessManager.get_instance().get_session_id(),
        memory_type,
        key,
        value,
        priority,
        relevance,
        source,
        tags
    )

    scratchpad_tool.add_scratchpad_tool_output(
        "insert_working_memory",
        f"Inserted working memory key='{key}' type='{memory_type}' result={result}"
    )
    return result


def get_working_memory():
    curr_working_memory = working_memory_db_client.get_working_memory()
    scratchpad_tool.add_scratchpad_tool_output(
        "get_working_memory",
        "Latest working memory row retrieved successfully."
    )
    return curr_working_memory


def get_all_working_memory_current_session():
    all_working_memory = working_memory_db_client.get_all_current_session_working_memory(
        process_manager.ProcessManager.get_instance().get_session_id()
    )

    scratchpad_tool.add_scratchpad_tool_output(
        "get_all_working_memory",
        "Retrieved all working memory for current session."
    )
    return all_working_memory


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

    scratchpad_tool.add_scratchpad_tool_output(
        "update_working_memory",
        f"Updated working memory with ID: {memory_id}"
    )

def add_scratchpad_memory_update(type, data, update=False):
    data_parsed = json.loads(data)

    if type == "working_memory" and not update:
        insert_working_memory(
            memory_type=data_parsed.get("memory_type"),
            key=data_parsed.get("key"),
            value=data_parsed.get("value"),
            priority=data_parsed.get("priority", 0.5),
            relevance=data_parsed.get("relevance", 0.5),
            source=data_parsed.get("source"),
            tags=data_parsed.get("tags")
        )
    elif type == "working_memory" and update:
        update_working_memory(
            memory_id=data_parsed.get("memory_id"),
            key=data_parsed.get("key"),
            value=data_parsed.get("value"),
            priority=data_parsed.get("priority"),
            relevance=data_parsed.get("relevance"),
            expires_at=data_parsed.get("expires_at"),
            source=data_parsed.get("source"),
            tags=data_parsed.get("tags")
        )

    import MemoryManagement.shortterm_memory.scratchpad as scratchpad_module
    scratchpad_module.scratchpad.add_memory_update(type, data, update)
    scratchpad_tool.add_scratchpad_tool_output(
        "add_scratchpad_memory_update",
        f"Memory {'update' if update else 'insert'} completed for type='{type}'"
    )

# def delete_working_memory(memory_id):
#     working_memory_db_client.delete_working_memory(memory_id)
#     scratchpad_tool.add_scratchpad_tool_output(f"Deleted working memory with ID: {memory_id}")