
import GlobalHelpers.token_counter as token_counter
import requests
import os


# def summarize(history_text, conversation_summary):
#     if token_counter.count_tokens(history_text + conversation_summary) < 10000:
#         return conversation_summary + "\n" + history_text
#     # Call LLM to summarize

#     prompt = f"""Summarize the following conversation between a user and an assistant. 
#         Update summary.
#         Keep:
#         - goals
#         - completed tasks
#         - unresolved issues
#         - preferences"""
#     prompt += "\n\nConversation History:\n" + history_text
#     prompt += "\n\nExisting Summary:\n" + conversation_summary

#     response = requests.post(
#         "http://localhost:11434/api/chat",
#         json={
#             "model": os.getenv("LLM_MODEL"),
#             "messages": [
#                 {
#                     "role": "system",
#                     "content": prompt
#                 }
#             ],
#             "stream": False
#         },
#         timeout=30,
#     )
#     response.raise_for_status()
#     data = response.json()
#     new_summary = data["message"]["content"]
#     return new_summary


def compiled_scratchpad_memory(
    current_goal, subtasks, completed_subtasks, current_step, next_action,
    summary, seven_notes,
    active_tool, tool_outputs,
    memory_updates,
    retry_count, last_error
):
    compiled = "SEVEN'S SHORT-TERM MEMORY\n\n"

    # Reasoning
    compiled += "SEVEN NOTES:\n" + seven_notes + "\n\n"
    compiled += "CONVERSATION SUMMARY:\n" + summary + "\n\n"

    # Planning
    compiled += "CURRENT GOAL:\n" + current_goal + "\n\n"
    compiled += "SUBTASKS:\n" + "\n".join(subtasks) if subtasks else "SUBTASKS:\nNone\n\n"
    compiled += "\n\nCOMPLETED SUBTASKS:\n" + "\n".join(completed_subtasks) if completed_subtasks else "COMPLETED SUBTASKS:\nNone\n\n"
    compiled += "\n\nCURRENT STEP:\n" + current_step + "\n\n"
    compiled += "NEXT ACTION:\n" + next_action + "\n\n"

    # Execution
    compiled += "ACTIVE TOOL:\n" + active_tool + "\n\n"
    if tool_outputs:
        compiled += "TOOL OUTPUTS:\n"
        for tool_name, output in tool_outputs.items():
            compiled += f"  {tool_name}: {output}\n"
        compiled += "\n"
    else:
        compiled += "TOOL OUTPUTS:\nNone\n\n"

    # Memory
    if memory_updates:
        compiled += "MEMORY UPDATES:\n"
        for update in memory_updates:
            compiled += f"  {update}\n"
        compiled += "\n"
    else:
        compiled += "MEMORY UPDATES:\nNone\n\n"

    # Robustness
    compiled += "RETRY COUNT:\n" + str(retry_count) + "\n\n"
    compiled += "LAST ERROR:\n" + (str(last_error) if last_error else "None") + "\n\n"

    return compiled
