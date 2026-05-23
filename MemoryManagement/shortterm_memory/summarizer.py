import GlobalHelpers.token_counter as token_counter

def summarize(history_text, conversation_summary):
    if token_counter.count_tokens(history_text + conversation_summary) < 10000:
        return conversation_summary + "\n" + history_text
    # Call LLM to summarize

    prompt = f"""Summarize the following conversation between a user and an assistant. 
        Update summary.
        Keep:
        - goals
        - completed tasks
        - unresolved issues
        - preferences"""
    prompt += "\n\nConversation History:\n" + history_text
    prompt += "\n\nExisting Summary:\n" + conversation_summary

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": os.getenv("LLM_MODEL"),
            "messages": [
                {
                    "role": "system",
                    "content": prompt
                }
            ],
            "stream": False
        },
    )
    response.raise_for_status()
    data = response.json()
    new_summary = data["message"]["content"]
    return new_summary


def compiled_scratchpad_memory(seven_notes, conversation_summary, current_goal, subtasks, retrieved_context, tool_outputs, memory_updates, response_drafts):
    compiled = "SEVEN'S SHORT-TERM MEMORY\n\n"
    compiled += "SEVEN NOTES:\n" + seven_notes + "\n\n"
    compiled += "CONVERSATION SUMMARY:\n" + conversation_summary + "\n\n"
    compiled += "CURRENT GOAL:\n" + current_goal + "\n\n"
    compiled += "SUBTASKS:\n" + "\n".join(subtasks) + "\n\n"
    compiled += "RETRIEVED CONTEXT:\n" + "\n".join(retrieved_context) + "\n\n"
    compiled += "TOOL OUTPUTS:\n" + "\n".join(tool_outputs) + "\n\n"
    compiled += "MEMORY UPDATES:\n" + "\n".join(memory_updates) + "\n\n"
    compiled += "RESPONSE DRAFTS:\n" + "\n".join(response_drafts) + "\n\n"

    return compiled