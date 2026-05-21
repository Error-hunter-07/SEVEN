def trim_context(messages):

    # Keep system prompt safe
    system_message = messages[0]

    # Conversation messages only
    conversation = messages[1:]

    # If limit exceeded
    if len(conversation) > 15:

        # Drop oldest 10 messages
        conversation = conversation[10:]

        # Rebuild messages list
        messages.clear()

        messages.append(system_message)

        messages.extend(conversation)
    
