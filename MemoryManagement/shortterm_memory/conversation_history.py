# from .summarizer import summarize

# conversation_summary = ""


# def trim_context(messages):

#     global conversation_summary

#     if not messages:
#         return

#     if messages[0].get("role") == "system":
#         system_message = messages[0]
#         conversation = messages[1:]
#     else:
#         system_message = None
#         conversation = messages

#     if len(conversation) > 15:

#         # Take oldest messages
#         old_messages = conversation[:10]

#         # Convert to text
#         history_text = ""

#         for msg in old_messages:
#             history_text += (
#                 f"{msg['role']}: "
#                 f"{msg['content']}\n"
#             )

#         # Summarize
#         new_summary = summarize(
#             history_text,
#             conversation_summary
#         )

#         conversation_summary = new_summary

#         # Keep recent messages
#         conversation = conversation[10:]

#         messages.clear()

#         if system_message is not None:
#             messages.append(system_message)

#         messages.extend(conversation)