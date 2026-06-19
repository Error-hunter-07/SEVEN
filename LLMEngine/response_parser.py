import re

def parse_response(message: str) -> str:
    """
    Convert LLM markdown-like output into terminal-friendly formatting.
    Returns formatted string.
    """

    message = re.sub(
        r"<tool_call>.*?</tool_call>",
        "",
        message,
        flags=re.DOTALL,
    ).strip()

    # Bold: **text**
    message = re.sub(
        r"\*\*(.*?)\*\*",
        r"\033[1m\1\033[0m",
        message
    )

    # Italic: *text*
    message = re.sub(
        r"(?<!\*)\*(.*?)\*(?!\*)",
        r"\033[3m\1\033[0m",
        message
    )

    # Headers (#, ##, ###)
    message = re.sub(
        r"^### (.+)$",
        r"\n\033[1m→ \1\033[0m",
        message,
        flags=re.MULTILINE
    )

    message = re.sub(
        r"^## (.+)$",
        r"\n\033[1m▶ \1\033[0m",
        message,
        flags=re.MULTILINE
    )

    message = re.sub(
        r"^# (.+)$",
        r"\n\033[1m◆ \1\033[0m",
        message,
        flags=re.MULTILINE
    )

    # Convert markdown bullet points
    message = re.sub(
        r"^- ",
        "• ",
        message,
        flags=re.MULTILINE
    )

    # Inline code: `code`
    message = re.sub(
        r"`(.*?)`",
        r"\033[96m\1\033[0m",
        message
    )

    # Triple backtick code blocks
    def code_block(match):
        code = match.group(1)
        return f"\n\033[90m{'-'*40}\n{code}\n{'-'*40}\033[0m"

    message = re.sub(
        r"```(.*?)```",
        code_block,
        message,
        flags=re.DOTALL
    )

    return message