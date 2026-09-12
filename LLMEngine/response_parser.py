import re

# Mirrors ToolCalling/parser.py's tolerant tag matcher — see that file for
# why: some local models emit a pipe-delimited variant of the tool-call
# wrapper instead of the canonical <tool_call>...</tool_call>. Whether or
# not the payload inside parses as valid JSON and actually executes, the
# wrapper itself should never be shown to the user or stored anywhere.
_TOOL_CALL_TAG = re.compile(
    r"<\|?/?tool_call\|?>.*?<\|?/?tool_call\|?>",
    re.DOTALL,
)


def strip_tool_call_tags(message: str) -> str:
    """
    Returns the CANONICAL, storage-safe text: tool-call wrapper tags
    removed, otherwise byte-for-byte what the model said.

    This — NOT format_for_display() below — is what must be used for
    anything that gets persisted or re-fed into another LLM call:
    conversation history (history_manager), the rolling chunk-summary
    accumulator (_current_chunk_turns / chunk_summary_worker), the
    full_conversation crash backup, and extraction_worker's queued
    turns. All of those get read back later — either as prompt context
    for the next turn, or as literal input text to another model
    (the background chunk/episode summarizer, the entity extractor).

    FIX (root cause of degraded chunk-summary / knowledge-graph
    quality): this function used to be format_for_display() itself —
    i.e. ANSI terminal escape codes (\\033[1m, \\033[3m, \\033[96m, ...)
    were baked into the *stored* text, not just the printed text. That
    meant every downstream consumer — future chat turns, the chunk
    summarizer, the episode summarizer, and ultimately the entity
    extractor that treats chunk summaries as its "primary extraction
    signal" — was reading control-character-riddled text. A small
    background model fed literal ESC-bracket-digit-m sequences
    mid-sentence produces incoherent/garbled summaries, which then
    propagate straight into the knowledge graph. Terminal formatting
    must only be applied at the point of printing (see
    format_for_display()), never to what gets saved or reused as
    context.
    """
    if not message:
        return message
    return _TOOL_CALL_TAG.sub("", message).strip()


def format_for_display(message: str) -> str:
    """
    Convert clean, already tag-stripped LLM output into terminal-friendly
    ANSI formatting for printing. Call this ONLY at the point where text
    is about to be printed to the terminal (see LLMEngine/llm_client.py's
    ask_llm() return value, printed by LLMEngine/cli.py). Never pass the
    result of this function to history_manager, chunk_summary_worker,
    extraction_worker, or any DB/ChromaDB write — those must all use the
    plain string from strip_tool_call_tags() instead.
    """
    if not message:
        return message

    # FIX: code blocks must be extracted BEFORE the inline single-backtick
    # regex runs. Previously the inline-code regex ran first and, being
    # non-greedy, matched the two backticks of a ```lang fence as if they
    # were inline code, mangling every triple-backtick code block before
    # the code-block regex ever got a chance to match it. Pulling code
    # blocks out first (and stashing them) avoids that entirely.
    code_blocks: list[str] = []

    def _stash_code_block(match: "re.Match[str]") -> str:
        code = match.group(1)
        code_blocks.append(f"\n\033[90m{'-' * 40}\n{code}\n{'-' * 40}\033[0m")
        return f"\x00CODEBLOCK{len(code_blocks) - 1}\x00"

    message = re.sub(r"```(.*?)```", _stash_code_block, message, flags=re.DOTALL)

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

    # Inline code: `code` (safe now — real code-block backticks were
    # already pulled out above)
    message = re.sub(
        r"`(.*?)`",
        r"\033[96m\1\033[0m",
        message
    )

    # Restore stashed code blocks
    for i, block in enumerate(code_blocks):
        message = message.replace(f"\x00CODEBLOCK{i}\x00", block)

    return message


# Backwards-compat alias: anything importing the old name gets the
# storage-safe behaviour, not the ANSI one — this is deliberate. If code
# elsewhere still calls parse_response() expecting terminal formatting,
# it needs to be updated to call format_for_display() explicitly instead;
# silently keeping the old name pointed at the ANSI formatter is exactly
# the trap that caused this bug in the first place.
def parse_response(message: str) -> str:
    return strip_tool_call_tags(message)
