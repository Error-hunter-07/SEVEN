"""
LLMEngine/cli.py

The interactive REPL loop split out of llm_client.py so llm_client
can be imported as a pure library (by tests, or by a future non-CLI
frontend) without triggering an input() loop as a side effect of import.

Run with: python -m LLMEngine.cli
"""

from LLMEngine.llm_client import ask_llm, process_manager
from SessionManager.session_lifecycle import on_session_end


def run() -> None:
    while True:
        user_query = input("You: ")

        if user_query.strip().lower() == "/stop":
            on_session_end(process_manager.session_id)
            process_manager.stop_from_cli()
            break

        answer = ask_llm(user_query)

        print("\nAssistant:")
        print(answer)


if __name__ == "__main__":
    run()
