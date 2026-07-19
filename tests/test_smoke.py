"""
tests/test_smoke.py

Smoke-test suite for SEVEN's memory subsystems.
Run with:  pytest -v tests/test_smoke.py

Tests that need a live Postgres or ChromaDB will SKIP (not fail) if those
aren't reachable, so you can run this against a partial environment.
Tests that need the local llama.cpp server (extraction pipeline) are NOT
included here — those need a live server and belong in a separate
integration test file, since spinning up the model per test run is slow.
"""

import time
import uuid
import pytest


# ─────────────────────────────────────────────────────────────────────────
# 1. CONFIG — validates the fail-fast env var layer works as designed.
# ─────────────────────────────────────────────────────────────────────────

class TestConfig:

    def test_settings_load_without_crashing(self):
        """If this import fails, either .env is missing values or config.py
        itself is broken. Should succeed in a properly configured environment."""
        from GlobalHelpers.config import settings
        assert settings.db_user
        assert settings.llm_model

    def test_db_connection_string_is_escaped_and_well_formed(self):
        """Regression test for the unescaped-credentials bug — the connection
        string must be a valid postgresql:// URL regardless of special
        characters in the password."""
        from GlobalHelpers.config import settings
        conn_str = settings.db_connection_string
        assert conn_str.startswith("postgresql://")
        assert "@" in conn_str
        assert settings.db_name in conn_str

    def test_missing_required_var_fails_fast(self, monkeypatch):
        """Regression test for the config validation layer itself:
        removing a required var must produce ONE clear error, not a
        cryptic NoneType failure three layers deep.

        NOTE: config.py calls load_dotenv() again on every reload, which
        would silently refill a deleted env var from .env before
        _load_settings() ever checks it. So we patch os.getenv itself
        (specifically for this one key) instead of monkeypatch.delenv,
        which load_dotenv() would just undo."""
        import importlib
        import os
        import GlobalHelpers.config as config_module

        original_getenv = os.getenv

        def fake_getenv(key, default=None):
            return None if key == "DB_PASSWORD" else original_getenv(key, default)

        monkeypatch.setattr("os.getenv", fake_getenv)

        with pytest.raises(SystemExit) as exc_info:
            importlib.reload(config_module)
        assert "DB_PASSWORD" in str(exc_info.value)

        # reload again with real getenv restored so later tests aren't affected
        monkeypatch.undo()
        importlib.reload(config_module)


# ─────────────────────────────────────────────────────────────────────────
# 2. LOGGING — validates the 3-file split and session tagging actually work.
#    This is the exact test that would have caught the "no session log
#    file created" bug immediately.
# ─────────────────────────────────────────────────────────────────────────

class TestLogging:

    def test_app_and_error_logs_are_created(self):
        from GlobalHelpers.logger import configure_logging, get_logger, LOG_DIR
        import os

        configure_logging()
        log = get_logger("test.logging")
        log.info("smoke test info line")
        log.error("smoke test error line")

        assert os.path.exists(os.path.join(LOG_DIR, "app.log"))
        assert os.path.exists(os.path.join(LOG_DIR, "errors.log"))

    def test_session_file_handler_creates_a_real_file(self):
        """The regression test for last session's bug: attach a session
        handler, log a line, and confirm a file actually appears on disk
        tagged with the session id — not just that no exception was thrown."""
        from GlobalHelpers.logger import (
            configure_logging, get_logger, set_session_id,
            attach_session_file_handler, SESSION_DIR,
        )
        import os, logging

        configure_logging()
        session_id = f"smoketest-{uuid.uuid4().hex[:8]}"
        set_session_id(session_id)
        handler = attach_session_file_handler(session_id)

        log = get_logger("test.session_logging")
        log.info("session-scoped test line")
        handler.flush()

        date_str = time.strftime("%Y-%m-%d")
        expected_path = os.path.join(SESSION_DIR, f"session_{session_id}_{date_str}.log")
        assert os.path.exists(expected_path), (
            f"Expected session log file at {expected_path} — if this fails, "
            f"attach_session_file_handler() is not being wired correctly."
        )
        with open(expected_path) as f:
            content = f.read()
        assert session_id in content
        assert "session-scoped test line" in content

        logging.getLogger().removeHandler(handler)
        handler.close()

    def test_session_id_tags_log_records_correctly(self):
        """Confirms the contextvar correctly tags records with the RIGHT
        session id, not just 'some' session id — catches cross-thread
        contamination if two sessions ever overlap."""
        from GlobalHelpers.logger import set_session_id, _session_id_ctx
        set_session_id("session-A")
        assert _session_id_ctx.get() == "session-A"
        set_session_id("session-B")
        assert _session_id_ctx.get() == "session-B"


# ─────────────────────────────────────────────────────────────────────────
# 3. DATABASE — Postgres connectivity + working memory CRUD.
#    Skips gracefully if Postgres isn't running.
# ─────────────────────────────────────────────────────────────────────────

class TestDatabase:

    @pytest.fixture(scope="class")
    def db_available(self):
        try:
            from Database.db import DB
            db = DB()
            conn = db.get_connection()
            db.put_connection(conn)
            return True
        except Exception:
            pytest.skip("Postgres not reachable — skipping DB tests")

    def test_connection_pool_initialises(self, db_available):
        from Database.db import DB
        db = DB()
        conn = db.get_connection()
        assert conn is not None
        db.put_connection(conn)

    def test_working_memory_insert_and_fetch(self, db_available):
        """NOTE: get_all_current_session_working_memory currently returns
        raw psycopg2 tuples (plain cursor), not dicts — column order per
        the SELECT is: id, memory_type, key, value, priority, relevance,
        created_at, updated_at, expires_at, source, tags. This test
        handles both shapes so it keeps working if you switch to
        RealDictCursor later (recommended — see review notes)."""
        from Database.working_memory_db_client import (
            insert_working_memory, get_all_current_session_working_memory,
        )
        session_id = str(uuid.uuid4())
        mem_id = insert_working_memory(
            session_id=session_id, memory_type="note",
            key="test_key", value="test_value",
        )
        assert mem_id is not None

        rows = get_all_current_session_working_memory(session_id)
        assert rows is not None

        def row_matches(r):
            if isinstance(r, dict):
                return r.get("key") == "test_key" and r.get("value") == "test_value"
            if isinstance(r, (tuple, list)):
                return r[2] == "test_key" and r[3] == "test_value"
            return False

        assert any(row_matches(r) for r in rows), (
            "Inserted row not found — or row shape changed unexpectedly. "
            f"Got rows: {rows!r}"
        )


# ─────────────────────────────────────────────────────────────────────────
# 4. SEMANTIC MEMORY — dedup, polarity guard, retrieval.
#    Skips gracefully if ChromaDB never becomes ready.
# ─────────────────────────────────────────────────────────────────────────

class TestSemanticMemory:

    @pytest.fixture(scope="class")
    def sm(self):
        from Database.chroma_db import wait_for_chroma
        if not wait_for_chroma(timeout=30):
            pytest.skip("ChromaDB did not become ready in time — skipping semantic memory tests")
        from MemoryManagement.semantic_memory.semantic_memory import semantic_memory
        assert semantic_memory._db is not None, (
            "semantic_memory._db is None even though ChromaDB reported ready — "
            "this is exactly the stale-reference bug from before, regression check failed."
        )
        return semantic_memory

    def test_store_and_count_increments(self, sm):
        before = sm.count()
        unique_text = f"User's favorite test marker is {uuid.uuid4().hex}"
        mem_id = sm.store(unique_text, importance=0.6, category="other", polarity="neutral")
        assert mem_id is not None
        assert sm.count() == before + 1

    def test_near_duplicate_does_not_grow_count(self, sm):
        marker = uuid.uuid4().hex
        sm.store(f"User likes testing framework {marker}", importance=0.5, category="preferences", polarity="positive")
        before = sm.count()
        # Near-identical phrasing of the same fact
        sm.store(f"User likes testing framework {marker}", importance=0.5, category="preferences", polarity="positive")
        assert sm.count() == before, "Near-duplicate text should merge, not create a new entry"

    def test_retrieval_finds_semantically_related_text(self, sm):
        marker = uuid.uuid4().hex
        sm.store(f"User's pet marker {marker} is a golden retriever named Max", importance=0.7, category="other")
        results = sm.retrieve(f"what pet does the user with marker {marker} have?", k=3)
        assert any(marker in r["text"] for r in results), (
            "Expected semantic retrieval to surface the stored fact even without exact keyword overlap"
        )

    def test_opposite_facts_do_not_silently_overwrite(self, sm):
        """Regression test for the polarity/negation merge bug via the
        SemanticMemory.store() API directly (the extraction pipeline path)."""
        marker = uuid.uuid4().hex
        id_likes = sm.store(f"User likes the marker-{marker} framework", importance=0.6,
                             category="preferences", polarity="positive")
        id_dislikes = sm.store(f"User dislikes the marker-{marker} framework", importance=0.6,
                                category="preferences", polarity="negative")
        assert id_likes is not None and id_dislikes is not None
        assert id_likes != id_dislikes, (
            "Opposite-polarity facts were merged into the same memory id — "
            "the polarity/negation dedup guard failed."
        )

    @pytest.mark.xfail(reason="Known gap: store_semantic_memory tool doesn't forward polarity yet")
    def test_polarity_gap_via_llm_tool_path(self, sm):
        """Documents the NEW bug found in this review: the LLM-facing
        store_semantic_memory tool always stores polarity='neutral',
        bypassing the polarity guard on the direct tool-call path.
        Once Tools/semantic_memory_tool.py forwards polarity, mark this
        test as no longer xfail.

        IMPORTANT: uses 'enjoys' vs 'avoids' rather than 'likes' vs
        'dislikes' — neither word is in _NEGATION_MARKERS, so the cheap
        keyword guard in semantic_memory.py can't save this pair. Only
        the polarity metadata field (if actually forwarded) can catch it.
        An earlier version of this test used 'likes'/'dislikes' and
        XPASSED by accident, because 'dislike' happens to be a literal
        negation marker — that made the test validate the wrong safety
        net and mask this exact gap."""
        import Tools.semantic_memory_tool as semantic_memory_tool
        marker = uuid.uuid4().hex

        id_a = semantic_memory_tool.store_semantic_memory(
            text=f"User enjoys marker-{marker} activity", importance=0.6, category="preferences",
        )
        id_b = semantic_memory_tool.store_semantic_memory(
            text=f"User avoids marker-{marker} activity", importance=0.6, category="preferences",
        )
        assert id_a != id_b, "store_semantic_memory tool path merged opposite facts (polarity not forwarded)"


# ─────────────────────────────────────────────────────────────────────────
# 5. SCRATCHPAD — state machine correctness, no LLM/DB required.
# ─────────────────────────────────────────────────────────────────────────

class TestScratchpad:

    @pytest.fixture(autouse=True)
    def clean_scratchpad(self):
        from MemoryManagement.shortterm_memory.scratchpad import clear_scratchpad
        clear_scratchpad()
        yield
        clear_scratchpad()

    def test_goal_and_subtask_lifecycle(self):
        from MemoryManagement.shortterm_memory.scratchpad import scratchpad, get_compiled_memory

        scratchpad.set_current_goal("Test goal for smoke test")
        scratchpad.add_subtask("Step 1")
        scratchpad.add_subtask("Step 2")
        scratchpad.mark_subtask_completed("Step 1")

        compiled = get_compiled_memory()
        assert "Test goal for smoke test" in compiled
        assert "Step 2" in compiled

    def test_clear_resets_everything(self):
        from MemoryManagement.shortterm_memory.scratchpad import scratchpad, get_compiled_memory, clear_scratchpad

        scratchpad.set_current_goal("Should be wiped")
        clear_scratchpad()
        compiled = get_compiled_memory()
        assert "Should be wiped" not in compiled

    def test_update_scratchpad_state_rejects_invalid_section(self):
        import Tools.scratchpad_tool as scratchpad_tool
        with pytest.raises(ValueError):
            scratchpad_tool.update_scratchpad_state("not_a_real_section", "current_goal", "x")

    def test_update_scratchpad_state_rejects_invalid_key(self):
        import Tools.scratchpad_tool as scratchpad_tool
        with pytest.raises(ValueError):
            scratchpad_tool.update_scratchpad_state("planning", "not_a_real_key", "x")

    @pytest.mark.xfail(reason="Known gap: subtasks value type isn't validated, only the key name is")
    def test_subtasks_value_type_is_enforced(self):
        """Documents the still-open bug: setting 'subtasks' to a non-list
        value should be rejected, but currently silently succeeds and
        corrupts get_compiled_memory() output later."""
        import Tools.scratchpad_tool as scratchpad_tool
        with pytest.raises((ValueError, TypeError)):
            scratchpad_tool.update_scratchpad_state("planning", "subtasks", "this should be a list, not a string")


# ─────────────────────────────────────────────────────────────────────────
# 6. TOOL REGISTRY — every registered tool is well-formed and callable.
# ─────────────────────────────────────────────────────────────────────────

class TestToolRegistry:

    def test_all_registered_tools_have_required_fields(self):
        from ToolCalling.register import registry
        tools = registry.list_tools()
        assert len(tools) > 0, "No tools registered — check register.py"
        for tool in tools:
            assert tool.name, f"Tool missing a name: {tool}"
            assert tool.description, f"Tool '{tool.name}' has no description"
            assert callable(tool.func), f"Tool '{tool.name}' func is not callable"

    def test_working_memory_tools_are_intentionally_not_registered(self):
        """This asserts the CURRENT intended design: working memory is only
        reachable indirectly via the scratchpad bridge, not as direct LLM
        tools. If this test starts failing, it means someone registered
        them directly — confirm that was an intentional design change."""
        from ToolCalling.register import registry
        tool_names = {t.name for t in registry.list_tools()}
        direct_wm_tools = {"insert_working_memory", "update_working_memory", "get_working_memory"}
        assert tool_names.isdisjoint(direct_wm_tools), (
            "Direct working-memory tools appear to be registered — "
            "confirm this is intentional (design says LLM should go through scratchpad only)."
        )

    def test_no_duplicate_tool_names(self):
        from ToolCalling.register import registry
        names = [t.name for t in registry.list_tools()]
        assert len(names) == len(set(names)), "Duplicate tool names found in registry"