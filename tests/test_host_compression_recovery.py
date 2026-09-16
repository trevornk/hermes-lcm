"""Cross-boundary recovery contracts for Hermes-LCM compaction."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import hermes_lcm.escalation as escalation
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_tokens


def test_escalation_uses_one_timeout_budget_across_l1_l2_and_l3(monkeypatch):
    """A slow unusable L1 consumes the caller's only budget; L2 must not start
    another full provider request after that deadline, and L3 stays local."""
    clock = [10.0]
    calls = []

    def unusable_call(_prompt, _max_tokens, model="", timeout=None):
        calls.append((model, timeout))
        clock[0] += float(timeout)
        return None

    monkeypatch.setattr(escalation.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(escalation, "_invoke_summary_llm", unusable_call)

    source = "durable raw history " * 600
    summary, level = escalation.summarize_with_escalation(
        text=source,
        source_tokens=count_tokens(source),
        token_budget=100,
        timeout=5.0,
        l3_truncate_tokens=40,
    )

    assert level == 3
    assert calls == [("", 5.0)]
    assert count_tokens(summary) <= 40


def test_host_rejection_cools_lcm_without_dropping_raw_history(tmp_path):
    """When the core declines a non-reducing candidate, LCM must temporarily
    stop requesting the same boundary while its immutable raw store remains
    available for a later useful recovery."""
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
    )
    engine = LCMEngine(config=config)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "durable historical request"},
        {"role": "assistant", "content": "durable historical reply"},
        {"role": "user", "content": "protected recent request"},
    ]
    try:
        engine.on_session_start("host-rejection", platform="cli", context_length=100)
        engine.threshold_tokens = 1
        engine.ingest(messages)
        raw_count = engine._store.get_session_count("host-rejection")

        engine.record_rejected_compaction()

        assert engine.should_compress_preflight(messages) is False
        assert engine._store.get_session_count("host-rejection") == raw_count
        assert engine.last_compression_was_noop is True
        assert "host rejected" in engine.last_compression_noop_reason
    finally:
        engine.shutdown()


def _prepare_host_import(monkeypatch) -> None:
    """Give Hermes' ``tools`` package precedence without leaking test state."""
    spec = importlib.util.find_spec("agent")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("Hermes host source is not installed for this plugin integration test")
    core_root = Path(next(iter(spec.submodule_search_locations))).parent
    monkeypatch.syspath_prepend(str(core_root))
    # pytest adds this plugin root to sys.path, where tools.py would otherwise
    # shadow Hermes' tools package. monkeypatch restores the exact prior module.
    monkeypatch.delitem(sys.modules, "tools", raising=False)


def _host_agent_with_lcm(db, engine, session_id):
    """Build the real host boundary without network/model setup."""
    try:
        from hermes_state import SessionDB
        from run_agent import AIAgent
    except ImportError:
        pytest.skip("Hermes host source is not installed for this plugin integration test")
    assert isinstance(db, SessionDB)
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "system"
    agent._use_prompt_caching = False
    agent._compression_feasibility_checked = True
    agent.compression_enabled = True
    agent.compression_in_place = True
    agent.context_compressor = engine
    return agent


def test_host_idle_deadline_bounds_real_lcm_aux_chain_and_commits_l3(tmp_path, monkeypatch):
    """The host's idle budget, not LCM's 300s configured provider timeout,
    bounds both silent LLM levels so the deterministic fallback returns before
    the fence and the real host can commit useful compacted context."""
    _prepare_host_import(monkeypatch)
    from agent.conversation_compression import (
        compress_context,
        run_compress_context_with_progress_timeout,
    )
    from hermes_state import SessionDB

    session_id = "host-idle-lcm"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "durable history " * 40},
        {"role": "assistant", "content": "durable reply " * 40},
        {"role": "user", "content": "protected recent request"},
    ]
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id, source="cli")
    for message in messages:
        db.append_message(session_id, message["role"], message["content"])
    engine = LCMEngine(LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
        summary_timeout_ms=300_000,
        l3_truncate_tokens=24,
    ))
    agent = _host_agent_with_lcm(db, engine, session_id)
    timeouts = []

    def silent_auxiliary(_prompt, _max_tokens, model="", timeout=None):
        del model
        timeouts.append(timeout)
        time.sleep(0.02)
        return None

    monkeypatch.setattr(escalation, "_call_llm_for_summary", silent_auxiliary)
    try:
        engine.on_session_start(session_id, platform="cli", context_length=100)
        engine.threshold_tokens = 1
        engine.ingest(messages)
        raw_count = engine._store.get_session_count(session_id)
        compacted, _prompt = run_compress_context_with_progress_timeout(
            worker=lambda fence: compress_context(
                agent,
                messages,
                "system",
                approx_tokens=1_000,
                commit_fence=fence,
            ),
            messages=messages,
            system_prompt_fallback="system",
            idle_timeout_seconds=0.5,
            total_ceiling_seconds=5.0,
            stall_fallback=False,
            telemetry_agent=agent,
        )

        assert 1 <= len(timeouts) <= 2
        assert all(timeout is not None and timeout < 0.5 for timeout in timeouts), timeouts
        assert compacted != messages
        assert engine.compression_count == 1
        nodes = engine._dag.get_session_nodes(session_id)
        assert len(nodes) == 1
        assert "deterministic truncation" in nodes[0].summary
        assert engine._store.get_session_count(session_id) >= raw_count
    finally:
        engine.shutdown()


def test_full_host_rejection_preserves_preexisting_rows_and_cools_real_lcm(tmp_path, monkeypatch):
    """A real LCM assembly that cannot reduce the host-visible budget is rejected
    by _compress_context: existing host rows and protected context survive, raw
    LCM rows remain available, and the next automatic preflight does not retry."""
    _prepare_host_import(monkeypatch)
    from hermes_state import SessionDB

    session_id = "host-lcm-rejection"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "durable historical request " * 8},
        {"role": "assistant", "content": "durable historical reply " * 8},
        {"role": "user", "content": "protected recent request"},
    ]
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id, source="cli")
    for message in messages:
        db.append_message(session_id, message["role"], message["content"])
    before_rows = db.get_messages(session_id)
    engine = LCMEngine(LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
    ))
    agent = _host_agent_with_lcm(db, engine, session_id)

    def nonreducing_summary(**kwargs):
        return kwargs["text"], 1

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", nonreducing_summary)
    try:
        engine.on_session_start(session_id, platform="cli", context_length=100)
        engine.threshold_tokens = 1
        engine.ingest(messages)
        raw_count = engine._store.get_session_count(session_id)

        compacted, _prompt = agent._compress_context(messages, "system", approx_tokens=1_000)

        assert compacted == messages
        assert compacted[-1]["content"] == "protected recent request"
        assert db.get_messages(session_id) == before_rows
        assert engine._store.get_session_count(session_id) == raw_count
        assert engine.should_compress_preflight(messages) is False
    finally:
        engine.shutdown()
