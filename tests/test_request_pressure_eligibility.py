"""LCM request-pressure eligibility is side-effect-free and tail-aware."""

import importlib

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


def _tail_heavy_messages():
    return [
        {"role": "user", "content": "old backlog"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "fresh " + "x" * 500},
        {"role": "assistant", "content": "fresh " + "y" * 500},
        {"role": "user", "content": "fresh " + "z" * 500},
    ]


def _engine(tmp_path, **config_overrides):
    defaults = {
        "database_path": str(tmp_path / "request-pressure.db"),
        "fresh_tail_count": 4,
        "leaf_chunk_tokens": 100,
    }
    defaults.update(config_overrides)
    config = LCMConfig(**defaults)
    engine = LCMEngine(config=config)
    engine._session_id = "request-pressure"
    engine.context_length = 1_000
    engine.threshold_tokens = 100
    return engine


def test_request_pressure_defers_when_only_protected_tail_exceeds_threshold(tmp_path):
    engine = _engine(tmp_path)
    messages = _tail_heavy_messages()
    try:
        assert count_messages_tokens(messages) >= engine.threshold_tokens
        assert engine.should_compress_request_pressure(messages, engine.threshold_tokens) is False
        assert engine._store.get_session_count(engine._session_id) == 0
    finally:
        engine.shutdown()


def test_request_pressure_preserves_forced_overflow_recovery(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    messages = _tail_heavy_messages()
    try:
        monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: True)
        assert engine.should_compress_request_pressure(messages, engine.threshold_tokens) is True
    finally:
        engine.shutdown()


def test_request_pressure_forced_overflow_bypasses_boundary_cooldown(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    messages = _tail_heavy_messages()
    try:
        monkeypatch.setattr(engine, "_compression_boundary_cooldown_active", lambda: True)
        monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: True)
        assert engine.should_compress_request_pressure(messages, engine.threshold_tokens) is True
        assert engine._store.get_session_count(engine._session_id) == 0
    finally:
        engine.shutdown()


def test_request_pressure_eligible_raw_backlog_is_true_without_ingestion(tmp_path):
    engine = _engine(tmp_path)
    messages = [
        {"role": "user", "content": "raw backlog " + "a" * 1_000},
        {"role": "assistant", "content": "raw answer " + "b" * 1_000},
        *_tail_heavy_messages()[1:],
    ]
    try:
        assert engine.should_compress_request_pressure(messages, engine.threshold_tokens) is True
        assert engine._store.get_session_count(engine._session_id) == 0
    finally:
        engine.shutdown()


def test_request_pressure_ignored_backlog_is_true_without_ingestion(tmp_path):
    engine = _engine(tmp_path, ignore_message_patterns=["^IGNORE"])
    messages = [
        {"role": "user", "content": "IGNORE maintenance marker"},
        *_tail_heavy_messages()[1:],
    ]
    try:
        assert engine.should_compress_request_pressure(messages, engine.threshold_tokens) is True
        assert engine._store.get_session_count(engine._session_id) == 0
    finally:
        engine.shutdown()


def test_request_pressure_debt_is_true_without_lifecycle_mutation(tmp_path):
    engine = _engine(
        tmp_path,
        deferred_maintenance_enabled=True,
        leaf_chunk_tokens=10_000,
        critical_budget_pressure_ratio=0.7,
    )
    messages = _tail_heavy_messages()
    try:
        engine._conversation_id = engine._session_id
        engine._lifecycle.bind_session(engine._session_id, conversation_id=engine._conversation_id)
        engine._lifecycle.record_debt(engine._conversation_id, kind="raw_backlog", size_estimate=1)
        before = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert engine._has_raw_backlog_debt()
        assert engine._critical_budget_pressure_reached(observed_tokens=800, messages=messages)

        assert engine.should_compress_request_pressure(messages, 800) is True

        after = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert after == before
        assert engine._store.get_session_count(engine._session_id) == 0
    finally:
        engine.shutdown()


def test_deterministic_eligible_compaction_shrinks_payload_and_retains_canary(tmp_path, monkeypatch):
    """A real eligible LCM pass stores raw history and returns a smaller payload."""
    engine = _engine(tmp_path, fresh_tail_count=2, leaf_chunk_tokens=50)
    messages = [
        {"role": "user", "content": "CANARY-ALPHA original request " + "a" * 1_000},
        {"role": "assistant", "content": "CANARY-ALPHA original response " + "b" * 1_000},
        {"role": "user", "content": "fresh question"},
        {"role": "assistant", "content": "fresh answer"},
    ]
    try:
        engine._conversation_id = engine._session_id
        engine._lifecycle.bind_session(engine._session_id, conversation_id=engine._conversation_id)
        engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            engine_module,
            "summarize_with_escalation",
            lambda **_kwargs: ("Summary retains CANARY-ALPHA for retrieval.", 1),
        )

        before = count_messages_tokens(messages)
        compacted = engine.compress(messages, current_tokens=before)

        assert count_messages_tokens(compacted) < before
        assert engine._store.get_session_count(engine._session_id) == len(messages)
        rows = engine._store.search("CANARY-ALPHA", session_id=engine._session_id)
        assert any("CANARY-ALPHA" in str(row.get("content", "")) for row in rows)
    finally:
        engine.shutdown()
