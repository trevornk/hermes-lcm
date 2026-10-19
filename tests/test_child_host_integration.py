"""Offline integration: real host gate and compressor, no provider calls."""
from unittest.mock import Mock

import pytest
from hermes_lcm.engine import LCMEngine

from agent.context_compressor import (
    ContextCompressor, DETERMINISTIC_SUMMARY_ROUTE, pin_summary_route,
)
from agent.turn_preflight import _request_pressure_compaction_eligible
from hermes_lcm.tokens import count_messages_tokens
from .test_child_compaction_budget import ChildAgentFrame, prototype


def test_real_host_gate_and_compressor(prototype: LCMEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    child = ChildAgentFrame(prototype)
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Turn {i}: " + "historical work detail " * 2000}
        for i in range(40)
    ]
    forbidden_network = Mock(side_effect=AssertionError("Integration must not call an LLM"))
    monkeypatch.setattr(ContextCompressor, "_generate_summary", forbidden_network)

    class ActiveFrame:
        session_id = "child"
        _parent_session_id = "parent"
        log_prefix = "[subagent-0]"

        def exercise(self) -> None:
            engine = child.engine
            assert _request_pressure_compaction_eligible(engine, messages, 468_000)
            assert engine.should_compress_preflight(messages)
            before = count_messages_tokens(messages)
            with pin_summary_route(DETERMINISTIC_SUMMARY_ROUTE):
                output = engine.compress(messages, current_tokens=468_000)
            compressor = engine._host_fallback_compressor
            assert isinstance(compressor, ContextCompressor)
            assert engine.context_length == 1_000_000
            assert engine.threshold_tokens == 220_000
            assert output != messages
            assert count_messages_tokens(output) < before
            assert engine.compression_count > 0
            assert engine._store.get_session_count("child") == 0
            forbidden_network.assert_not_called()
            print(f"Real host gate passed; ContextCompressor reduced {before} to {count_messages_tokens(output)} tokens; compression_count={engine.compression_count}; child LCM rows=0")
    try:
        ActiveFrame().exercise()
    finally:
        child.engine.shutdown()
