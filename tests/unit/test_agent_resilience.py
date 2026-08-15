import pytest

from tools.agent_resilience import (
    AgentCostLedger,
    AgentFailureKind,
    AgentTask,
    BoundedAgentRunner,
    Pricing,
    RetryPolicy,
    UsageRecord,
    classify_agent_failure,
    execute_with_retry,
)


class HttpFailure(RuntimeError):
    def __init__(self, status_code):
        super().__init__(f"http {status_code}")
        self.status_code = status_code


def test_failure_classifier_is_provider_neutral_and_fail_closed():
    assert classify_agent_failure(TimeoutError()) == AgentFailureKind.TIMEOUT
    assert classify_agent_failure(ConnectionError()) == AgentFailureKind.TRANSIENT
    assert classify_agent_failure(HttpFailure(429)) == AgentFailureKind.RATE_LIMITED
    assert classify_agent_failure(HttpFailure(503)) == AgentFailureKind.TRANSIENT
    assert classify_agent_failure(HttpFailure(400)) == AgentFailureKind.PERMANENT
    assert classify_agent_failure(ValueError("bad input")) == AgentFailureKind.PERMANENT


def test_retry_is_bounded_and_backoff_is_deterministic_with_injected_sleep():
    calls = []
    sleeps = []

    def operation():
        calls.append(len(calls) + 1)
        if len(calls) < 3:
            raise ConnectionError("temporary")
        return "ok"

    result = execute_with_retry(
        operation,
        policy=RetryPolicy(max_attempts=4, base_delay_seconds=0.5, max_delay_seconds=2, multiplier=2),
        sleeper=sleeps.append,
    )

    assert result.succeeded is True
    assert result.value == "ok"
    assert len(result.attempts) == 3
    assert sleeps == [0.5, 1.0]


def test_permanent_failure_is_not_retried():
    result = execute_with_retry(lambda: (_ for _ in ()).throw(ValueError("invalid")))
    assert result.succeeded is False
    assert len(result.attempts) == 1
    assert result.attempts[0].failure_kind == AgentFailureKind.PERMANENT


def test_runner_preserves_partial_success_and_failure_metadata():
    runner = BoundedAgentRunner(max_workers=2, max_pending=4)
    try:
        outcomes = runner.run(
            [
                AgentTask("good", lambda: 7),
                AgentTask("bad", lambda: (_ for _ in ()).throw(ValueError("invalid"))),
            ],
            retry_policy=RetryPolicy(max_attempts=2, base_delay_seconds=0, max_delay_seconds=0),
            timeout_seconds=2,
        )
    finally:
        runner.shutdown()

    by_id = {item.task_id: item for item in outcomes}
    assert by_id["good"].status == "succeeded"
    assert by_id["good"].value == 7
    assert by_id["bad"].status == "failed"
    assert by_id["bad"].failure_kind == AgentFailureKind.PERMANENT
    assert by_id["bad"].error_type == "ValueError"


def test_runner_rejects_duplicate_task_ids():
    runner = BoundedAgentRunner(max_workers=1, max_pending=1)
    try:
        with pytest.raises(ValueError, match="duplicate task_id"):
            runner.run([AgentTask("x", lambda: 1), AgentTask("x", lambda: 2)])
    finally:
        runner.shutdown()


def test_cost_ledger_never_invents_prices():
    ledger = AgentCostLedger()
    ledger.add(UsageRecord("planner", input_tokens=1000, output_tokens=500, requests=2, latency_seconds=0.4))
    summary = ledger.summarize()
    assert summary["total_cost"] == 0.0
    assert summary["agents"]["planner"]["cost"] == 0.0


def test_cost_ledger_uses_only_caller_supplied_prices():
    ledger = AgentCostLedger(
        pricing={"planner": Pricing(input_per_million=2.0, output_per_million=4.0, per_request=0.01)}
    )
    ledger.add(UsageRecord("planner", input_tokens=1_000_000, output_tokens=500_000, requests=2, latency_seconds=0.4))
    summary = ledger.summarize()
    assert summary["total_cost"] == pytest.approx(4.02)
    assert summary["coordination_latency_seconds"] == pytest.approx(0.4)


def test_retry_policy_rejects_unsafe_limits():
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(base_delay_seconds=2, max_delay_seconds=1)
