import pytest

from tools.multihop_budget import allocate_multihop_budget
from tools.query_decomposition import build_decomposition_plan


def test_global_budget_preserves_minima_and_exact_accounting():
    plan = build_decomposition_plan("Compare E5 and BGE-M3 for retrieval.")
    budget = allocate_multihop_budget(
        plan,
        top_k=5,
        total_limit=300,
        per_hop_limit=150,
    )
    assert budget.allocated_cost + budget.unallocated_cost == 300
    assert budget.allocated_cost == sum(
        item.max_estimated_cost for item in budget.budgets
    )
    assert all(item.max_estimated_cost >= item.minimum_cost for item in budget.budgets)
    assert all(item.max_estimated_cost <= 150 for item in budget.budgets)
    assert set(budget.by_id()) == {"q1", "q2", "q3"}


def test_dependent_comparison_receives_at_least_as_much_as_simple_leaves():
    plan = build_decomposition_plan("Compare E5 and BGE-M3 for retrieval.")
    budget = allocate_multihop_budget(plan, total_limit=600, per_hop_limit=300)
    by_id = budget.by_id()
    assert by_id["q3"].weight > by_id["q1"].weight
    assert by_id["q3"].max_estimated_cost >= by_id["q1"].max_estimated_cost


def test_impossible_global_or_per_hop_limits_fail_before_retrieval():
    plan = build_decomposition_plan("Compare E5 and BGE-M3 for retrieval.")
    with pytest.raises(ValueError, match="minimum required"):
        allocate_multihop_budget(plan, total_limit=2, per_hop_limit=500)
    with pytest.raises(ValueError, match="initial retrieval"):
        allocate_multihop_budget(plan, total_limit=1000, per_hop_limit=1)


def test_excess_total_is_reported_as_unallocated_after_caps():
    plan = build_decomposition_plan("Question")
    budget = allocate_multihop_budget(plan, total_limit=1000, per_hop_limit=100)
    assert budget.allocated_cost == 100
    assert budget.unallocated_cost == 900


def test_boolean_limits_fail_closed():
    plan = build_decomposition_plan("Question")
    with pytest.raises(ValueError, match="total_limit"):
        allocate_multihop_budget(plan, total_limit=True)
