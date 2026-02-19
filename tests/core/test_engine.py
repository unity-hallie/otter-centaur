"""
Property-based and unit tests for the Otter engine.

Core invariants:
    - Items never re-appear in set_of_support after being moved to usable
    - set_of_support shrinks by exactly one per step (plus new items added)
    - to_dict() -> from_dict() is a round-trip identity
    - run_otter with a stop_fn halts when the condition is met
"""

import pytest
import json
from hypothesis import given, settings
from hypothesis import strategies as st

from otter.core.state import Item, OtterState
from otter.core.engine import otter_step, run_otter


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_item(name: str) -> Item:
    return Item(name=name, content=f"content of {name}")


def null_combine(x, y) -> list:
    """Never produces anything. Used to test loop mechanics."""
    return []


def always_combine(x, y) -> list:
    """Always produces one new item: x+y."""
    new_name = f"{x.name}+{y.name}"
    return [Item(name=new_name, content="combined")]


def make_simple_state(*names) -> OtterState:
    state = OtterState()
    for name in names:
        state.set_of_support.append(make_item(name))
    return state


# ── Unit tests ───────────────────────────────────────────────────────────────

class TestOtterStep:
    def test_empty_sos_halts(self):
        state = OtterState()
        state = otter_step(state, null_combine, verbose=False)
        assert state.halted
        assert "empty" in state.halt_reason

    def test_focus_moves_to_usable(self):
        state = make_simple_state("a", "b")
        state = otter_step(state, null_combine, verbose=False)
        assert len(state.usable) == 1
        assert state.usable[0].name == "a"  # FIFO

    def test_sos_shrinks_by_one_when_no_new_items(self):
        state = make_simple_state("a", "b", "c")
        initial_sos = len(state.set_of_support)
        state = otter_step(state, null_combine, verbose=False)
        # One moved to usable, no new items
        assert len(state.set_of_support) == initial_sos - 1

    def test_new_items_added_to_sos(self):
        state = make_simple_state("a", "b")
        # First step: focus=a, usable=[], no combinations possible yet
        state = otter_step(state, always_combine, verbose=False)
        # a is now in usable; b is still in sos; a+b not yet produced
        # Second step: focus=b, usable=[a], produces a+b
        state = otter_step(state, always_combine, verbose=False)
        names = {item.name for item in state.set_of_support} | {item.name for item in state.usable}
        assert "b+a" in names or "a+b" in names  # order may vary

    def test_duplicates_not_added(self):
        state = make_simple_state("a", "b", "c")
        # Run two steps so a+b gets produced once
        state = otter_step(state, always_combine, verbose=False)  # focus=a, usable=[]
        state = otter_step(state, always_combine, verbose=False)  # focus=b, produces b+a
        all_names = [i.name for i in state.set_of_support] + [i.name for i in state.usable]
        # The combined item should appear at most once
        combined = [n for n in all_names if "+" in n]
        assert len(combined) == len(set(combined))

    def test_step_counter_increments(self):
        state = make_simple_state("a")
        assert state.step == 0
        state = otter_step(state, null_combine, verbose=False)
        assert state.step == 1

    def test_history_recorded(self):
        state = make_simple_state("a", "b")
        state = otter_step(state, null_combine, verbose=False)
        assert len(state.history) == 1
        assert state.history[0]["focus"] == "a"


class TestRunOtter:
    def test_runs_max_steps(self):
        state = make_simple_state("a", "b", "c", "d", "e")
        state = run_otter(state, null_combine, max_steps=3, verbose=False)
        assert state.step == 3

    def test_halts_when_sos_empty(self):
        state = make_simple_state("a")
        state = run_otter(state, null_combine, max_steps=100, verbose=False)
        assert state.halted
        assert state.step == 1  # only one item to focus on

    def test_stop_fn_halts_early(self):
        state = make_simple_state("a", "b", "c", "d", "e")
        # Stop after 2 steps
        call_count = {"n": 0}
        def stop_after_2(s):
            call_count["n"] += 1
            return call_count["n"] >= 2
        state = run_otter(state, null_combine, max_steps=100,
                          stop_fn=stop_after_2, verbose=False)
        assert state.step <= 3  # stopped early


class TestStateSerialization:
    def test_item_round_trip(self):
        state = make_simple_state("a", "b", "c")
        state.usable.append(make_item("x"))
        d = state.to_dict()
        restored = OtterState.from_dict(d)
        assert list(i.name for i in restored.set_of_support) == \
               list(i.name for i in state.set_of_support)
        assert [i.name for i in restored.usable] == [i.name for i in state.usable]

    def test_json_round_trip(self, tmp_path):
        state = make_simple_state("alpha", "beta")
        path = str(tmp_path / "test_state.json")
        state.save(path)
        restored = OtterState.load(path)
        assert list(i.name for i in restored.set_of_support) == \
               list(i.name for i in state.set_of_support)

    def test_step_and_halted_preserved(self):
        state = make_simple_state("a", "b")
        state = run_otter(state, null_combine, max_steps=1, verbose=False)
        d = state.to_dict()
        restored = OtterState.from_dict(d)
        assert restored.step == state.step
        assert restored.halted == state.halted


# ── Property-based tests ─────────────────────────────────────────────────────

@st.composite
def simple_states(draw, min_items=1, max_items=5):
    n = draw(st.integers(min_value=min_items, max_value=max_items))
    names = draw(st.lists(
        st.text(alphabet="abcdefghij", min_size=1, max_size=3),
        min_size=n, max_size=n, unique=True,
    ))
    return make_simple_state(*names)


class TestEngineProperties:

    @given(simple_states(min_items=1, max_items=6))
    def test_focus_never_stays_in_sos(self, state):
        """Once an item is focused, it never stays in set_of_support."""
        initial_names = {item.name for item in state.set_of_support}
        state = otter_step(state, null_combine, verbose=False)
        if state.usable:
            focus_name = state.usable[-1].name
            sos_names = {item.name for item in state.set_of_support}
            assert focus_name not in sos_names

    @given(simple_states(min_items=1, max_items=6))
    def test_total_items_conserved_with_null_combine(self, state):
        """With null combine, total item count is conserved across a step."""
        total_before = len(state.set_of_support) + len(state.usable)
        state = otter_step(state, null_combine, verbose=False)
        total_after = len(state.set_of_support) + len(state.usable)
        assert total_after == total_before

    @given(simple_states(min_items=2, max_items=6))
    def test_serialization_round_trip(self, state):
        """to_dict -> from_dict is identity on item names."""
        d = state.to_dict()
        restored = OtterState.from_dict(d)
        assert list(i.name for i in restored.set_of_support) == \
               list(i.name for i in state.set_of_support)
        assert [i.name for i in restored.usable] == \
               [i.name for i in state.usable]

    @given(simple_states(min_items=1, max_items=5),
           st.integers(min_value=1, max_value=10))
    def test_steps_bounded_by_max(self, state, max_steps):
        """run_otter never exceeds max_steps."""
        state = run_otter(state, null_combine, max_steps=max_steps, verbose=False)
        assert state.step <= max_steps
