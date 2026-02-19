"""
The Otter main loop.

Based on Johnicholas Hines' formulation: pick a focus from set_of_support,
combine it with everything in usable, add new results back to set_of_support.
The combination function is entirely pluggable -- this loop knows nothing
about what the items mean.
"""

from typing import Callable, Optional
from .state import OtterState


def otter_step(
    state: OtterState,
    combine_fn: Callable,
    choose_focus_fn: Optional[Callable] = None,
    subsumes_fn: Optional[Callable] = None,
    prune_fn: Optional[Callable] = None,
    max_new_items: int = 50,
    verbose: bool = True,
) -> OtterState:
    """
    Execute one step of the Otter main loop.

    One step = pick a focus, try combining it with everything in usable,
    move focus to usable, add new results to set_of_support.

    Args:
        state:            current OtterState
        combine_fn:       combine_fn(x, y) -> list[Item|Edge|Clause]
                          The pluggable heart of the whole thing.
        choose_focus_fn:  choose_focus_fn(set_of_support) -> item
                          Default: FIFO (breadth-first).
        subsumes_fn:      subsumes_fn(a, b) -> bool
                          Does a subsume b? Default: no subsumption.
        prune_fn:         prune_fn(item, state) -> bool
                          Should we discard this item? Default: no pruning.
        max_new_items:    safety valve per step.
        verbose:          print progress.
    """
    if not state.set_of_support:
        state.halted = True
        state.halt_reason = "set_of_support empty"
        return state

    if choose_focus_fn:
        focus = choose_focus_fn(state.set_of_support)
        state.set_of_support.remove(focus)
    else:
        focus = state.set_of_support.popleft()

    state.step += 1
    if verbose:
        print(f"\n--- Step {state.step}: Focus on {focus.name} ---")

    new_items = []

    for y in state.usable:
        results = combine_fn(focus, y)
        for result in results:
            all_known = set(state.set_of_support) | set(state.usable) | set(new_items)
            if result in all_known:
                continue

            if subsumes_fn:
                if any(subsumes_fn(known, result) for known in all_known):
                    if verbose:
                        print(f"  [subsumed] {result.name}")
                    continue

            if prune_fn and prune_fn(result, state):
                if verbose:
                    print(f"  [pruned] {result.name}")
                continue

            result.step = state.step
            new_items.append(result)
            if verbose:
                print(f"  [new] {result.name} (from {focus.name} + {y.name})")

            if len(new_items) >= max_new_items:
                if verbose:
                    print(f"  [safety valve] max_new_items reached")
                break
        if len(new_items) >= max_new_items:
            break

    if subsumes_fn:
        for new_item in new_items:
            to_remove = []
            for existing in list(state.set_of_support) + state.usable:
                if subsumes_fn(new_item, existing):
                    to_remove.append(existing)
            for item in to_remove:
                if item in state.set_of_support:
                    state.set_of_support.remove(item)
                if item in state.usable:
                    state.usable.remove(item)
                if verbose:
                    print(f"  [back-subsumed] {item.name} by {new_item.name}")

    state.usable.append(focus)

    for item in new_items:
        state.set_of_support.append(item)

    entry = {
        "step": state.step,
        "focus": focus.name,
        "combined_with": len(state.usable) - 1,
        "produced": [i.name for i in new_items],
        "set_of_support_size": len(state.set_of_support),
        "usable_size": len(state.usable),
    }
    state.history.append(entry)

    if verbose:
        print(f"  Set of support: {len(state.set_of_support)} | Usable: {len(state.usable)}")

    return state


def run_otter(
    state: OtterState,
    combine_fn: Callable,
    max_steps: int = 100,
    stop_fn: Optional[Callable] = None,
    save_path: Optional[str] = None,
    **kwargs,
) -> OtterState:
    """
    Run the Otter loop until halted, stop condition met, or max_steps reached.

    Args:
        state:      initial state
        combine_fn: combination function
        max_steps:  safety limit
        stop_fn:    stop_fn(state) -> bool; halt early if True
        save_path:  if set, checkpoint state after each step
        **kwargs:   passed through to otter_step
    """
    for _ in range(max_steps):
        if state.halted:
            break
        if stop_fn and stop_fn(state):
            state.halted = True
            state.halt_reason = "stop condition met"
            break
        state = otter_step(state, combine_fn, **kwargs)
        if save_path:
            state.save(save_path)
    return state
