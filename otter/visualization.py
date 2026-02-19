"""
Visualization and reporting utilities.
"""

from .core.state import OtterState


def print_state(state: OtterState):
    """Print a summary of the current Otter state."""
    print(f"\n{'='*60}")
    print(f"Step: {state.step}")
    print(f"Set of support ({len(state.set_of_support)}):")
    for item in state.set_of_support:
        src = f" (from {item.source[0]} + {item.source[1]})" if item.source else ""
        print(f"  {item.name}{src}")
    print(f"Usable ({len(state.usable)}):")
    for item in state.usable:
        print(f"  {item.name}")
    print(f"{'='*60}")


def print_history(state: OtterState):
    """Print the combination history."""
    print(f"\n{'='*60}")
    print("Combination history:")
    print(f"{'='*60}")
    for entry in state.history:
        produced = ", ".join(entry["produced"]) if entry["produced"] else "(nothing new)"
        print(f"  Step {entry['step']}: Focused on {entry['focus']} -> {produced}")


def export_dot(state: OtterState, path="otter_graph.dot"):
    """Export the derivation graph as a DOT file for Graphviz visualization."""
    with open(path, "w") as f:
        f.write("digraph otter {\n")
        f.write("  rankdir=BT;\n")
        f.write("  node [shape=box, style=rounded];\n")

        all_items = list(state.usable) + list(state.set_of_support)
        for item in all_items:
            label = item.name.replace('"', '\\"')
            color = "lightblue" if item in state.set_of_support else "lightgray"
            f.write(f'  "{label}" [fillcolor={color}, style=filled];\n')
            if item.source:
                for parent in item.source:
                    parent_label = parent.replace('"', '\\"')
                    f.write(f'  "{parent_label}" -> "{label}";\n')
        f.write("}\n")
    print(f"Graph exported to {path}")
