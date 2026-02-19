"""
Domain: Interactive (human in the loop).

The human chooses what to focus on and decides what combinations produce.
Demonstrates the Otter loop as a collaborative exploration tool.
"""

from ..core.state import Item, Edge


def interactive_choose_focus(set_of_support):
    """Let the human choose what to focus on next."""
    print("\nWhat would you like to focus on?")
    items = list(set_of_support)
    for i, item in enumerate(items):
        print(f"  [{i}] {item.name}")
    while True:
        try:
            choice = input("> ").strip()
            if choice.lower() in ('q', 'quit', 'exit'):
                raise KeyboardInterrupt
            idx = int(choice)
            return items[idx]
        except (ValueError, IndexError):
            print("Enter a number, or 'q' to quit.")


def interactive_combine(x, y) -> list:
    """Let the human decide what two items combine to produce."""
    print(f"\nCombine: {x.name} + {y.name}")
    print(f"  {x.name}: {x.content}")
    print(f"  {y.name}: {y.content}")
    result = input("What does this produce? (empty to skip, name:description) > ").strip()
    if not result:
        return []

    parts = result.split(":", 1)
    name = parts[0].strip()
    desc = parts[1].strip() if len(parts) > 1 else name

    if isinstance(x, Edge):
        edge_parts = name.split()
        if len(edge_parts) >= 3:
            return [Edge(
                edge_parts[0], edge_parts[1], " ".join(edge_parts[2:]),
                confidence=0.5, source=(x.name, y.name),
            )]

    return [Item(name=name, content=desc, source=(x.name, y.name))]
