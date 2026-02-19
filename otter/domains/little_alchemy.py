"""
Domain: Little Alchemy.

Classic Otter demo: items are things, combination rules are a lookup table.
Demonstrates the engine working at its simplest.
"""

from ..core.state import Item, OtterState


LITTLE_ALCHEMY_RULES = {
    frozenset({"air", "water"}):   [("rain",     "Water falling from sky")],
    frozenset({"air", "fire"}):    [("energy",   "Pure force")],
    frozenset({"air", "earth"}):   [("dust",     "Fine particles in the air")],
    frozenset({"earth", "water"}): [("mud",      "Wet earth")],
    frozenset({"earth", "fire"}):  [("lava",     "Molten rock")],
    frozenset({"fire", "water"}):  [("steam",    "Heated water vapor")],
    frozenset({"rain", "earth"}):  [("plant",    "Growing life")],
    frozenset({"mud", "plant"}):   [("swamp",    "Wetland ecosystem")],
    frozenset({"lava", "water"}):  [("stone",    "Cooled volcanic rock"),
                                    ("steam",    "Heated water vapor")],
    frozenset({"energy", "air"}):  [("wind",     "Moving air")],
    frozenset({"plant", "water"}): [("algae",    "Simple aquatic life")],
    frozenset({"stone", "fire"}):  [("metal",    "Refined mineral")],
    frozenset({"stone", "air"}):   [("sand",     "Eroded stone")],
    frozenset({"sand", "fire"}):   [("glass",    "Transparent solid")],
    frozenset({"plant", "fire"}):  [("tobacco",  "Dried burning leaf"),
                                    ("ash",      "Remnant of fire")],
    frozenset({"steam", "air"}):   [("cloud",    "Condensed vapor")],
    frozenset({"cloud", "water"}): [("rain",     "Water falling from sky")],
    frozenset({"energy", "plant"}): [("tree",    "Large plant")],
    frozenset({"tree", "fire"}):   [("ash",      "Remnant of fire"),
                                    ("charcoal", "Burned wood")],
}


def little_alchemy_combine(x: Item, y: Item) -> list:
    """Combination function for Little Alchemy: look up the pair in the rule table."""
    key = frozenset({x.name, y.name})
    results = LITTLE_ALCHEMY_RULES.get(key, [])
    return [
        Item(name=name, content=desc, source=(x.name, y.name))
        for name, desc in results
    ]


def make_little_alchemy_state() -> OtterState:
    state = OtterState()
    for name, desc in [
        ("air",   "The atmosphere"),
        ("earth", "Solid ground"),
        ("fire",  "Combustion"),
        ("water", "H2O in liquid form"),
    ]:
        state.set_of_support.append(Item(name=name, content=desc))
    return state
