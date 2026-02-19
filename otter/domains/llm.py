"""
Domain: LLM combination.

Uses the Claude API to decide whether and how two items combine.
This is the centaur mode: the LLM acts as the combination oracle.

Requires: pip install anthropic
          ANTHROPIC_API_KEY environment variable
"""

from ..core.state import Item, Edge


def make_llm_combine(api_key=None, model="claude-sonnet-4-20250514"):
    """
    Returns a combination function that consults Claude.

    The LLM sees two items and decides:
    - NO: they don't meaningfully combine
    - YES: produces a NAME and CONTENT for the result
    """
    def llm_combine(x, y) -> list:
        try:
            import anthropic
        except ImportError:
            print("pip install anthropic")
            return []

        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""You are participating in a combinatorial exploration.

You will be shown two items. Decide if they can be meaningfully combined
to produce something new -- a new idea, connection, or synthesis.

Item A: {x.name}
{x.content}

Item B: {y.name}
{y.content}

Can these be combined into something new and interesting?

If NO, respond with just: NO

If YES, respond with exactly two lines:
NAME: (brief name, a few words)
CONTENT: (one paragraph describing the combination)"""

        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()

        if text.upper().startswith("NO"):
            return []

        lines = text.split("\n")
        name = ""
        content = ""
        for line in lines:
            if line.startswith("NAME:"):
                name = line[5:].strip()
            elif line.startswith("CONTENT:"):
                content = line[8:].strip()

        if name and content:
            if isinstance(x, Edge):
                words = name.split()
                if len(words) >= 3:
                    return [Edge(
                        words[0], words[1], " ".join(words[2:]),
                        confidence=0.5, source=(x.name, y.name),
                    )]
            return [Item(name=name, content=content, source=(x.name, y.name))]

        return []

    return llm_combine
