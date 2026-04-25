from crewai import Agent


def build_router_agent() -> Agent:
    return Agent(
        role="Vietnam Legal Router",
        goal="Classify the legal request and determine the right workflow",
        backstory="Specializes in triaging legal support requests for consumers.",
        verbose=False,
    )


def build_reasoner_agent() -> Agent:
    return Agent(
        role="Vietnam Legal Reasoner",
        goal="Produce a simple, citation-grounded legal answer",
        backstory="Explains Vietnamese legal material to non-lawyers.",
        verbose=False,
    )
