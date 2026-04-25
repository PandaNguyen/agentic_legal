from crewai import Task


def placeholder_task(description: str) -> Task:
    return Task(description=description, expected_output="Structured result")
