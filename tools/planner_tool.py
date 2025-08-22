from crewai_tools import tool

@tool("planner_tool")
def planner_tool(inputs: str) -> str:
    """
    Creates structured execution plan:
    1. Summarize JSON/PDF structure
    2. Map JSON -> UI components
    3. Decide framework (Streamlit/React)
    4. Outline deliverables
    """
    return f"📋 Plan created for inputs:\n{inputs}"
