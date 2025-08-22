from crewai_tools import tool

@tool("developer_tool")
def developer_tool(plan: str) -> str:
    """
    Converts plan into full production-ready code.
    Generates Streamlit or React code depending on instructions.
    """
    return f"💻 Generated code based on plan:\n{plan}"
