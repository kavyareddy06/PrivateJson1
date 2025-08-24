# agents.py
from crewai import Agent
from config import SYSTEM_INSTRUCTIONS, LLM_MODEL
from tools.retriever_tool import retriever_tool
from tools.planner_tool import planner_tool
from tools.developer_tool import developer_tool
from utils.llm_client import llm_complete

# Define agents
retriever_agent = Agent(
    role="Retriever Agent",
    goal="Fetch JSON/PDF context from knowledge base.",
    backstory=SYSTEM_INSTRUCTIONS.get("retriever", "You fetch context."),
    llm=LLM_MODEL,   # just pass model string, crewai will handle
    tools=[retriever_tool],
    verbose=True,
)

planner_agent = Agent(
    role="Planner Agent",
    goal="Interpret the query + context into a structured plan (LIST, EXPLAIN, GENERATE).",
    backstory=SYSTEM_INSTRUCTIONS.get("planner", "You plan dynamically based on studio.json."),
    llm=LLM_MODEL,
    tools=[planner_tool],
    verbose=True,
)

developer_agent = Agent(
    role="Developer Agent",
    goal="Execute the plan and produce final structured output (JSON layouts, lists, or explanations).",
    backstory=SYSTEM_INSTRUCTIONS.get("developer", "You generate valid JSON or clear answers."),
    llm=LLM_MODEL,
    tools=[developer_tool],
    verbose=True,
)
