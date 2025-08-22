from crewai import Agent
from langchain_ollama import OllamaLLM
from tools.retriever_tool import retriever_tool
from tools.planner_tool import planner_tool
from tools.developer_tool import developer_tool
from config import SYSTEM_INSTRUCTIONS, LLM_MODEL

llm = OllamaLLM(model=LLM_MODEL)

retriever_agent = Agent(
    role="Retriever Agent",
    goal="Fetch JSON/PDF context from knowledge base.",
    backstory=SYSTEM_INSTRUCTIONS["retriever"],
    llm=llm,
    tools=[retriever_tool],
    verbose=True,
)

planner_agent = Agent(
    role="Planner Agent",
    goal="Plan structured steps for implementation.",
    backstory=SYSTEM_INSTRUCTIONS["planner"],
    llm=llm,
    tools=[planner_tool],
    verbose=True,
)

developer_agent = Agent(
    role="Developer Agent",
    goal="Generate production-ready code grounded in JSON context.",
    backstory=SYSTEM_INSTRUCTIONS["developer"],
    llm=llm,
    tools=[developer_tool],
    verbose=True,
)
