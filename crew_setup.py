# crew_setup.py
from crewai import Crew, Task
from agents import retriever_agent, planner_agent, developer_agent

# 1) Retrieval Task
task_retrieve = Task(
    agent=retriever_agent,
    description="Retrieve context from studio.json knowledge base.",
    expected_output="Relevant text or JSON snippets",
    output_key="context",
)

# 2) Planning Task
task_plan = Task(
    agent=planner_agent,
    description="Interpret query + context into a structured plan (intent, targets, constraints, expected_output).",
    expected_output="JSON plan with intent, targets, constraints, expected_output",
    inputs={"query": "{{ query }}", "context": "{{ context }}"},
    output_key="plan",
)

# 3) Developer Task
task_dev = Task(
    agent=developer_agent,
    description="Generate final structured answer (JSON layout, list of commands, or explanation).",
    expected_output="End-to-end JSON or explanatory text",
    inputs={"query": "{{ query }}", "context": "{{ context }}", "plan": "{{ plan }}"},
    output_key="final_code",
)

crew = Crew(
    agents=[retriever_agent, planner_agent, developer_agent],
    tasks=[task_retrieve, task_plan, task_dev],
    verbose=True,
)

def agentic_rag_answer(query: str) -> str:
    """
    Kick off the crew pipeline:
    - Retrieve context
    - Plan intent + targets
    - Develop structured answer
    """
    result = crew.kickoff(inputs={"query": query})
    return result.get("final_code", "❌ No final output produced")
