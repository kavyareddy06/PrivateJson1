from crewai import Crew, Task
from agents import retriever_agent, planner_agent, developer_agent

task_retrieve = Task(
    agent=retriever_agent,
    description="Retrieve context from JSON/PDF KB.",
    expected_output="Relevant text chunks",
    output_key="context",
)

task_plan = Task(
    agent=planner_agent,
    description="Create plan from query + context.",
    expected_output="Step-by-step plan",
    inputs={"query": "{{ query }}", "context": "{{ context }}"},
    output_key="plan",
)

task_dev = Task(
    agent=developer_agent,
    description="Generate final production-ready code.",
    expected_output="End-to-end code",
    inputs={"query": "{{ query }}", "context": "{{ context }}", "plan": "{{ plan }}"},
    output_key="final_code",
)

crew = Crew(
    agents=[retriever_agent, planner_agent, developer_agent],
    tasks=[task_retrieve, task_plan, task_dev],
    verbose=True,
)

def agentic_rag_answer(query: str) -> str:
    return crew.kickoff(inputs={"query": query})
