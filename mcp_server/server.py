from fastapi import FastAPI
from tools.retriever_tool import retriever_tool
from tools.planner_tool import planner_tool
from tools.developer_tool import developer_tool

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/retrieve")
def retrieve(query: str):
    return {"result": retriever_tool(query)}

@app.post("/plan")
def plan(data: dict):
    return {"plan": planner_tool(data.get("context", ""))}

@app.post("/develop")
def develop(data: dict):
    return {"code": developer_tool(data.get("plan", ""))}
