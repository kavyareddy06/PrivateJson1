# tools/planner_tool.py
import json
import re
from typing import Dict, Any, List
from crewai_tools import tool
from utils.llm_client import llm_complete
from utils.json_index import (
    detect_signals_from_context,
    soft_intent_heuristics,
)

SYSTEM_PROMPT = """You are a planning agent that understands a large Studio JSON schema.
You receive:
- user query
- retrieved 'context' (raw text snippets and/or JSON path lines)

Your job:
1) Infer the user's INTENT: one of ["LIST","EXPLAIN","GENERATE"].
   - LIST: user wants lists (e.g., all commandNames, all styles, etc.)
   - EXPLAIN: user wants explanations/definitions, behavior, usage, relationships.
   - GENERATE: user wants you to produce or transform a JSON layout/snippet.

2) Identify TARGETS relevant to the query (e.g., ["commandName"], ["styles"], ["components"], ["layouts"], etc.).
   Targets must be short, concrete nouns from the domain of Studio JSON.

3) Extract CONSTRAINTS (filters, matchers, parts of the query like "only those used in layouts[2]", "starting with Transform", "responsive only").

4) Decide EXPECTED_OUTPUT: "json" when you expect to produce a JSON output; otherwise "text".

Return a strict JSON object with fields:
{
  "intent": "LIST" | "EXPLAIN" | "GENERATE",
  "targets": string[],
  "constraints": string[],
  "expected_output": "json" | "text",
  "notes": string
}
Do not include markdown. Return only valid JSON.
"""

def _llm_plan(query: str, context_preview: str) -> Dict[str, Any]:
    user = f"QUERY:\n{query}\n\nCONTEXT (preview):\n{context_preview[:4000]}"
    raw = llm_complete(system=SYSTEM_PROMPT, prompt=user, max_tokens=800)
    try:
        return json.loads(raw)
    except Exception:
        return {}

@tool("planner_tool")
def planner_tool(inputs: Dict[str, Any]) -> str:
    """
    Planner Tool:
    - Accepts {"query": str, "context": str}
    - Uses heuristics + LLM to build a dynamic, executable plan for the developer tool.
    - Returns a JSON string.
    """
    query: str = (inputs or {}).get("query", "") or ""
    context: str = (inputs or {}).get("context", "") or ""

    # 1) quick heuristics on query + context to anchor the LLM
    heuristic = soft_intent_heuristics(query)
    detected = detect_signals_from_context(context)

    # 2) LLM planning pass (robust & dynamic)
    llm_plan = _llm_plan(query, context)

    # 3) Merge heuristics + LLM signal
    intent = llm_plan.get("intent") or heuristic["intent"]
    targets = llm_plan.get("targets") or heuristic["targets"] or detected["targets"]
    expected_output = llm_plan.get("expected_output") or heuristic["expected_output"]
    constraints = list(set((llm_plan.get("constraints") or []) + heuristic["constraints"]))

    # 4) Hard fallback defaults
    if intent not in {"LIST", "EXPLAIN", "GENERATE"}:
        intent = heuristic["intent"]
    if expected_output not in {"json", "text"}:
        # LIST (JSON list), GENERATE (JSON), EXPLAIN (text)
        expected_output = "json" if intent in {"LIST", "GENERATE"} else "text"

    plan = {
        "intent": intent,
        "targets": targets or [],
        "constraints": constraints,
        "expected_output": expected_output,
        "notes": llm_plan.get("notes", "auto-generated plan"),
        # pass raw query/context forward too (developer may need it)
        "_inputs": {
            "query": query,
            "context_preview": context[:2000],
        },
    }
    return json.dumps(plan, indent=2)
