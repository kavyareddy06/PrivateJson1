# tools/developer_tool.py
import json
from typing import Dict, Any, List
from crewai_tools import tool
from utils.json_index import (
    load_kb_json_objects,
    list_unique_values_for_key,
    list_style_blocks,
    list_component_types,
    filtered_select,
)
from utils.llm_client import llm_complete

DEV_SYSTEM_PROMPT = """You are a developer agent that produces accurate JSON or clear explanations.
Rules:
- If asked to LIST, return a compact JSON object with arrays (no prose unless requested).
- If asked to EXPLAIN, provide textual explanation grounded by evidence.
- If asked to GENERATE, produce a syntactically-valid JSON layout. Avoid hallucinating fields.
- Prefer fields that actually exist in the provided Studio JSONs unless user asks for new ones.
- If generation is requested, ensure "page" root with "title" and "sections" is valid.
"""

def _explain_with_llm(query: str, evidence: str) -> str:
    prompt = f"Explain the following for Studio.json\n\nUser question:\n{query}\n\nRelevant evidence:\n{evidence}"
    return llm_complete(system=DEV_SYSTEM_PROMPT, prompt=prompt, max_tokens=900)

def _generate_layout_with_llm(query: str, evidence: str) -> Dict[str, Any]:
    prompt = (
        "Generate a valid JSON layout matching the user's request. "
        "Keep keys present in the evidence when appropriate. "
        "Return JSON only (no markdown). "
        f"\nUser request:\n{query}\n\nEvidence (excerpts):\n{evidence}"
    )
    raw = llm_complete(system=DEV_SYSTEM_PROMPT, prompt=prompt, max_tokens=1200)
    # JSON repair pass:
    try:
        return json.loads(raw)
    except Exception:
        # fallback skeleton if LLM fails
        return {
            "page": {
                "title": "Generated Layout",
                "sections": [
                    {"type": "header", "content": "Auto-generated"},
                    {"type": "body", "content": "See evidence-based notes."},
                    {"type": "footer", "content": "—"}
                ]
            },
            "_llm_raw": raw[:4000]
        }

@tool("developer_tool")
def developer_tool(inputs: Dict[str, Any]) -> str:
    """
    Developer Tool:
    Inputs: {"plan": str|dict, "query": str, "context": str}
    - Loads studio JSONs from knowledge_base/
    - Executes LIST / EXPLAIN / GENERATE with deterministic parsing + LLM where needed
    Returns JSON string or plain text depending on plan.expected_output
    """
    plan_in = (inputs or {}).get("plan") or {}
    if isinstance(plan_in, str):
        try:
            plan = json.loads(plan_in)
        except Exception:
            plan = {}
    else:
        plan = plan_in

    query = (inputs or {}).get("query", "")
    context_preview = (inputs or {}).get("context", "")[:2000]

    intent = plan.get("intent", "EXPLAIN")
    targets = plan.get("targets", [])
    expected_output = plan.get("expected_output", "json")
    constraints = plan.get("constraints", [])

    # Load Studio JSON(s) deterministically
    json_objs = load_kb_json_objects()

    # Build evidence snippets (short extracts) from Studio JSON
    evidence_chunks: List[str] = []
    if "commandName" in targets or "commands" in targets:
        cmds = list_unique_values_for_key(json_objs, "commandName", limit=5000)
        evidence_chunks.append("COMMANDS=" + json.dumps(cmds[:200]))
    if "styles" in targets:
        styles = list_style_blocks(json_objs, limit=50)
        evidence_chunks.append("STYLES=" + json.dumps(styles[:50])[:3000])
    if "components" in targets or "widgets" in targets:
        comps = list_component_types(json_objs, limit=200)
        evidence_chunks.append("COMPONENT_TYPES=" + json.dumps(comps[:200]))

    # If user asked filtered things (constraints), try simple filtering:
    filtered = filtered_select(json_objs, constraints)
    if filtered:
        evidence_chunks.append("FILTERED=" + json.dumps(filtered)[:3000])

    evidence_text = "\n\n".join(evidence_chunks) or context_preview

    # Branch by intent
    if intent == "LIST":
        # Deterministic lists from JSON
        out = {"type": "list", "targets": targets, "results": {}}
        if "commandName" in targets or "commands" in targets:
            out["results"]["commandNames"] = list_unique_values_for_key(json_objs, "commandName", limit=10000)
        if "styles" in targets:
            out["results"]["styles"] = list_style_blocks(json_objs, limit=500)
        if "components" in targets or "widgets" in targets:
            out["results"]["components"] = list_component_types(json_objs, limit=10000)

        # If no specific targets, try to infer from query words
        if not out["results"]:
            # fallback: provide commandNames as most useful
            out["results"]["commandNames"] = list_unique_values_for_key(json_objs, "commandName", limit=10000)

        return json.dumps(out, indent=2)

    elif intent == "EXPLAIN":
        explanation = _explain_with_llm(query, evidence_text)
        # EXPLAIN returns text by default
        return explanation

    elif intent == "GENERATE":
        layout = _generate_layout_with_llm(query, evidence_text)
        # Ensure minimal shape
        if not isinstance(layout, dict) or "page" not in layout:
            layout = {
                "page": {
                    "title": "Generated Layout (safe fallback)",
                    "sections": [{"type": "body", "content": "Generation fallback"}]
                },
                "_note": "LLM output could not be parsed; returned safe structure."
            }
        return json.dumps({"type": "layout", "layout": layout}, indent=2)

    # Unknown intent — graceful fallback
    fallback = _explain_with_llm(query, evidence_text)
    return fallback
