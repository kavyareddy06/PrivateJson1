# tools/planner_tool.py
import json
import re
from typing import Dict, Any, List
from crewai.tools import tool

def _extract_candidate_keys(raw_text: str, limit: int = 32) -> List[str]:
    # Catch JSONPath-like keys (`$.something[0].key`) and simple "key: value" lines
    keys = set()

    # JSONPath-ish
    for m in re.finditer(r'(\$\.[\w\.\[\]]+)', raw_text):
        keys.add(m.group(1))

    # key: value pairs
    for line in raw_text.splitlines():
        if ":" in line and len(line) < 160:
            k = line.split(":", 1)[0].strip().strip('"').strip("'")
            if k and len(k) < 60 and all(ch.isprintable() for ch in k):
                keys.add(k)

    return list(keys)[:limit]

@tool("planner_tool")
def planner_tool(inputs: str) -> str:
    """
    Planner Tool:
    Input (stringified JSON): { "query": str, "context": str }
    Always produces a JSON design spec for the developer, even if context is not pure JSON.
    """
    try:
        data = json.loads(inputs) if isinstance(inputs, str) else (inputs or {})
    except Exception:
        data = {}

    query = data.get("query", "")
    context = data.get("context", "")  # may be raw_text from retriever

    candidate_keys = _extract_candidate_keys(context or "")

    design_spec: Dict[str, Any] = {
        "type": "design_spec",
        "query": query,
        "assumptions": [
            "Context may include JSONPath-like lines and text from PDFs.",
            "We must generate a JSON layout and production-grade code from this.",
        ],
        "json_targets": {
            "wants_json_layout": True,
            "suggested_top_level": ["page", "sections", "styles", "responsive"],
            "candidate_keys_from_context": candidate_keys,
        },
        "ui_mapping_strategy": {
            "header": "Use title/logo/navigation if present or requested.",
            "body": "Map candidate keys into fields/components.",
            "footer": "Static attribution or metadata.",
        },
        "deliverables": [
            "A JSON layout object",
            "End-to-end code file(s) (default: Streamlit app.py) that use the layout",
            "Notes on how fields map from the KB"
        ]
    }

    return json.dumps(design_spec, indent=2, ensure_ascii=False)
