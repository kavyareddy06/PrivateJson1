# utils/json_index.py
import os
from typing import Any, Dict, List, Set
import orjson

KB_DIR = os.getenv("KB_DIR", "knowledge_base")

# -------- Loading ----------
def _walk(obj: Any):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)

def load_kb_json_objects() -> List[Dict[str, Any]]:
    objs: List[Dict[str, Any]] = []
    if not os.path.isdir(KB_DIR):
        return objs
    for f in os.listdir(KB_DIR):
        if f.lower().endswith(".json"):
            path = os.path.join(KB_DIR, f)
            try:
                with open(path, "rb") as fh:
                    objs.append(orjson.loads(fh.read()))
            except Exception:
                # skip corrupted
                pass
    return objs

# -------- Signals / Heuristics ----------
def detect_signals_from_context(context: str) -> Dict[str, List[str]]:
    targets: Set[str] = set()
    if "commandName" in context:
        targets.add("commandName")
    if "styles" in context or '"style"' in context:
        targets.add("styles")
    if "component" in context or "widget" in context:
        targets.add("components")
    return {"targets": list(targets)}

def soft_intent_heuristics(query: str) -> Dict[str, Any]:
    q = (query or "").lower()
    intent = "EXPLAIN"
    targets: List[str] = []
    constraints: List[str] = []
    expected_output = "text"

    if any(w in q for w in ["list", "all", "enumerate", "show me", "extract"]):
        intent = "LIST"; expected_output = "json"
    if any(w in q for w in ["json", "schema", "layout", "generate", "create", "produce", "build"]):
        intent = "GENERATE"; expected_output = "json"
    if any(w in q for w in ["explain", "what is", "how does", "meaning", "difference", "describe"]):
        intent = "EXPLAIN"; expected_output = "text"

    if "command" in q: targets.append("commandName")
    if "style" in q or "css" in q or "responsive" in q: targets.append("styles")
    if "component" in q or "widget" in q: targets.append("components")

    return {
        "intent": intent,
        "targets": list(dict.fromkeys(targets)),
        "constraints": constraints,
        "expected_output": expected_output,
    }

# -------- Deterministic extraction ----------
def list_unique_values_for_key(objs: List[Dict[str, Any]], key_name: str, limit: int = 10000) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for obj in objs:
        for k, v in _walk(obj):
            if k == key_name and isinstance(v, str):
                if v not in seen:
                    seen.add(v); out.append(v)
                    if len(out) >= limit:
                        return out
    return out

def list_style_blocks(objs: List[Dict[str, Any]], limit: int = 500) -> List[Dict[str, Any]]:
    styles: List[Dict[str, Any]] = []
    for obj in objs:
        for k, v in _walk(obj):
            if k in ("style", "styles") and isinstance(v, (dict, list)):
                styles.append({k: v})
                if len(styles) >= limit:
                    return styles
    return styles

def list_component_types(objs: List[Dict[str, Any]], limit: int = 10000) -> List[str]:
    types: List[str] = []
    seen: Set[str] = set()
    for obj in objs:
        for k, v in _walk(obj):
            if k in ("type", "component", "widget") and isinstance(v, str):
                if v not in seen:
                    seen.add(v); types.append(v)
                    if len(types) >= limit:
                        return types
    return types

def filtered_select(objs: List[Dict[str, Any]], constraints: List[str]) -> List[Dict[str, Any]]:
    """
    Super simple filter engine:
    - supports contains-matching for key:value pairs written as "key=foo"
    """
    if not constraints:
        return []
    pairs = []
    for c in constraints:
        if "=" in c:
            k, v = c.split("=", 1)
            pairs.append((k.strip(), v.strip().lower()))
    if not pairs:
        return []

    hits: List[Dict[str, Any]] = []
    for obj in objs:
        ok = True
        bag: List[str] = []
        for k, v in _walk(obj):
            if isinstance(v, (str, int, float, bool)):
                bag.append(f"{k}:{str(v).lower()}")
        for kq, vq in pairs:
            if not any(s.startswith(f"{kq.lower()}:") and s.endswith(vq) for s in bag):
                ok = False; break
        if ok:
            hits.append({"match": pairs, "sample": str(obj)[:800]})
            if len(hits) >= 25:
                break
    return hits
