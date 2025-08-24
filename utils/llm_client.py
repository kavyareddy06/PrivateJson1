# utils/llm_client.py
import os
import json
import requests

"""
Unified LLM client for:
- Prodigy ADK MCP server (HTTP) -> set LLM_PROVIDER=prodigy and PRODIGY_ENDPOINT=http://localhost:8000/complete
- OpenAI -> set LLM_PROVIDER=openai and OPENAI_API_KEY=...
- Ollama (local) -> set LLM_PROVIDER=ollama and OLLAMA_MODEL=llama3.1 (or similar)

Usage: llm_complete(system, prompt, max_tokens)
"""

def _post_json(url: str, payload: dict, headers: dict = None, timeout: int = 60) -> str:
    r = requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
    r.raise_for_status()
    return r.text

def llm_complete(system: str, prompt: str, max_tokens: int = 700) -> str:
    provider = (os.getenv("LLM_PROVIDER") or "prodigy").lower()

    if provider == "prodigy":
        # Expect your MCP/ADK HTTP bridge to accept:
        # { "system": "...", "prompt": "...", "max_tokens": 700 }
        endpoint = os.getenv("PRODIGY_ENDPOINT", "http://localhost:8000/complete")
        try:
            resp = requests.post(
                endpoint,
                json={"system": system, "prompt": prompt, "max_tokens": max_tokens},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            # Expect { "completion": "..." } or raw text
            if isinstance(data, dict) and "completion" in data:
                return data["completion"]
            # fallbacks
            return data if isinstance(data, str) else json.dumps(data)
        except Exception as e:
            return f"LLM(PRODIGY)_ERROR: {e}"

    if provider == "openai":
        # Minimal OpenAI REST call (completions style) – adjust to your SDK if needed.
        from openai import OpenAI  # pip install openai>=1.0
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"LLM(OPENAI)_ERROR: {e}"

    if provider == "ollama":
        # Simple Ollama REST call
        url = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/chat")
        model = os.getenv("OLLAMA_MODEL", "llama3.1")
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.1},
            }
            txt = _post_json(url, payload)
            data = json.loads(txt)
            return data.get("message", {}).get("content", txt)
        except Exception as e:
            return f"LLM(OLLAMA)_ERROR: {e}"

    return "LLM_ERROR: Unknown provider"
