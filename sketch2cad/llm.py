"""Client minimal pour Ollama (http://localhost:11434), sans dépendance externe.

Toutes les requêtes restent sur la machine locale. Si Ollama n'est pas lancé,
``available()`` renvoie False et le pipeline bascule sur ses replis déterministes.
"""
from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


class Ollama:
    def __init__(self, host: str = "http://localhost:11434", timeout: float = 600):
        self.host = host.rstrip("/")
        if not (self.host.startswith("http://localhost") or self.host.startswith("http://127.0.0.1")):
            log.warning("Hôte Ollama non local (%s) : les croquis quitteraient la machine.", self.host)
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(self.host + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def available(self, model: Optional[str] = None) -> bool:
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=3) as r:
                tags = json.loads(r.read().decode())
        except (urllib.error.URLError, OSError, ValueError):
            return False
        if model is None:
            return True
        names = {m.get("name", "") for m in tags.get("models", [])}
        return model in names or f"{model}:latest" in names

    def chat(self, model: str, prompt: str, images: Optional[list[Path]] = None,
             schema: Optional[dict] = None, system: Optional[str] = None,
             keep_alive: Any = 0, temperature: float = 0.0, num_ctx: int = 8192) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msg: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            msg["images"] = [base64.b64encode(Path(p).read_bytes()).decode() for p in images]
        msgs.append(msg)
        payload: dict[str, Any] = {
            "model": model, "messages": msgs, "stream": False, "keep_alive": keep_alive,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        if schema is not None:
            payload["format"] = schema
        res = self._post("/api/chat", payload)
        return res["message"]["content"]

    def chat_json(self, model: str, prompt: str, schema: dict, **kw) -> dict:
        txt = self.chat(model, prompt, schema=schema, **kw)
        txt = txt.strip()
        if txt.startswith("```"):
            txt = txt.strip("`").split("\n", 1)[1] if "\n" in txt else txt
        return json.loads(txt)
