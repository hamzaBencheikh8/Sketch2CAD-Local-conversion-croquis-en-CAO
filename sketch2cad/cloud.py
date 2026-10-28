"""Profil « api » : lecture des cotes par un modèle de vision distant GRATUIT.

Fournisseurs à offre gratuite, tous compatibles avec l'API OpenAI (``/chat/completions``) :

* Google Gemini (Google AI Studio) — clé gratuite, quotas par minute et par jour ;
* Groq — clé gratuite, très rapide ;
* OpenRouter — modèles ``:free`` et routeur ``openrouter/free``.

Seule la LECTURE des textes passe par l'API. Le solveur, la critique et la génération CAO
restent locaux et exacts. Deux portées :

* ``textes`` (par défaut) : seules les petites zones de texte (« 150 », « R10 »…) sont
  envoyées, rassemblées dans une seule image ; le dessin de la pièce ne quitte pas le PC ;
* ``complet`` : l'image entière est envoyée (meilleure lecture, cotes oubliées par l'OCR
  retrouvées, et le modèle distant remplace aussi le VLM local pour rattacher les cotes).

La clé est lue dans la variable d'environnement du fournisseur (``GEMINI_API_KEY``…) ou dans
``api_settings.json`` à la racine du projet (fichier écrit par l'interface web, ignoré par git).
En cas d'échec (pas de clé, quota, réseau), le pipeline continue avec l'OCR local.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / "api_settings.json"

PROVIDERS: dict[str, dict[str, str]] = {
    "gemini": {
        "label": "Google Gemini (gratuit)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.5-flash",
        "env": "GEMINI_API_KEY",
        "key_url": "https://aistudio.google.com/apikey",
    },
    "groq": {
        "label": "Groq (gratuit)",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "qwen/qwen3.8-27b",
        "env": "GROQ_API_KEY",
        "key_url": "https://console.groq.com/keys",
    },
    "openrouter": {
        "label": "OpenRouter (modèles gratuits)",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openrouter/free",
        "env": "OPENROUTER_API_KEY",
        "key_url": "https://openrouter.ai/keys",
    },
}
SCOPES = ("textes", "complet")


class CloudError(RuntimeError):
    pass


# --------------------------------------------------------------------------- réglages

def load_settings(path: Path = SETTINGS_PATH) -> dict:
    s = {"provider": "gemini", "model": "", "scope": "textes", "keys": {}}
    try:
        s.update(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    if s["provider"] not in PROVIDERS:
        s["provider"] = "gemini"
    if s["scope"] not in SCOPES:
        s["scope"] = "textes"
    return s


def save_settings(s: dict, path: Path = SETTINGS_PATH) -> None:
    Path(path).write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve(cfg_api: Optional[dict] = None, settings: Optional[dict] = None) -> dict:
    """Réglages effectifs : config.yaml < api_settings.json (interface) < variables d'environnement."""
    st = settings if settings is not None else load_settings()
    r = {"provider": "gemini", "model": "", "scope": "textes", "base_url": "", "key": ""}
    for src in (cfg_api or {}, st):
        for k in ("provider", "model", "scope"):
            if src.get(k):
                r[k] = src[k]
    if r["provider"] not in PROVIDERS:
        r["provider"] = "gemini"
    prov = PROVIDERS[r["provider"]]
    r["model"] = r["model"] or prov["model"]
    # adresse propre à un fournisseur (avancé : proxy d'entreprise, serveur compatible OpenAI)
    urls = {**(cfg_api or {}).get("base_urls", {}), **st.get("base_urls", {})}
    r["base_url"] = (urls.get(r["provider"]) or prov["base_url"]).rstrip("/")
    keys = st.get("keys", {})
    r["key"] = os.environ.get(prov["env"], "") or keys.get(r["provider"], "")
    return r


def mask_key(k: str) -> str:
    return "" if not k else (k[:4] + "…" + k[-4:] if len(k) > 10 else "…")


# --------------------------------------------------------------------------- client

def _data_url(img: Any) -> str:
    if isinstance(img, (bytes, bytearray)):
        data, mime = bytes(img), "image/png"
    elif isinstance(img, np.ndarray):
        data, mime = cv2.imencode(".png", img)[1].tobytes(), "image/png"
    else:
        p = Path(img)
        data = p.read_bytes()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(p.suffix.lower(),
                                                                                       "image/png")
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def parse_json(txt: str) -> Any:
    """JSON tolérant : retire les ```json … ``` et le texte autour du premier objet."""
    t = txt.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except ValueError:
        pass
    m = re.search(r"\{.*\}|\[.*\]", t, re.S)
    if not m:
        raise CloudError(f"Réponse non JSON : {txt[:200]!r}")
    return json.loads(m.group(0))


class CloudVLM:
    """Client OpenAI-compatible. Même interface que ``llm.Ollama`` (chat, chat_json, available)."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 90, provider: str = ""):
        self.base_url = base_url.rstrip("/")
        self.key = api_key
        self.model = model
        self.timeout = timeout
        self.provider = provider
        self.calls = 0

    @classmethod
    def from_settings(cls, r: dict) -> "CloudVLM":
        if not r.get("key"):
            prov = PROVIDERS.get(r["provider"], {})
            raise CloudError(f"Aucune clé API pour {prov.get('label', r['provider'])} : créez-en une gratuitement "
                             f"sur {prov.get('key_url', '')} puis collez-la dans l'interface (mode API).")
        return cls(r["base_url"], r["key"], r["model"], provider=r["provider"])

    def _req(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                   "User-Agent": "Sketch2CAD-Local"}
        if "openrouter.ai" in self.base_url:
            headers["X-Title"] = "Sketch2CAD-Local"
        data = json.dumps(payload).encode() if payload is not None else None
        for attempt in range(2):
            req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:300]
                if e.code == 429 and attempt == 0:  # quota gratuit par minute : une seule nouvelle tentative
                    wait = min(float(e.headers.get("Retry-After") or 15), 30)
                    log.warning("API : quota atteint, nouvelle tentative dans %.0f s", wait)
                    time.sleep(wait)
                    continue
                msg = {401: "clé API refusée", 403: "accès refusé", 404: "modèle ou adresse introuvable",
                       429: "quota gratuit épuisé (réessayez plus tard)"}.get(e.code, f"erreur HTTP {e.code}")
                raise CloudError(f"API {self.provider or self.base_url} : {msg} — {body}") from None
            except (urllib.error.URLError, OSError) as e:
                raise CloudError(f"API injoignable ({self.base_url}) : {e}") from None
        raise CloudError("API : quota gratuit épuisé")

    # -- interface commune avec llm.Ollama
    def available(self, model: Optional[str] = None) -> bool:
        return bool(self.key)

    def chat(self, model: Optional[str], prompt: str, images: Optional[list] = None,
             schema: Optional[dict] = None, system: Optional[str] = None, temperature: float = 0.0,
             **_ignored) -> str:
        if schema is not None:
            prompt += ("\n\nRéponds UNIQUEMENT par un objet JSON valide conforme à ce schéma :\n"
                       + json.dumps(schema, ensure_ascii=False))
        content: Any = prompt
        if images:
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": _data_url(im)}} for im in images]
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        res = self._req("POST", "/chat/completions",
                        {"model": model or self.model, "messages": msgs, "temperature": temperature})
        self.calls += 1
        try:
            out = res["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise CloudError(f"Réponse inattendue : {str(res)[:200]}") from None
        if isinstance(out, list):  # certains fournisseurs renvoient des morceaux
            out = "".join(p.get("text", "") for p in out if isinstance(p, dict))
        return out or ""

    def chat_json(self, model: Optional[str], prompt: str, schema: dict, **kw) -> Any:
        return parse_json(self.chat(model, prompt, schema=schema, **kw))

    def list_models(self) -> list[str]:
        res = self._req("GET", "/models")
        ids = [m.get("id", "") for m in res.get("data", []) if isinstance(m, dict)]
        return sorted(i.removeprefix("models/") for i in ids if i)


# --------------------------------------------------------------------------- lecture des cotes

_CONVENTIONS = (
    "Conventions of mechanical drawings: Ø = diameter (may look like a slashed circle, ⌀, or 'D'); "
    "R = fillet radius; 'r=' radius, 'd=' diameter; ° = degrees; 'ep 5' or 'e=5' = thickness; "
    "units mm, cm or m; a decimal comma is allowed (0,5). Keep the text exactly as written "
    "(same symbols, letters, units); do not convert or invent anything."
)


def build_mosaic(gray: np.ndarray, boxes: list[tuple[float, float, float, float]],
                 row_h: int = 96) -> np.ndarray:
    """Une image : chaque zone de texte sur sa propre ligne, numérotée #1, #2… à gauche."""
    H, W = gray.shape[:2]
    rows = []
    for k, (x0, y0, x1, y1) in enumerate(boxes, 1):
        h = max(y1 - y0, 8)
        pad = 0.35 * h + 6
        a, b = int(max(0, x0 - pad)), int(min(W, x1 + pad))
        c, d = int(max(0, y0 - pad)), int(min(H, y1 + pad))
        crop = gray[c:d, a:b]
        if crop.size == 0:
            crop = np.full((row_h, row_h), 255, np.uint8)
        else:  # encre pâle d'une photo : contraste étiré (encre -> noir, papier -> blanc)
            lo, hi = np.percentile(crop, 1), np.percentile(crop, 60)
            if hi - lo > 8:
                crop = np.clip((crop.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        s = (row_h - 12) / crop.shape[0]
        crop = cv2.resize(crop, (max(8, int(crop.shape[1] * s)), row_h - 12),
                          interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        crop = crop[:, : 1400]
        row = np.full((row_h, 110 + crop.shape[1] + 12), 255, np.uint8)
        cv2.putText(row, f"#{k}", (8, row_h // 2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2, cv2.LINE_AA)
        row[6:6 + crop.shape[0], 110:110 + crop.shape[1]] = crop
        cv2.line(row, (100, 0), (100, row_h), 160, 1)
        rows.append(row)
    width = max(r.shape[1] for r in rows)
    out = []
    for r in rows:
        r = np.pad(r, ((0, 0), (0, width - r.shape[1])), constant_values=255)
        out += [r, np.full((3, width), 120, np.uint8)]
    return np.vstack(out[:-1])


def read_texts(client: CloudVLM, gray: np.ndarray, texts: list[dict], scope: str = "textes",
               max_items: int = 40) -> tuple[list[dict], dict]:
    """Relit les textes de l'OCR (et, en portée « complet », cherche ceux qu'il a oubliés).

    ``texts`` : [{id, raw, bbox, conf}] de l'OCR local. Retourne (textes mis à jour, note pour le rapport).
    """
    from .ocr import parse_dimension_ex

    note: dict[str, Any] = {"provider": client.provider, "model": client.model, "scope": scope,
                            "changed": [], "added": [], "sent": ""}
    items = texts[:max_items]
    out = [dict(t) for t in texts]
    by_id = {t["id"]: t for t in out}

    if scope == "complet":
        img = gray
        lo, hi = np.percentile(gray, 0.5), np.percentile(gray, 50)
        if hi - lo > 8:
            img = np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        s = 1600 / max(gray.shape[:2])
        if s < 1:
            img = cv2.resize(img, (int(gray.shape[1] * s), int(gray.shape[0] * s)), interpolation=cv2.INTER_AREA)
        else:
            s = 1.0
        listing = "\n".join(
            f'{t["id"]}: OCR read "{t["raw"]}" at x={100 * (t["bbox"][0] + t["bbox"][2]) / 2 / gray.shape[1]:.0f}%,'
            f' y={100 * (t["bbox"][1] + t["bbox"][3]) / 2 / gray.shape[0]:.0f}%' for t in items) or "(none)"
        prompt = (
            "This is a photo of a hand-drawn, dimensioned mechanical sketch. A local OCR found these "
            "annotations (position = centre, in % of image width/height):\n" + listing + "\n\n"
            "1) For each id, give the correct transcription of the handwritten annotation at that place "
            "(\"\" if it is not a dimension/annotation).\n"
            "2) List the dimension annotations the OCR MISSED, with the centre of each (x, y in % of image "
            "width/height).\n" + _CONVENTIONS + "\n"
            'Answer ONLY with JSON: {"texts": {"T1": "...", ...}, "missing": [{"text": "...", "x": 0, "y": 0}]}')
        note["sent"] = "image entière"
        res = parse_json(client.chat(None, prompt, images=[img]))
        fixes = res.get("texts", {}) if isinstance(res, dict) else {}
        missing = res.get("missing", []) if isinstance(res, dict) else []
    else:
        if not items:
            return out, note
        mosaic = build_mosaic(gray, [tuple(t["bbox"]) for t in items])
        prompt = (
            f"This image shows {len(items)} handwritten annotations cut out of a hand-drawn mechanical "
            f"drawing, one per row, numbered #1 to #{len(items)} on the left. Transcribe each one. "
            + _CONVENTIONS + " If a row is unreadable or is not an annotation, answer \"\".\n"
            'Answer ONLY with JSON: {"1": "...", "2": "..."}')
        note["sent"] = f"{len(items)} zones de texte (mosaïque {mosaic.shape[1]}×{mosaic.shape[0]} px)"
        res = parse_json(client.chat(None, prompt, images=[mosaic]))
        res = res.get("texts", res) if isinstance(res, dict) else {}
        fixes = {items[int(k.strip("#")) - 1]["id"]: v for k, v in res.items()
                 if str(k).strip("#").isdigit() and 1 <= int(str(k).strip("#")) <= len(items)}
        missing = []

    for tid, new in fixes.items():
        t = by_id.get(str(tid))
        if t is None or not isinstance(new, str):
            continue
        new = new.strip()
        old = t["raw"]
        # on ne remplace que par une cote lisible ; une lecture OCR valide n'est pas effacée
        if new and new != old and parse_dimension_ex(new) is not None:
            t["raw"], t["ocr_raw"], t["src"] = new, old, "api"
            note["changed"].append({"id": t["id"], "ocr": old, "api": new})

    H, W = gray.shape[:2]
    n = max([int(t["id"][1:]) for t in out if t["id"][1:].isdigit()] + [0])
    for m in missing if isinstance(missing, list) else []:
        try:
            txt = str(m["text"]).strip()
            x, y = float(m["x"]) / 100 * W, float(m["y"]) / 100 * H
        except (KeyError, TypeError, ValueError):
            continue
        if not txt or parse_dimension_ex(txt) is None or not (0 <= x <= W and 0 <= y <= H):
            continue
        # déjà présent tout près ? (le modèle répète parfois une cote connue)
        key = re.sub(r"\s+", "", txt.lower())
        kind = parse_dimension_ex(txt)[0]
        if any(abs((t["bbox"][0] + t["bbox"][2]) / 2 - x) < 0.04 * W and
               abs((t["bbox"][1] + t["bbox"][3]) / 2 - y) < 0.04 * H for t in out):
            continue
        if kind == "thickness" and any((parse_dimension_ex(t["raw"]) or ("",))[0] == "thickness" for t in out):
            continue  # une seule épaisseur par pièce
        if any(re.sub(r"\s+", "", t["raw"].lower()) == key and
               np.hypot((t["bbox"][0] + t["bbox"][2]) / 2 - x, (t["bbox"][1] + t["bbox"][3]) / 2 - y)
               < 0.15 * max(W, H) for t in out):
            continue  # même texte tout près : position approximative du modèle
        n += 1
        hw, hh = max(20, 0.012 * W * len(txt)), max(12, 0.015 * H)
        out.append({"id": f"T{n}", "raw": txt, "bbox": [x - hw, y - hh, x + hw, y + hh], "conf": 0.5,
                    "src": "api", "added": True})
        note["added"].append({"id": f"T{n}", "api": txt})
    note["calls"] = client.calls
    return out, note


def make_text_reader(cfg: dict):
    """Fonction passée à ``perception.perceive`` : relit les textes via l'API, ou None si mode local."""
    if not cfg.get("active", {}).get("cloud"):
        return None
    r = resolve(cfg.get("api"))

    def reader(gray: np.ndarray, dims: list) -> tuple[list[dict], dict]:
        texts = [{"id": d.id, "raw": d.raw, "bbox": list(d.bbox), "conf": d.conf} for d in dims]
        try:
            client = CloudVLM.from_settings(r)
            t0 = time.time()
            texts, note = read_texts(client, gray, texts, r["scope"])
            note["seconds"] = round(time.time() - t0, 1)
            note["ok"] = True
        except (CloudError, ValueError, KeyError, TypeError) as exc:
            log.warning("API : %s", exc)
            note = {"provider": r["provider"], "model": r["model"], "scope": r["scope"], "ok": False,
                    "error": str(exc)}
        return texts, note

    return reader
