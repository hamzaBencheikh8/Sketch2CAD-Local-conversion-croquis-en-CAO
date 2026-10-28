"""Agent 2 — Construction du graphe de contraintes.

Le VLM (ou, à défaut, l'heuristique de proximité) se contente de RATTACHER chaque
cote OCR (T*) à un élément du croquis (E*, V*, C*). Il ne produit jamais de coordonnée.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .llm import Ollama
from .schema import Constraint, Dimension, Graph, Perception

log = logging.getLogger(__name__)

HV_TOL_DEG = 6.0


# --------------------------------------------------------------------------- géométrie px

def _seg_dist(p, a, b) -> tuple[float, float]:
    p, a, b = map(np.asarray, (p, a, b))
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab))), t


def _box_dist(d: Dimension, p) -> float:
    """Distance d'un point à la boîte du texte (0 si dedans)."""
    x0, y0, x1, y1 = d.bbox
    dx = max(x0 - p[0], 0, p[0] - x1)
    dy = max(y0 - p[1], 0, p[1] - y1)
    return float(math.hypot(dx, dy))


def _diag(p: Perception) -> float:
    pts = np.array([v.px for v in p.vertices])
    return float(np.linalg.norm(pts.max(0) - pts.min(0)))


def _inside(p: Perception, pt) -> bool:
    import cv2
    poly = np.array([v.px for v in p.vertices], np.float32)
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) > 0


def _edge_cost(p: Perception, d: Dimension, eid: str) -> float:
    e = next(e for e in p.edges if e.id == eid)
    vm = {v.id: v.px for v in p.vertices}
    a, b = np.array(vm[e.a]), np.array(vm[e.b])
    dist, t = _seg_dist(d.center, a, b)
    return dist + 0.35 * abs(t - 0.5) * float(np.linalg.norm(b - a))


# --------------------------------------------------------------------------- heuristique

def heuristic_assign(p: Perception, exclude: Optional[set[str]] = None) -> dict[str, str]:
    """Rattache chaque cote à l'élément le plus proche (affectation optimale par type)."""
    exclude = exclude or set()
    diag = _diag(p)
    vm = {v.id: v.px for v in p.vertices}
    out: dict[str, str] = {}
    dims = [d for d in p.dimensions if d.id not in exclude]

    # cotes ambiguës (« 012 ») : diamètre si un cercle est plus proche qu'une arête
    resolved: list[tuple[Dimension, str]] = []
    for d in dims:
        kind = d.kind
        if kind == "ambiguous":
            de = min((_edge_cost(p, d, e.id) for e in p.edges), default=1e9)
            dc = min((max(0.0, math.dist(d.center, c.px) - c.r_px) for c in p.circles), default=1e9)
            kind = "diameter" if dc < de else "length"
        elif kind == "length" and p.circles and _inside(p, d.center):
            # nombre seul écrit dans la matière, plus près d'un perçage que d'une arête : Ø oublié/mal lu
            de = min(_edge_cost(p, d, e.id) for e in p.edges)
            dc = min(max(0.0, math.dist(d.center, c.px) - c.r_px) for c in p.circles)
            if dc < de:
                kind = "diameter"
                d.kind = "ambiguous"  # sera retypé plus bas
        resolved.append((d, kind))

    def assign(kind: str, targets: list[str], cost_fn, max_cost: float):
        ds = [d for d, k in resolved if k == kind]
        if not ds or not targets:
            return
        C = np.array([[cost_fn(d, t) for t in targets] for d in ds])
        rows, cols = linear_sum_assignment(C)
        for r, c in zip(rows, cols):
            if C[r, c] <= max_cost:
                out[ds[r].id] = targets[c]

    assign("length", [e.id for e in p.edges], lambda d, t: _edge_cost(p, d, t), 0.6 * diag)
    assign("angle", [v.id for v in p.vertices], lambda d, t: _box_dist(d, vm[t]), 0.4 * diag)
    assign("radius", [v.id for v in p.vertices], lambda d, t: _box_dist(d, vm[t]), 0.4 * diag)
    cm = {c.id: c for c in p.circles}
    assign("diameter", list(cm), lambda d, t: max(0.0, _box_dist(d, cm[t].px) - cm[t].r_px), 0.5 * diag)
    for d, k in resolved:
        if k == "thickness":
            out[d.id] = "PART"
    # mémorise le type retenu pour les ambiguës
    for d, k in resolved:
        if d.kind == "ambiguous" and d.id in out:
            d.kind = k  # type: ignore[assignment]
    return out


# --------------------------------------------------------------------------- VLM

_SCHEMA = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target": {"type": "string"},
                    "kind": {"type": "string",
                             "enum": ["length", "angle", "radius", "diameter", "thickness"]},
                    "value": {"type": "number"},
                },
                "required": ["text", "target", "kind", "value"],
            },
        },
        "extra": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["perpendicular", "parallel"]},
                    "edges": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["type", "edges"],
            },
        },
    },
    "required": ["assignments"],
}

_SYSTEM = (
    "Tu es un dessinateur industriel. Tu lis des croquis mécaniques faits à main levée. "
    "Le croquis n'est PAS à l'échelle : ne déduis jamais une cote de la longueur dessinée. "
    "Tu réponds uniquement en JSON conforme au schéma."
)


def _prompt(p: Perception, hint: str = "") -> str:
    lines = ["Image : croquis annoté (Set-of-Mark).",
             "- Sommets du contour (rouge) : " + ", ".join(f"{v.id}({v.px[0]:.0f},{v.px[1]:.0f})" for v in p.vertices),
             "- Arêtes (vert) : " + ", ".join(f"{e.id}={e.a}-{e.b}" for e in p.edges),
             "- Perçages (violet) : " + (", ".join(f"{c.id}({c.px[0]:.0f},{c.px[1]:.0f}) r={c.r_px:.0f}px" for c in p.circles) or "aucun"),
             "- Textes lus par OCR (orange) :"]
    for d in p.dimensions:
        lines.append(f"  {d.id}: \"{d.raw}\" -> {d.kind} {d.value:g} en ({d.center[0]:.0f},{d.center[1]:.0f})")
    lines += [
        "",
        "Pour CHAQUE texte T*, indique l'élément qu'il cote :",
        "- longueur -> une arête E* (suis les lignes de cote et les flèches, pas seulement la proximité) ;",
        "- angle ou rayon R -> un sommet V* ;",
        "- diamètre Ø -> un perçage C* ;",
        "- épaisseur (ep, e=, t=) -> target \"PART\".",
        "Corrige 'value' uniquement si l'OCR a manifestement mal lu le chiffre sur l'image.",
        "Dans 'extra', ajoute les perpendicularités ou parallélismes évidents non cotés.",
    ]
    if hint:
        lines += ["", "ATTENTION, une vérification précédente a signalé : " + hint]
    return "\n".join(lines)


def vlm_assign(p: Perception, som_path: Path, client: Ollama, model: str,
               hint: str = "", retries: int = 2) -> tuple[dict[str, str], list[dict], dict[str, float]]:
    """Retourne (rattachements valides, contraintes extra, valeurs corrigées)."""
    ids = {e.id for e in p.edges} | {v.id for v in p.vertices} | {c.id for c in p.circles}
    kinds_ok = {"length": "E", "angle": "V", "radius": "V", "diameter": "C"}
    prompt = _prompt(p, hint)
    last_err = ""
    for attempt in range(retries):
        try:
            q = prompt if not last_err else prompt + f"\n\nTa réponse précédente était invalide : {last_err}"
            res = client.chat_json(model, q, _SCHEMA, images=[som_path], system=_SYSTEM)
            out, fixes = {}, {}
            dmap = {d.id: d for d in p.dimensions}
            for a in res.get("assignments", []):
                t, tgt, kind = a.get("text"), a.get("target"), a.get("kind")
                if t not in dmap:
                    continue
                if kind == "thickness" and tgt == "PART":
                    out[t] = "PART"
                elif tgt in ids and kinds_ok.get(kind, "?") == tgt[0]:
                    out[t] = tgt
                else:
                    continue
                d = dmap[t]
                if d.kind in ("ambiguous",) or d.kind != kind:
                    d.kind = kind
                v = a.get("value")
                if isinstance(v, (int, float)) and v > 0 and abs(v - d.value) > 1e-9:
                    fixes[t] = float(v)
            extra = [x for x in res.get("extra", []) if all(e in ids for e in x.get("edges", []))]
            return out, extra, fixes
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)[:300]
            log.warning("VLM tentative %d échouée : %s", attempt + 1, last_err)
    raise RuntimeError(f"VLM indisponible ou réponses invalides : {last_err}")


# --------------------------------------------------------------------------- graphe

def build_graph(p: Perception, assign: dict[str, str], extra: Optional[list[dict]] = None,
                fixes: Optional[dict[str, float]] = None, method: str = "heuristic",
                default_thickness: float = 5.0) -> Graph:
    fixes = fixes or {}
    vm = {v.id: np.array(v.px) for v in p.vertices}
    cons: list[Constraint] = []
    thickness: Optional[float] = None
    warnings: list[str] = []
    for d in p.dimensions:
        tgt = assign.get(d.id)
        if tgt is None:
            continue
        val = fixes.get(d.id, d.value)
        if d.id in fixes:
            warnings.append(f"{d.id} : valeur OCR {d.value:g} corrigée en {val:g} par le VLM.")
        if d.kind == "length":
            cons.append(Constraint(type="length", edge=tgt, value=val, source=d.id))
        elif d.kind == "angle":
            cons.append(Constraint(type="angle", vertex=tgt, value=val, source=d.id))
        elif d.kind == "radius":
            cons.append(Constraint(type="fillet", vertex=tgt, value=val, source=d.id))
        elif d.kind == "diameter":
            cons.append(Constraint(type="diameter", circle=tgt, value=val, source=d.id))
        elif d.kind == "thickness":
            thickness = val
    # contraintes implicites : arêtes quasi horizontales / verticales
    for e in p.edges:
        v = vm[e.b] - vm[e.a]
        ang = math.degrees(math.atan2(-v[1], v[0])) % 180
        if min(ang, 180 - ang) < HV_TOL_DEG:
            cons.append(Constraint(type="horizontal", edge=e.id, implicit=True))
        elif abs(ang - 90) < HV_TOL_DEG:
            cons.append(Constraint(type="vertical", edge=e.id, implicit=True))
    for x in extra or []:
        cons.append(Constraint(type=x["type"], edges=x["edges"][:2], implicit=True, note="VLM"))
    if thickness is None:
        thickness = default_thickness
        warnings.append(f"Aucune épaisseur lue : {default_thickness:g} mm par défaut.")
    unassigned = [d.id for d in p.dimensions if d.id not in assign]
    for u in unassigned:
        d = next(d for d in p.dimensions if d.id == u)
        warnings.append(f"Cote {u} « {d.raw} » non rattachée.")
    return Graph(vertices=p.vertices, edges=p.edges, circles=p.circles, constraints=cons,
                 thickness=thickness, unassigned=unassigned, warnings=warnings, method=method)


def extract_graph(p: Perception, out_dir: Path, client: Optional[Ollama], model: Optional[str],
                  default_thickness: float, hint: str = "",
                  forced: Optional[dict[str, str]] = None) -> Graph:
    """Exécute l'agent 2 (VLM si disponible, sinon heuristique)."""
    method, extra, fixes = "heuristic", [], {}
    assign: dict[str, str] = {}
    if client is not None and model:
        try:
            assign, extra, fixes = vlm_assign(p, out_dir / "som.png", client, model, hint)
            method = "vlm"
        except RuntimeError as exc:
            log.warning("%s — repli sur l'heuristique.", exc)
    # le repli heuristique complète ce que le VLM n'a pas rattaché
    missing = {d.id for d in p.dimensions} - set(assign)
    if missing:
        h = heuristic_assign(p, exclude=set(assign))
        assign.update({k: v for k, v in h.items() if k in missing})
    if forced:
        assign.update(forced)
    g = build_graph(p, assign, extra, fixes, method, default_thickness)
    (out_dir / "graph.json").write_text(g.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
    log.info("Graphe (%s) : %d contraintes", method, len(g.constraints))
    return g


def load_graph(path: Path) -> Graph:
    return Graph.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
