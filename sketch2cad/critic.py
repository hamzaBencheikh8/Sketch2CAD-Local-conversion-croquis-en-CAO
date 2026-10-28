"""Agent 5 — Critique et auto-correction.

Compare le profil résolu au croquis original et propose des corrections :
- angle pris dans le mauvais sens (convexité inversée)   -> θ devient 360° − θ ;
- cote de longueur rattachée à la mauvaise arête          -> nouveau rattachement ;
- profil auto-sécant, perçage hors matière                -> signalés.
Une critique visuelle par VLM (optionnelle) ne fait que CONFIRMER ces signaux.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from .graph_builder import _edge_cost
from .llm import Ollama
from .profile import profile_polyline
from .schema import Graph, Perception, Solution
from .solver import interior_angle

log = logging.getLogger(__name__)


@dataclass
class Critique:
    ok: bool = True
    iou: float = 1.0
    flip_angles: list[str] = field(default_factory=list)        # ids de contraintes (source)
    reassign: dict[str, str] = field(default_factory=dict)      # T* -> E*
    issues: list[str] = field(default_factory=list)
    vlm: Optional[str] = None

    def actionable(self) -> bool:
        return bool(self.flip_angles or self.reassign)


def _drawn(g: Graph) -> np.ndarray:
    return np.array([[v.px[0], -v.px[1]] for v in g.vertices], float)


def _affine_iou(A: np.ndarray, B: np.ndarray) -> float:
    X = np.hstack([A, np.ones((len(A), 1))])
    M, *_ = np.linalg.lstsq(X, B, rcond=None)
    pa = Polygon(X @ M)
    pb = Polygon(B)
    if not pa.is_valid or not pb.is_valid:
        pa, pb = pa.buffer(0), pb.buffer(0)
    u = pa.union(pb).area
    return float(pa.intersection(pb).area / u) if u > 0 else 0.0


def critique(g: Graph, sol: Solution, p: Perception, iou_min: float = 0.6) -> Critique:
    cr = Critique()
    D = _drawn(g)
    S = np.array(sol.points)
    n = len(S)

    # 1. signature de convexité
    for k in range(n):
        dconv = interior_angle(D, k) < math.pi
        sconv = interior_angle(S, k) < math.pi
        if dconv != sconv:
            vid = g.vertices[k].id
            ang = [c for c in g.constraints if c.type == "angle" and c.vertex == vid]
            if ang:
                cr.flip_angles += [c.source or vid for c in ang]
                cr.issues.append(f"{vid} : dessiné {'convexe' if dconv else 'rentrant'}, résolu "
                                 f"{'convexe' if sconv else 'rentrant'} -> angle {ang[0].value:g}° inversé.")
            else:
                cr.issues.append(f"{vid} : convexité différente du dessin (cote voisine suspecte).")

    # 2. profil simple et orienté
    poly = Polygon(profile_polyline(sol.points, sol.fillets))
    if not poly.is_valid:
        cr.issues.append("Le profil résolu se recoupe (auto-intersection).")
    sa = 0.5 * float(np.sum(S[:, 0] * np.roll(S[:, 1], -1) - np.roll(S[:, 0], -1) * S[:, 1]))
    if sa < 0:
        cr.issues.append("Profil miroir (sens horaire).")

    # 3. ressemblance de forme + cotes de longueur suspectes
    cr.iou = _affine_iou(D, S) if poly.is_valid else 0.0
    lens = {}
    vidx = {v.id: i for i, v in enumerate(g.vertices)}
    for c in g.constraints:
        if c.type == "length" and c.edge and c.source:
            e = g.edge(c.edge)
            dl = float(np.linalg.norm(D[vidx[e.b]] - D[vidx[e.a]]))
            lens[c.source] = (c.edge, c.value, dl)
    if lens:
        med = float(np.median([v / dl for _, v, dl in lens.values()]))
        used = {e for e, _, _ in lens.values()}
        dmap = {d.id: d for d in p.dimensions}
        for tid, (eid, val, dl) in lens.items():
            dev = abs(math.log((val / dl) / med))
            if dev < math.log(3) or tid not in dmap:
                continue
            best = None
            for e in g.edges:
                if e.id in used:
                    continue
                dl2 = float(np.linalg.norm(D[vidx[e.b]] - D[vidx[e.a]]))
                dev2 = abs(math.log((val / dl2) / med))
                near = _edge_cost(p, dmap[tid], e.id) <= 2 * _edge_cost(p, dmap[tid], eid) + 1
                if dev2 < math.log(1.5) and near and (best is None or dev2 < best[1]):
                    best = (e.id, dev2)
            if best:
                cr.reassign[tid] = best[0]
                cr.issues.append(f"{tid} ({val:g}) : rapport cote/dessin aberrant sur {eid}, "
                                 f"rattachée à {best[0]}.")
            else:
                ratio = (val / dl) / med
                cr.issues.append(f"{tid} ({val:g}) sur {eid} : cote {ratio:.1f}× "
                                 f"{'plus grande' if ratio > 1 else 'plus petite'} que ne le suggère le dessin "
                                 f"(lecture OCR ou rattachement à vérifier).")
    if cr.iou < iou_min:
        cr.issues.append(f"Forme résolue éloignée du dessin (IoU {cr.iou:.2f} < {iou_min}).")

    # 4. perçages dans la matière, paroi >= 1 mm
    for i, (x, y, d) in enumerate(sol.holes):
        if poly.is_valid and not poly.buffer(-(d / 2 + 1.0)).contains(Point(x, y)):
            cr.issues.append(f"C{i + 1} (Ø{d:g}) déborde du contour ou paroi < 1 mm.")

    cr.ok = not cr.issues
    return cr


def render_comparison(sketch_bgr: np.ndarray, sol: Solution, path: Path) -> Path:
    """Image côte à côte croquis / profil résolu (pour l'utilisateur et le VLM)."""
    h = 600
    s = cv2.resize(sketch_bgr, (int(sketch_bgr.shape[1] * h / sketch_bgr.shape[0]), h))
    pts = np.array(profile_polyline(sol.points, sol.fillets))
    lo, hi = pts.min(0), pts.max(0)
    sc = (h - 80) / max(hi - lo)
    W = int((hi - lo)[0] * sc) + 80
    canvas = np.full((h, max(W, 300), 3), 255, np.uint8)

    def tr(p):
        return (int(40 + (p[0] - lo[0]) * sc), int(h - 40 - (p[1] - lo[1]) * sc))
    cv2.polylines(canvas, [np.array([tr(p) for p in pts], np.int32)], True, (40, 40, 40), 2, cv2.LINE_AA)
    for x, y, d in sol.holes:
        cv2.circle(canvas, tr((x, y)), int(d / 2 * sc), (180, 0, 180), 2, cv2.LINE_AA)
    for i, p in enumerate(sol.points):
        cv2.putText(canvas, f"V{i + 1}", tr(p), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
    out = np.hstack([s, np.full((h, 10, 3), 200, np.uint8), canvas])
    cv2.imencode(".png", out)[1].tofile(str(path))
    return path


def vlm_confirm(client: Ollama, model: str, image: Path, cr: Critique) -> Optional[str]:
    """Demande au VLM de confirmer (ou non) les problèmes géométriques détectés."""
    if not cr.issues:
        return None
    q = ("À gauche : croquis original. À droite : profil reconstruit (étiquettes V*). "
         "Pour chacun des points suivants, réponds OUI s'il est visible, NON sinon, une ligne par point :\n"
         + "\n".join(f"- {i}" for i in cr.issues))
    try:
        return client.chat(model, q, images=[image])
    except Exception as exc:  # noqa: BLE001
        log.warning("Critique VLM indisponible : %s", exc)
        return None
