"""Corps de révolution dessinés en perspective : demi-profil -> solveur -> révolution.

Couvre cylindre, cône, tronc de cône, sphère/hémisphère, gélule, bouteille... : tout solide
dont la silhouette est symétrique par rapport à un axe vertical. Indice 3D exigé : au moins
une ellipse (base vue en perspective) ou un dôme ; sinon le dessin reste une pièce plate.

Le demi-profil (côté droit de l'axe) devient un polygone fermé par l'axe :
    A (axe, bas) -> B (bord de la base) -> ... -> D (fin de dôme) -> E (axe, haut) -> A
résolu par le même solveur que les pièces plates, puis tourné de 360° autour de l'axe.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import cv2
import numpy as np

from .primitives import _clean
from .schema import Constraint, Dimension, Edge, Graph, Vertex

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- détection

def _silhouette(strokes: np.ndarray) -> Optional[np.ndarray]:
    H, W = strokes.shape
    k = max(5, int(0.012 * max(H, W))) | 1
    closed = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    fill = np.zeros_like(strokes)
    if cv2.contourArea(c) < 0.25 * w * h:
        # contour pas tout à fait fermé (coin non rejoint, base non tracée) : enveloppe convexe
        pts = cv2.findNonZero(strokes)
        if pts is None or len(pts) < 20:
            return None
        hull = cv2.convexHull(pts)
        cv2.drawContours(fill, [hull], -1, 255, -1)
        return fill
    cv2.drawContours(fill, [c], -1, 255, -1)
    return fill


def _ellipses(strokes: np.ndarray, fill: np.ndarray) -> list[dict]:
    """Bases elliptiques : régions fermées intérieures, plus larges que hautes."""
    kc = max(5, int(0.012 * max(strokes.shape))) | 1   # referme les petites ellipses mal jointes
    closed = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    inner = cv2.bitwise_and(fill, cv2.bitwise_not(cv2.dilate(closed, np.ones((3, 3), np.uint8))))
    n, lab, st, _ = cv2.connectedComponentsWithStats(inner, 8)
    fx, fy, fw, fh = cv2.boundingRect(fill)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < 0.15 * fw or a < 50:
            continue
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = cnts[0]
        if len(c) < 5:
            continue
        (cx, cy), (a1, a2), ang = cv2.fitEllipse(c)
        major, minor = max(a1, a2), min(a1, a2)
        horiz = (a1 >= a2 and (ang < 25 or ang > 155)) or (a2 > a1 and 65 < ang < 115)
        # la région doit vraiment être une ellipse (une bande rectangulaire entre deux traits, non)
        em = np.zeros_like(fill)
        cv2.ellipse(em, ((cx, cy), (a1, a2), ang), 255, -1)
        reg = (lab == i)
        iou = np.logical_and(reg, em > 0).sum() / max(np.logical_or(reg, em > 0).sum(), 1)
        if horiz and minor < 0.65 * major and iou > 0.85 and h < 0.45 * fh:
            out.append({"cx": float(cx), "cy": float(cy), "major": float(major + 2), "minor": float(minor + 2),
                        "top": float(y), "bottom": float(y + h)})
    return out


def detect_revolution(strokes_in: np.ndarray) -> Optional[dict]:
    strokes = _clean(strokes_in)
    fill = _silhouette(strokes)
    if fill is None:
        return None
    ys, xs = np.nonzero(fill)
    y0, y1 = int(ys.min()), int(ys.max())
    H = y1 - y0
    rows = []
    for y in range(y0, y1 + 1):
        xx = np.nonzero(fill[y])[0]
        if xx.size:
            rows.append((y, float(xx.min()), float(xx.max())))
    rows = np.array(rows)
    centers = (rows[:, 1] + rows[:, 2]) / 2
    x_axis = float(np.median(centers))
    # symétrie : IoU silhouette / miroir autour de l'axe
    mirror = np.zeros_like(fill)
    for y, xl, xr in rows:
        ml, mr = 2 * x_axis - xr, 2 * x_axis - xl
        mirror[int(y), max(0, int(ml)):max(0, int(mr)) + 1] = 255
    inter = np.logical_and(fill > 0, mirror > 0).sum()
    union = np.logical_or(fill > 0, mirror > 0).sum()
    sym = inter / max(union, 1)
    ells = [e for e in _ellipses(strokes, fill) if abs(e["cx"] - x_axis) < 0.15 * (rows[:, 2] - rows[:, 1]).max()]
    if sym < 0.82:
        return None
    # profil r(y) lissé
    r = (rows[:, 2] - rows[:, 1]) / 2
    r = np.convolve(np.pad(r, 3, mode="edge"), np.ones(7) / 7, mode="valid")
    y_top, y_bot = float(rows[0, 0]), float(rows[-1, 0])
    top_ell = next((e for e in ells if e["top"] - y_top < 0.12 * H), None)
    bot_ell = next((e for e in ells if y_bot - e["bottom"] < 0.12 * H), None)
    # les bases vues en perspective : la face plane est au niveau du centre de l'ellipse
    cut_top = top_ell["cy"] if top_ell else y_top
    cut_bot = bot_ell["cy"] if bot_ell else y_bot
    if not bot_ell:
        # base plane tracée de biais (photo inclinée) : la face est là où le profil atteint sa largeur
        n_ = len(r)
        tail = r[int(0.8 * n_):]
        if tail.size:
            ref = 0.9 * tail.max()
            i = n_ - 1
            while i > 0 and r[i] < ref:
                i -= 1
            cut_bot = float(rows[i, 0])
    # les lignes couvertes par une ellipse de base appartiennent à l'ellipse, pas au flanc
    lo = cut_top + (top_ell["minor"] / 2 if top_ell else 0)
    hi = cut_bot - (bot_ell["minor"] / 2 if bot_ell else 0)
    sel = (rows[:, 0] >= lo) & (rows[:, 0] <= hi)
    prof_y, prof_r = rows[sel, 0], r[sel]
    if top_ell:
        prof_y, prof_r = np.r_[cut_top, prof_y], np.r_[top_ell["major"] / 2, prof_r]
    if bot_ell:
        prof_y, prof_r = np.r_[prof_y, cut_bot], np.r_[prof_r, bot_ell["major"] / 2]
    dome = None
    if not top_ell:
        dome = _fit_dome(prof_y, prof_r)
    has_3d_cue = bool(ells) or dome is not None
    if not has_3d_cue:
        return None
    return {"x_axis": x_axis, "symmetry": float(sym), "y_top": cut_top, "y_bot": cut_bot,
            "profile_y": prof_y.tolist(), "profile_r": prof_r.tolist(), "top_ellipse": top_ell,
            "bottom_ellipse": bot_ell, "dome": dome, "height_px": float(cut_bot - cut_top)}


def _fit_dome(py: np.ndarray, pr: np.ndarray) -> Optional[dict]:
    """Calotte en haut du profil : de l'axe jusqu'à l'endroit où le flanc devient vertical,
    ajustée par un cercle centré sur l'axe : r² + (y - yc)² = ρ²."""
    n = len(py)
    if n < 20 or pr[0] > 0.35 * pr.max():
        return None  # le profil ne part pas de l'axe : sommet plat, pas de dôme
    sm = np.convolve(np.pad(pr, 4, mode="edge"), np.ones(9) / 9, mode="valid")
    slope = np.abs(np.gradient(sm, py))
    k = None
    for i in range(5, n - 5):
        if sm[i] > 0.6 * pr.max() and np.all(slope[i:i + 5] < 0.15):
            k = i
            break
    if k is None or k < 6:
        return None
    yy, rr = py[:k + 1], pr[:k + 1]
    A = np.stack([2 * yy, np.ones_like(yy)], 1)
    sol, *_ = np.linalg.lstsq(A, rr ** 2 + yy ** 2, rcond=None)
    yc = sol[0]
    rho2 = sol[1] + yc ** 2
    if rho2 <= 0:
        return None
    rho = math.sqrt(rho2)
    res = np.abs(np.sqrt(rr ** 2 + (yy - yc) ** 2) - rho)
    if res.mean() > 0.05 * rho or not (0.85 < rho / max(rr[-1], 1e-9) < 1.3):
        return None  # pas un dôme sphérique qui raccorde le flanc
    return {"k": int(k + 1), "yc": float(yc), "rho": float(rho), "end_y": float(yy[-1]), "end_r": float(rr[-1])}


# --------------------------------------------------------------------------- demi-profil

def half_profile(rev: dict) -> tuple[list[tuple[float, float]], Optional[int]]:
    """Polygone CCW (px, y vers le bas) du demi-profil + index de l'arête « dôme » (D -> E)."""
    xa = rev["x_axis"]
    py, pr = np.array(rev["profile_y"]), np.array(rev["profile_r"])
    dome = rev["dome"]
    start = dome["k"] - 1 if dome else 0
    # partie droite du profil (hors dôme), simplifiée
    curve = np.stack([xa + pr[start:], py[start:]], 1).astype(np.float32)
    eps = max(0.025 * rev["height_px"], 3.0)
    simp = cv2.approxPolyDP(curve.reshape(-1, 1, 2), eps, False).reshape(-1, 2)
    simp = [tuple(map(float, p)) for p in simp]
    # de bas en haut pour un parcours CCW (en y-haut)
    side = simp[::-1]
    y_bot, y_top = rev["y_bot"], rev["y_top"]
    pts: list[tuple[float, float]] = [(xa, y_bot)]
    if side[0][0] - xa > 0.03 * rev["height_px"]:
        pts.append((side[0][0], y_bot))
    pts += side[1:]
    # sommets presque alignés (trait tremblé) : supprimés
    i = 1
    while i < len(pts) - 1:
        a, b, c = np.array(pts[i - 1]), np.array(pts[i]), np.array(pts[i + 1])
        u, v = b - a, c - b
        cosang = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
        if cosang > math.cos(math.radians(15)) and i != 1:
            pts.pop(i)
        else:
            i += 1
    # segments trop courts (bavure de trait aux coins) : sommet supprimé
    lim = max(0.06 * rev["height_px"], 8.0)
    i = 2
    while i < len(pts) - 1:
        if math.dist(pts[i - 1], pts[i]) < lim or (i == len(pts) - 2 and math.dist(pts[i], pts[i + 1]) < lim
                                                   and not dome):
            pts.pop(i)
        else:
            i += 1
    if len(pts) > 2 and math.dist(pts[1], pts[2]) < lim:
        pts.pop(2)
    arc_edge = None
    if dome:
        arc_edge = len(pts) - 1  # arête du dernier point du profil vers le sommet E
        pts.append((xa, y_top))
    else:
        if pts[-1][0] - xa > 0.03 * rev["height_px"]:
            pts[-1] = (pts[-1][0], y_top)
        pts.append((xa, y_top))
    return pts, arc_edge


def _seg_cost(p, a, b):
    p, a, b = map(np.asarray, (p, a, b))
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab))) + 0.3 * abs(t - 0.5) * float(np.linalg.norm(ab))


def build_revolution_graph(rev: dict, dims: list[Dimension], default_unit_note: str = "") -> tuple[Graph, list[str]]:
    pts, arc_edge = half_profile(rev)
    xa = rev["x_axis"]
    n = len(pts)
    V = [Vertex(id=f"V{i + 1}", px=p) for i, p in enumerate(pts)]
    E = [Edge(id=f"E{i + 1}", a=V[i].id, b=V[(i + 1) % n].id) for i in range(n)]
    axis_e = n - 1                           # E_n : sommet haut -> sommet bas (axe)
    cons: list[Constraint] = [Constraint(type="vertical", edge=E[axis_e].id, note="axe de révolution")]
    end_edges = []                           # faces planes (bas / haut) : horizontales
    if pts[1][1] == pts[0][1] and pts[1][0] != xa:
        end_edges.append(0)
    if arc_edge is None and n >= 3 and pts[-2][1] == pts[-1][1]:
        end_edges.append(n - 2)
    for k in end_edges:
        cons.append(Constraint(type="horizontal", edge=E[k].id, note="face plane"))
    warns = []
    profile_edges = [k for k in range(n - 1) if k not in end_edges and k != arc_edge]
    for k in profile_edges:
        a, b = np.array(pts[k]), np.array(pts[k + 1])
        ang = math.degrees(math.atan2(-(b - a)[1], (b - a)[0])) % 180
        if abs(ang - 90) < 6:
            cons.append(Constraint(type="vertical", edge=E[k].id, implicit=True))
        elif min(ang, 180 - ang) < 6:
            cons.append(Constraint(type="horizontal", edge=E[k].id, implicit=True))
    if arc_edge is not None:
        cons.append(Constraint(type="arc_tangent", edge=E[arc_edge].id, implicit=True, note="dôme hémisphérique"))

    # ---- rattachement des cotes (textes de gauche ramenés à droite par symétrie)
    ells = [e for e in (rev["top_ellipse"], rev["bottom_ellipse"]) if e]
    used_edges: set[int] = set()
    thickness = None
    for d in dims:
        if d.kind == "unknown":
            continue
        cx, cy = d.center
        mx = xa + abs(cx - xa)
        near_ell = None
        for e in ells:
            if abs(cy - e["cy"]) < max(e["minor"], 0.06 * rev["height_px"]) + 0.5 * (d.bbox[3] - d.bbox[1]) \
                    and abs(mx - xa) < e["major"] / 2 + 0.25 * rev["height_px"]:
                near_ell = e
        if d.kind == "angle":
            cands = [i for i in range(1, n - 1) if i != len(pts) - 1]
            k = min(cands, key=lambda i: math.dist((mx, cy), pts[i]))
            cons.append(Constraint(type="angle", vertex=V[k].id, value=d.value, source=d.id))
            continue
        if d.kind == "radius" and arc_edge is not None and d.kind != "diameter":
            cons.append(Constraint(type="arc_radius", edge=E[arc_edge].id, value=d.value, source=d.id))
            continue
        if d.kind in ("diameter", "radius") or (d.kind in ("length", "ambiguous") and near_ell):
            val = d.value / 2 if d.kind != "radius" else d.value
            # bord plan le plus proche ; à défaut, rayon au sommet de profil le plus proche
            if end_edges:
                k = min(end_edges, key=lambda i: abs(cy - pts[i][1]))
                if k not in used_edges:
                    cons.append(Constraint(type="length", edge=E[k].id, value=val, source=d.id,
                                           note="rayon de la base (Ø/2)"))
                    used_edges.add(k)
                    continue
            k = min(range(1, n - 1), key=lambda i: math.dist((mx, cy), pts[i]))
            cons.append(Constraint(type="radius_at", vertex=V[k].id, value=val, source=d.id))
            continue
        if d.kind in ("length", "ambiguous"):
            free = [k for k in profile_edges if k not in used_edges] or profile_edges
            k = min(free, key=lambda i: _seg_cost((mx, cy), pts[i], pts[i + 1]))
            cons.append(Constraint(type="length", edge=E[k].id, value=d.value, source=d.id))
            used_edges.add(k)
            if d.kind == "ambiguous":
                warns.append(f"{d.id} « {d.raw} » : lu comme la longueur de {E[k].id} (position sur le dessin).")
            continue
        if d.kind == "thickness":
            thickness = d.value
    g = Graph(vertices=V, edges=E, constraints=cons, thickness=thickness or 0.0, method="revolution",
              unassigned=[d.id for d in dims if d.kind == "unknown"])
    return g, warns


def draw_revolution_som(img: np.ndarray, rev: dict, dims: list[Dimension]) -> np.ndarray:
    from .perception import _draw_texts
    out = img.copy()
    sc = max(0.5, max(out.shape[:2]) / 1600)
    th = max(2, int(round(3 * sc)))
    xa = rev["x_axis"]
    cv2.line(out, (int(xa), int(rev["y_top"] - 20)), (int(xa), int(rev["y_bot"] + 20)), (0, 160, 0), th)
    pts, arc = half_profile(rev)
    for i, p in enumerate(pts):
        q = pts[(i + 1) % len(pts)]
        col = (200, 0, 200) if i == arc else (0, 0, 220)
        cv2.line(out, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), col, th)
        cv2.putText(out, f"V{i + 1}", (int(p[0]) + 6, int(p[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * sc + 0.3, (0, 0, 200), th, cv2.LINE_AA)
    for e in (rev["top_ellipse"], rev["bottom_ellipse"]):
        if e:
            cv2.ellipse(out, (int(e["cx"]), int(e["cy"])), (int(e["major"] / 2), int(e["minor"] / 2)), 0, 0, 360,
                        (200, 0, 200), th)
    _draw_texts(out, dims)
    return out


def rev_polyline(points, dome_edge, step_deg: float = 4.0) -> list[tuple[float, float]]:
    """Demi-profil échantillonné (arc du dôme compris), pour aperçus et contrôles."""
    out = []
    n = len(points)
    for k in range(n):
        p0, p1 = points[k], points[(k + 1) % n]
        out.append(tuple(p0))
        if k == dome_edge:
            dx, dy = p0[0] - p1[0], p1[1] - p0[1]
            if dy <= 0:
                continue
            rho = (dx * dx + dy * dy) / (2 * dy)
            c = (p1[0], p1[1] - rho)
            a0 = math.atan2(p0[1] - c[1], p0[0] - c[0])
            a1 = math.atan2(p1[1] - c[1], p1[0] - c[0])
            m = max(2, int(abs(a1 - a0) / math.radians(step_deg)))
            for i in range(1, m):
                a = a0 + (a1 - a0) * i / m
                out.append((c[0] + rho * math.cos(a), c[1] + rho * math.sin(a)))
    return out
