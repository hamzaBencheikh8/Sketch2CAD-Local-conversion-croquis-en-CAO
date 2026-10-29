"""Primitives 3D dessinées en perspective (v1.1 : cylindre).

Un cylindre dessiné à main levée = deux génératrices droites quasi parallèles + une
ellipse (ou demi-ellipse) à chaque extrémité. La détection est géométrique et tolère
un trait fin et fragmenté (stylo pâle, papier froissé) : segments par transformée de
Hough probabiliste, puis ajustement d'ellipses sur les traits au voisinage des extrémités.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import cv2
import numpy as np

from .schema import Dimension

log = logging.getLogger(__name__)


def _clean(binary: np.ndarray) -> np.ndarray:
    """Retire les points isolés (grain du papier, poussières)."""
    H, W = binary.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep = np.zeros_like(binary)
    lim = 0.02 * max(H, W)
    for i in range(1, n):
        if max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) >= lim:
            keep[lab == i] = 255
    return keep


def _vertical_lines(strokes: np.ndarray) -> list[dict]:
    """Segments quasi verticaux fusionnés : [{'x', 'y0', 'y1', 'len'}]."""
    H, W = strokes.shape
    S = max(H, W)
    segs = cv2.HoughLinesP(strokes, 1, np.pi / 180, threshold=25, minLineLength=int(0.04 * S),
                           maxLineGap=int(0.05 * S))
    if segs is None:
        return []
    vs = []
    for x1, y1, x2, y2 in segs.reshape(-1, 4):
        ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if abs(ang - 90) < 14:
            vs.append((min(y1, y2), max(y1, y2), (x1 + x2) / 2))
    # regroupe les segments alignés (même x à ±2 % de l'image)
    vs.sort(key=lambda s: s[2])
    groups: list[list] = []
    for s in vs:
        if groups and abs(s[2] - np.mean([g[2] for g in groups[-1]])) < 0.02 * S:
            groups[-1].append(s)
        else:
            groups.append([s])
    out = []
    for g in groups:
        y0, y1 = min(s[0] for s in g), max(s[1] for s in g)
        cover = sum(s[1] - s[0] for s in g)
        out.append({"x": float(np.mean([s[2] for s in g])), "y0": float(y0), "y1": float(y1),
                    "len": float(y1 - y0), "ink": float(min(cover, y1 - y0))})
    return [o for o in out if o["len"] > 0.08 * S]


def _cap(strokes: np.ndarray, xl: float, xr: float, y: float, side: str) -> Optional[dict]:
    """Ellipse d'extrémité autour de la hauteur y, entre les génératrices xl et xr."""
    D = xr - xl
    ys, xs = np.nonzero(strokes)
    band_x = (xs > xl - 0.06 * D) & (xs < xr + 0.06 * D)
    band_y = (ys > y - 0.35 * D) & (ys < y + 0.35 * D)
    # on exclut les génératrices elles-mêmes
    not_side = (np.abs(xs - xl) > 0.06 * D) & (np.abs(xs - xr) > 0.06 * D)
    sel = band_x & band_y & not_side
    px, py = xs[sel].astype(np.float32), ys[sel].astype(np.float32)
    if px.size < 30:
        return None
    bins = np.histogram(px, bins=10, range=(xl, xr))[0]
    coverage = float(np.mean(bins > 0))
    spread = float(np.std(py))
    try:
        (cx, cy), (a1, a2), ang = cv2.fitEllipse(np.stack([px, py], 1))
    except cv2.error:
        return None
    major, minor = max(a1, a2), min(a1, a2)
    # part des points du trait qui tombent sur l'ellipse ajustée : une vraie base elliptique
    # s'ajuste bien, un mélange de bords droits et de lignes de cote, non
    t = math.radians(ang)
    dx, dy = px - cx, py - cy
    u = dx * math.cos(t) + dy * math.sin(t)
    v = -dx * math.sin(t) + dy * math.cos(t)
    r = np.sqrt((u / (a1 / 2 + 1e-9)) ** 2 + (v / (a2 / 2 + 1e-9)) ** 2)
    inliers = float(np.mean(np.abs(r - 1) < 0.12))
    # part du trait au-delà de l'extrémité des génératrices, côté intérieur : une base du haut
    # est vue entière (son arc avant passe SOUS le haut des génératrices) ; un dôme non
    inner = float(np.mean(py > y + 0.03 * D)) if side == "haut" else float(np.mean(py < y - 0.03 * D))
    return {"inner": inner, "cx": float(cx), "cy": float(cy), "major": float(major), "minor": float(minor),
            "angle": float(ang), "coverage": coverage, "spread": spread, "n": int(px.size), "side": side,
            "inliers": inliers, "axes": [float(a1), float(a2)]}


def detect_cylinder(binary: np.ndarray) -> Optional[dict]:
    """Retourne les mesures px du cylindre, ou None si le dessin n'en est pas un."""
    strokes = _clean(binary)
    lines = _vertical_lines(strokes)
    best = None
    for i, L in enumerate(lines):
        for R in lines[i + 1:]:
            D = R["x"] - L["x"]
            ov = min(L["y1"], R["y1"]) - max(L["y0"], R["y0"])
            span = max(L["len"], R["len"])
            if D < 0.25 * span or ov < 0.6 * min(L["len"], R["len"]):
                continue
            if min(L["len"], R["len"]) < 0.6 * span or L["ink"] < 0.5 * L["len"] or R["ink"] < 0.5 * R["len"]:
                continue
            top = _cap(strokes, L["x"], R["x"], min(L["y0"], R["y0"]), "haut")
            bot = _cap(strokes, L["x"], R["x"], max(L["y1"], R["y1"]), "bas")
            if not top or not bot:
                continue
            xm = (L["x"] + R["x"]) / 2
            ok = all(c["coverage"] >= 0.6 and c["inliers"] >= 0.5 and abs(c["cx"] - xm) < 0.2 * D
                     and 0.75 * D < c["major"] < 1.35 * D and 0.1 * D < c["minor"] < 0.8 * D
                     for c in (top, bot))
            if top["inner"] < 0.15:
                ok = False  # sommet en dôme (gélule, ogive) : corps de révolution, pas cylindre
            # les deux bases doivent être distinctes (un cercle seul donne deux « ellipses » superposées)
            if (bot["cy"] - top["cy"]) < 0.5 * (top["minor"] + bot["minor"]):
                ok = False
            if not ok:
                continue
            score = (top["inliers"] + bot["inliers"]) * span
            if best is None or score > best["score"]:
                best = {"left": L, "right": R, "top": top, "bottom": bot, "score": score,
                        "D_px": float(D), "H_px": float(bot["cy"] - top["cy"])}
    if best:
        log.info("Cylindre détecté : Ø %.0f px, hauteur %.0f px", best["D_px"], best["H_px"])
    return best


def assign_cylinder_dims(cyl: dict, dims: list[Dimension]) -> tuple[dict, list[str], list[str]]:
    """Rattache les cotes au rayon / diamètre / hauteur. Retourne (valeurs mm, avertissements, conflits)."""
    L, R, top, bot = cyl["left"], cyl["right"], cyl["top"], cyl["bottom"]
    vals: dict[str, tuple[float, str]] = {}
    warns, confl = [], []
    for d in dims:
        cx, cy = d.center
        D_px = cyl["D_px"]
        between = L["x"] - 0.05 * D_px <= cx <= R["x"] + 0.05 * D_px
        in_cap = between and any(abs(cy - c["cy"]) < max(c["minor"], 0.25 * D_px) for c in (top, bot))
        beside = not between
        if d.kind == "radius":
            key, v = "diameter", 2 * d.value
        elif d.kind == "diameter":
            key, v = "diameter", d.value
        elif d.kind in ("length", "ambiguous"):
            if beside:
                key, v = "height", d.value
            elif in_cap:
                key, v = "diameter", d.value
                if d.kind == "length":
                    warns.append(f"{d.id} « {d.raw} » dans une base : lue comme le diamètre.")
            else:
                key, v = "height", d.value
            if d.kind == "ambiguous":
                warns.append(f"{d.id} « {d.raw} » : lu comme {'la hauteur' if key == 'height' else 'le diamètre'} "
                             f"(position sur le dessin).")
        else:
            continue
        if key in vals and abs(vals[key][0] - v) > 1e-6:
            confl.append(f"{key} coté deux fois : {vals[key][0]:g} mm ({vals[key][1]}) et {v:g} mm ({d.id})")
            continue
        vals[key] = (v, d.id)
    out = {"diameter": vals.get("diameter", (None, None))[0], "height": vals.get("height", (None, None))[0],
           "sources": {k: v[1] for k, v in vals.items()}}
    ratio = cyl["H_px"] / cyl["D_px"]
    if out["diameter"] is None and out["height"] is not None:
        out["diameter"] = round(out["height"] / ratio, 1)
        warns.append(f"Diamètre non coté : {out['diameter']:g} mm d'après les proportions du dessin.")
    elif out["height"] is None and out["diameter"] is not None:
        out["height"] = round(out["diameter"] * ratio, 1)
        warns.append(f"Hauteur non cotée : {out['height']:g} mm d'après les proportions du dessin.")
    elif out["height"] is None and out["diameter"] is None:
        out["diameter"], out["height"] = round(cyl["D_px"] * 0.25, 1), round(cyl["H_px"] * 0.25, 1)
        warns.append("Aucune cote lue : dimensions arbitraires (0,25 mm/px).")
    else:
        drawn = ratio
        written = out["height"] / out["diameter"]
        if not 0.33 < written / drawn < 3:
            warns.append(f"Proportions dessinées (H/Ø = {drawn:.2f}) très différentes des cotes "
                         f"(H/Ø = {written:.2f}) : vérifier la lecture des cotes.")
    return out, warns, confl


def draw_cylinder_som(img: np.ndarray, cyl: dict, dims: list[Dimension]) -> np.ndarray:
    out = img.copy()
    sc = max(0.5, max(out.shape[:2]) / 1600)
    th = max(2, int(round(3 * sc)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for s in ("left", "right"):
        L = cyl[s]
        cv2.line(out, (int(L["x"]), int(L["y0"])), (int(L["x"]), int(L["y1"])), (0, 160, 0), th)
    for c in (cyl["top"], cyl["bottom"]):
        cv2.ellipse(out, ((c["cx"], c["cy"]), tuple(c["axes"]), c["angle"]), (200, 0, 200), th)
    for d in dims:
        x0, y0, x1, y1 = (int(v) for v in d.bbox)
        cv2.rectangle(out, (x0, y0), (x1, y1), (200, 120, 0), th)
        cv2.putText(out, d.id, (x0, y0 - 6), font, 0.7 * sc + 0.3, (200, 120, 0), th, cv2.LINE_AA)
    xm = int((cyl["left"]["x"] + cyl["right"]["x"]) / 2)
    cv2.putText(out, f"CYLINDRE  D={cyl['D_px']:.0f}px  H={cyl['H_px']:.0f}px", (int(cyl["left"]["x"]),
                int(cyl["bottom"]["cy"] + cyl["bottom"]["minor"] / 2 + 40 * sc)), font, 0.7 * sc + 0.3,
                (0, 0, 200), th, cv2.LINE_AA)
    _ = xm
    return out
