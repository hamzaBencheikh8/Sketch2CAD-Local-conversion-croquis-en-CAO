"""Réservoir / cuve vertical vu de face (v1.3).

Un réservoir dessiné à main levée = deux génératrices verticales (le corps cylindrique), un fond
bombé au-dessus et un fond bombé en dessous, et des accessoires :

* ouvertures en façade (trou d'homme, piquage radial) = cercles dans le corps, une bride = deux
  cercles concentriques ;
* piquages axiaux = petits « ⊓ » posés sur le fond du haut (ou « ⊔ » sous le fond du bas) ;
* pieds = traits verticaux qui partent du fond du bas jusqu'au sol.

La détection est géométrique (comme le cylindre) ; les cotes donnent les vraies dimensions :
Ø du corps, hauteur droite du corps (entre les lignes de raccordement des fonds), R des fonds
(calotte sphérique), Ø des ouvertures, ep = épaisseur de paroi. Ce qui n'est pas coté est estimé
à l'échelle du dessin et signalé.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import cv2
import numpy as np

from .schema import Dimension

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- détection

def _first_hit(st: np.ndarray, x: int, y0: int, y1: int, half: int = 1) -> Optional[int]:
    """Premier pixel de trait dans la colonne x en allant de y0 vers y1 (inclus)."""
    H, W = st.shape
    x0, x1 = max(0, x - half), min(W, x + half + 1)
    step = 1 if y1 >= y0 else -1
    for y in range(y0, y1 + step, step):
        if 0 <= y < H and st[y, x0:x1].any():
            return y
    return None


def _fit_cap(xs: np.ndarray, ys: np.ndarray, xm: float) -> Optional[tuple[float, float, np.ndarray]]:
    """Cercle centré sur l'axe x = xm : (x-xm)² + (y-yc)² = ρ², robuste (rejet itératif)."""
    keep = np.ones(len(xs), bool)
    for _ in range(4):
        if keep.sum() < 8:
            return None
        X, Y = xs[keep], ys[keep]
        A = np.stack([2 * Y, np.ones_like(Y)], 1)
        b = (X - xm) ** 2 + Y ** 2
        (yc, c), *_ = np.linalg.lstsq(A, b, rcond=None)
        rho2 = c + yc ** 2
        if rho2 <= 0:
            return None
        rho = math.sqrt(rho2)
        res = np.abs(np.sqrt((xs - xm) ** 2 + (ys - yc) ** 2) - rho)
        lim = max(2.5 * float(np.median(res[keep])), 3.0)
        keep = res < lim
    return float(yc), float(rho), keep


def _head(st: np.ndarray, xl: float, xr: float, y_join: float, up: bool) -> Optional[dict]:
    """Fond bombé : profil du premier trait rencontré au-delà de la ligne de raccordement."""
    D = xr - xl
    xm = (xl + xr) / 2
    cols = np.linspace(xl + 0.06 * D, xr - 0.06 * D, 61).astype(int)
    start = int(y_join - 0.015 * D) if up else int(y_join + 0.015 * D)
    end = int(y_join - 1.1 * D) if up else int(y_join + 1.1 * D)
    pts = [(x, _first_hit(st, int(x), start, end)) for x in cols]
    pts = [(x, y) for x, y in pts if y is not None]
    if len(pts) < 0.7 * len(cols):
        return None
    xs, ys = np.array(pts, float).T
    fit = _fit_cap(xs, ys, xm)
    if fit is None:
        return None
    yc, rho, keep = fit
    if keep.mean() < 0.6:
        return None
    # convexité : le centre du cercle est du côté du corps
    if (up and yc <= y_join - 0.05 * D) or (not up and yc >= y_join + 0.05 * D):
        return None
    apex = yc - rho if up else yc + rho
    depth = (y_join - apex) if up else (apex - y_join)
    if not 0.1 * D <= depth <= 0.75 * D:
        return None
    # le fond rejoint les génératrices : aux bords, le profil est près de la ligne de raccordement
    edge = np.abs(xs[keep] - xm) > 0.38 * D
    if edge.any() and np.median(np.abs(ys[keep][edge] - y_join)) > 0.6 * depth + 0.05 * D:
        return None
    res = np.abs(np.sqrt((xs[keep] - xm) ** 2 + (ys[keep] - yc) ** 2) - rho)
    if res.mean() > 0.025 * D:
        return None
    # bombé sur toute la largeur : un bord plat aux coins arrondis (plaque à congés) a un milieu plat
    kx, ky = xs[keep], ys[keep]
    mid, side = np.abs(kx - xm) < 0.08 * D, (np.abs(kx - xm) > 0.2 * D) & (np.abs(kx - xm) < 0.3 * D)
    if mid.any() and side.any():
        expect = math.sqrt(rho ** 2 - (0.25 * D) ** 2) if rho > 0.25 * D else 0.0
        expect = rho - expect                          # écart attendu entre le milieu et x = ±D/4
        seen = abs(float(np.median(ky[side])) - float(np.median(ky[mid])))
        if seen < 0.4 * expect:
            return None
    return {"yc": yc, "rho": rho, "apex": float(apex), "depth_px": float(depth), "join": float(y_join)}


def head_y(h: dict, x: float, xm: float, up: bool) -> float:
    """Ordonnée (px) du fond à l'abscisse x."""
    d2 = max(0.0, h["rho"] ** 2 - (x - xm) ** 2)
    return h["yc"] - math.sqrt(d2) if up else h["yc"] + math.sqrt(d2)


def _has_back_arc(st: np.ndarray, xm: float, D: float, y_join: float, up: bool) -> bool:
    """Arc arrière (souvent en pointillés) côté corps = base elliptique vue en perspective,
    donc cylindre / gélule, pas un fond bombé."""
    x0, x1 = int(xm - 0.12 * D), int(xm + 0.12 * D)
    if up:
        y0, y1 = int(y_join + 0.02 * D), int(y_join + 0.2 * D)
    else:
        y0, y1 = int(y_join - 0.2 * D), int(y_join - 0.02 * D)
    band = st[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    return band.size > 0 and int((band.any(axis=1)).sum()) >= 3


def _openings(st: np.ndarray, xl: float, xr: float, y_top: float, y_bot: float) -> list[dict]:
    """Cercles fermés dans le corps ; deux cercles concentriques = ouverture à bride."""
    D = xr - xl
    k = max(3, int(0.006 * D) | 1)
    closed = cv2.morphologyEx(st, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnts, hier = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    circles = []
    if hier is None:
        return []
    for c, h in zip(cnts, hier[0]):
        if h[3] < 0:  # on ne garde que les trous (intérieurs de traits fermés)
            continue
        area = cv2.contourArea(c)
        if area < 20:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        peri = cv2.arcLength(c, True)
        circ = 4 * math.pi * area / (peri * peri + 1e-9)
        if circ < 0.78 or area < 0.6 * math.pi * r * r:
            continue
        if not (0.035 * D <= r <= 0.45 * D and xl < cx < xr and y_top < cy < y_bot):
            continue
        circles.append((float(cx), float(cy), float(r)))
    circles.sort(key=lambda t: t[2])
    groups: list[list] = []
    for c in circles:
        for g in groups:
            if math.hypot(c[0] - g[0][0], c[1] - g[0][1]) < 0.25 * g[0][2]:
                g.append(c)
                break
        else:
            groups.append([c])
    out = []
    for g in groups:
        bore, outer = g[0], g[-1]
        out.append({"cx": bore[0], "cy": bore[1], "r_bore": bore[2],
                    "r_flange": outer[2] if len(g) > 1 and outer[2] > 1.12 * bore[2] else None})
    return out


def _accessories(st: np.ndarray, xl: float, xr: float, head: dict, up: bool, legs_x: list[float]) -> list[dict]:
    """Piquages axiaux au-delà du fond (au-dessus du haut, en dessous du bas).

    Un piquage vu de face = deux parois verticales proches, reliées par un trait à leur extrémité
    (« ⊓ » sur le fond du haut, « ⊔ » sous le fond du bas). On cherche ces paires de parois :
    un renvoi de cote oblique, une ligne de rappel ou un pied isolé n'en forment pas.
    """
    D = xr - xl
    xm = (xl + xr) / 2
    H, W = st.shape
    band = max(3, int(0.012 * D))
    mask = np.zeros_like(st)
    for x in range(max(0, int(xl - 0.02 * D)), min(W, int(xr + 0.02 * D))):
        hy = head_y(head, min(max(x, xl), xr), xm, up)
        y0, y1 = (int(hy - 0.5 * D), int(hy - band)) if up else (int(hy + band), int(hy + 0.5 * D))
        mask[max(0, y0):max(0, y1), x] = 255
    acc = cv2.bitwise_and(st, mask)
    # parois : colonnes qui portent une longue course verticale de trait
    min_run, max_run = max(0.025 * D, 8.0), 0.4 * D
    walls_cols = []
    for x in range(W):
        ys = np.nonzero(acc[:, x])[0]
        if ys.size < min_run:
            continue
        brk = np.nonzero(np.diff(ys) > 3)[0]
        starts = np.r_[0, brk + 1]
        ends = np.r_[brk, ys.size - 1]
        k = int(np.argmax(ys[ends] - ys[starts]))
        y0, y1 = int(ys[starts[k]]), int(ys[ends[k]])
        if min_run <= y1 - y0 <= max_run:
            walls_cols.append((x, y0, y1))
    walls = []
    for x, y0, y1 in walls_cols:
        if walls and x - walls[-1]["x1"] <= 3:
            w = walls[-1]
            w["x1"] = x
            w["ys"].append((y0, y1))
        else:
            walls.append({"x0": x, "x1": x, "ys": [(y0, y1)]})
    for w in walls:
        w["x"] = (w["x0"] + w["x1"]) / 2
        w["y0"] = float(np.median([a for a, _ in w["ys"]]))
        w["y1"] = float(np.median([b for _, b in w["ys"]]))
    out, used = [], set()
    pairs = []
    for i, a in enumerate(walls):
        for j in range(i + 1, len(walls)):
            b = walls[j]
            gap = b["x"] - a["x"]
            if not 0.03 * D <= gap <= 0.35 * D:
                continue
            ov = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
            if ov < 0.5 * min(a["y1"] - a["y0"], b["y1"] - b["y0"]):
                continue
            # trait de fermeture à l'extrémité libre (en haut pour « ⊓ », en bas pour « ⊔ »)
            yend = min(a["y0"], b["y0"]) if up else max(a["y1"], b["y1"])
            strip = acc[max(0, int(yend - band)):int(yend + band) + 1, int(a["x"]):int(b["x"]) + 1]
            if strip.size == 0 or strip.any(axis=0).mean() < 0.6:
                continue
            pairs.append((gap, i, j, yend))
    for gap, i, j, yend in sorted(pairs):
        if i in used or j in used:
            continue
        a, b = walls[i], walls[j]
        xc = (a["x"] + b["x"]) / 2
        if any(abs(xc - lx) < 0.04 * D for lx in legs_x):
            continue
        base = head_y(head, xc, xm, up)
        height = (base - yend) if up else (yend - base)
        if height < 0.02 * D:
            continue
        used |= {i, j}
        out.append({"cx": float(xc), "w_px": float(gap), "h_px": float(height), "y_end": float(yend)})
    return sorted(out, key=lambda n: n["cx"])


def _straight(st: np.ndarray, x: float, y0: float, y1: float, D: float) -> bool:
    """La génératrice est un trait droit sur toute sa hauteur (pas le flanc d'un cercle)."""
    H, W = st.shape
    win = int(max(4, 0.05 * D))
    xs = []
    for y in np.linspace(y0 + 0.12 * (y1 - y0), y1 - 0.12 * (y1 - y0), 25).astype(int):
        row = np.nonzero(st[y, max(0, int(x) - win):min(W, int(x) + win + 1)])[0] + max(0, int(x) - win)
        if row.size:
            # le tronçon de trait le plus proche (une pointe de flèche voisine ne compte pas)
            brk = np.nonzero(np.diff(row) > 1)[0]
            runs = np.split(row, brk + 1)
            xs.append(float(min((r.mean() for r in runs), key=lambda m: abs(m - x))))
    if len(xs) < 20:
        return False
    xs = np.array(xs)
    yy = np.arange(len(xs), dtype=float)
    a, b = np.polyfit(yy, xs, 1)             # une photo légèrement penchée garde un trait droit
    res = np.sort(np.abs(xs - (a * yy + b)))
    return float(res[-3]) <= max(3.0, 0.012 * D)   # deux points aberrants tolérés


def detect_vessel(binary: np.ndarray) -> Optional[dict]:
    """Retourne les mesures px du réservoir, ou None si le dessin n'en est pas un."""
    from .primitives import _clean, _vertical_lines
    st = _clean(binary)
    # regroupement serré : un pied juste sous une génératrice ne doit pas la prolonger
    lines = _vertical_lines(st, tol=0.008)
    best = None
    for i, L in enumerate(lines):
        for R in lines[i + 1:]:
            D = R["x"] - L["x"]
            ov = min(L["y1"], R["y1"]) - max(L["y0"], R["y0"])
            span = max(L["len"], R["len"])
            if D < 0.2 * span or ov < 0.6 * min(L["len"], R["len"]) or min(L["len"], R["len"]) < 0.6 * span:
                continue
            if L["ink"] < 0.6 * L["len"] or R["ink"] < 0.6 * R["len"]:
                continue
            y_top = (L["y0"] + R["y0"]) / 2
            y_bot = (L["y1"] + R["y1"]) / 2
            xm = (L["x"] + R["x"]) / 2
            # corps cylindrique : une vraie partie droite (un cercle n'a que deux petits flancs verticaux)
            if y_bot - y_top < 0.5 * D:
                continue
            if not (_straight(st, L["x"], y_top, y_bot, D) and _straight(st, R["x"], y_top, y_bot, D)):
                continue
            top = _head(st, L["x"], R["x"], y_top, True)
            bot = _head(st, L["x"], R["x"], y_bot, False)
            if top is None or bot is None:
                continue
            if _has_back_arc(st, xm, D, y_top, True) or _has_back_arc(st, xm, D, y_bot, False):
                continue
            score = min(L["len"], R["len"]) + 0.5 * D
            if best is None or score > best["score"]:
                best = {"score": score, "xl": float(L["x"]), "xr": float(R["x"]), "y_top": float(y_top),
                        "y_bot": float(y_bot), "top": top, "bottom": bot}
    if best is None:
        return None
    xl, xr = best["xl"], best["xr"]
    D = xr - xl
    xm = (xl + xr) / 2
    # pieds : traits verticaux qui partent du fond du bas
    legs = []
    for ln in lines:
        if abs(ln["x"] - xl) < 0.03 * D or abs(ln["x"] - xr) < 0.03 * D:
            continue
        if xl - 0.15 * D <= ln["x"] <= xr + 0.15 * D and ln["y0"] >= best["y_bot"] - 0.05 * D \
                and ln["y1"] > best["bottom"]["apex"] + 0.05 * D and ln["len"] > 0.15 * D:
            legs.append({"x": float(ln["x"]), "y_ground": float(ln["y1"])})
    best["legs"] = legs
    best["openings"] = _openings(st, xl, xr, best["y_top"], best["y_bot"])
    lx = [lg["x"] for lg in legs]
    best["nozzles_top"] = _accessories(st, xl, xr, best["top"], True, lx)
    best["nozzles_bottom"] = _accessories(st, xl, xr, best["bottom"], False, lx)
    best["D_px"] = float(D)
    best["H_px"] = float(best["y_bot"] - best["y_top"])
    best["x_axis"] = float(xm)
    best.pop("score")
    log.info("Réservoir détecté : Ø %.0f px, corps %.0f px, %d ouverture(s), %d+%d piquage(s), %d pied(s)",
             D, best["H_px"], len(best["openings"]), len(best["nozzles_top"]), len(best["nozzles_bottom"]), len(legs))
    return best


# --------------------------------------------------------------------------- cotes

def cap_depth(R: float, a: float) -> float:
    """Flèche d'une calotte sphérique de rayon R sur un corps de rayon a."""
    return R - math.sqrt(max(0.0, R * R - a * a))


def assign_vessel_dims(v: dict, dims: list[Dimension], default_wall: float = 5.0
                       ) -> tuple[dict, list[str], list[str]]:
    """Valeurs en mm à partir des cotes ; ce qui n'est pas coté est estimé à l'échelle du dessin."""
    xl, xr, yt, yb = v["xl"], v["xr"], v["y_top"], v["y_bot"]
    D_px, H_px, xm = v["D_px"], v["H_px"], v["x_axis"]
    warns: list[str] = []
    confl: list[str] = []
    src: dict[str, str] = {}
    D = H = wall = None
    radii: list[tuple[float, float, str]] = []           # (valeur, y du texte, id)
    op_bore: dict[int, tuple[float, str]] = {}
    noz_d: dict[str, tuple[float, str]] = {}
    # le Ø du corps est le plus grand Ø écrit sur le dessin ; les autres vont aux ouvertures / piquages
    dia_like = [d for d in dims if d.kind in ("diameter", "ambiguous")]
    body_d = max(dia_like, key=lambda d: d.value) if dia_like else None
    for d in dims:
        cx, cy = d.center
        th = d.bbox[3] - d.bbox[1]
        if body_d is not None and d is body_d:
            D, src["diameter"] = d.value, d.id
            if d.kind != "diameter":
                warns.append(f"{d.id} « {d.raw} » : lu comme le diamètre du corps (plus grand Ø du dessin).")
            continue
        if d.kind == "thickness":
            wall, src["wall"] = d.value, d.id
            continue
        if d.kind == "radius":
            radii.append((d.value, cy, d.id))
            continue
        if d.kind == "angle" or d.kind == "unknown":
            continue
        inside = xl - 0.03 * D_px <= cx <= xr + 0.03 * D_px
        # Ø près d'une ouverture : son alésage
        near = [(math.hypot(cx - o["cx"], cy - o["cy"]) - (o["r_flange"] or o["r_bore"]), k)
                for k, o in enumerate(v["openings"])]
        near = [t for t in near if t[0] < 1.6 * th + 0.04 * D_px]
        if d.kind in ("diameter", "ambiguous") and near:
            k = min(near)[1]
            if k not in op_bore:
                op_bore[k] = (d.value, d.id)
                continue
        nz = [(abs(cx - n["cx"]) + abs(cy - n["y_end"]), f"{side}{k}")
              for side in ("t", "b") for k, n in enumerate(v["nozzles_top" if side == "t" else "nozzles_bottom"])]
        nz = [t for t in nz if t[0] < 0.15 * D_px + 2 * th]
        if d.kind == "diameter" and nz and (D is not None or not inside):
            noz_d[min(nz)[1]] = (d.value, d.id)
            continue
        if d.kind in ("diameter", "ambiguous") or (d.kind == "length" and inside and yt < cy < yb):
            if D is None:
                D, src["diameter"] = d.value, d.id
                if d.kind != "diameter":
                    warns.append(f"{d.id} « {d.raw} » dans le corps : lu comme son diamètre.")
            elif abs(D - d.value) > 1e-6:
                confl.append(f"diamètre du corps coté deux fois : {D:g} mm ({src['diameter']}) et {d.value:g} mm ({d.id})")
            continue
        if d.kind == "length":
            if not inside and yt - 0.1 * D_px < cy < yb + 0.1 * D_px:
                if H is None:
                    H, src["height"] = d.value, d.id
                elif abs(H - d.value) > 1e-6:
                    confl.append(f"hauteur du corps cotée deux fois : {H:g} mm et {d.value:g} mm ({d.id})")
            else:
                warns.append(f"{d.id} « {d.raw} » non rattachée (position inattendue pour un réservoir).")
    # échelles : le croquis peut être hors échelle -> une échelle par direction quand c'est possible
    if D is None and H is None:
        sx = sy = 1.0
        D, H = D_px, H_px
        warns.append("Aucune cote du corps : dimensions arbitraires (1 mm par pixel). Ajoutez Ø et hauteur.")
    elif D is None:
        sx = sy = H / H_px
        D = round(D_px * sx, 1)
        warns.append(f"Diamètre du corps non coté : {D:g} mm d'après les proportions du dessin.")
    elif H is None:
        sx = sy = D / D_px
        H = round(H_px * sy, 1)
        warns.append(f"Hauteur du corps non cotée : {H:g} mm d'après les proportions du dessin.")
    else:
        sx, sy = D / D_px, H / H_px
    a = D / 2
    heads = {}
    for key, h, up in (("top", v["top"], True), ("bottom", v["bottom"], False)):
        cand = [r for r in radii]
        if len(radii) >= 2:
            cand = [min(radii, key=lambda r: abs(r[1] - (yt if up else yb)))]
        if cand:
            R, _, rid = cand[0]
            src[f"R_{key}"] = rid
            if R < a:
                warns.append(f"R{R:g} ({rid}) plus petit que le rayon du corps ({a:g} mm) : "
                             f"fond {('haut' if up else 'bas')} pris hémisphérique (R{a:g}).")
                R = a
        else:
            depth = h["depth_px"] * sy
            depth = min(max(depth, 0.05 * a), a)
            R = (a * a + depth * depth) / (2 * depth)
            warns.append(f"Fond {('haut' if up else 'bas')} non coté : R{R:.0f} mm estimé d'après le dessin.")
        heads[key] = round(R, 3)
    if wall is None:
        wall = default_wall
        warns.append(f"Épaisseur de paroi non cotée : {wall:g} mm (écrivez « ep » sur le croquis).")
    if wall >= 0.25 * a:
        warns.append(f"Paroi de {wall:g} mm trop épaisse pour Ø{D:g} : ramenée à {0.1 * a:g} mm.")
        wall = round(0.1 * a, 3)
    ht, hb = cap_depth(heads["top"], a), cap_depth(heads["bottom"], a)
    # ouvertures en façade
    openings = []
    for k, o in enumerate(v["openings"]):
        if k in op_bore:
            bore, oid = op_bore[k]
            src[f"opening{k + 1}"] = oid
        else:
            bore = round(2 * o["r_bore"] * sx, 1)
            warns.append(f"Ouverture {k + 1} non cotée : Ø{bore:g} mm d'après le dessin.")
        ratio = (o["r_flange"] / o["r_bore"]) if o["r_flange"] else None
        flange = round(bore * ratio, 1) if ratio else None
        x = round((o["cx"] - xm) * sx, 1)
        z = round((yb - o["cy"]) * sy, 1)
        if abs(x) + bore / 2 > a * 0.98:
            warns.append(f"Ouverture {k + 1} (Ø{bore:g}) dépasse du corps : décalée vers l'axe.")
            x = math.copysign(max(0.0, a * 0.98 - bore / 2), x)
        openings.append({"x": x, "z": z, "bore": bore, "flange": flange,
                         "length": round(max(0.08 * D, 0.6 * bore), 1)})
    # piquages axiaux
    nozzles = []
    for side, lst in (("top", v["nozzles_top"]), ("bottom", v["nozzles_bottom"])):
        for k, n in enumerate(lst):
            key = f"{side[0]}{k}"
            if key in noz_d:
                dia = noz_d[key][0]
                src[f"nozzle_{key}"] = noz_d[key][1]
            else:
                dia = round(n["w_px"] * sx, 1)
            x = round((n["cx"] - xm) * sx, 1)
            if abs(x) + dia / 2 > a * 0.95:
                x = math.copysign(max(0.0, a * 0.95 - dia / 2), x)
            nozzles.append({"side": side, "x": x, "dia": dia, "height": round(max(n["h_px"] * sy, 0.04 * D), 1)})
    if nozzles and not any(f"nozzle_{n}" in src for n in noz_d):
        warns.append(f"{len(nozzles)} piquage(s) non coté(s) : Ø et hauteur d'après le dessin.")
    # pieds : 4 pieds à 90°, vus deux par deux de face
    legs = None
    if v["legs"]:
        dx = float(np.median([abs(lg["x"] - xm) for lg in v["legs"]])) * sx
        r = min(max(dx * math.sqrt(2), 0.35 * a), a * 0.92)
        ground = float(np.median([lg["y_ground"] for lg in v["legs"]]))
        below = (ground - yb) * sy                      # sous la ligne de raccordement basse
        if below < hb + 0.05 * D:
            below = hb + 0.15 * D
            warns.append("Pieds plus courts que le fond du bas : allongés pour toucher le sol sous la cuve.")
        legs = {"count": 4, "radius": round(r, 1), "dia": round(max(0.05 * D, 20.0), 1),
                "ground": round(-below, 1)}
        warns.append(f"Pieds non cotés : 4 pieds Ø{legs['dia']:g} mm à 90°, sol à {below:.0f} mm sous le corps.")
    out = {"diameter": D, "height": H, "R_top": heads["top"], "R_bottom": heads["bottom"], "wall": wall,
           "head_top": round(ht, 3), "head_bottom": round(hb, 3), "openings": openings,
           "nozzles": nozzles, "legs": legs, "sources": src}
    return out, warns, confl


# --------------------------------------------------------------------------- Set-of-Mark

def draw_vessel_som(img: np.ndarray, v: dict, dims: list[Dimension]) -> np.ndarray:
    from .perception import _draw_texts
    out = img.copy()
    sc = max(1, int(round(max(out.shape[:2]) / 900)))
    xl, xr, yt, yb, xm = v["xl"], v["xr"], v["y_top"], v["y_bot"], v["x_axis"]
    red, mag, grn = (0, 0, 220), (200, 0, 200), (0, 150, 0)
    for x in (xl, xr):
        cv2.line(out, (int(x), int(yt)), (int(x), int(yb)), grn, 2 * sc)
    for h, up in ((v["top"], True), (v["bottom"], False)):
        pts = [(int(x), int(head_y(h, x, xm, up))) for x in np.linspace(xl, xr, 50)]
        cv2.polylines(out, [np.array(pts, np.int32)], False, mag, 2 * sc)
    for k, o in enumerate(v["openings"]):
        cv2.circle(out, (int(o["cx"]), int(o["cy"])), int(o["r_bore"]), red, 2 * sc)
        if o["r_flange"]:
            cv2.circle(out, (int(o["cx"]), int(o["cy"])), int(o["r_flange"]), red, sc)
        cv2.putText(out, f"O{k + 1}", (int(o["cx"] + o["r_bore"] + 6), int(o["cy"])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * sc, red, sc, cv2.LINE_AA)
    for k, n in enumerate(v["nozzles_top"] + v["nozzles_bottom"]):
        x0 = int(n["cx"] - n["w_px"] / 2)
        cv2.rectangle(out, (x0, int(n["y_end"]) - 4), (int(x0 + n["w_px"]), int(n["y_end"]) + 4), (200, 120, 0), 2 * sc)
        cv2.putText(out, f"P{k + 1}", (int(x0 + n["w_px"] + 4), int(n["y_end"])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * sc, (200, 120, 0), sc, cv2.LINE_AA)
    for lg in v["legs"]:
        cv2.line(out, (int(lg["x"]), int(yb)), (int(lg["x"]), int(lg["y_ground"])), (0, 120, 220), 2 * sc)
    cv2.putText(out, f"RESERVOIR D={v['D_px']:.0f}px H={v['H_px']:.0f}px", (int(xl), int(yb + 0.05 * v["D_px"]) + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * sc, mag, 2 * sc, cv2.LINE_AA)
    _draw_texts(out, dims)
    return out
