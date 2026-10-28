"""Agent 3 — Résolution algébrique des contraintes (mise à l'échelle mathématique).

Moindres carrés non linéaires (scipy) en deux phases :
1. avec une faible régularisation vers la forme dessinée (lève l'indétermination
   des croquis sous-cotés) ;
2. sans régularisation, à partir de la solution 1, pour satisfaire exactement les cotes.
Les contraintes implicites (horizontal/vertical déduits du dessin) sont relâchées
si elles contredisent une cote écrite.
"""
from __future__ import annotations

import logging
import math
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from .schema import Constraint, Graph, Solution

log = logging.getLogger(__name__)

TOL_MM = 0.01


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def interior_angle(P: np.ndarray, k: int) -> float:
    """Angle intérieur (rad, 0..2π) au sommet k d'un polygone CCW (repère y-haut)."""
    n = len(P)
    u = P[(k + 1) % n] - P[k]
    v = P[k - 1] - P[k]
    a = math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])
    return a % (2 * math.pi)


def signed_area(P: np.ndarray) -> float:
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


class _Problem:
    def __init__(self, g: Graph):
        self.g = g
        self.n = len(g.vertices)
        self.vidx = {v.id: i for i, v in enumerate(g.vertices)}
        self.eidx = {e.id: (self.vidx[e.a], self.vidx[e.b]) for e in g.edges}
        self.drawn = np.array([[v.px[0], -v.px[1]] for v in g.vertices], float)  # y vers le haut
        self.drawn -= self.drawn[0]
        # perçages positionnés par cotes -> inconnues supplémentaires
        self.free_circles = sorted({c.circle for c in g.constraints if c.type == "distance" and c.circle})
        self.cidx = {cid: i for i, cid in enumerate(self.free_circles)}
        self.cdrawn = {c.id: np.array([c.px[0], -c.px[1]]) - np.array([g.vertices[0].px[0], -g.vertices[0].px[1]])
                       for c in g.circles}

    # -- vecteur d'inconnues <-> points
    def unpack(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        P = np.zeros((self.n, 2))
        P[1:] = x[: 2 * (self.n - 1)].reshape(-1, 2)
        C = x[2 * (self.n - 1):].reshape(-1, 2)
        return P, C

    def pack(self, P: np.ndarray, C: np.ndarray) -> np.ndarray:
        return np.concatenate([P[1:].ravel(), C.ravel()])

    def residual_fns(self, cons: list[Constraint], lref: float) -> list[tuple[Constraint, Callable]]:
        out = []
        for c in cons:
            if c.type == "length" and c.edge in self.eidx:
                i, j = self.eidx[c.edge]
                out.append((c, lambda P, C, i=i, j=j, L=c.value: np.linalg.norm(P[j] - P[i]) - L))
            elif c.type == "angle" and c.vertex in self.vidx:
                k = self.vidx[c.vertex]
                th = math.radians(c.value)
                out.append((c, lambda P, C, k=k, th=th: _wrap(interior_angle(P, k) - th) * lref))
            elif c.type == "horizontal" and c.edge in self.eidx:
                i, j = self.eidx[c.edge]
                out.append((c, lambda P, C, i=i, j=j: P[j][1] - P[i][1]))
            elif c.type == "vertical" and c.edge in self.eidx:
                i, j = self.eidx[c.edge]
                out.append((c, lambda P, C, i=i, j=j: P[j][0] - P[i][0]))
            elif c.type in ("perpendicular", "parallel") and c.edges and len(c.edges) == 2 \
                    and all(e in self.eidx for e in c.edges):
                (a, b), (p, q) = self.eidx[c.edges[0]], self.eidx[c.edges[1]]

                def f(P, C, a=a, b=b, p=p, q=q, t=c.type):
                    u, v = P[b] - P[a], P[q] - P[p]
                    nu = np.linalg.norm(u) * np.linalg.norm(v) + 1e-9
                    val = np.dot(u, v) if t == "perpendicular" else u[0] * v[1] - u[1] * v[0]
                    return val / nu * lref
                out.append((c, f))
            elif c.type == "arc_radius" and c.edge in self.eidx:
                # dôme : arc de D (i) au sommet E (j) sur l'axe, centre sur l'axe sous E
                i, j = self.eidx[c.edge]
                out.append((c, lambda P, C, i=i, j=j, rho=c.value:
                            math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1] + rho) - rho))
            elif c.type == "arc_tangent" and c.edge in self.eidx:
                i, j = self.eidx[c.edge]  # dôme hémisphérique : hauteur = rayon en D
                out.append((c, lambda P, C, i=i, j=j: (P[j][1] - P[i][1]) - (P[i][0] - P[j][0])))
            elif c.type == "radius_at" and c.vertex in self.vidx:
                k = self.vidx[c.vertex]  # distance à l'axe (V1 sur l'axe, x = 0)
                out.append((c, lambda P, C, k=k, v=c.value: P[k][0] - P[0][0] - v))
            elif c.type == "distance" and c.circle in self.cidx and c.target in self.eidx:
                ci = self.cidx[c.circle]
                i, j = self.eidx[c.target]

                def f(P, C, ci=ci, i=i, j=j, d=c.value):
                    u = P[j] - P[i]
                    w = C[ci] - P[i]
                    return abs(u[0] * w[1] - u[1] * w[0]) / (np.linalg.norm(u) + 1e-9) - d
                out.append((c, f))
        return out


def _estimate_scale(pb: _Problem, g: Graph) -> tuple[float, list[str]]:
    ratios = []
    for c in g.constraints:
        if c.type == "length" and c.edge in pb.eidx:
            i, j = pb.eidx[c.edge]
            dl = np.linalg.norm(pb.drawn[j] - pb.drawn[i])
            if dl > 1:
                ratios.append(c.value / dl)
    for c in g.constraints:
        if c.type == "arc_radius" and c.edge in pb.eidx:
            i, j = pb.eidx[c.edge]
            dx = abs(pb.drawn[i][0] - pb.drawn[j][0])
            if dx > 1:
                ratios.append(c.value / dx)
        if c.type == "radius_at" and c.vertex in pb.vidx:
            dx = abs(pb.drawn[pb.vidx[c.vertex]][0] - pb.drawn[0][0])
            if dx > 1:
                ratios.append(c.value / dx)
    if not ratios:
        cm = {c.id: c for c in g.circles}
        ratios = [c.value / (2 * cm[c.circle].r_px) for c in g.constraints
                  if c.type == "diameter" and c.circle in cm]
    if not ratios:
        return 0.25, ["Aucune cote de longueur : échelle arbitraire 0,25 mm/px."]
    return float(np.median(ratios)), []


def solve(g: Graph, max_relax_rounds: int = 5) -> Solution:
    pb = _Problem(g)
    warnings: list[str] = []
    s, w = _estimate_scale(pb, g)
    warnings += w
    P0 = pb.drawn * s
    C0 = np.array([pb.cdrawn[cid] * s for cid in pb.free_circles]).reshape(-1, 2)
    x0 = pb.pack(P0, C0)
    lengths = [np.linalg.norm(P0[(k + 1) % pb.n] - P0[k]) for k in range(pb.n)]
    lref = float(np.median(lengths)) if lengths else 1.0

    active = [c for c in g.constraints if c.type not in ("fillet", "diameter")]
    relaxed: list[str] = []
    x = x0
    nfev = 0
    for _round in range(max_relax_rounds):
        fns = pb.residual_fns(active, lref)

        def res(xv, reg):
            P, C = pb.unpack(xv)
            r = [(1.0 if not c.implicit else 0.1) * f(P, C) for c, f in fns]
            if reg:
                r += list(reg * (xv - x0))
            return np.array(r, float)

        r1 = least_squares(res, x, args=(1e-3,), method="trf", xtol=1e-12, ftol=1e-12, gtol=1e-12,
                           max_nfev=5000)
        x = r1.x
        if fns:
            r2 = least_squares(res, x, args=(0.0,), method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14,
                               max_nfev=5000)
            x = r2.x
            nfev += r2.nfev
        nfev += r1.nfev
        P, C = pb.unpack(x)
        bad_implicit = [(c, abs(f(P, C))) for c, f in fns if c.implicit and abs(f(P, C)) > TOL_MM]
        bad_explicit = [c for c, f in fns if not c.implicit and abs(f(P, C)) > TOL_MM]
        if bad_explicit and bad_implicit:
            worst = max(bad_implicit, key=lambda t: t[1])[0]
            active = [c for c in active if c is not worst]
            relaxed.append(f"{worst.type} {worst.edge or worst.edges}")
            x = x0  # relance depuis le dessin sans la contrainte fautive
            continue
        break

    P, C = pb.unpack(x)
    fns = pb.residual_fns(active, lref)
    conflicts = []
    max_res = 0.0
    for c, f in fns:
        r = abs(f(P, C))
        if c.type == "arc_tangent":
            continue
        if c.type == "angle":
            r = r / lref  # rad
            if math.degrees(r) > 0.01:
                conflicts.append(f"angle {c.vertex} = {c.value:g}° ({c.source}) non satisfait "
                                 f"(écart {math.degrees(r):.3f}°)")
            continue
        if not c.implicit:
            max_res = max(max_res, r)
            if r > TOL_MM:
                conflicts.append(f"{c.type} {c.edge or c.circle or c.vertex} = {c.value:g} ({c.source}) "
                                 f"non satisfait (écart {r:.3f} mm)")

    # --- degrés de liberté restants
    dof = _dof(pb, fns, x)
    if dof > 0:
        warnings.append(f"Croquis sous-coté : {dof} degré(s) de liberté complété(s) d'après le dessin.")

    # --- congés
    fillets: dict[int, float] = {}
    for c in g.constraints:
        if c.type == "fillet" and c.vertex in pb.vidx and c.value:
            fillets[pb.vidx[c.vertex]] = float(c.value)
    warnings += _check_fillets(P, fillets)

    # --- perçages : cotés, sinon placés par l'application affine dessin -> solution
    A, t = _affine(pb.drawn, P)
    sim_scale = math.sqrt(abs(np.linalg.det(A)))
    diam = {c.circle: c.value for c in g.constraints if c.type == "diameter" and c.circle}
    holes = []
    for circ in g.circles:
        if circ.id in pb.cidx:
            cx, cy = C[pb.cidx[circ.id]]
        else:
            cx, cy = A @ pb.cdrawn[circ.id] + t
        d = diam.get(circ.id)
        if d is None:
            d = round(2 * circ.r_px * sim_scale, 1)
            warnings.append(f"{circ.id} sans cote Ø : diamètre estimé {d:g} mm d'après le dessin.")
        holes.append((float(cx), float(cy), float(d)))

    if signed_area(P) < 0:
        warnings.append("Profil résolu orienté en sens horaire (miroir probable).")
    return Solution(points=[(float(a), float(b)) for a, b in P], fillets=fillets, holes=holes,
                    thickness=float(g.thickness or 5.0), dof=dof, max_residual_mm=max_res,
                    conflicts=conflicts, relaxed=relaxed, warnings=warnings, iterations=nfev)


def _dof(pb: _Problem, fns, x: np.ndarray) -> int:
    if not fns:
        return 2 * pb.n - 3
    def r(xv):
        P, C = pb.unpack(xv)
        return np.array([f(P, C) for _, f in fns])
    eps = 1e-6
    J = np.array([(r(x + eps * e) - r(x - eps * e)) / (2 * eps) for e in np.eye(len(x))]).T
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(sv > 1e-6 * max(sv.max(), 1e-12)))
    P, C = pb.unpack(x)
    rot = pb.pack(np.stack([-P[:, 1], P[:, 0]], 1), np.stack([-C[:, 1], C[:, 0]], 1) if len(C) else C)
    rot_free = np.linalg.norm(J @ rot) < 1e-6 * (np.linalg.norm(rot) + 1e-9) * max(sv.max(), 1)
    return max(0, len(x) - rank - (1 if rot_free else 0))


def _affine(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.hstack([src, np.ones((len(src), 1))])
    M, *_ = np.linalg.lstsq(X, dst, rcond=None)
    return M[:2].T, M[2]


def _check_fillets(P: np.ndarray, fillets: dict[int, float]) -> list[str]:
    n = len(P)
    tang = {}
    for k, R in fillets.items():
        a = interior_angle(P, k)
        beta = min(a, 2 * math.pi - a)
        tang[k] = R / math.tan(beta / 2) if beta > 1e-6 else float("inf")
    out = []
    for i in range(n):
        j = (i + 1) % n
        L = float(np.linalg.norm(P[j] - P[i]))
        if tang.get(i, 0) + tang.get(j, 0) > L + 1e-6:
            out.append(f"Congés trop grands pour l'arête V{i + 1}-V{j + 1} ({L:.2f} mm).")
    return out
