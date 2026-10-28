"""Géométrie du profil : contour polygonal + congés -> segments (lignes et arcs).

Ce module est sans dépendance (math seul) : son code source est recopié tel quel
dans chaque ``part.py`` généré, pour que le script CAO reste autonome.
"""
import math


def _unit(v):
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n)


def profile_segments(points, fillets):
    """Retourne une liste de segments du contour CCW.

    ('line', p0, p1)
    ('arc', p0, p1, center, radius, mid, ccw)   # arc de p0 à p1 passant par mid
    """
    n = len(points)
    P = [tuple(map(float, p)) for p in points]
    corners = []  # pour chaque sommet : (point d'entrée, point de sortie, arc ou None)
    for k in range(n):
        R = float(fillets.get(k, 0) or fillets.get(str(k), 0) or 0)
        if R <= 0:
            corners.append((P[k], P[k], None))
            continue
        d_in = _unit((P[k][0] - P[k - 1][0], P[k][1] - P[k - 1][1]))
        d_out = _unit((P[(k + 1) % n][0] - P[k][0], P[(k + 1) % n][1] - P[k][1]))
        cross = d_in[0] * d_out[1] - d_in[1] * d_out[0]          # > 0 : virage à gauche (convexe)
        cosb = max(-1.0, min(1.0, -(d_in[0] * d_out[0] + d_in[1] * d_out[1])))
        beta = math.acos(cosb)                                     # angle entre les deux arêtes
        t = R / math.tan(beta / 2)
        t1 = (P[k][0] - d_in[0] * t, P[k][1] - d_in[1] * t)
        t2 = (P[k][0] + d_out[0] * t, P[k][1] + d_out[1] * t)
        bis = _unit((-d_in[0] + d_out[0], -d_in[1] + d_out[1]))
        h = R / math.sin(beta / 2)
        c = (P[k][0] + bis[0] * h, P[k][1] + bis[1] * h)
        to_corner = _unit((P[k][0] - c[0], P[k][1] - c[1]))
        mid = (c[0] + to_corner[0] * R, c[1] + to_corner[1] * R)
        corners.append((t1, t2, (c, R, mid, cross > 0)))
    segs = []
    for k in range(n):
        j = (k + 1) % n
        a = corners[k][1]
        b = corners[j][0]
        if math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-9:
            segs.append(("line", a, b))
        arc = corners[j][2]
        if arc is not None:
            c, R, mid, ccw = arc
            segs.append(("arc", corners[j][0], corners[j][1], c, R, mid, ccw))
    return segs


def profile_polyline(points, fillets, step_deg=5.0):
    """Approximation polyligne du profil (aperçus, aires, contrôles)."""
    out = []
    for s in profile_segments(points, fillets):
        if s[0] == "line":
            out.append(s[1])
        else:
            _, p0, p1, c, R, _mid, ccw = s
            a0 = math.atan2(p0[1] - c[1], p0[0] - c[0])
            a1 = math.atan2(p1[1] - c[1], p1[0] - c[0])
            if ccw and a1 < a0:
                a1 += 2 * math.pi
            if not ccw and a1 > a0:
                a1 -= 2 * math.pi
            m = max(2, int(abs(a1 - a0) / math.radians(step_deg)))
            for i in range(m):
                a = a0 + (a1 - a0) * i / m
                out.append((c[0] + R * math.cos(a), c[1] + R * math.sin(a)))
    return out
