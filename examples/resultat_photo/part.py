#!/usr/bin/env python3
"""Pièce générée par Sketch2CAD-Local le 2026-10-28 11:48.

Script autonome : modifiez les paramètres ci-dessous puis relancez
    python part.py
pour régénérer part.step, part.stl, part.dxf et part.svg.
Pour changer une cote en conservant toutes les contraintes, éditez plutôt
graph.json puis relancez « python -m sketch2cad solve graph.json ».
"""
import math
import sys
from pathlib import Path

NAME = 'part'
THICKNESS = 5.0          # mm, extrusion
POINTS = [
    (0.0, 0.0), (150.0, -0.0), (150.0, 50.0),
    (110.0, 50.0), (90.0, 84.641016), (0.0, 84.641016),
]                  # sommets du contour (mm, sens trigonométrique)
FILLETS = {1: 10.0}              # index de sommet -> rayon de congé (mm)
HOLES = [
    (131.828422, 32.034486, 8.0), (51.411584, 22.593359, 12.0),
]                    # perçages traversants (x, y, diamètre) en mm
DIMS = [
    {'type': 'length', 'i': 2, 'j': 3, 'value': 40.0},
    {'type': 'length', 'i': 3, 'j': 4, 'value': 40.0},
    {'type': 'angle', 'k': 3, 'value': 240.0},
    {'type': 'length', 'i': 1, 'j': 2, 'value': 50.0},
    {'type': 'diameter', 'h': 0, 'value': 8.0},
    {'type': 'diameter', 'h': 1, 'value': 12.0},
    {'type': 'radius', 'k': 1, 'value': 10.0},
    {'type': 'length', 'i': 0, 'j': 1, 'value': 150.0},
]                      # cotes reportées sur la mise en plan


# ----------------------------------------------------------------- géométrie du profil


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


# ----------------------------------------------------------------- solide 3D (build123d)
def build_solid():
    from build123d import (BuildLine, BuildPart, BuildSketch, Circle, Line, Locations, Mode,
                           ThreePointArc, extrude, make_face)
    with BuildPart() as part:
        with BuildSketch():
            with BuildLine():
                for s in profile_segments(POINTS, FILLETS):
                    if s[0] == "line":
                        Line(s[1], s[2])
                    else:
                        ThreePointArc(s[1], s[5], s[2])
            make_face()
            for x, y, d in HOLES:
                with Locations((x, y)):
                    Circle(d / 2, mode=Mode.SUBTRACT)
        extrude(amount=THICKNESS)
    return part.part


# ----------------------------------------------------------------- mise en plan 2D (ezdxf)
def write_dxf(path):
    import ezdxf
    doc = ezdxf.new("R2018", setup=True)
    doc.units = ezdxf.units.MM
    for name, color in (("CONTOUR", 7), ("PERCAGES", 1), ("COTATION", 3), ("AXES", 4)):
        doc.layers.add(name, color=color)
    msp = doc.modelspace()
    for s in profile_segments(POINTS, FILLETS):
        if s[0] == "line":
            msp.add_line(s[1], s[2], dxfattribs={"layer": "CONTOUR"})
        else:
            _, p0, p1, c, r, _mid, ccw = s
            a0 = math.degrees(math.atan2(p0[1] - c[1], p0[0] - c[0]))
            a1 = math.degrees(math.atan2(p1[1] - c[1], p1[0] - c[0]))
            if not ccw:
                a0, a1 = a1, a0
            msp.add_arc(c, r, a0, a1, dxfattribs={"layer": "CONTOUR"})
    xs = [p[0] for p in POINTS]
    ys = [p[1] for p in POINTS]
    size = max(max(xs) - min(xs), max(ys) - min(ys))
    h = max(2.5, round(size * 0.03, 1))
    ov = {"dimtxt": h, "dimasz": h * 0.8, "dimexe": h * 0.4, "dimexo": h * 0.4,
          "dimgap": h * 0.3, "dimdec": 2, "dimzin": 8, "dimadec": 1, "dimazin": 2,
          "dimlfac": 1.0, "dimtih": 0, "dimtoh": 0}
    att = {"layer": "COTATION"}
    for x, y, d in HOLES:
        msp.add_circle((x, y), d / 2, dxfattribs={"layer": "PERCAGES"})
        e = d / 2 + h
        msp.add_line((x - e, y), (x + e, y), dxfattribs={"layer": "AXES", "linetype": "CENTER"})
        msp.add_line((x, y - e), (x, y + e), dxfattribs={"layer": "AXES", "linetype": "CENTER"})
    n = len(POINTS)
    for dm in DIMS:
        t = dm["type"]
        if t == "length":
            p1, p2 = POINTS[dm["i"]], POINTS[dm["j"]]
            out_n = (p2[1] - p1[1], -(p2[0] - p1[0]))      # normale extérieure (contour CCW)
            ang = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
            if not -90 < ang <= 90:                         # texte toujours lisible
                p1, p2 = p2, p1
            left = (-(p2[1] - p1[1]), p2[0] - p1[0])
            side = 1 if left[0] * out_n[0] + left[1] * out_n[1] > 0 else -1
            msp.add_aligned_dim(p1=p1, p2=p2, distance=side * 2.5 * h, override=ov,
                                dxfattribs=att).render()
        elif t == "angle":
            k = dm["k"]
            c = POINTS[k]
            u, v = POINTS[(k + 1) % n], POINTS[k - 1]
            a = math.radians(dm["value"])
            if a > math.pi:                                 # angle rentrant : coté côté extérieur
                u, v, a = v, u, 2 * math.pi - a
            a_u = math.atan2(u[1] - c[1], u[0] - c[0])
            rr = 2.2 * h
            base = (c[0] + rr * math.cos(a_u + a / 2), c[1] + rr * math.sin(a_u + a / 2))
            msp.add_angular_dim_3p(base=base, center=c, p1=u, p2=v, override=ov,
                                   dxfattribs=att).render()
        elif t == "radius":
            k = dm["k"]
            for s in profile_segments(POINTS, FILLETS):
                if s[0] == "arc" and math.hypot(s[5][0] - POINTS[k][0], s[5][1] - POINTS[k][1]) <= s[4] + 1e-6:
                    c, mid = s[3], s[5]
                    ang = math.degrees(math.atan2(mid[1] - c[1], mid[0] - c[0]))
                    msp.add_radius_dim(center=c, radius=s[4], angle=ang, override=ov,
                                       dxfattribs=att).render()
                    break
        elif t == "diameter":
            x, y, d = HOLES[dm["h"]]
            msp.add_diameter_dim(center=(x, y), radius=d / 2, angle=45, override=ov,
                                 dxfattribs=att).render()
    msp.add_text(f"ep = {THICKNESS:g} mm", height=h,
                 dxfattribs={"layer": "COTATION"}).set_placement((min(xs), min(ys) - 6 * h))
    doc.saveas(path)
    return doc


def write_svg(doc, path):
    from ezdxf.addons.drawing import Frontend, RenderContext, config, layout, svg
    backend = svg.SVGBackend()
    cfg = config.Configuration(background_policy=config.BackgroundPolicy.WHITE,
                               color_policy=config.ColorPolicy.BLACK)
    Frontend(RenderContext(doc), backend, config=cfg).draw_layout(doc.modelspace())
    page = layout.Page(0, 0, layout.Units.mm, margins=layout.Margins.all(10))
    Path(path).write_text(backend.get_string(page), encoding="utf-8")


def main(out_dir="."):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = write_dxf(str(out / f"{NAME}.dxf"))
    write_svg(doc, out / f"{NAME}.svg")
    try:
        from build123d import export_step, export_stl
    except ImportError:
        print("build123d absent : seuls .dxf et .svg ont été produits.", file=sys.stderr)
        return 2
    solid = build_solid()
    export_step(solid, str(out / f"{NAME}.step"))
    export_stl(solid, str(out / f"{NAME}.stl"), tolerance=0.01, angular_tolerance=0.1)
    bb = solid.bounding_box()
    print(f"OK volume={solid.volume:.3f} valid={solid.is_valid} "
          f"bbox={bb.size.X:.3f}x{bb.size.Y:.3f}x{bb.size.Z:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
