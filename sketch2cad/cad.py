"""Agent 4 — Synthèse « Code-as-CAD ».

Écrit un script ``part.py`` autonome et paramétrique (build123d + ezdxf), puis
l'exécute dans un sous-processus pour produire .step, .stl, .dxf et .svg.
Le gabarit couvre le périmètre v1 ; un LLM codeur local (optionnel) peut ensuite
modifier le script pour des fonctions hors gabarit, avec ré-essais sur erreur.
"""
from __future__ import annotations

import datetime as _dt
import inspect
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import profile as _profile
from .llm import Ollama
from .schema import Graph, Solution

log = logging.getLogger(__name__)

_TEMPLATE = r'''#!/usr/bin/env python3
"""Pièce générée par Sketch2CAD-Local le {date}.

Script autonome : modifiez les paramètres ci-dessous puis relancez
    python part.py
pour régénérer {name}.step, {name}.stl, {name}.dxf et {name}.svg.
Pour changer une cote en conservant toutes les contraintes, éditez plutôt
graph.json puis relancez « python -m sketch2cad solve graph.json ».
"""
import math
import sys
from pathlib import Path

NAME = {name!r}
THICKNESS = {thickness!r}          # mm, extrusion
POINTS = {points}                  # sommets du contour (mm, sens trigonométrique)
FILLETS = {fillets!r}              # index de sommet -> rayon de congé (mm)
HOLES = {holes}                    # perçages traversants (x, y, diamètre) en mm
DIMS = {dims}                      # cotes reportées sur la mise en plan


# ----------------------------------------------------------------- géométrie du profil
{profile_src}

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
    for name, color, lw in (("CONTOUR", 7, 50), ("PERCAGES", 1, 50), ("COTATION", 3, 18), ("AXES", 4, 13)):
        doc.layers.add(name, color=color, lineweight=lw)
    msp = doc.modelspace()
    for s in profile_segments(POINTS, FILLETS):
        if s[0] == "line":
            msp.add_line(s[1], s[2], dxfattribs={{"layer": "CONTOUR"}})
        else:
            _, p0, p1, c, r, _mid, ccw = s
            a0 = math.degrees(math.atan2(p0[1] - c[1], p0[0] - c[0]))
            a1 = math.degrees(math.atan2(p1[1] - c[1], p1[0] - c[0]))
            if not ccw:
                a0, a1 = a1, a0
            msp.add_arc(c, r, a0, a1, dxfattribs={{"layer": "CONTOUR"}})
    xs = [p[0] for p in POINTS]
    ys = [p[1] for p in POINTS]
    size = max(max(xs) - min(xs), max(ys) - min(ys))
    h = max(2.5, round(size * 0.03, 1))
    ov = {{"dimtxt": h, "dimasz": h * 0.8, "dimexe": h * 0.4, "dimexo": h * 0.4,
          "dimgap": h * 0.3, "dimdec": 2, "dimzin": 8, "dimadec": 1, "dimazin": 2,
          "dimlfac": 1.0, "dimtih": 0, "dimtoh": 0}}
    att = {{"layer": "COTATION"}}
    for x, y, d in HOLES:
        msp.add_circle((x, y), d / 2, dxfattribs={{"layer": "PERCAGES"}})
        e = d / 2 + h
        msp.add_line((x - e, y), (x + e, y), dxfattribs={{"layer": "AXES", "linetype": "CENTER"}})
        msp.add_line((x, y - e), (x, y + e), dxfattribs={{"layer": "AXES", "linetype": "CENTER"}})
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
    msp.add_text(f"ep = {{THICKNESS:g}} mm", height=h,
                 dxfattribs={{"layer": "COTATION"}}).set_placement((min(xs), min(ys) - 6 * h))
    doc.saveas(path)
    return doc


def write_svg(doc, path):
    from ezdxf.addons.drawing import Frontend, RenderContext, config, layout, svg
    backend = svg.SVGBackend()
    cfg = config.Configuration(background_policy=config.BackgroundPolicy.WHITE,
                               color_policy=config.ColorPolicy.BLACK,
                               lineweight_policy=config.LineweightPolicy.ABSOLUTE)
    Frontend(RenderContext(doc), backend, config=cfg).draw_layout(doc.modelspace())
    page = layout.Page(297, 210, layout.Units.mm, margins=layout.Margins.all(10))  # A4 paysage
    Path(path).write_text(backend.get_string(page, settings=layout.Settings(fit_page=True)), encoding="utf-8")


def main(out_dir="."):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = write_dxf(str(out / f"{{NAME}}.dxf"))
    write_svg(doc, out / f"{{NAME}}.svg")
    try:
        from build123d import export_step, export_stl
    except ImportError:
        print("build123d absent : seuls .dxf et .svg ont été produits.", file=sys.stderr)
        return 2
    solid = build_solid()
    export_step(solid, str(out / f"{{NAME}}.step"))
    export_stl(solid, str(out / f"{{NAME}}.stl"), tolerance=0.01, angular_tolerance=0.1)
    bb = solid.bounding_box()
    print(f"OK volume={{solid.volume:.3f}} valid={{solid.is_valid}} "
          f"bbox={{bb.size.X:.3f}}x{{bb.size.Y:.3f}}x{{bb.size.Z:.3f}}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
'''


def _fmt_list(items, per_line=3) -> str:
    if not items:
        return "[]"
    parts = [repr(tuple(round(v, 6) for v in it)) if isinstance(it, (tuple, list)) else repr(it)
             for it in items]
    lines = [", ".join(parts[i:i + per_line]) for i in range(0, len(parts), per_line)]
    return "[\n    " + ",\n    ".join(lines) + ",\n]"


def dims_for_drawing(g: Graph, sol: Solution) -> list[dict]:
    vidx = {v.id: i for i, v in enumerate(g.vertices)}
    eidx = {e.id: (vidx[e.a], vidx[e.b]) for e in g.edges}
    cidx = {c.id: i for i, c in enumerate(g.circles)}
    out = []
    for c in g.constraints:
        if c.implicit:
            continue
        if c.type == "length" and c.edge in eidx:
            i, j = eidx[c.edge]
            out.append({"type": "length", "i": i, "j": j, "value": c.value})
        elif c.type == "angle" and c.vertex in vidx:
            out.append({"type": "angle", "k": vidx[c.vertex], "value": c.value})
        elif c.type == "fillet" and c.vertex in vidx:
            out.append({"type": "radius", "k": vidx[c.vertex], "value": c.value})
        elif c.type == "diameter" and c.circle in cidx:
            out.append({"type": "diameter", "h": cidx[c.circle], "value": c.value})
    return out


def generate_script(g: Graph, sol: Solution, name: str = "part") -> str:
    src = inspect.getsource(_profile)
    src = src.split('"""', 2)[2].lstrip()  # retire la docstring du module
    src = src.replace("import math\n", "", 1)
    return _TEMPLATE.format(
        date=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        name=name,
        thickness=round(sol.thickness, 6),
        points=_fmt_list(sol.points),
        fillets={int(k): v for k, v in sol.fillets.items()},
        holes=_fmt_list(sol.holes, 2),
        dims=_fmt_list(dims_for_drawing(g, sol), 1),
        profile_src=src,
    )


def run_script(script: Path, out_dir: Path, timeout: int = 180) -> tuple[bool, str]:
    script, out_dir = Path(script).resolve(), Path(out_dir).resolve()
    try:
        r = subprocess.run([sys.executable, str(script), str(out_dir)], capture_output=True,
                           text=True, timeout=timeout, cwd=str(out_dir))
    except subprocess.TimeoutExpired:
        return False, f"Délai de {timeout} s dépassé."
    return r.returncode == 0, (r.stdout + "\n" + r.stderr).strip()


_CODER_SYSTEM = ("Tu modifies un script Python build123d/ezdxf existant. Réponds uniquement avec "
                 "le script complet corrigé, sans explication, dans un bloc ```python.")


def llm_patch(client: Ollama, model: str, script: str, request: str, error: str = "") -> str:
    prompt = f"Script actuel :\n```python\n{script}\n```\n\nDemande : {request}\n"
    if error:
        prompt += f"\nL'exécution précédente a échoué avec :\n{error[-3000:]}\n"
    txt = client.chat(model, prompt, system=_CODER_SYSTEM, num_ctx=16384)
    if "```" in txt:
        txt = txt.split("```python", 1)[-1] if "```python" in txt else txt.split("```", 1)[1]
        txt = txt.split("```", 1)[0]
    return txt.strip() + "\n"


def synthesize(g: Graph, sol: Solution, out_dir: Path, name: str = "part",
               client: Optional[Ollama] = None, coder_model: Optional[str] = None,
               extra_request: str = "", max_attempts: int = 3) -> dict:
    """Exécute l'agent 4 et retourne un rapport."""
    out_dir = Path(out_dir).resolve()
    script = generate_script(g, sol, name)
    path = out_dir / f"{name}.py"
    path.write_text(script, encoding="utf-8")
    ok, log_txt = run_script(path, out_dir)
    report = {"script": str(path), "ok": ok, "log": log_txt, "llm_attempts": 0}
    if ok and extra_request and client is not None and coder_model:
        base = script
        err = ""
        for i in range(max_attempts):
            report["llm_attempts"] = i + 1
            patched = llm_patch(client, coder_model, base, extra_request, err)
            cand = out_dir / f"{name}_llm.py"
            cand.write_text(patched, encoding="utf-8")
            ok2, log2 = run_script(cand, out_dir)
            if ok2:
                cand.replace(path)
                report.update(ok=True, log=log2, llm="appliqué")
                break
            err = log2
        else:
            report["llm"] = "échec : script gabarit conservé"
            run_script(path, out_dir)  # restaure les fichiers du gabarit
    if not ok:
        log.error("part.py a échoué :\n%s", log_txt)
    for line in log_txt.splitlines():
        if line.startswith("OK "):
            report["solid"] = dict(kv.split("=") for kv in line[3:].split())
    (out_dir / "cad_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    return report


# =========================================================================== cylindre
_CYL_TEMPLATE = r'''#!/usr/bin/env python3
"""Cylindre généré par Sketch2CAD-Local le {date}.

Modifiez DIAMETER / HEIGHT puis relancez « python {name}.py » pour régénérer
{name}.step, {name}.stl, {name}.dxf et {name}.svg.
"""
import sys
from pathlib import Path

NAME = {name!r}
DIAMETER = {diameter!r}   # mm
HEIGHT = {height!r}       # mm


def build_solid():
    from build123d import Align, Cylinder
    return Cylinder(radius=DIAMETER / 2, height=HEIGHT, align=(Align.CENTER, Align.CENTER, Align.MIN))


def write_dxf(path):
    """Mise en plan 2 vues : vue de face (rectangle) et vue de dessus (cercle)."""
    import ezdxf
    doc = ezdxf.new("R2018", setup=True)
    doc.units = ezdxf.units.MM
    for n, col, lw in (("CONTOUR", 7, 50), ("COTATION", 3, 18), ("AXES", 4, 13)):
        doc.layers.add(n, color=col, lineweight=lw)
    msp = doc.modelspace()
    R, H = DIAMETER / 2, HEIGHT
    h = max(2.5, round(max(DIAMETER, HEIGHT) * 0.03, 1))
    ov = {{"dimtxt": h, "dimasz": h * 0.8, "dimexe": h * 0.4, "dimexo": h * 0.4, "dimgap": h * 0.3,
          "dimdec": 2, "dimzin": 8, "dimlfac": 1.0, "dimtih": 0, "dimtoh": 0}}
    att = {{"layer": "COTATION"}}
    # vue de face
    msp.add_lwpolyline([(-R, 0), (R, 0), (R, H), (-R, H)], close=True, dxfattribs={{"layer": "CONTOUR"}})
    msp.add_line((0, -h), (0, H + h), dxfattribs={{"layer": "AXES", "linetype": "CENTER"}})
    d = msp.add_linear_dim(base=(0, -3 * h), p1=(-R, 0), p2=(R, 0), override=ov, dxfattribs=att)
    d.set_text("%%c<>")
    d.render()
    msp.add_linear_dim(base=(R + 3 * h, 0), p1=(R, 0), p2=(R, H), angle=90, override=ov, dxfattribs=att).render()
    # vue de dessus, à droite
    cx = R * 2 + 8 * h + R
    cy = H / 2
    msp.add_circle((cx, cy), R, dxfattribs={{"layer": "CONTOUR"}})
    msp.add_line((cx - R - h, cy), (cx + R + h, cy), dxfattribs={{"layer": "AXES", "linetype": "CENTER"}})
    msp.add_line((cx, cy - R - h), (cx, cy + R + h), dxfattribs={{"layer": "AXES", "linetype": "CENTER"}})
    msp.add_diameter_dim(center=(cx, cy), radius=R, angle=45, override=ov, dxfattribs=att).render()
    msp.add_text("Vue de face", height=h, dxfattribs=att).set_placement((-R, H + 2 * h))
    msp.add_text("Vue de dessus", height=h, dxfattribs=att).set_placement((cx - R, cy + R + 2 * h))
    doc.saveas(path)
    return doc


def write_svg(doc, path):
    from ezdxf.addons.drawing import Frontend, RenderContext, config, layout, svg
    backend = svg.SVGBackend()
    cfg = config.Configuration(background_policy=config.BackgroundPolicy.WHITE,
                               color_policy=config.ColorPolicy.BLACK,
                               lineweight_policy=config.LineweightPolicy.ABSOLUTE)
    Frontend(RenderContext(doc), backend, config=cfg).draw_layout(doc.modelspace())
    page = layout.Page(297, 210, layout.Units.mm, margins=layout.Margins.all(10))  # A4 paysage
    Path(path).write_text(backend.get_string(page, settings=layout.Settings(fit_page=True)), encoding="utf-8")


def main(out_dir="."):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = write_dxf(str(out / f"{{NAME}}.dxf"))
    write_svg(doc, out / f"{{NAME}}.svg")
    from build123d import export_step, export_stl
    solid = build_solid()
    export_step(solid, str(out / f"{{NAME}}.step"))
    tol = max(0.01, DIAMETER / 2000)
    export_stl(solid, str(out / f"{{NAME}}.stl"), tolerance=tol, angular_tolerance=0.1)
    bb = solid.bounding_box()
    print(f"OK volume={{solid.volume:.3f}} valid={{solid.is_valid}} "
          f"bbox={{bb.size.X:.3f}}x{{bb.size.Y:.3f}}x{{bb.size.Z:.3f}}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
'''


def synthesize_cylinder(diameter: float, height: float, out_dir: Path, name: str = "part") -> dict:
    out_dir = Path(out_dir).resolve()
    script = _CYL_TEMPLATE.format(date=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"), name=name,
                                  diameter=round(float(diameter), 6), height=round(float(height), 6))
    path = out_dir / f"{name}.py"
    path.write_text(script, encoding="utf-8")
    ok, log_txt = run_script(path, out_dir)
    report = {"script": str(path), "ok": ok, "log": log_txt}
    if not ok:
        log.error("%s a échoué :\n%s", path.name, log_txt)
    for line in log_txt.splitlines():
        if line.startswith("OK "):
            report["solid"] = dict(kv.split("=") for kv in line[3:].split())
    (out_dir / "cad_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


# =========================================================================== révolution
_REV_TEMPLATE = r'''#!/usr/bin/env python3
"""Corps de révolution généré par Sketch2CAD-Local le {date}.

PROFILE = demi-profil (x = rayon, y = hauteur, mm), fermé par l'axe x = 0.
DOME = index de l'arête « dôme » (arc centré sur l'axe) ou None.
Relancez « python {name}.py » après modification.
"""
import math
import sys
from pathlib import Path

NAME = {name!r}
PROFILE = {points}
DOME = {dome!r}
DIMS = {dims}


def segments():
    """Lignes et arc du demi-profil : ('line', p0, p1) | ('arc', p0, p1, centre, rayon, milieu)."""
    n = len(PROFILE)
    out = []
    for k in range(n):
        p0, p1 = PROFILE[k], PROFILE[(k + 1) % n]
        if k == DOME:
            dx, dy = p0[0] - p1[0], p1[1] - p0[1]          # p1 = sommet sur l'axe
            rho = (dx * dx + dy * dy) / (2 * dy)
            c = (p1[0], p1[1] - rho)
            a0 = math.atan2(p0[1] - c[1], p0[0] - c[0])
            a1 = math.atan2(p1[1] - c[1], p1[0] - c[0])
            am = (a0 + a1) / 2
            out.append(("arc", p0, p1, c, rho, (c[0] + rho * math.cos(am), c[1] + rho * math.sin(am))))
        elif math.dist(p0, p1) > 1e-9:
            out.append(("line", p0, p1))
    return out


def build_solid():
    from build123d import (Axis, BuildLine, BuildPart, BuildSketch, Line, Plane, ThreePointArc,
                           make_face, revolve)
    with BuildPart() as part:
        with BuildSketch(Plane.XZ):
            with BuildLine():
                for s in segments():
                    if s[0] == "line":
                        Line(s[1], s[2])
                    else:
                        ThreePointArc(s[1], s[5], s[2])
            make_face()
        revolve(axis=Axis.Z)
    return part.part


def write_dxf(path):
    """Vue de face : silhouette complète (profil + symétrique), axe, cotes."""
    import ezdxf
    doc = ezdxf.new("R2018", setup=True)
    doc.units = ezdxf.units.MM
    for n_, col, lw in (("CONTOUR", 7, 50), ("COTATION", 3, 18), ("AXES", 4, 13)):
        doc.layers.add(n_, color=col, lineweight=lw)
    msp = doc.modelspace()
    att_c = {{"layer": "CONTOUR"}}
    for sign in (1, -1):
        for s in segments():
            if s[0] == "line":
                a, b = s[1], s[2]
                if abs(a[0]) < 1e-9 and abs(b[0]) < 1e-9:
                    continue  # l'axe n'est pas une arête
                msp.add_line((sign * a[0], a[1]), (sign * b[0], b[1]), dxfattribs=att_c)
            else:
                _, p0, p1, c, rho, _m = s
                a0 = math.degrees(math.atan2(p0[1] - c[1], p0[0] - c[0]))
                a1 = math.degrees(math.atan2(p1[1] - c[1], p1[0] - c[0]))
                if sign == 1:
                    msp.add_arc(c, rho, a0, a1, dxfattribs=att_c)
                else:
                    msp.add_arc((-c[0], c[1]), rho, 180 - a1, 180 - a0, dxfattribs=att_c)
    ys = [p[1] for p in PROFILE]
    xs = [p[0] for p in PROFILE]
    size = max(max(ys) - min(ys), 2 * max(xs))
    h = max(2.5, round(size * 0.03, 1))
    msp.add_line((0, min(ys) - h), (0, max(ys) + h), dxfattribs={{"layer": "AXES", "linetype": "CENTER"}})
    ov = {{"dimtxt": h, "dimasz": h * 0.8, "dimexe": h * 0.4, "dimexo": h * 0.4, "dimgap": h * 0.3,
          "dimdec": 2, "dimzin": 8, "dimadec": 1, "dimazin": 2, "dimlfac": 1.0, "dimtih": 0, "dimtoh": 0}}
    att = {{"layer": "COTATION"}}
    n = len(PROFILE)
    for dm in DIMS:
        t = dm["type"]
        if t == "length":
            p1, p2 = PROFILE[dm["i"]], PROFILE[dm["j"]]
            if abs(p1[1] - p2[1]) < 1e-6 and (abs(p1[0]) < 1e-9 or abs(p2[0]) < 1e-9):
                r = max(p1[0], p2[0])   # rayon d'une face plane -> cote Ø sur toute la largeur
                d = msp.add_linear_dim(base=(0, p1[1] - 2.5 * h if p1[1] <= min(ys) + 1e-6 else p1[1] + 2.5 * h),
                                       p1=(-r, p1[1]), p2=(r, p1[1]), override=ov, dxfattribs=att)
                d.set_text("%%c<>")
                d.render()
                continue
            ang = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
            if not -90 < ang <= 90:
                p1, p2 = p2, p1
            out_n = (p2[1] - p1[1], -(p2[0] - p1[0]))
            left = (-(p2[1] - p1[1]), p2[0] - p1[0])
            side = 1 if left[0] * out_n[0] + left[1] * out_n[1] > 0 else -1
            msp.add_aligned_dim(p1=p1, p2=p2, distance=side * 2.5 * h, override=ov, dxfattribs=att).render()
        elif t == "angle":
            k = dm["k"]
            c = PROFILE[k]
            u, v = PROFILE[(k + 1) % n], PROFILE[k - 1]
            a = math.radians(dm["value"])
            if a > math.pi:
                u, v, a = v, u, 2 * math.pi - a
            a_u = math.atan2(u[1] - c[1], u[0] - c[0])
            base = (c[0] + 2.2 * h * math.cos(a_u + a / 2), c[1] + 2.2 * h * math.sin(a_u + a / 2))
            msp.add_angular_dim_3p(base=base, center=c, p1=u, p2=v, override=ov, dxfattribs=att).render()
        elif t == "arc_radius" and DOME is not None:
            s = [x for x in segments() if x[0] == "arc"][0]
            c, rho, m = s[3], s[4], s[5]
            ang = math.degrees(math.atan2(m[1] - c[1], m[0] - c[0]))
            msp.add_radius_dim(center=c, radius=rho, angle=ang, override=ov, dxfattribs=att).render()
        elif t == "radius_at":
            p = PROFILE[dm["k"]]
            d = msp.add_linear_dim(base=(0, p[1]), p1=(-p[0], p[1]), p2=(p[0], p[1]), override=ov, dxfattribs=att)
            d.set_text("%%c<>")
            d.render()
    doc.saveas(path)
    return doc


def write_svg(doc, path):
    from ezdxf.addons.drawing import Frontend, RenderContext, config, layout, svg
    backend = svg.SVGBackend()
    cfg = config.Configuration(background_policy=config.BackgroundPolicy.WHITE,
                               color_policy=config.ColorPolicy.BLACK,
                               lineweight_policy=config.LineweightPolicy.ABSOLUTE)
    Frontend(RenderContext(doc), backend, config=cfg).draw_layout(doc.modelspace())
    page = layout.Page(297, 210, layout.Units.mm, margins=layout.Margins.all(10))
    Path(path).write_text(backend.get_string(page, settings=layout.Settings(fit_page=True)), encoding="utf-8")


def main(out_dir="."):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = write_dxf(str(out / f"{{NAME}}.dxf"))
    write_svg(doc, out / f"{{NAME}}.svg")
    from build123d import export_step, export_stl
    solid = build_solid()
    export_step(solid, str(out / f"{{NAME}}.step"))
    tol = max(0.01, max(p[0] for p in PROFILE) / 1000)
    export_stl(solid, str(out / f"{{NAME}}.stl"), tolerance=tol, angular_tolerance=0.1)
    bb = solid.bounding_box()
    print(f"OK volume={{solid.volume:.3f}} valid={{solid.is_valid}} "
          f"bbox={{bb.size.X:.3f}}x{{bb.size.Y:.3f}}x{{bb.size.Z:.3f}}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
'''


def synthesize_revolution(g: Graph, sol: Solution, dome_edge: Optional[int], out_dir: Path,
                          name: str = "part") -> dict:
    out_dir = Path(out_dir).resolve()
    vidx = {v.id: i for i, v in enumerate(g.vertices)}
    eidx = {e.id: (vidx[e.a], vidx[e.b]) for e in g.edges}
    dims = []
    for c in g.constraints:
        if c.implicit or c.value is None:
            continue
        if c.type == "length" and c.edge in eidx:
            i, j = eidx[c.edge]
            dims.append({"type": "length", "i": i, "j": j, "value": c.value})
        elif c.type == "angle" and c.vertex in vidx:
            dims.append({"type": "angle", "k": vidx[c.vertex], "value": c.value})
        elif c.type == "arc_radius":
            dims.append({"type": "arc_radius", "value": c.value})
        elif c.type == "radius_at" and c.vertex in vidx:
            dims.append({"type": "radius_at", "k": vidx[c.vertex], "value": c.value})
    pts = [(round(max(0.0, x), 6) if abs(x) > 1e-7 else 0.0, round(y, 6)) for x, y in sol.points]
    script = _REV_TEMPLATE.format(date=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"), name=name,
                                  points=_fmt_list(pts), dome=dome_edge, dims=_fmt_list(dims, 1))
    path = out_dir / f"{name}.py"
    path.write_text(script, encoding="utf-8")
    ok, log_txt = run_script(path, out_dir)
    report = {"script": str(path), "ok": ok, "log": log_txt}
    if not ok:
        log.error("%s a échoué :\n%s", path.name, log_txt)
    for line in log_txt.splitlines():
        if line.startswith("OK "):
            report["solid"] = dict(kv.split("=") for kv in line[3:].split())
    (out_dir / "cad_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


_SPHERE_TEMPLATE = r'''#!/usr/bin/env python3
"""Sphère générée par Sketch2CAD-Local le {date}. Modifiez DIAMETER puis relancez."""
import sys
from pathlib import Path

NAME = {name!r}
DIAMETER = {diameter!r}  # mm


def build_solid():
    from build123d import Sphere
    return Sphere(radius=DIAMETER / 2)


def write_dxf(path):
    import ezdxf
    doc = ezdxf.new("R2018", setup=True)
    doc.units = ezdxf.units.MM
    for n, col, lw in (("CONTOUR", 7, 50), ("COTATION", 3, 18), ("AXES", 4, 13)):
        doc.layers.add(n, color=col, lineweight=lw)
    msp = doc.modelspace()
    R = DIAMETER / 2
    h = max(2.5, round(DIAMETER * 0.04, 1))
    msp.add_circle((0, 0), R, dxfattribs={{"layer": "CONTOUR"}})
    msp.add_line((-R - h, 0), (R + h, 0), dxfattribs={{"layer": "AXES", "linetype": "CENTER"}})
    msp.add_line((0, -R - h), (0, R + h), dxfattribs={{"layer": "AXES", "linetype": "CENTER"}})
    ov = {{"dimtxt": h, "dimasz": h * 0.8, "dimdec": 2, "dimzin": 8, "dimlfac": 1.0, "dimtih": 0, "dimtoh": 0}}
    d = msp.add_diameter_dim(center=(0, 0), radius=R, angle=45, override=ov, dxfattribs={{"layer": "COTATION"}})
    d.set_text("S<>")
    d.render()
    doc.saveas(path)
    return doc


def write_svg(doc, path):
    from ezdxf.addons.drawing import Frontend, RenderContext, config, layout, svg
    backend = svg.SVGBackend()
    cfg = config.Configuration(background_policy=config.BackgroundPolicy.WHITE,
                               color_policy=config.ColorPolicy.BLACK,
                               lineweight_policy=config.LineweightPolicy.ABSOLUTE)
    Frontend(RenderContext(doc), backend, config=cfg).draw_layout(doc.modelspace())
    page = layout.Page(297, 210, layout.Units.mm, margins=layout.Margins.all(10))
    Path(path).write_text(backend.get_string(page, settings=layout.Settings(fit_page=True)), encoding="utf-8")


def main(out_dir="."):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = write_dxf(str(out / f"{{NAME}}.dxf"))
    write_svg(doc, out / f"{{NAME}}.svg")
    from build123d import export_step, export_stl
    solid = build_solid()
    export_step(solid, str(out / f"{{NAME}}.step"))
    export_stl(solid, str(out / f"{{NAME}}.stl"), tolerance=max(0.01, DIAMETER / 2000), angular_tolerance=0.1)
    bb = solid.bounding_box()
    print(f"OK volume={{solid.volume:.3f}} valid={{solid.is_valid}} "
          f"bbox={{bb.size.X:.3f}}x{{bb.size.Y:.3f}}x{{bb.size.Z:.3f}}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
'''


def synthesize_sphere(diameter: float, out_dir: Path, name: str = "part") -> dict:
    out_dir = Path(out_dir).resolve()
    script = _SPHERE_TEMPLATE.format(date=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"), name=name,
                                     diameter=round(float(diameter), 6))
    path = out_dir / f"{name}.py"
    path.write_text(script, encoding="utf-8")
    ok, log_txt = run_script(path, out_dir)
    report = {"script": str(path), "ok": ok, "log": log_txt}
    for line in log_txt.splitlines():
        if line.startswith("OK "):
            report["solid"] = dict(kv.split("=") for kv in line[3:].split())
    (out_dir / "cad_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


# =========================================================================== réservoir
_VESSEL_TEMPLATE = r'''#!/usr/bin/env python3
"""Réservoir généré par Sketch2CAD-Local le {date}.

Corps cylindrique vertical (axe Z, ligne de raccordement basse en z = 0), deux fonds bombés en
calotte sphérique, coque creuse d'épaisseur WALL, ouvertures à bride en façade (vers -Y),
piquages axiaux et pieds. Toutes les cotes sont en mm : modifiez VESSEL puis relancez
« python {name}.py ».
"""
import math
import sys
from pathlib import Path

NAME = {name!r}
VESSEL = {vessel!r}


def _caps(a, R_top, R_bot, H):
    """Centres (z) des calottes et flèches des fonds pour un corps de rayon a."""
    ht = R_top - math.sqrt(R_top ** 2 - a ** 2)
    hb = R_bot - math.sqrt(R_bot ** 2 - a ** 2)
    return H + ht - R_top, R_bot - hb, ht, hb


def _profile(a, zcb, Rb, zct, Rt):
    """Demi-profil (x, z) : fond bas, flanc, fond haut, fermé par l'axe."""
    zb = zcb - math.sqrt(Rb ** 2 - a ** 2)
    zt = zct + math.sqrt(Rt ** 2 - a ** 2)
    mb = (a / 2, zcb - math.sqrt(Rb ** 2 - (a / 2) ** 2))
    mt = (a / 2, zct + math.sqrt(Rt ** 2 - (a / 2) ** 2))
    return (0.0, zcb - Rb), mb, (a, zb), (a, zt), mt, (0.0, zct + Rt)


def _revolve(pts):
    from build123d import Axis, BuildLine, BuildPart, BuildSketch, Line, Plane, ThreePointArc, make_face, revolve
    p0, mb, pb, pt, mt, p1 = pts
    with BuildPart() as part:
        with BuildSketch(Plane.XZ):
            with BuildLine():
                ThreePointArc(p0, mb, pb)
                Line(pb, pt)
                ThreePointArc(pt, mt, p1)
                Line(p1, p0)
            make_face()
        revolve(axis=Axis.Z)
    return part.part


def head_z(r, top=True):
    """Altitude de la surface extérieure d'un fond à la distance r de l'axe."""
    v = VESSEL
    zct, zcb, _, _ = _caps(v["diameter"] / 2, v["R_top"], v["R_bottom"], v["height"])
    if top:
        return zct + math.sqrt(v["R_top"] ** 2 - r ** 2)
    return zcb - math.sqrt(v["R_bottom"] ** 2 - r ** 2)


def build_solid():
    from build123d import Cylinder, Pos, Rot, Align
    v = VESSEL
    a, H, t = v["diameter"] / 2, v["height"], v["wall"]
    zct, zcb, ht, hb = _caps(a, v["R_top"], v["R_bottom"], H)
    outer = _revolve(_profile(a, zcb, v["R_bottom"], zct, v["R_top"]))
    inner = _revolve(_profile(a - t, zcb, v["R_bottom"] - t, zct, v["R_top"] - t))
    solid = outer - inner
    cuts = []
    al = (Align.CENTER, Align.CENTER, Align.MIN)
    # ouvertures en façade : col + bride vers -Y, alésage à travers la paroi
    for o in v["openings"]:
        x, z, bore = o["x"], o["z"], o["bore"]
        s = math.sqrt(max(a ** 2 - x ** 2, 1.0))           # profondeur de la paroi à cette abscisse
        L = o["length"]
        neck = Pos(x, -(s - 2 * t), z) * Rot(90, 0, 0) * Cylinder(bore / 2 + t, L + 2 * t, align=al)
        solid = solid + neck
        if o.get("flange"):
            tf = max(2 * t, 0.04 * bore)
            solid = solid + Pos(x, -(s + L - tf), z) * Rot(90, 0, 0) * Cylinder(o["flange"] / 2, tf, align=al)
        cuts.append(Pos(x, -(s - 3 * t), z) * Rot(90, 0, 0) * Cylinder(bore / 2, L + 4 * t, align=al))
    # piquages axiaux (haut / bas)
    for n in v["nozzles"]:
        x, dia, hgt = n["x"], n["dia"], n["height"]
        top = n["side"] == "top"
        zs = head_z(abs(x), top)
        if top:
            z0, z1 = zs - 2 * t, zs + hgt
        else:
            z0, z1 = zs - hgt, zs + 2 * t
        solid = solid + Pos(x, 0, z0) * Cylinder(dia / 2 + t, z1 - z0, align=al)
        bz0, bz1 = (zs - 3 * t, z1 + 1) if top else (z0 - 1, zs + 3 * t)
        cuts.append(Pos(x, 0, bz0) * Cylinder(dia / 2, bz1 - bz0, align=al))
    for c in cuts:
        solid = solid - c
    # pieds : n pieds à 360/n degrés, soudés sous le fond (ou contre le corps)
    lg = v.get("legs")
    if lg:
        r, d, zg = lg["radius"], lg["dia"], lg["ground"]
        for k in range(lg["count"]):
            ang = math.radians(45 + 360 * k / lg["count"])
            px, py = r * math.cos(ang), r * math.sin(ang)
            ztop = head_z(min(r, a - d / 2), False) + 0.5 * t
            solid = solid + Pos(px, py, zg) * Cylinder(d / 2, ztop - zg, align=al)
    return solid


def _arc_pts(cx, cz, R, a, top, n=40):
    """Points d'un fond (vue de face) de x = -a à x = +a."""
    out = []
    for i in range(n + 1):
        x = -a + 2 * a * i / n
        dz = math.sqrt(max(R ** 2 - x ** 2, 0.0))
        out.append((x, cz + dz if top else cz - dz))
    return out


def write_dxf(path):
    """Vue de face cotée : coque, fonds, ouvertures, piquages, pieds."""
    import ezdxf
    v = VESSEL
    a, H, t = v["diameter"] / 2, v["height"], v["wall"]
    zct, zcb, ht, hb = _caps(a, v["R_top"], v["R_bottom"], H)
    doc = ezdxf.new("R2018", setup=True)
    doc.units = ezdxf.units.MM
    for nme, col, lw in (("CONTOUR", 7, 50), ("PERCAGES", 1, 35), ("COTATION", 3, 18), ("AXES", 4, 13)):
        doc.layers.add(nme, color=col, lineweight=lw)
    msp = doc.modelspace()
    C = {{"layer": "CONTOUR"}}
    h = max(2.5, round(max(v["diameter"], H) * 0.02, 1))
    ov = {{"dimtxt": h, "dimasz": h * 0.8, "dimexe": h * 0.4, "dimexo": h * 0.4, "dimgap": h * 0.3,
          "dimdec": 1, "dimzin": 8, "dimlfac": 1.0, "dimtih": 0, "dimtoh": 0}}
    att = {{"layer": "COTATION"}}
    msp.add_line((-a, 0), (-a, H), dxfattribs=C)
    msp.add_line((a, 0), (a, H), dxfattribs=C)
    msp.add_lwpolyline(_arc_pts(0, zct, v["R_top"], a, True), dxfattribs=C)
    msp.add_lwpolyline(_arc_pts(0, zcb, v["R_bottom"], a, False), dxfattribs=C)
    msp.add_line((0, -hb - 2 * h), (0, H + ht + 2 * h), dxfattribs={{"layer": "AXES", "linetype": "CENTER"}})
    for o in v["openings"]:
        msp.add_circle((o["x"], o["z"]), o["bore"] / 2, dxfattribs={{"layer": "PERCAGES"}})
        if o.get("flange"):
            msp.add_circle((o["x"], o["z"]), o["flange"] / 2, dxfattribs=C)
        msp.add_text(f"%%c{{o['bore']:g}}", height=h, dxfattribs=att).set_placement(
            (o["x"] - h, o["z"] - (o.get("flange") or o["bore"]) / 2 - 2 * h))
    for n in v["nozzles"]:
        top = n["side"] == "top"
        zs = head_z(abs(n["x"]), top)
        r = n["dia"] / 2 + t
        z1 = zs + n["height"] if top else zs - n["height"]
        msp.add_lwpolyline([(n["x"] - r, zs), (n["x"] - r, z1), (n["x"] + r, z1), (n["x"] + r, zs)], dxfattribs=C)
    lg = v.get("legs")
    if lg:
        for s in (-1, 1):
            x = s * lg["radius"] * math.cos(math.radians(45))
            ztop = head_z(min(abs(x), a), False)
            for dx in (-lg["dia"] / 2, lg["dia"] / 2):
                msp.add_line((x + dx, lg["ground"]), (x + dx, ztop), dxfattribs=C)
        msp.add_line((-a - 2 * h, lg["ground"]), (a + 2 * h, lg["ground"]), dxfattribs={{"layer": "AXES"}})
    d = msp.add_linear_dim(base=(0, H * 0.75), p1=(-a, H * 0.75), p2=(a, H * 0.75), override=ov, dxfattribs=att)
    d.set_text("%%c<>")
    d.render()
    msp.add_linear_dim(base=(a + 4 * h, 0), p1=(a, 0), p2=(a, H), angle=90, override=ov, dxfattribs=att).render()
    msp.add_linear_dim(base=(a + 8 * h, 0), p1=(0, -hb), p2=(0, H + ht), angle=90, override=ov,
                       dxfattribs=att).render()
    msp.add_text(f"R{{v['R_top']:g}}", height=h, dxfattribs=att).set_placement((a * 0.55, H + ht + h))
    msp.add_text(f"R{{v['R_bottom']:g}}", height=h, dxfattribs=att).set_placement((a * 0.55, -hb - 2.2 * h))
    msp.add_text(f"Paroi ep {{t:g}} mm", height=h, dxfattribs=att).set_placement((-a, -hb - 4.5 * h
                 + (lg["ground"] + hb if lg else 0)))
    doc.saveas(path)
    return doc


def write_svg(doc, path):
    from ezdxf.addons.drawing import Frontend, RenderContext, config, layout, svg
    backend = svg.SVGBackend()
    cfg = config.Configuration(background_policy=config.BackgroundPolicy.WHITE,
                               color_policy=config.ColorPolicy.BLACK,
                               lineweight_policy=config.LineweightPolicy.ABSOLUTE)
    Frontend(RenderContext(doc), backend, config=cfg).draw_layout(doc.modelspace())
    page = layout.Page(210, 297, layout.Units.mm, margins=layout.Margins.all(10))  # A4 portrait
    Path(path).write_text(backend.get_string(page, settings=layout.Settings(fit_page=True)), encoding="utf-8")


def main(out_dir="."):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = write_dxf(str(out / f"{{NAME}}.dxf"))
    write_svg(doc, out / f"{{NAME}}.svg")
    from build123d import export_step, export_stl
    solid = build_solid()
    export_step(solid, str(out / f"{{NAME}}.step"))
    tol = max(0.05, VESSEL["diameter"] / 1500)
    export_stl(solid, str(out / f"{{NAME}}.stl"), tolerance=tol, angular_tolerance=0.1)
    bb = solid.bounding_box()
    print(f"OK volume={{solid.volume:.3f}} valid={{solid.is_valid}} "
          f"bbox={{bb.size.X:.3f}}x{{bb.size.Y:.3f}}x{{bb.size.Z:.3f}}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
'''


def synthesize_vessel(values: dict, out_dir: Path, name: str = "part") -> dict:
    out_dir = Path(out_dir).resolve()
    keep = ("diameter", "height", "R_top", "R_bottom", "wall", "openings", "nozzles", "legs")
    vessel = {k: values[k] for k in keep}
    script = _VESSEL_TEMPLATE.format(date=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"), name=name, vessel=vessel)
    path = out_dir / f"{name}.py"
    path.write_text(script, encoding="utf-8")
    ok, log_txt = run_script(path, out_dir, timeout=300)
    report = {"script": str(path), "ok": ok, "log": log_txt}
    if not ok:
        log.error("%s a échoué :\n%s", path.name, log_txt)
    for line in log_txt.splitlines():
        if line.startswith("OK "):
            report["solid"] = dict(kv.split("=") for kv in line[3:].split())
    (out_dir / "cad_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
