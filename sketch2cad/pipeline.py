"""Orchestrateur multi-agents : perception -> graphe -> solveur <-> critique -> CAO."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import yaml

from .cad import synthesize
from .critic import critique, render_comparison, vlm_confirm
from .graph_builder import extract_graph, load_graph
from .llm import Ollama
from .perception import load_image, perceive
from .schema import Dimension, Graph, Perception, Solution
from .solver import solve

log = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "profile": "offline",
    "ollama_host": "http://localhost:11434",
    "default_thickness": 5.0,
    "ocr_backends": ["paddle", "easyocr", "tesseract"],
    "max_critic_iterations": 3,
    "iou_min": 0.6,
    "profiles": {
        "gpu8": {"vlm": "qwen2.5vl:7b", "coder": "qwen2.5-coder:7b", "vlm_critic": True},
        "gpu4": {"vlm": "qwen2.5vl:3b", "coder": "qwen2.5-coder:3b", "vlm_critic": True},
        "cpu": {"vlm": "qwen2.5vl:3b", "coder": None, "vlm_critic": False},
        "offline": {"vlm": None, "coder": None, "vlm_critic": False},
    },
}


def load_config(path: Optional[Path] = None, profile: Optional[str] = None) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    for cand in [path, Path("config.yaml"), Path(__file__).resolve().parent.parent / "config.yaml"]:
        if cand and Path(cand).exists():
            user = yaml.safe_load(Path(cand).read_text(encoding="utf-8")) or {}
            for k, v in user.items():
                if k == "profiles":
                    cfg["profiles"].update(v)
                else:
                    cfg[k] = v
            break
    if profile:
        cfg["profile"] = profile
    if cfg["profile"] not in cfg["profiles"]:
        raise ValueError(f"Profil inconnu : {cfg['profile']} ({', '.join(cfg['profiles'])})")
    cfg["active"] = cfg["profiles"][cfg["profile"]]
    return cfg


def _models(cfg: dict) -> tuple[Optional[Ollama], Optional[str], Optional[str]]:
    act = cfg["active"]
    if not act.get("vlm") and not act.get("coder"):
        return None, None, None
    client = Ollama(cfg["ollama_host"])
    if not client.available():
        log.warning("Ollama injoignable sur %s : mode hors-ligne.", cfg["ollama_host"])
        return None, None, None
    vlm = act.get("vlm") if act.get("vlm") and client.available(act["vlm"]) else None
    coder = act.get("coder") if act.get("coder") and client.available(act["coder"]) else None
    for want, got in ((act.get("vlm"), vlm), (act.get("coder"), coder)):
        if want and not got:
            log.warning("Modèle %s absent : lancez « ollama pull %s ».", want, want)
    return client, vlm, coder


def solve_and_correct(g: Graph, p: Perception, cfg: dict, out_dir: Path) -> tuple[Graph, Solution, list]:
    """Boucle agent 3 <-> agent 5."""
    history = []
    sol = solve(g)
    for it in range(cfg["max_critic_iterations"]):
        cr = critique(g, sol, p, cfg["iou_min"])
        history.append({"iteration": it + 1, "iou": round(cr.iou, 3), "issues": cr.issues,
                        "conflicts": sol.conflicts})
        if not cr.actionable():
            break
        for c in g.constraints:
            if c.type == "angle" and (c.source or c.vertex) in cr.flip_angles:
                c.value = 360.0 - c.value
                c.note = "sens corrigé par l'agent critique"
            if c.type == "length" and c.source in cr.reassign:
                c.edge = cr.reassign[c.source]
                c.note = "rattachement corrigé par l'agent critique"
        log.info("Critique %d : %d correction(s), nouvelle résolution", it + 1,
                 len(cr.flip_angles) + len(cr.reassign))
        sol = solve(g)
    else:
        cr = critique(g, sol, p, cfg["iou_min"])
        history.append({"iteration": len(history) + 1, "iou": round(cr.iou, 3), "issues": cr.issues,
                        "conflicts": sol.conflicts})
    (out_dir / "graph.json").write_text(g.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
    return g, sol, history


OUTPUTS = ("part.py", "part.step", "part.stl", "part.dxf", "part.svg", "solution.json", "graph.json",
           "report.json", "comparison.png", "som.png", "cad_report.json", "part_llm.py")


def _text_issues(p: Perception) -> list[str]:
    """Problèmes de lecture qui rendent le résultat douteux (affichés, statut « À VÉRIFIER »)."""
    out = [f"Texte illisible {d.id} « {d.raw} » : corrigez-le dans l'onglet Textes."
           for d in p.dimensions if d.kind == "unknown"]
    if not any(d.kind != "unknown" for d in p.dimensions):
        out.append("Aucune cote lue sur ce dessin : dimensions arbitraires. Corrigez/ajoutez les textes.")
    return out


def run(image: Path, out_dir: Path, cfg: dict, dims_json: Optional[Path] = None,
        name: str = "part", extra_request: str = "", drawing: Optional[int] = None,
        interpret: Optional[str] = None, ask: bool = False) -> dict:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in OUTPUTS:  # pas de fichiers d'un essai précédent qui traînent
        (out_dir / f).unlink(missing_ok=True)
    client, vlm, coder = _models(cfg)
    dims = None
    if dims_json:
        dims = [Dimension(**d) for d in json.loads(Path(dims_json).read_text(encoding="utf-8"))]

    p, img = perceive(image, out_dir, cfg["ocr_backends"], dims, drawing=drawing, ask=ask)   # agent 1
    if p.shape == "choose_drawing":
        return _choice_report(out_dir, cfg, p, "drawing",
                              [f"dessin {d['index'] + 1}" for d in p.drawings],
                              f"{len(p.drawings)} dessins trouvés sur la feuille : choisissez celui à traiter.")
    if p.shape == "circle":
        if interpret not in ("sphere", "disc"):
            return _choice_report(out_dir, cfg, p, "interpretation", ["sphere", "disc"],
                                  "Cercle seul : s'agit-il d'une sphère ou d'un disque ?")
        return run_circle(p, img, out_dir, cfg, interpret, name)
    if p.shape == "open":
        return _simple_report(out_dir, cfg, p, "open", "-", {"ok": False}, [],
                              ["Dessin ouvert : le contour n'est pas fermé, on ne peut pas en faire un solide. "
                               "Fermez le trait sur le papier, ou choisissez un autre dessin."], {})
    if p.shape == "cylinder":
        return run_cylinder(p, img, out_dir, cfg, name)
    if p.shape == "revolution":
        return run_revolution(p, img, out_dir, cfg, name)
    g = extract_graph(p, out_dir, client, vlm, cfg["default_thickness"])                  # agent 2
    g, sol, history = solve_and_correct(g, p, cfg, out_dir)                               # agents 3+5
    (out_dir / "solution.json").write_text(sol.model_dump_json(indent=2), encoding="utf-8")
    cmp_path = render_comparison(img, sol, out_dir / "comparison.png")
    vlm_note = None
    if cfg["active"].get("vlm_critic") and client and vlm:
        cr = critique(g, sol, p, cfg["iou_min"])
        vlm_note = vlm_confirm(client, vlm, cmp_path, cr)
    cad = synthesize(g, sol, out_dir, name, client, coder, extra_request)                 # agent 4
    return _report(out_dir, cfg, p, g, sol, history, cad, vlm_note)


def _choice_report(out_dir: Path, cfg: dict, p: Perception, kind: str, options: list[str], msg: str) -> dict:
    rep = {"status": "CHOIX REQUIS", "choice": {"type": kind, "options": options, "message": msg},
           "shape": p.shape, "profile": cfg["profile"], "graph_method": "-", "vertices": 0, "holes": 0,
           "thickness_mm": 0, "dof_remaining": 0, "max_residual_mm": 0.0, "conflicts": [],
           "relaxed_implicit": [], "warnings": p.warnings, "critic": [], "vlm_critic": None,
           "cad_ok": False, "solid": None, "drawings": p.drawings}
    rep["files"] = sorted(f.name for f in out_dir.iterdir() if f.is_file()) + ["report.json"]
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return rep


def _simple_report(out_dir, cfg, p, shape, method, cad, warns, confl, extra) -> dict:
    issues = _text_issues(p) if p else []
    status = ("À VÉRIFIER (voir report.json)" if not cad.get("ok") or confl or issues
              else "OK avec avertissements" if warns else "OK")
    rep = {"status": status, "shape": shape, "profile": cfg["profile"], "graph_method": method,
           "vertices": 0, "holes": 0, "thickness_mm": 0, "dof_remaining": 0, "max_residual_mm": 0.0,
           "conflicts": confl, "relaxed_implicit": [], "warnings": (p.warnings if p else []) + issues + warns,
           "critic": [], "vlm_critic": None, "cad_ok": cad.get("ok"), "solid": cad.get("solid"),
           "drawing": p.drawing if p else None, "drawings": p.drawings if p else []}
    rep.update(extra)
    rep["files"] = sorted(f.name for f in out_dir.iterdir() if f.is_file()) + ["report.json"]
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return rep


# --------------------------------------------------------------------------- cercle

def run_circle(p: Perception, img, out_dir: Path, cfg: dict, interpret: str, name: str = "part",
               values: Optional[dict] = None) -> dict:
    from .cad import synthesize_cylinder, synthesize_sphere
    import math as _m
    warns, confl = [], []
    c = p.circle
    if values is None:
        R, thick, src = None, None, {}
        for d in p.dimensions:
            if d.kind == "radius":
                r_ = d.value
            elif d.kind == "diameter" or (d.kind == "ambiguous"):
                r_ = d.value / 2
            elif d.kind == "length":
                inside = _m.dist(d.center, (c["cx"], c["cy"])) < c["r_px"]
                r_ = d.value if inside else d.value / 2
                warns.append(f"{d.id} « {d.raw} » lu comme {'le rayon (écrit sur le rayon tracé)' if inside else 'le diamètre'}.")
            elif d.kind == "thickness":
                thick = d.value
                continue
            else:
                continue
            if R is not None and abs(R - r_) > 1e-6:
                confl.append(f"rayon coté deux fois : {R:g} et {r_:g} mm")
                continue
            R, src["radius"] = r_, d.id
        if R is None:
            R = round(c["r_px"] * 0.25, 1)
            warns.append("Aucune cote de rayon lue : rayon arbitraire (0,25 mm/px).")
        if interpret == "disc" and thick is None:
            thick = cfg["default_thickness"]
            warns.append(f"Épaisseur du disque non écrite : {thick:g} mm par défaut.")
        values = {"radius": R, "thickness": thick, "sources": src}
    R, thick = float(values["radius"]), values.get("thickness")
    sol = {"shape": interpret, "radius": R, "diameter": 2 * R, "thickness": thick,
           "sources": values.get("sources", {}), "warnings": warns, "conflicts": confl}
    (out_dir / "solution.json").write_text(json.dumps(sol, indent=2, ensure_ascii=False), encoding="utf-8")
    cad = synthesize_sphere(2 * R, out_dir, name) if interpret == "sphere" else \
        synthesize_cylinder(2 * R, float(thick), out_dir, name)
    extra = {"diameter_mm": 2 * R, "thickness_mm": float(thick) if interpret == "disc" else 2 * R,
             "interpretation": interpret}
    return _simple_report(out_dir, cfg, p, interpret, "primitive", cad, warns, confl, extra)


# --------------------------------------------------------------------------- révolution

def run_revolution(p: Perception, img, out_dir: Path, cfg: dict, name: str = "part",
                   graph: Optional[Graph] = None) -> dict:
    import math as _m
    import numpy as np
    from .cad import synthesize_revolution
    from .revolution import build_revolution_graph, half_profile, rev_polyline
    from .solver import interior_angle
    warns: list[str] = []
    if graph is None:
        g, w = build_revolution_graph(p.revolution, p.dimensions)
        warns += w
    else:
        g = graph
    _, dome_edge = half_profile(p.revolution)
    if not any(c.type == "arc_tangent" for c in g.constraints):
        dome_edge = None
    sol = solve(g)
    # angle écrit très différent du dessin : sens de mesure à vérifier
    drawn = np.array([[v.px[0], -v.px[1]] for v in g.vertices])
    P = np.array(sol.points)
    for c in g.constraints:
        if c.type == "angle" and c.vertex:
            k = g.vidx(c.vertex)
            a_d = _m.degrees(interior_angle(drawn, k))
            if abs(a_d - c.value) > 30:
                warns.append(f"Angle {c.value:g}° ({c.source}) en {c.vertex} : le dessin montre environ {a_d:.0f}°. "
                             "Vérifiez de quel côté l'angle est mesuré.")
    g.method = "revolution"
    (out_dir / "graph.json").write_text(g.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
    (out_dir / "solution.json").write_text(json.dumps({**json.loads(sol.model_dump_json()), "shape": "revolution",
                                                        "dome_edge": dome_edge}, indent=2), encoding="utf-8")
    if img is not None:
        half = rev_polyline(sol.points, dome_edge)
        full = [(x, y) for x, y in half if x > 1e-9] + [(-x, y) for x, y in reversed(half) if x > 1e-9]
        from .schema import Solution as _S
        render_comparison(img, _S(points=full, thickness=0.0), out_dir / "comparison.png")
    cad = synthesize_revolution(g, sol, dome_edge, out_dir, name)
    ys = [y for _, y in sol.points]
    extra = {"vertices": len(g.vertices), "diameter_mm": 2 * max(x for x, _ in sol.points),
             "thickness_mm": max(ys) - min(ys), "dof_remaining": sol.dof, "max_residual_mm": sol.max_residual_mm,
             "relaxed_implicit": sol.relaxed}
    return _simple_report(out_dir, cfg, p, "revolution", "revolution", cad, warns + g.warnings + sol.warnings
                          + ([f"Profil sous-coté : {sol.dof} degré(s) de liberté complété(s) d'après le dessin."]
                             if sol.dof else []), sol.conflicts, extra)


def run_from_graph(graph_path: Path, out_dir: Path, cfg: dict, name: str = "part") -> dict:
    """Relance solveur + CAO depuis un graph.json corrigé à la main."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    g = load_graph(graph_path)
    sol = solve(g)
    (out_dir / "solution.json").write_text(sol.model_dump_json(indent=2), encoding="utf-8")
    cad = synthesize(g, sol, out_dir, name)
    return _report(out_dir, cfg, None, g, sol, [], cad, None)


# --------------------------------------------------------------------------- cylindre

def run_cylinder(p: Perception, img, out_dir: Path, cfg: dict, name: str = "part",
                 values: Optional[dict] = None) -> dict:
    """Branche « primitive » : le cylindre est entièrement défini par Ø et hauteur."""
    from .cad import synthesize_cylinder
    from .primitives import assign_cylinder_dims
    if values is None:
        values, warns, confl = assign_cylinder_dims(p.cylinder, p.dimensions)
    else:
        warns, confl = [], []
    D, H = float(values["diameter"]), float(values["height"])
    sol = {"shape": "cylinder", "diameter": D, "radius": D / 2, "height": H,
           "sources": values.get("sources", {}), "warnings": warns, "conflicts": confl}
    (out_dir / "solution.json").write_text(json.dumps(sol, indent=2, ensure_ascii=False), encoding="utf-8")
    if img is not None:
        _cylinder_comparison(img, D, H, out_dir / "comparison.png")
    cad = synthesize_cylinder(D, H, out_dir, name)
    issues = _text_issues(p) if p else []
    warns = issues + warns
    status = ("À VÉRIFIER (voir report.json)" if not cad.get("ok") or confl or issues
              else "OK avec avertissements" if warns else "OK")
    rep = {
        "status": status, "shape": "cylinder", "drawing": p.drawing if p else None,
        "drawings": p.drawings if p else [], "profile": cfg["profile"], "graph_method": "primitive",
        "vertices": 0, "holes": 0, "thickness_mm": H, "diameter_mm": D, "dof_remaining": 0,
        "max_residual_mm": 0.0, "conflicts": confl, "relaxed_implicit": [],
        "warnings": (p.warnings if p else []) + warns, "critic": [], "vlm_critic": None,
        "cad_ok": cad.get("ok"), "solid": cad.get("solid"),
    }
    rep["files"] = sorted(f.name for f in out_dir.iterdir() if f.is_file()) + ["report.json"]
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return rep


def _cylinder_comparison(sketch, D: float, H: float, path: Path) -> None:
    import numpy as np
    h = 600
    s = cv2.resize(sketch, (int(sketch.shape[1] * h / sketch.shape[0]), h))
    canvas = np.full((h, 420, 3), 255, np.uint8)
    sc = min(300 / D, 440 / H)
    w, hh, e = D * sc, H * sc, 0.28 * D * sc
    x0, y0 = int(210 - w / 2), int(80 + e / 2)
    ink = (40, 40, 40)
    cv2.ellipse(canvas, (210, y0), (int(w / 2), int(e / 2)), 0, 0, 360, ink, 2, cv2.LINE_AA)
    cv2.ellipse(canvas, (210, int(y0 + hh)), (int(w / 2), int(e / 2)), 0, 0, 180, ink, 2, cv2.LINE_AA)
    cv2.line(canvas, (x0, y0), (x0, int(y0 + hh)), ink, 2, cv2.LINE_AA)
    cv2.line(canvas, (int(x0 + w), y0), (int(x0 + w), int(y0 + hh)), ink, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"D {D:g} mm  x  H {H:g} mm", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 200), 2, cv2.LINE_AA)
    out = np.hstack([s, np.full((h, 10, 3), 200, np.uint8), canvas])
    cv2.imencode(".png", out)[1].tofile(str(path))


def _report(out_dir, cfg, p, g, sol, history, cad, vlm_note) -> dict:
    last_issues = history[-1]["issues"] if history else []
    text_issues = _text_issues(p) if p else []
    if not cad.get("ok") or sol.conflicts or last_issues or text_issues:
        status = "À VÉRIFIER (voir report.json)"
    elif [u for u in g.unassigned if any(d.id == u and d.kind != "unknown" for d in (p.dimensions if p else []))]:
        status = "À VÉRIFIER (voir report.json)"  # des cotes lues n'ont pas trouvé leur élément
    elif sol.dof > 0 or g.unassigned:
        status = "OK avec avertissements"
    else:
        status = "OK"
    rep = {
        "status": status,
        "profile": cfg["profile"],
        "graph_method": g.method,
        "vertices": len(g.vertices),
        "holes": len(sol.holes),
        "thickness_mm": sol.thickness,
        "dof_remaining": sol.dof,
        "max_residual_mm": sol.max_residual_mm,
        "conflicts": sol.conflicts,
        "relaxed_implicit": sol.relaxed,
        "warnings": (p.warnings if p else []) + text_issues + g.warnings + sol.warnings,
        "drawing": p.drawing if p else None, "drawings": p.drawings if p else [],
        "critic": history,
        "vlm_critic": vlm_note,
        "cad_ok": cad.get("ok"),
        "solid": cad.get("solid"),
        "files": sorted(f.name for f in out_dir.iterdir() if f.is_file()),
    }
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return rep
