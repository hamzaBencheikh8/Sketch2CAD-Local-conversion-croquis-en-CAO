"""Interface en ligne de commande.

    python -m sketch2cad run croquis.jpg -o sortie/ [--profile gpu4] [--dims cotes.json]
    python -m sketch2cad solve sortie/graph.json -o sortie2/
    python -m sketch2cad doctor
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import shutil
import sys
from pathlib import Path

from .pipeline import load_config, run, run_from_graph


def _print_report(rep: dict) -> None:
    ok = rep["status"]
    print("\n=== Sketch2CAD-Local ===")
    print(f"Profil : {rep['profile']}   graphe : {rep['graph_method']}")
    if rep.get("choice"):
        print(rep["choice"]["message"])
        if rep["choice"]["type"] == "interpretation":
            print("Relancez avec --interpret sphere  ou  --interpret disc")
        print("Résultat : " + rep["status"])
        return
    if rep.get("shape") in ("cylinder", "revolution", "sphere", "disc"):
        print(f"Forme : {rep['shape']}   Ø max {rep.get('diameter_mm', 0):g} mm   hauteur : {rep['thickness_mm']:g} mm")
    else:
        print(f"Sommets : {rep['vertices']}   perçages : {rep['holes']}   épaisseur : {rep['thickness_mm']:g} mm")
    print(f"Résidu max des cotes : {rep['max_residual_mm']:.2e} mm   DDL restants : {rep['dof_remaining']}")
    for c in rep["critic"]:
        print(f"Critique #{c['iteration']} : IoU {c['iou']}  " + ("; ".join(c["issues"]) or "RAS"))
    for w in rep["warnings"]:
        print(f"  ! {w}")
    for c in rep["conflicts"]:
        print(f"  X {c}")
    if rep.get("solid"):
        print(f"Solide : {rep['solid']}")
    print("Fichiers : " + ", ".join(rep["files"]))
    print("Résultat : " + ok)


def _doctor(cfg: dict) -> int:
    print(f"Python {sys.version.split()[0]}")
    for mod in ["cv2", "numpy", "scipy", "shapely", "ezdxf", "pydantic", "yaml", "build123d",
                "paddleocr", "easyocr", "pytesseract"]:
        try:
            m = importlib.import_module(mod)
            print(f"  [ok] {mod} {getattr(m, '__version__', '')}")
        except Exception:  # noqa: BLE001
            print(f"  [--] {mod}")
    print(f"  tesseract binaire : {shutil.which('tesseract') or 'absent'}")
    from .llm import Ollama
    cl = Ollama(cfg["ollama_host"])
    if cl.available():
        print(f"  [ok] Ollama {cfg['ollama_host']}")
        for k in ("vlm", "coder"):
            m = cfg["active"].get(k)
            if m:
                print(f"     {'[ok]' if cl.available(m) else '[--]'} {k} = {m}"
                      + ("" if cl.available(m) else f"  ->  ollama pull {m}"))
    else:
        print(f"  [--] Ollama injoignable ({cfg['ollama_host']}) : mode offline uniquement")
    print(f"Profil actif : {cfg['profile']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sketch2cad", description="Croquis coté -> DXF/SVG/STEP/STL, 100 % local")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--profile", choices=["gpu8", "gpu4", "cpu", "offline"])
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="pipeline complet depuis une photo")
    r.add_argument("image", type=Path)
    r.add_argument("-o", "--out", type=Path, default=Path("sortie"))
    r.add_argument("--dims", type=Path, help="cotes saisies à la main (JSON) au lieu de l'OCR")
    r.add_argument("--name", default="part")
    r.add_argument("--extra", default="", help="fonction hors gabarit à demander au LLM codeur")
    r.add_argument("--drawing", type=int, help="n° du dessin à traiter si la feuille en contient plusieurs (1 = le plus grand)")
    r.add_argument("--interpret", choices=["sphere", "disc"], help="pour un cercle seul : sphère ou disque")

    s = sub.add_parser("solve", help="relancer solveur + CAO depuis un graph.json corrigé")
    s.add_argument("graph", type=Path)
    s.add_argument("-o", "--out", type=Path, default=Path("sortie"))
    s.add_argument("--name", default="part")

    sub.add_parser("doctor", help="vérifier l'installation et les modèles")

    w = sub.add_parser("web", help="interface web de test (http://127.0.0.1:8765)")
    w.add_argument("--port", type=int, default=8765)
    w.add_argument("--no-browser", action="store_true")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config, a.profile)
    if a.cmd == "doctor":
        return _doctor(cfg)
    if a.cmd == "web":
        from .webapp import serve
        serve(port=a.port, open_browser=not a.no_browser, config_path=a.config)
        return 0
    if a.cmd == "run":
        rep = run(a.image, a.out, cfg, a.dims, a.name, a.extra,
                  drawing=(a.drawing - 1) if a.drawing else None, interpret=a.interpret)
    else:
        rep = run_from_graph(a.graph, a.out, cfg, a.name)
    _print_report(rep)
    return 0 if rep["cad_ok"] or rep.get("choice") else 1


if __name__ == "__main__":
    sys.exit(main())
