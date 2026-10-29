"""Passe tout le jeu de test dans le pipeline et compare au corrigé (attendu.json).

    python tools/bench_test_set.py croquis_test                 # profil offline
    python tools/bench_test_set.py croquis_test --profile api   # lecture des cotes par l'API

Écrit croquis_test/resultats.html (même page que le corrigé, avec le verdict de chaque croquis)
et croquis_test/resultats.json ; affiche le score dans la console.
Les choix (sphère/disque, dessin de la feuille) sont vérifiés comme dans l'interface web.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

TOL = 0.005  # 0,5 % sur chaque dimension


def _size(out: Path):
    p = out / "solution.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text(encoding="utf-8"))
    sh = s.get("shape")
    if sh == "cylinder":
        return [s["diameter"], s["diameter"], s["height"]]
    if sh == "sphere":
        return [s["diameter"]] * 3
    if sh == "disc":
        return [s["diameter"], s["diameter"], s["thickness"]]
    xs = [q[0] for q in s["points"]]
    ys = [q[1] for q in s["points"]]
    if sh == "revolution":
        return [2 * max(xs), 2 * max(xs), max(ys) - min(ys)]
    return [max(xs) - min(xs), max(ys) - min(ys), s["thickness"]]


def _close(a, b):
    return a is not None and b is not None and len(a) == len(b) and all(
        abs(x - y) <= TOL * max(abs(y), 1) for x, y in zip(sorted(a), sorted(b)))


def judge(k: dict, rep: dict, size) -> tuple[str, str]:
    """Retourne (classe ok|part|bad, libellé)."""
    code = k["code"]
    if rep.get("error"):
        return "bad", "✗ erreur : " + rep["error"][:120]
    if code.startswith("choice_"):
        want = code.split("_", 1)[1]
        ok = rep.get("status") == "CHOIX REQUIS" and rep.get("choice", {}).get("type") == want
        return ("ok", f"✓ le programme demande le choix ({want})") if ok else \
               ("bad", f"✗ pas de choix demandé (statut {rep.get('status')})")
    if code == "open":
        return ("ok", "✓ dessin ouvert signalé, pas de solide") if rep.get("shape") == "open" else \
               ("bad", f"✗ forme {rep.get('shape')} au lieu de « dessin ouvert »")
    got = " × ".join(f"{v:g}" for v in size) if size else "—"
    shape_ok = (code == "prism" and rep.get("shape") in (None, "prism")) or \
               (code == "cylinder" and rep.get("shape") in ("cylinder", "revolution")) or \
               (code == "revolution" and rep.get("shape") == "revolution")
    size_ok = _close(size, k.get("size"))
    holes_ok = k.get("holes") is None or rep.get("holes") == len(k["holes"])
    if shape_ok and size_ok and holes_ok:
        return "ok", f"✓ {got} mm" + (f", {rep.get('holes')} perçage(s)" if k.get("holes") else "")
    why = []
    if not shape_ok:
        why.append(f"forme {rep.get('shape') or 'plate'}")
    if not size_ok:
        why.append(f"obtenu {got} mm")
    if not holes_ok:
        why.append(f"{rep.get('holes')} perçage(s) au lieu de {len(k['holes'])}")
    return ("part" if size_ok else "bad"), ("~ " if size_ok else "✗ ") + " ; ".join(why)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path, nargs="?", default=ROOT / "croquis_test")
    ap.add_argument("--profile", default="offline")
    ap.add_argument("-o", "--out", type=Path, help="dossier des sorties (défaut : <folder>/_sorties)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    from make_test_set import write_corrige
    from sketch2cad.pipeline import load_config, run

    folder = a.folder.resolve()
    outs = (a.out or folder / "_sorties").resolve()
    key = json.loads((folder / "attendu.json").read_text(encoding="utf-8"))
    cfg = load_config(profile=a.profile)
    results, table = {}, []
    t_all = time.time()
    for k in key:
        out = outs / Path(k["file"]).stem
        t0 = time.time()
        try:
            rep = run(folder / k["file"], out, cfg, ask=True)
        except Exception as exc:  # noqa: BLE001
            rep = {"error": f"{type(exc).__name__}: {exc}"}
            (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        cls, label = judge(k, rep, None if rep.get("error") else _size(out))
        results[k["file"]] = {"cls": cls, "label": label, "status": rep.get("status"),
                              "seconds": round(time.time() - t0, 1)}
        table.append((k["file"], cls, label))
        print(f"{'✓' if cls == 'ok' else '~' if cls == 'part' else '✗'} {k['file']:34s} {label}")
    n_ok = sum(1 for _, c, _ in table if c == "ok")
    print(f"\nScore : {n_ok}/{len(table)} réussis   (profil {a.profile}, {time.time() - t_all:.0f} s)")
    (folder / "resultats.json").write_text(json.dumps(
        {"profile": a.profile, "score": f"{n_ok}/{len(table)}", "date": time.strftime("%Y-%m-%d %H:%M"),
         "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    page = write_corrige(folder, key, results,
                         title=f"Résultats — {n_ok}/{len(table)} réussis (profil {a.profile})")
    print(f"Page : {page}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
