"""Interface web locale de test (Flask), servie uniquement sur 127.0.0.1.

    python -m sketch2cad web            ->  http://127.0.0.1:8765

Les photos et les résultats restent sur la machine, dans le dossier ``sortie_web/``.
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from .graph_builder import load_graph
from .ocr import dims_from_texts
from .pipeline import load_config, run, run_from_graph
from .schema import Perception

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "sketch_photo.jpg"
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EDITABLE = {"length", "angle", "fillet", "diameter", "arc_radius", "radius_at", "height"}


def create_app(work_dir: Path | None = None, config_path: Path | None = None) -> Flask:
    app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
    app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024
    app.jinja_env.auto_reload = True  # la page suit les mises à jour du projet sans redémarrage
    work = Path(work_dir or Path.cwd() / "sortie_web").resolve()
    work.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, dict] = {}
    gpu_lock = threading.Lock()  # un seul calcul à la fois (modèles et OCR lourds)

    # -- reprise des essais précédents (liste de l'historique)
    for d in sorted(work.iterdir()):
        rep = d / "report.json"
        if d.is_dir() and rep.exists():
            jobs[d.name] = {"id": d.name, "state": "done", "name": _meta(d).get("name", d.name),
                            "created": _meta(d).get("created", rep.stat().st_mtime)}

    def _job_dir(jid: str) -> Path:
        if jid not in jobs:
            abort(404)
        return work / jid

    def _worker(jid: str, fn) -> None:
        job = jobs[jid]
        job["state"] = "queued"
        with gpu_lock:
            job["state"] = "running"
            job["started"] = time.time()
            try:
                fn()
                job["state"] = "done"
            except Exception as exc:  # noqa: BLE001
                job["state"] = "error"
                job["error"] = f"{type(exc).__name__}: {exc}"
                (work / jid / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            job["elapsed"] = round(time.time() - job["started"], 1)

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        cfg = load_config(config_path)
        return render_template("index.html", profiles=[p for p in cfg["profiles"] if p != "api"],
                               default_profile=cfg["profile"],
                               default_thickness=cfg["default_thickness"], has_example=EXAMPLE.exists())

    # ------------------------------------------------------------------ réglages du profil « api »
    def _settings_view() -> dict:
        from .cloud import PROVIDERS, load_settings, mask_key, resolve
        cfg = load_config(config_path)
        st = load_settings()
        r = resolve(cfg.get("api"), st)
        import os
        provs = {k: {"label": v["label"], "model": v["model"], "key_url": v["key_url"], "env": v["env"],
                     "has_key": bool(os.environ.get(v["env"]) or st.get("keys", {}).get(k)),
                     "key_masked": mask_key(os.environ.get(v["env"]) or st.get("keys", {}).get(k, "")),
                     "from_env": bool(os.environ.get(v["env"]))}
                 for k, v in PROVIDERS.items()}
        return {"provider": r["provider"], "model": st.get("model") or "", "effective_model": r["model"],
                "scope": r["scope"], "providers": provs}

    @app.get("/api/settings")
    def get_settings():
        return jsonify(_settings_view())

    @app.post("/api/settings")
    def set_settings():
        from .cloud import PROVIDERS, SCOPES, load_settings, save_settings
        body = request.get_json(force=True)
        st = load_settings()
        if body.get("provider") in PROVIDERS:
            if body["provider"] != st.get("provider"):
                st["model"] = ""  # modèle propre à chaque fournisseur
            st["provider"] = body["provider"]
        if "model" in body:
            st["model"] = (body.get("model") or "").strip()
        if body.get("scope") in SCOPES:
            st["scope"] = body["scope"]
        if body.get("key"):
            st.setdefault("keys", {})[st["provider"]] = body["key"].strip()
        if body.get("forget_key"):
            st.setdefault("keys", {}).pop(st["provider"], None)
        save_settings(st)
        return jsonify(_settings_view())

    @app.post("/api/settings/test")
    def test_settings():
        from .cloud import CloudError, CloudVLM, resolve
        r = resolve(load_config(config_path).get("api"))
        t0 = time.time()
        try:
            reply = CloudVLM.from_settings(r).chat(None, "Réponds seulement : OK")
        except CloudError as exc:
            return jsonify(ok=False, error=str(exc)[:400])
        return jsonify(ok=True, reply=reply.strip()[:80], seconds=round(time.time() - t0, 1), model=r["model"])

    @app.get("/api/settings/models")
    def list_models():
        from .cloud import CloudError, CloudVLM, resolve
        r = resolve(load_config(config_path).get("api"))
        try:
            return jsonify(ok=True, models=CloudVLM.from_settings(r).list_models())
        except CloudError as exc:
            return jsonify(ok=False, error=str(exc)[:400])

    # ------------------------------------------------------------------ API
    @app.post("/api/jobs")
    def create_job():
        profile = request.form.get("profile") or None
        try:
            cfg = load_config(config_path, profile)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        if request.form.get("thickness"):
            cfg["default_thickness"] = float(request.form["thickness"].replace(",", "."))
        jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        d = work / jid
        d.mkdir()
        if request.form.get("example") == "1":
            if not EXAMPLE.exists():
                return jsonify(error="Image d'exemple absente."), 400
            src = d / ("input" + EXAMPLE.suffix)
            shutil.copy(EXAMPLE, src)
            name = EXAMPLE.name
        else:
            f = request.files.get("image")
            if not f or not f.filename:
                return jsonify(error="Aucune image reçue."), 400
            ext = Path(f.filename).suffix.lower()
            if ext not in ALLOWED:
                return jsonify(error=f"Format {ext} non pris en charge."), 400
            src = d / ("input" + ext)
            f.save(src)
            name = Path(f.filename).name
        meta = {"name": name, "profile": cfg["profile"], "created": time.time(), "input": src.name,
                "default_thickness": cfg["default_thickness"], "drawing": None, "interpret": None}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        jobs[jid] = {"id": jid, "state": "queued", "name": name, "created": meta["created"]}
        threading.Thread(target=_worker, args=(jid, lambda: run(src, d, cfg, ask=True)), daemon=True).start()
        return jsonify(id=jid)

    @app.get("/api/jobs/<jid>/texts")
    def get_texts(jid):
        """Textes lus sur le dessin traité (ou sur toute la feuille si aucun dessin n'est choisi)."""
        d = _job_dir(jid)
        p = Perception.model_validate_json((d / "perception.json").read_text(encoding="utf-8"))
        tpath = d / "texts_all.json"
        extra = {t["id"]: t for t in json.loads(tpath.read_text(encoding="utf-8"))} if tpath.exists() else {}
        return jsonify(texts=[{"id": t.id, "raw": t.raw, "kind": t.kind, "value": t.value,
                               "src": extra.get(t.id, {}).get("src"), "ocr_raw": extra.get(t.id, {}).get("ocr_raw")}
                              for t in p.dimensions])

    @app.post("/api/jobs/<jid>/rerun")
    def rerun(jid):
        """Relance complète sans OCR : autre dessin, autre interprétation, ou textes corrigés."""
        d = _job_dir(jid)
        if jobs[jid]["state"] in ("queued", "running"):
            return jsonify(error="Calcul déjà en cours."), 409
        body = request.get_json(force=True)
        meta = _meta(d)
        if "drawing" in body:
            meta["drawing"] = None if body["drawing"] is None else int(body["drawing"])
            meta["interpret"] = None  # autre dessin : nouvelle interprétation
        if "interpret" in body:
            meta["interpret"] = body["interpret"]
        tpath = d / "texts_all.json"
        texts = json.loads(tpath.read_text(encoding="utf-8")) if tpath.exists() else []
        for e in body.get("texts", []):
            for t in texts:
                if t["id"] == e["id"]:
                    t["raw"] = e["raw"]
        add = [a for a in body.get("add", []) if (a.get("raw") or "").strip()]
        if add:
            p = Perception.model_validate_json((d / "perception.json").read_text(encoding="utf-8"))
            dr = p.drawings[p.drawing] if p.drawings and p.drawing is not None else None
            W, H = p.image_size
            x0, y0, x1, y1 = dr["bbox"] if dr else (0, 0, W, H)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            pos = {"right": (x1 + 15, cy), "left": (x0 - 15, cy), "top": (cx, y0 - 15),
                   "bottom": (cx, y1 + 15), "center": (cx, cy)}
            n = max([int(t["id"][1:]) for t in texts if t["id"][1:].isdigit()] + [0])
            for a in add:
                n += 1
                px, py = pos.get(a.get("where"), pos["center"])
                texts.append({"id": f"T{n}", "raw": a["raw"].strip(), "conf": 1.0,
                              "bbox": [px - 20, py - 12, px + 20, py + 12], "added": True})
        tpath.write_text(json.dumps(texts, indent=2, ensure_ascii=False), encoding="utf-8")
        (d / "dims.json").write_text(json.dumps([x.model_dump() for x in dims_from_texts(texts)], indent=2,
                                                ensure_ascii=False), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        cfg = load_config(config_path, meta.get("profile"))
        cfg["default_thickness"] = meta.get("default_thickness", cfg["default_thickness"])
        src = d / meta.get("input", "input.jpg")
        threading.Thread(target=_worker, args=(jid, lambda: run(
            src, d, cfg, dims_json=d / "dims.json", drawing=meta.get("drawing"),
            interpret=meta.get("interpret"), ask=True)), daemon=True).start()
        return jsonify(ok=True)

    @app.get("/api/jobs")
    def list_jobs():
        items = sorted(jobs.values(), key=lambda j: -j.get("created", 0))
        return jsonify([{k: j.get(k) for k in ("id", "state", "name", "created")} for j in items[:30]])

    @app.get("/api/jobs/<jid>")
    def job_status(jid):
        d = _job_dir(jid)
        job = dict(jobs[jid])
        if job["state"] == "running":
            job["elapsed"] = round(time.time() - job["started"], 1)
        rep = d / "report.json"
        if job["state"] == "done" and rep.exists():
            job["report"] = json.loads(rep.read_text(encoding="utf-8"))
            sol = d / "solution.json"
            if sol.exists():
                s = json.loads(sol.read_text(encoding="utf-8"))
            shape = s.get("shape") if sol.exists() else None
            if shape == "cylinder":
                job["size"] = [s["diameter"], s["diameter"], s["height"]]
            elif shape == "sphere":
                job["size"] = [s["diameter"]] * 3
            elif shape == "disc":
                job["size"] = [s["diameter"], s["diameter"], s["thickness"]]
            elif shape == "revolution":
                xs = [p[0] for p in s["points"]]
                ys = [p[1] for p in s["points"]]
                job["size"] = [round(2 * max(xs), 3), round(2 * max(xs), 3), round(max(ys) - min(ys), 3)]
            elif sol.exists():
                xs = [p[0] for p in s["points"]]
                ys = [p[1] for p in s["points"]]
                job["size"] = [round(max(xs) - min(xs), 3), round(max(ys) - min(ys), 3), s["thickness"]]
            job["files"] = sorted(f.name for f in d.iterdir() if f.is_file())
            job["meta"] = _meta(d)
        return jsonify(job)

    @app.get("/api/jobs/<jid>/graph")
    def get_graph(jid):
        d = _job_dir(jid)
        sol_p = d / "solution.json"
        if sol_p.exists():
            s = json.loads(sol_p.read_text(encoding="utf-8"))
            if s.get("shape") == "cylinder":
                src = s.get("sources", {})
                return jsonify(shape="cylinder", thickness=None, edges=[], vertices=[], circles=[], rows=[
                    {"index": 0, "type": "cyl_diameter", "target": None, "value": s["diameter"],
                     "source": src.get("diameter"), "note": None},
                    {"index": 1, "type": "cyl_height", "target": None, "value": s["height"],
                     "source": src.get("height"), "note": None}])
            if s.get("shape") in ("sphere", "disc"):
                rows = [{"index": 0, "type": "circ_radius", "target": None, "value": s["radius"],
                         "source": s.get("sources", {}).get("radius"), "note": None}]
                if s["shape"] == "disc":
                    rows.append({"index": 1, "type": "circ_thickness", "target": None, "value": s["thickness"],
                                 "source": None, "note": None})
                return jsonify(shape=s["shape"], thickness=None, edges=[], vertices=[], circles=[], rows=rows)
        g = load_graph(d / "graph.json")
        rows = []
        for i, c in enumerate(g.constraints):
            if c.type in EDITABLE:
                rows.append({"index": i, "type": c.type, "target": c.edge or c.vertex or c.circle,
                             "value": c.value, "source": c.source, "note": c.note})
        return jsonify(rows=rows, thickness=g.thickness, shape=g.method, edges=[e.id for e in g.edges],
                       vertices=[v.id for v in g.vertices], circles=[c.id for c in g.circles])

    @app.post("/api/jobs/<jid>/resolve")
    def resolve(jid):
        d = _job_dir(jid)
        if jobs[jid]["state"] in ("queued", "running"):
            return jsonify(error="Calcul déjà en cours."), 409
        body = request.get_json(force=True)
        sol_p = d / "solution.json"
        if sol_p.exists() and json.loads(sol_p.read_text(encoding="utf-8")).get("shape") == "cylinder":
            vals = {r["index"]: float(r["value"]) for r in body.get("rows", [])}
            old_sol = json.loads(sol_p.read_text(encoding="utf-8"))
            values = {"diameter": vals.get(0, old_sol["diameter"]), "height": vals.get(1, old_sol["height"]),
                      "sources": old_sol.get("sources", {})}
            cfg = load_config(config_path)

            def cyl_task():
                from .perception import load_image
                from .pipeline import run_cylinder
                from .schema import Perception
                p = Perception.model_validate_json((d / "perception.json").read_text(encoding="utf-8"))
                img = load_image(d / "rectified.png") if (d / "rectified.png").exists() else None
                rep = run_cylinder(p, img, d, cfg, values=values)
                rep["edited"] = True
                (d / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
            threading.Thread(target=_worker, args=(jid, cyl_task), daemon=True).start()
            return jsonify(ok=True)
        old_sol = json.loads(sol_p.read_text(encoding="utf-8")) if sol_p.exists() else {}
        if old_sol.get("shape") in ("sphere", "disc"):
            vals = {r["index"]: float(r["value"]) for r in body.get("rows", [])}
            values = {"radius": vals.get(0, old_sol["radius"]), "thickness": vals.get(1, old_sol.get("thickness")),
                      "sources": old_sol.get("sources", {})}
            cfg = load_config(config_path)

            def circ_task():
                from .perception import load_image
                from .pipeline import run_circle
                p = Perception.model_validate_json((d / "perception.json").read_text(encoding="utf-8"))
                rep = run_circle(p, None, d, cfg, old_sol["shape"], values=values)
                rep["edited"] = True
                (d / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
            threading.Thread(target=_worker, args=(jid, circ_task), daemon=True).start()
            return jsonify(ok=True)
        g = load_graph(d / "graph.json")
        targets = {"length": "edge", "height": "edge", "angle": "vertex", "fillet": "vertex", "diameter": "circle",
                   "arc_radius": "edge", "radius_at": "vertex"}
        deleted: set[int] = set()
        for row in body.get("rows", []):
            i = int(row["index"])
            if not 0 <= i < len(g.constraints) or g.constraints[i].type not in EDITABLE:
                continue
            c = g.constraints[i]
            if row.get("delete"):
                deleted.add(i)
                continue
            new_val, new_tgt = float(row["value"]), row.get("target") or getattr(c, targets[c.type])
            if new_val != c.value or new_tgt != getattr(c, targets[c.type]):
                c.value = new_val
                setattr(c, targets[c.type], new_tgt)
                c.note = "modifiée dans l'interface web"
        g.constraints = [c for i, c in enumerate(g.constraints) if i not in deleted]
        if body.get("thickness"):
            g.thickness = float(body["thickness"])
        (d / "graph.json").write_text(g.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
        cfg = load_config(config_path)
        old = json.loads((d / "report.json").read_text(encoding="utf-8")) if (d / "report.json").exists() else {}
        if g.method == "revolution":
            def rev_task():
                from .perception import load_image
                from .pipeline import run_revolution
                p = Perception.model_validate_json((d / "perception.json").read_text(encoding="utf-8"))
                img = load_image(d / "rectified.png") if (d / "rectified.png").exists() else None
                rep = run_revolution(p, img, d, cfg, graph=g)
                rep["edited"] = True
                (d / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
            threading.Thread(target=_worker, args=(jid, rev_task), daemon=True).start()
            return jsonify(ok=True)

        def task():
            rep = run_from_graph(d / "graph.json", d, cfg)
            from .critic import render_comparison
            from .perception import load_image
            from .schema import Solution
            sol = Solution.model_validate_json((d / "solution.json").read_text(encoding="utf-8"))
            if (d / "rectified.png").exists():
                render_comparison(load_image(d / "rectified.png"), sol, d / "comparison.png")
            rep["critic"] = old.get("critic", [])  # la critique porte sur l'extraction initiale
            rep["edited"] = True
            (d / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        threading.Thread(target=_worker, args=(jid, task), daemon=True).start()
        return jsonify(ok=True)

    @app.get("/files/<jid>/<path:name>")
    def files(jid, name):
        d = _job_dir(jid)
        as_dl = request.args.get("dl") == "1"
        resp = send_from_directory(d, name, as_attachment=as_dl, max_age=0)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    return app


def _meta(d: Path) -> dict:
    m = d / "meta.json"
    return json.loads(m.read_text(encoding="utf-8")) if m.exists() else {}


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True,
          config_path: Path | None = None) -> None:
    if host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("Interface exposée sur %s : les croquis seront accessibles depuis le réseau.", host)
    app = create_app(config_path=config_path)
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    print(f"\nSketch2CAD-Local — interface de test : {url}\n(Ctrl+C pour arrêter)\n")
    if open_browser:
        import webbrowser
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=False, threaded=True)
