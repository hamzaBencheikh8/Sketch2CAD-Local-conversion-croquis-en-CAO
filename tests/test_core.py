"""Tests unitaires et bout-en-bout (mode offline, sans GPU ni OCR)."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sketch2cad.ocr import parse_dimension  # noqa: E402
from sketch2cad.profile import profile_segments  # noqa: E402
from sketch2cad.schema import Constraint, Edge, Graph, Vertex  # noqa: E402
from sketch2cad.solver import interior_angle, solve  # noqa: E402


@pytest.mark.parametrize("txt,exp", [
    ("150", ("length", 150.0)), ("80,5", ("length", 80.5)), ("120°", ("angle", 120.0)),
    ("R10", ("radius", 10.0)), ("Ø12", ("diameter", 12.0)), ("⌀ 8", ("diameter", 8.0)),
    ("ep 5", ("thickness", 5.0)), ("e=3", ("thickness", 3.0)), ("012", ("ambiguous", 12.0)),
    ("l5O", ("length", 150.0)), ("abc", None), ("O", None),
])
def test_parse_dimension(txt, exp):
    assert parse_dimension(txt) == exp


def _graph(px, cons):
    vs = [Vertex(id=f"V{i + 1}", px=p) for i, p in enumerate(px)]
    n = len(vs)
    es = [Edge(id=f"E{i + 1}", a=vs[i].id, b=vs[(i + 1) % n].id) for i in range(n)]
    return Graph(vertices=vs, edges=es, constraints=[Constraint(**c) for c in cons], thickness=5)


# rectangle dessiné hors échelle (y-bas, sens CCW en y-haut) : 150 x 80 attendu
RECT_PX = [(100, 400), (300, 400), (300, 100), (100, 100)]
RECT_CONS = [
    {"type": "length", "edge": "E1", "value": 150}, {"type": "length", "edge": "E2", "value": 80},
    {"type": "horizontal", "edge": "E1", "implicit": True}, {"type": "vertical", "edge": "E2", "implicit": True},
    {"type": "horizontal", "edge": "E3", "implicit": True}, {"type": "vertical", "edge": "E4", "implicit": True},
]


def test_solver_rectangle_out_of_scale():
    sol = solve(_graph(RECT_PX, RECT_CONS))
    P = np.array(sol.points)
    assert np.allclose(P, [[0, 0], [150, 0], [150, 80], [0, 80]], atol=1e-6)
    assert sol.dof == 0 and not sol.conflicts


def test_solver_reports_underconstrained():
    sol = solve(_graph(RECT_PX, RECT_CONS[:1] + RECT_CONS[2:]))
    assert sol.dof == 1


def test_solver_relaxes_implicit_before_explicit():
    # 150 en bas, 120 en haut : la verticale implicite de E4 cède, pas les cotes écrites
    cons = RECT_CONS + [{"type": "length", "edge": "E3", "value": 120, "source": "T9"}]
    sol = solve(_graph(RECT_PX, cons))
    assert not sol.conflicts and sol.relaxed
    P = np.array(sol.points)
    assert math.isclose(np.linalg.norm(P[3] - P[2]), 120, abs_tol=1e-6)


def test_solver_reports_conflict():
    # triangle impossible : 10 + 10 < 30
    px = [(0, 300), (300, 300), (150, 0)]
    cons = [{"type": "length", "edge": e, "value": v, "source": s}
            for e, v, s in (("E1", 30, "T1"), ("E2", 10, "T2"), ("E3", 10, "T3"))]
    sol = solve(_graph(px, cons))
    assert sol.conflicts


def test_angle_constraint():
    # triangle : base 100, angle 60° en V1 et V2 -> équilatéral
    px = [(0, 300), (300, 300), (150, 0)]
    cons = [{"type": "length", "edge": "E1", "value": 100},
            {"type": "angle", "vertex": "V1", "value": 60}, {"type": "angle", "vertex": "V2", "value": 60}]
    sol = solve(_graph(px, cons))
    P = np.array(sol.points)
    assert math.isclose(np.linalg.norm(P[2] - P[0]), 100, abs_tol=1e-6)
    assert math.isclose(math.degrees(interior_angle(P, 2)), 60, abs_tol=1e-6)


def test_fillet_tangent_points():
    segs = profile_segments([(0, 0), (100, 0), (100, 50), (0, 50)], {1: 10})
    arc = next(s for s in segs if s[0] == "arc")
    assert np.allclose(arc[1], (90, 0)) and np.allclose(arc[2], (100, 10)) and np.allclose(arc[3], (90, 10))


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_synthetic_sketch.py"), str(d)], check=True)
    return d


def test_end_to_end_offline(synthetic, tmp_path):
    from sketch2cad.pipeline import load_config, run
    cfg = load_config(profile="offline")
    rep = run(synthetic / "sketch_flat.png", tmp_path, cfg, dims_json=synthetic / "dims.json")
    truth = json.loads((synthetic / "truth.json").read_text())
    sol = json.loads((tmp_path / "solution.json").read_text())
    assert np.allclose(sol["points"], truth["points"], atol=0.01)
    assert sorted(h[2] for h in sol["holes"]) == sorted(truth["hole_diameters"])
    assert sol["fillets"] == {"1": 10.0}
    assert any("inversé" in i for c in rep["critic"] for i in c["issues"]), "l'agent critique doit corriger 120° -> 240°"
    assert (tmp_path / "part.dxf").exists() and (tmp_path / "part.svg").exists()
    if rep["cad_ok"] and rep.get("solid"):
        assert (tmp_path / "part.step").exists() and (tmp_path / "part.stl").exists()


class _FakeOllama:
    """Simule un VLM qui rattache correctement les cotes et propose une valeur corrigée."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def chat_json(self, model, prompt, schema, **kw):
        self.calls += 1
        return self.answer


def test_vlm_assignment_validated(tmp_path):
    from sketch2cad.graph_builder import extract_graph
    from sketch2cad.schema import Dimension, Perception
    vs = [Vertex(id=f"V{i + 1}", px=p) for i, p in enumerate(RECT_PX)]
    es = [Edge(id=f"E{i + 1}", a=vs[i].id, b=vs[(i + 1) % 4].id) for i in range(4)]
    dims = [Dimension(id="T1", raw="15O", kind="length", value=15, bbox=(180, 420, 220, 440)),
            Dimension(id="T2", raw="80", kind="length", value=80, bbox=(320, 240, 350, 260))]
    p = Perception(image_size=(400, 500), stroke_px=3, vertices=vs, edges=es, circles=[], dimensions=dims)
    (tmp_path / "som.png").write_bytes(b"")
    fake = _FakeOllama({"assignments": [
        {"text": "T1", "target": "E1", "kind": "length", "value": 150},   # corrige la lecture OCR
        {"text": "T2", "target": "C7", "kind": "length", "value": 80},    # cible invalide -> repli
    ]})
    g = extract_graph(p, tmp_path, fake, "qwen2.5vl:3b", 5.0)
    lens = {c.source: (c.edge, c.value) for c in g.constraints if c.type == "length"}
    assert g.method == "vlm"
    assert lens["T1"] == ("E1", 150)          # valeur corrigée par le VLM
    assert lens["T2"] == ("E2", 80)           # rattaché par l'heuristique de repli


def test_web_interface(tmp_path):
    from sketch2cad.webapp import create_app
    app = create_app(work_dir=tmp_path, config_path=ROOT / "config.yaml")
    c = app.test_client()
    r = c.get("/")
    assert r.status_code == 200 and "Sketch2CAD-Local" in r.get_data(as_text=True)
    assert c.get("/api/jobs").get_json() == []
    assert c.get("/static/vendor/three.module.min.js").status_code == 200
    assert c.post("/api/jobs", data={"profile": "offline"}).status_code == 400  # pas d'image


@pytest.mark.parametrize("txt,exp", [
    ("F=0.5", ("radius", 0.5, None)), ("d=2m", ("ambiguous", 2.0, "m")), ("h = 40 cm", ("length", 40.0, "cm")),
    ("r=12", ("radius", 12.0, None)), ("Ø12mm", ("diameter", 12.0, "mm")),
])
def test_parse_labels_units(txt, exp):
    from sketch2cad.ocr import parse_dimension_ex
    assert parse_dimension_ex(txt) == exp


def test_units_propagate():
    from sketch2cad.ocr import apply_units
    from sketch2cad.schema import Dimension
    d = [Dimension(id="T1", raw="r=0,5", kind="radius", value=0.5, bbox=(0, 0, 1, 1)),
         Dimension(id="T2", raw="d=2m", kind="ambiguous", value=2, unit="m", bbox=(0, 0, 1, 1))]
    apply_units(d)
    assert [x.value for x in d] == [500, 2000]


def _draw_cylinder(path, D=300, H=420, x0=250, y0=300):
    import cv2
    rng = np.random.default_rng(3)
    img = np.full((1100, 900, 3), 245, np.uint8)
    ink, e = (150, 60, 40), int(0.28 * D)
    cx = x0 + D // 2
    cv2.ellipse(img, (cx, y0), (D // 2, e // 2), 0, 0, 360, ink, 2, cv2.LINE_AA)
    cv2.ellipse(img, (cx, y0 + H), (D // 2, e // 2), 0, 0, 180, ink, 2, cv2.LINE_AA)
    for x in (x0, x0 + D):
        for ya in range(y0, y0 + H, 60):  # trait volontairement fragmenté
            cv2.line(img, (x + int(rng.integers(-2, 3)), ya), (x, min(ya + 45, y0 + H)), ink, 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def test_cylinder_detected_and_built(tmp_path):
    from sketch2cad.pipeline import load_config, run
    from sketch2cad.schema import Dimension
    img = tmp_path / "cyl.png"
    _draw_cylinder(img)
    dims = [{"id": "T1", "raw": "R=25", "kind": "radius", "value": 25, "bbox": [380, 290, 470, 320]},
            {"id": "T2", "raw": "h=120", "kind": "length", "value": 120, "bbox": [600, 480, 700, 520]}]
    (tmp_path / "dims.json").write_text(json.dumps(dims))
    rep = run(img, tmp_path / "out", load_config(profile="offline"), dims_json=tmp_path / "dims.json")
    assert rep["shape"] == "cylinder" and rep["diameter_mm"] == 50 and rep["thickness_mm"] == 120
    if rep.get("solid"):
        assert rep["solid"]["bbox"] == "50.000x50.000x120.000"


def test_rectangle_is_not_cylinder(tmp_path):
    import cv2
    from sketch2cad.primitives import detect_cylinder
    img = np.zeros((900, 900), np.uint8)
    cv2.rectangle(img, (250, 250), (600, 700), 255, 4)
    cv2.line(img, (250, 180), (600, 180), 255, 2)   # ligne de cote au-dessus
    assert detect_cylinder(img) is None


# --------------------------------------------------------------------------- v1.2 : révolution, cercle, feuille

def _draw_capsule(path):
    """Gélule : dôme R=100 px, flancs 250 px, cône vers une base elliptique."""
    import cv2
    img = np.full((900, 700, 3), 245, np.uint8)
    ink = (120, 60, 40)
    cx, R = 350, 100
    cv2.ellipse(img, (cx, 200), (R, R), 0, 180, 360, ink, 3, cv2.LINE_AA)       # dôme
    cv2.line(img, (cx - R, 200), (cx - R, 450), ink, 3, cv2.LINE_AA)
    cv2.line(img, (cx + R, 200), (cx + R, 450), ink, 3, cv2.LINE_AA)
    cv2.line(img, (cx - R, 450), (cx - 50, 540), ink, 3, cv2.LINE_AA)            # cône
    cv2.line(img, (cx + R, 450), (cx + 50, 540), ink, 3, cv2.LINE_AA)
    cv2.ellipse(img, (cx, 540), (50, 16), 0, 0, 360, ink, 3, cv2.LINE_AA)        # base
    cv2.imwrite(str(path), img)


def test_revolution_capsule(tmp_path):
    from sketch2cad.pipeline import load_config, run
    img = tmp_path / "capsule.png"
    _draw_capsule(img)
    texts = [{"id": "T1", "raw": "r=50", "bbox": [320, 60, 380, 90]},          # au-dessus du dôme
             {"id": "T2", "raw": "h=120", "bbox": [470, 310, 540, 340]},       # à droite du flanc
             {"id": "T3", "raw": "Ø40", "bbox": [420, 530, 470, 555]}]         # à côté de la base
    from sketch2cad.ocr import dims_from_texts
    (tmp_path / "dims.json").write_text(json.dumps([d.model_dump() for d in dims_from_texts(texts)]))
    rep = run(img, tmp_path / "out", load_config(profile="offline"), dims_json=tmp_path / "dims.json")
    assert rep["shape"] == "revolution"
    sol = json.loads((tmp_path / "out" / "solution.json").read_text())
    P = np.array(sol["points"])
    assert math.isclose(P[:, 0].max(), 50, abs_tol=1e-6)            # rayon du corps = rayon du dôme
    assert any(math.isclose(np.linalg.norm(P[i + 1] - P[i]), 120, abs_tol=1e-6) for i in range(len(P) - 1))
    assert math.isclose(P[1][0], 20, abs_tol=1e-6)                  # base Ø40
    if rep.get("solid"):
        assert rep["solid"]["valid"] == "True"


def test_circle_asks_sphere_or_disc(tmp_path):
    import cv2
    from sketch2cad.pipeline import load_config, run
    img = np.full((800, 800, 3), 245, np.uint8)
    cv2.circle(img, (400, 400), 200, (90, 40, 40), 3, cv2.LINE_AA)
    cv2.imwrite(str(tmp_path / "c.png"), img)
    (tmp_path / "dims.json").write_text(json.dumps([{"id": "T1", "raw": "r=5m", "kind": "radius", "value": 5000,
                                                     "bbox": [420, 380, 480, 410]}]))
    cfg = load_config(profile="offline")
    rep = run(tmp_path / "c.png", tmp_path / "o1", cfg, dims_json=tmp_path / "dims.json")
    assert rep["status"] == "CHOIX REQUIS" and rep["choice"]["options"] == ["sphere", "disc"]
    rep = run(tmp_path / "c.png", tmp_path / "o2", cfg, dims_json=tmp_path / "dims.json", interpret="sphere")
    assert rep["shape"] == "sphere" and rep["diameter_mm"] == 10000


def test_open_drawing_is_reported(tmp_path):
    import cv2
    from sketch2cad.pipeline import load_config, run
    img = np.full((800, 800, 3), 245, np.uint8)
    cv2.polylines(img, [np.array([[200, 200], [400, 600], [600, 200]])], False, (60, 40, 40), 3)
    cv2.imwrite(str(tmp_path / "v.png"), img)
    (tmp_path / "d.json").write_text("[]")
    rep = run(tmp_path / "v.png", tmp_path / "o", load_config(profile="offline"), dims_json=tmp_path / "d.json")
    assert rep["shape"] == "open" and not rep["cad_ok"]


def test_sheet_with_several_drawings(tmp_path):
    import cv2
    from sketch2cad.sheet import find_drawings
    img = np.zeros((1000, 1000), np.uint8)
    cv2.rectangle(img, (100, 100), (400, 400), 255, 3)
    cv2.circle(img, (750, 300), 120, 255, 3)
    cv2.rectangle(img, (300, 650), (700, 900), 255, 3)
    cv2.circle(img, (500, 775), 40, 255, 3)  # perçage : reste dans le même dessin
    d = find_drawings(img, [])
    assert len(d) == 3


def test_texts_reparsed_after_correction():
    from sketch2cad.ocr import dims_from_texts
    d = dims_from_texts([{"id": "T1", "raw": "r=0,5", "bbox": [0, 0, 1, 1]},
                         {"id": "T2", "raw": "d=1m", "bbox": [0, 0, 1, 1]},
                         {"id": "T3", "raw": "", "bbox": [0, 0, 1, 1]},
                         {"id": "T4", "raw": "u B", "bbox": [0, 0, 1, 1]}])
    assert [(x.id, x.kind, x.value) for x in d] == [("T1", "radius", 500), ("T2", "ambiguous", 1000),
                                                    ("T4", "unknown", 0)]


def test_split_by_darkness_low_res_capture():
    """Capture d'écran agrandie : même épaisseur partout, contour noir, cotation grise."""
    import cv2
    import numpy as np
    from sketch2cad.perception import split_strokes
    gray = np.full((800, 800), 255, np.uint8)
    cv2.rectangle(gray, (200, 200), (600, 500), 0, 5)            # contour : noir
    cv2.line(gray, (200, 580), (600, 580), 110, 5)               # ligne de cote : grise
    cv2.line(gray, (680, 200), (680, 500), 110, 5)
    binary = (gray < 200).astype(np.uint8) * 255
    strong, _sw, thr = split_strokes(binary, gray)
    assert thr is not None
    assert strong[580, 400] == 0 and strong[350, 680] == 0       # cotation écartée
    assert strong[200, 400] > 0 and strong[350, 600] > 0         # contour conservé


# --------------------------------------------------------------------------- profil « api »

class _FakeCloud:
    """Faux client : répond comme un modèle de vision, sans réseau."""
    provider, model, calls = "fake", "fake-vlm", 0

    def __init__(self, answer):
        self.answer = answer
        self.images = []

    def chat(self, model, prompt, images=None, **kw):
        self.calls += 1
        self.images += images or []
        return self.answer


def test_cloud_parse_json_tolerant():
    from sketch2cad.cloud import parse_json
    assert parse_json('```json\n{"1": "R10"}\n```') == {"1": "R10"}
    assert parse_json('Voici : {"1": "Ø12"} fin') == {"1": "Ø12"}


def test_cloud_read_texts_replaces_only_valid_readings():
    import numpy as np
    from sketch2cad.cloud import read_texts
    gray = np.full((400, 600), 255, np.uint8)
    texts = [{"id": "T1", "raw": "r= 0r", "bbox": [10, 10, 80, 40], "conf": 0.7},
             {"id": "T2", "raw": "150", "bbox": [100, 300, 160, 330], "conf": 0.95},
             {"id": "T3", "raw": "R10", "bbox": [300, 300, 360, 330], "conf": 0.9}]
    cl = _FakeCloud('```json\n{"1": "r=0,5", "2": "", "3": "n\'importe quoi"}\n```')
    out, note = read_texts(cl, gray, texts, "textes")
    assert [t["raw"] for t in out] == ["r=0,5", "150", "R10"]       # lecture OCR valide conservée
    assert out[0]["ocr_raw"] == "r= 0r" and out[0]["src"] == "api"
    assert note["changed"] == [{"id": "T1", "ocr": "r= 0r", "api": "r=0,5"}]
    assert cl.calls == 1 and cl.images[0].shape[0] < 400                  # une seule mosaïque envoyée


def test_cloud_full_scope_adds_missing_without_duplicates():
    import numpy as np
    from sketch2cad.cloud import read_texts
    gray = np.full((1000, 1000), 255, np.uint8)
    texts = [{"id": "T1", "raw": "ep 5", "bbox": [100, 900, 160, 930], "conf": 0.9}]
    cl = _FakeCloud('{"texts": {"T1": "ep 5"}, "missing": [{"text": "Ø8", "x": 50, "y": 40},'
                    ' {"text": "ep 5", "x": 80, "y": 20}, {"text": "bonjour", "x": 10, "y": 10}]}')
    out, note = read_texts(cl, gray, texts, "complet")
    assert [t["raw"] for t in out] == ["ep 5", "Ø8"]
    x0, y0, x1, y1 = out[1]["bbox"]
    assert abs((x0 + x1) / 2 - 500) < 1 and abs((y0 + y1) / 2 - 400) < 1


def test_cloud_settings_priority_and_missing_key(tmp_path, monkeypatch):
    from sketch2cad.cloud import CloudError, CloudVLM, resolve
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    st = {"provider": "groq", "model": "", "scope": "complet", "keys": {}}
    r = resolve({"provider": "gemini", "scope": "textes"}, st)
    assert r["provider"] == "groq" and r["scope"] == "complet" and "groq.com" in r["base_url"]
    try:
        CloudVLM.from_settings(r)
        assert False, "une clé absente doit lever CloudError"
    except CloudError as exc:
        assert "console.groq.com" in str(exc)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env")
    assert resolve(None, st)["key"] == "gsk_env"


def test_api_profile_falls_back_to_local_ocr(tmp_path, monkeypatch):
    """Profil api sans clé : le pipeline aboutit quand même avec l'OCR local (cotes fournies ici)."""
    import json
    import sketch2cad.cloud as cloud
    from sketch2cad.pipeline import load_config
    monkeypatch.setattr(cloud, "load_settings", lambda path=None: {"provider": "gemini", "model": "",
                                                                   "scope": "textes", "keys": {}})
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cfg = load_config(profile="api")
    reader = cloud.make_text_reader(cfg)
    assert reader is not None
    from sketch2cad.schema import Dimension
    import numpy as np
    texts, note = reader(np.full((50, 50), 255, np.uint8),
                         [Dimension(id="T1", raw="150", kind="length", value=150, bbox=(1, 1, 20, 10))])
    assert note["ok"] is False and "aistudio.google.com" in note["error"]
    assert texts[0]["raw"] == "150"
    assert cloud.make_text_reader(load_config(profile="offline")) is None
    json.dumps(note)


def test_ocr_confusions_on_handwritten_dimensions():
    from sketch2cad.ocr import parse_dimension
    assert parse_dimension("ØIO") == ("diameter", 10.0)      # I, O lus à la place de 1, 0
    assert parse_dimension("〇08") == ("diameter", 8.0)      # perçage collé à sa cote « Ø8 »
    assert parse_dimension("R1O") == ("radius", 10.0)
    assert parse_dimension("060") == ("ambiguous", 60.0)      # Ø lu comme 0
    assert parse_dimension("0,5") == ("length", 0.5)
    assert parse_dimension("Roo") is None


def test_solver_height_constraint_on_slanted_edge():
    """Cote verticale d'un flanc incliné (tronc de cône) : hauteur, pas longueur du flanc."""
    from sketch2cad.schema import Constraint, Edge, Graph, Vertex
    from sketch2cad.solver import solve
    px = [(0, 300), (260, 300), (110, 0), (0, 0)]                # demi-profil px (y vers le bas)
    V = [Vertex(id=f"V{i + 1}", px=p) for i, p in enumerate(px)]
    E = [Edge(id=f"E{i + 1}", a=V[i].id, b=V[(i + 1) % 4].id) for i in range(4)]
    cons = [Constraint(type="vertical", edge="E4"), Constraint(type="horizontal", edge="E1"),
            Constraint(type="horizontal", edge="E3"),
            Constraint(type="length", edge="E1", value=30), Constraint(type="length", edge="E3", value=15),
            Constraint(type="height", edge="E2", value=50)]
    sol = solve(Graph(vertices=V, edges=E, constraints=cons, thickness=0, method="revolution"))
    ys = [p[1] for p in sol.points]
    assert abs(max(ys) - min(ys) - 50) < 1e-6
    assert abs(max(p[0] for p in sol.points) - 30) < 1e-6


# --------------------------------------------------------------------------- réservoir

def _vessel_sketch():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from make_test_set import Sheet, draw_vessel
    sh = Sheet(W=1200, H=1600, seed=31, font=0, ink=0)
    draw_vessel(sh, dict(cx=580, a=210, yb=520, yt=1180, kt=0.45, kb=0.45,
                         openings=[(0, 900, 80, 115, None), (110, 640, 32, None, None)],
                         top=[(-90, 40, 45), (80, 40, 45)], bottom=[(0, 40, 40)], legs=(170, 150),
                         D="", H="", d_y=1080))
    sh.texts.clear()
    return sh.render()


def test_vessel_detection_finds_body_heads_and_accessories():
    from sketch2cad.perception import normalize
    from sketch2cad.vessel import detect_vessel
    _, binary = normalize(_vessel_sketch())
    v = detect_vessel(binary)
    assert v is not None
    assert len(v["openings"]) == 2 and sum(1 for o in v["openings"] if o["r_flange"]) == 1
    assert len(v["nozzles_top"]) == 2 and len(v["nozzles_bottom"]) == 1
    assert len(v["legs"]) == 2
    # le raccordement d'un fond tangent au corps est flou : la partie droite est mesurée à ~10 % près
    assert abs(v["D_px"] - 420) < 15 and abs(v["H_px"] - 660) < 0.1 * 660


def test_vessel_generatrix_end_follows_ink_not_leader_line():
    """Hough peut prolonger une génératrice jusqu'à une ligne de renvoi (« R1500 ») qui croise
    son aplomb plus haut : l'extrémité retenue doit être la fin réelle du trait (cas V03 / V14)."""
    import numpy as np
    from sketch2cad.vessel import _trace_end
    st = np.zeros((800, 300), np.uint8)
    st[300:700, 149:152] = 255             # génératrice x = 150, de y = 300 à 700
    st[150:156, 140:160] = 255             # ligne de renvoi qui passe à l'aplomb, 150 px plus haut
    st[420:428, 140:160] = 0               # petite levée de stylo dans le trait : elle est franchie
    D = 400.0
    assert abs(_trace_end(st, (0.0, 150.0), 500, 100, D) - 300) <= 1
    assert abs(_trace_end(st, (0.0, 150.0), 500, 760, D) - 699) <= 1


def test_vessel_manhole_near_head_is_not_a_back_arc():
    """Trou d'homme dessiné tout près du fond bas : ce n'est pas l'arc arrière d'un cylindre (cas V08)."""
    import cv2
    from sketch2cad.perception import normalize
    from sketch2cad.vessel import detect_vessel
    img = _vessel_sketch()
    v0 = detect_vessel(normalize(img)[1])
    D, xm, yb = v0["D_px"], int(v0["x_axis"]), v0["y_bot"]
    cv2.circle(img, (xm, int(yb - 0.15 * D)), int(0.09 * D), (30, 30, 30), 3)
    v = detect_vessel(normalize(img)[1])
    assert v is not None and len(v["openings"]) >= 1


def test_vessel_dims_and_cap_geometry():
    from sketch2cad.schema import Dimension
    from sketch2cad.vessel import assign_vessel_dims, cap_depth
    assert abs(cap_depth(1200, 600) - 160.7695) < 1e-3 and cap_depth(400, 400) == 400
    v = {"xl": 100, "xr": 500, "y_top": 200, "y_bot": 900, "D_px": 400, "H_px": 700, "x_axis": 300,
         "top": {"depth_px": 80}, "bottom": {"depth_px": 80},
         "openings": [{"cx": 300, "cy": 500, "r_bore": 60, "r_flange": 85}], "nozzles_top": [],
         "nozzles_bottom": [], "legs": []}
    dims = [Dimension(id="T1", raw="Ø1000", kind="diameter", value=1000, bbox=(260, 300, 340, 330)),
            Dimension(id="T2", raw="1800", kind="length", value=1800, bbox=(560, 530, 620, 560)),
            Dimension(id="T3", raw="Ø300", kind="diameter", value=300, bbox=(270, 600, 330, 630)),
            Dimension(id="T4", raw="R1000", kind="radius", value=1000, bbox=(520, 120, 600, 150)),
            Dimension(id="T5", raw="ep 8", kind="thickness", value=8, bbox=(20, 1000, 80, 1030))]
    out, warns, confl = assign_vessel_dims(v, dims)
    assert (out["diameter"], out["height"], out["R_top"], out["R_bottom"], out["wall"]) == (1000, 1800, 1000, 1000, 8)
    assert out["openings"][0]["bore"] == 300 and abs(out["openings"][0]["flange"] - 425) < 1
    assert not confl


def test_vessel_cad_builds_hollow_shell_with_accessories(tmp_path):
    from sketch2cad.cad import synthesize_vessel
    vals = {"diameter": 1000.0, "height": 1800.0, "R_top": 1000.0, "R_bottom": 1000.0, "wall": 5.0,
            "openings": [{"x": 0.0, "z": 1000.0, "bore": 450.0, "flange": 640.0, "length": 270.0}],
            "nozzles": [{"side": "top", "x": -200.0, "dia": 100.0, "height": 130.0}],
            "legs": {"count": 4, "radius": 460.0, "dia": 50.0, "ground": -900.0}}
    rep = synthesize_vessel(vals, tmp_path)
    assert rep["ok"], rep["log"][-800:]
    sx, sy, sz = (float(t) for t in rep["solid"]["bbox"].split("x"))
    assert abs(sx - 1000) < 1 and sz > 1800 + 2 * 133.9
    assert float(rep["solid"]["volume"]) < 0.1 * 3.1416 * 500 ** 2 * 1800   # coque creuse, pas un plein
    assert (tmp_path / "part.step").exists() and (tmp_path / "part.dxf").exists()
