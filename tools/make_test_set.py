"""Génère un jeu de croquis de test « à main levée », HORS ÉCHELLE, avec leur corrigé.

    python tools/make_test_set.py croquis_test   (18 croquis)

Chaque croquis a une vérité terrain (attendu.json) : dimensions exactes, forme, perçages.
Les tracés sont volontairement déformés (proportions fausses, traits tremblés) : seules
les cotes écrites font foi, comme sur un vrai croquis d'atelier.
Polices manuscrites : tools/fonts (SIL Open Font License).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sketch2cad.profile import profile_polyline  # noqa: E402

FONTS = sorted((Path(__file__).resolve().parent / "fonts").glob("*.ttf"))
INKS = [(70, 45, 30), (150, 70, 35), (40, 40, 40)]          # BGR : bleu-noir, bleu bille, noir


# --------------------------------------------------------------------------- feuille

class Sheet:
    """Une feuille blanche ; tout est dessiné en coordonnées px « y vers le haut »."""

    def __init__(self, W=1600, H=1200, seed=0, font=0, ink=0, thick=4, thin=2):
        self.W, self.H = W, H
        self.rng = np.random.default_rng(seed)
        paper = np.full((H, W, 3), 248, np.float32)
        paper -= np.linspace(0, 14, W)[None, :, None] * self.rng.uniform(0.5, 1.2)
        paper += self.rng.normal(0, 1.8, (H, W, 1))
        self.img = np.clip(paper, 0, 255).astype(np.uint8)
        self.ink = INKS[ink % len(INKS)]
        self.font = str(FONTS[font % len(FONTS)])
        self.thick, self.thin = thick, thin
        self.texts: list[tuple[str, tuple[float, float], int]] = []

    def P(self, p):
        return (int(round(p[0])), int(round(self.H - p[1])))

    def wobble(self, pts, closed=False, amp=2.2, step=7.0):
        """Densifie une polyligne et la fait trembler (main levée)."""
        pts = [np.asarray(p, float) for p in pts]
        if closed:
            pts = pts + [pts[0]]
        out = []
        for a, b in zip(pts[:-1], pts[1:]):
            L = np.linalg.norm(b - a)
            n = max(2, int(L / step))
            t = np.linspace(0, 1, n, endpoint=False)
            out += list(a + np.outer(t, b - a))
        out.append(pts[-1])
        out = np.array(out)
        k = len(out)
        s = np.arange(k)
        noise = (amp * np.sin(s / k * math.pi * self.rng.uniform(2, 6) + self.rng.uniform(0, 6))
                 + self.rng.normal(0, 0.35, k))
        d = np.gradient(out, axis=0)
        nrm = np.stack([-d[:, 1], d[:, 0]], 1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        if not closed:
            noise[0] = noise[-1] = 0
        return out + nrm * noise[:, None]

    def stroke(self, pts, w, closed=False, amp=2.2):
        q = self.wobble(pts, closed, amp)
        cv2.polylines(self.img, [np.array([self.P(p) for p in q], np.int32)], closed, self.ink, w, cv2.LINE_AA)

    def line(self, a, b, w=None):
        self.stroke([a, b], w or self.thin, amp=0.8)

    def arrow_dim(self, a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        self.line(a, b)
        for p, q in ((a, b), (b, a)):
            d = (q - p) / (np.linalg.norm(q - p) + 1e-9)
            n = np.array([-d[1], d[0]])
            for s in (1, -1):
                cv2.line(self.img, self.P(q), self.P(q - 17 * d + 6 * s * n), self.ink, self.thin, cv2.LINE_AA)

    def ellipse(self, c, ax, ay, w, a0=0, a1=360, dashed=False):
        t = np.radians(np.linspace(a0, a1, max(12, int(abs(a1 - a0) / 4))))
        pts = np.stack([c[0] + ax * np.cos(t), c[1] + ay * np.sin(t)], 1)
        if dashed:
            for i in range(0, len(pts) - 2, 4):
                self.stroke(pts[i:i + 3], w, amp=0.5)
        else:
            self.stroke(list(pts), w, closed=(a1 - a0 >= 360), amp=1.2)

    def text(self, txt, c, size=54):
        self.texts.append((txt, (float(c[0]), float(c[1])), size))

    def render(self) -> np.ndarray:
        pil = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        rgb = tuple(int(v) for v in self.ink[::-1])
        for txt, (x, y), size in self.texts:
            f = ImageFont.truetype(self.font, size)
            x0, y0, x1, y1 = f.getbbox(txt)
            tile = Image.new("RGBA", (x1 + 20, y1 + 20), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((10, 10), txt, font=f, fill=rgb + (255,))
            tile = tile.rotate(float(self.rng.uniform(-4, 4)), resample=Image.BICUBIC, expand=True)
            pil.paste(tile, (int(x - tile.width / 2), int(self.H - y - tile.height / 2)), tile)
        out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return cv2.GaussianBlur(out, (3, 3), 0)


# --------------------------------------------------------------------------- pièces plates

def _area(pts):
    p = np.asarray(pts)
    return 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])


def draw_plate(sh: Sheet, spec: dict, box=None) -> None:
    """Pièce plate : contour (trait fort), perçages, cotes (trait fin).

    spec : pts (mm, CCW y-haut), fillets {i: R}, holes [(x, y, D)], dims [...], thickness « ep »,
    distort (kx, ky) : proportions volontairement fausses.
    """
    pts = np.array(spec["pts"], float)
    assert _area(pts) > 0, "points CCW attendus"
    kx, ky = spec.get("distort", (1.0, 1.0))
    x0, y0, x1, y1 = box or (260, 230, sh.W - 260, sh.H - 230)
    span = np.ptp(pts, axis=0) * [kx, ky]
    s = min((x1 - x0) / span[0], (y1 - y0) / span[1])
    org = np.array([(x0 + x1) / 2, (y0 + y1) / 2]) - (pts.min(0) * [kx, ky] + span / 2) * s

    def M(p):
        return org + np.asarray(p, float) * [kx, ky] * s

    D = np.array([M(p) for p in pts]) + sh.rng.normal(0, 4, pts.shape)
    fil_px = {i: R * s * (kx + ky) / 2 for i, R in spec.get("fillets", {}).items()}
    sh.stroke(profile_polyline([tuple(p) for p in D], fil_px, step_deg=6), sh.thick, closed=True)
    cen = D.mean(0)
    hole_px = []
    for (hx, hy, Dm) in spec.get("holes", []):
        c = M((hx, hy))
        r = Dm / 2 * s * (kx + ky) / 2
        sh.ellipse(c, r, r * sh.rng.uniform(0.93, 1.05), sh.thick)
        hole_px.append((c, r))
    n = len(D)
    for d in spec.get("dims", []):
        kind = d[0]
        if kind == "len":
            _, i, txt = d[:3]
            off = d[3] if len(d) > 3 else 70
            p, q = D[i], D[(i + 1) % n]
            u = (q - p) / np.linalg.norm(q - p)
            nrm = np.array([u[1], -u[0]])                     # extérieur d'un contour CCW
            for e in (p, q):
                sh.line(e + nrm * 10, e + nrm * (off + 14))
            sh.arrow_dim(p + nrm * off, q + nrm * off)
            mid = (p + q) / 2 + nrm * (off + 30 + 12 * abs(nrm[0]) * len(txt))
            sh.text(txt, mid)
        elif kind == "ang":
            _, i, txt = d
            v, a, b = D[i], D[i - 1], D[(i + 1) % n]
            ua, ub = (a - v) / np.linalg.norm(a - v), (b - v) / np.linalg.norm(b - v)
            t0, t1 = math.degrees(math.atan2(ub[1], ub[0])), math.degrees(math.atan2(ua[1], ua[0]))
            if t1 < t0:
                t1 += 360
            sh.ellipse(v, 58, 58, sh.thin, t0, t1)
            bis = (ua + ub) / (np.linalg.norm(ua + ub) + 1e-9)
            sh.text(txt, v + bis * 150)
        elif kind == "rad":
            _, i, txt = d
            v = D[i]
            bis = (v - cen) / np.linalg.norm(v - cen)
            a = v - bis * (fil_px.get(i, 0) * 0.35)
            sh.line(a, v + bis * 70)
            sh.text(txt, v + bis * 70 + np.array([45 * np.sign(bis[0]), 0]))
        elif kind == "dia":
            _, k, txt = d
            c, r = hole_px[k]
            sh.text(txt, c + np.array([0, -(r + 38)]))
    if spec.get("thickness"):
        lo = D.min(0)
        sh.text(spec["thickness"], (max(120, lo[0] - 40), max(70, lo[1] - 150)))


def draw_cylinder(sh: Sheet, c, D_px, H_px, d_txt, h_txt):
    cx, cy = c
    a, b = D_px / 2, D_px * 0.14
    top, bot = cy + H_px / 2, cy - H_px / 2
    sh.ellipse((cx, top), a, b, sh.thick)
    sh.ellipse((cx, bot), a, b, sh.thick, 180, 360)
    sh.ellipse((cx, bot), a, b, sh.thin, 0, 180, dashed=True)
    sh.stroke([(cx - a, top), (cx - a, bot)], sh.thick)
    sh.stroke([(cx + a, top), (cx + a, bot)], sh.thick)
    if d_txt:
        sh.text(d_txt, (cx, top + 4), 46)
    if h_txt:
        x = cx + a + 70
        sh.line((cx + a + 12, top), (x + 14, top))
        sh.line((cx + a + 12, bot), (x + 14, bot))
        sh.arrow_dim((x, top), (x, bot))
        sh.text(h_txt, (x + 60, cy))


def draw_capsule(sh: Sheet, c, D_px, Hc_px, r_txt, h_txt, d_txt):
    """Gélule : base elliptique, flancs droits, dôme hémisphérique."""
    cx, cy = c
    a, b = D_px / 2, D_px * 0.14
    bot = cy - (Hc_px + a) / 2
    top = bot + Hc_px
    sh.ellipse((cx, bot), a, b, sh.thick, 180, 360)
    sh.ellipse((cx, bot), a, b, sh.thin, 0, 180, dashed=True)
    t = np.radians(np.linspace(0, 180, 60))
    dome = list(np.stack([cx + a * np.cos(t), top + a * np.sin(t)], 1))
    sh.stroke([(cx + a, bot)] + dome + [(cx - a, bot)], sh.thick)
    x = cx + a + 70
    sh.line((cx + a + 12, top), (x + 14, top))
    sh.line((cx + a + 12, bot), (x + 14, bot))
    sh.arrow_dim((x, top), (x, bot))
    sh.text(h_txt, (x + 30 + 14 * len(h_txt), (top + bot) / 2))
    sh.line((cx, top), (cx + a * 0.62, top + a * 0.78))
    sh.text(r_txt, (cx + a * 0.95, top + a * 1.15))
    sh.text(d_txt, (cx, bot - b - 45))


def draw_frustum(sh: Sheet, c, D1_px, D2_px, H_px, d1_txt, d2_txt, h_txt):
    """Tronc de cône : grande base en bas, petite en haut."""
    cx, cy = c
    a1, a2 = D1_px / 2, D2_px / 2
    bot, top = cy - H_px / 2, cy + H_px / 2
    sh.ellipse((cx, top), a2, a2 * 0.28, sh.thick)
    sh.ellipse((cx, bot), a1, a1 * 0.28, sh.thick, 180, 360)
    sh.ellipse((cx, bot), a1, a1 * 0.28, sh.thin, 0, 180, dashed=True)
    sh.stroke([(cx - a2, top), (cx - a1, bot)], sh.thick)
    sh.stroke([(cx + a2, top), (cx + a1, bot)], sh.thick)
    x = cx + a1 + 70
    sh.line((cx + a2 + 12, top), (x + 14, top))
    sh.line((cx + a1 + 12, bot), (x + 14, bot))
    sh.arrow_dim((x, top), (x, bot))
    sh.text(h_txt, (x + 55, cy))
    sh.text(d2_txt, (cx, top + a2 * 0.28 + 45))
    sh.text(d1_txt, (cx, bot - a1 * 0.28 - 45))


# --------------------------------------------------------------------------- photos

def draw_vessel(sh: Sheet, v: dict) -> None:
    """Réservoir vertical vu de face : corps, deux fonds bombés, ouvertures à bride, piquages, pieds.

    v : cx, a (demi-largeur px), yb / yt (bas / haut du corps, y vers le haut), kt / kb (profondeur
    des fonds / a), openings [(dx, y, r_in, r_out|None, texte|None)], top [(dx, w, h)],
    bottom [(dx, w, h)], legs (dx, y_sol) | None, textes D / H / R / ep et côté de la cote H.
    """
    cx, a, yb, yt = v["cx"], v["a"], v["yb"], v["yt"]
    t = np.radians(np.linspace(0, 180, 70))
    top = [(cx + a * math.cos(u), yt + v["kt"] * a * math.sin(u)) for u in t]
    bot = [(cx - a * math.cos(u), yb - v["kb"] * a * math.sin(u)) for u in t]
    sh.stroke([(cx + a, yb), (cx + a, yt)] + top + [(cx - a, yb)] + bot, sh.thick, closed=True)
    head_t = lambda x: yt + v["kt"] * a * math.sqrt(max(0.0, 1 - ((x - cx) / a) ** 2))
    head_b = lambda x: yb - v["kb"] * a * math.sqrt(max(0.0, 1 - ((x - cx) / a) ** 2))
    for dx, y, r1, r2, txt in v.get("openings", []):
        sh.ellipse((cx + dx, y), r1, r1, sh.thick)
        if r2:
            sh.ellipse((cx + dx, y), r2, r2, sh.thick)
        if txt:
            sh.text(txt, (cx + dx, y - (r2 or r1) - 40))
    for dx, w, h in v.get("top", []):
        x0, x1 = cx + dx - w / 2, cx + dx + w / 2
        sh.stroke([(x0, head_t(x0) - 2), (x0, head_t(cx + dx) + h), (x1, head_t(cx + dx) + h), (x1, head_t(x1) - 2)], sh.thick)
    for dx, w, h in v.get("bottom", []):
        x0, x1 = cx + dx - w / 2, cx + dx + w / 2
        sh.stroke([(x0, head_b(x0) + 2), (x0, head_b(cx + dx) - h), (x1, head_b(cx + dx) - h), (x1, head_b(x1) + 2)], sh.thick)
    if v.get("legs"):
        dx, yg = v["legs"]
        for x in (cx - dx, cx + dx):
            sh.stroke([(x, head_b(x) + 4), (x, yg)], sh.thick)
    side = 1 if v.get("h_side", "right") == "right" else -1
    xe = cx + side * (a + 70)
    sh.line((cx + side * (a + 12), yt), (xe + side * 14, yt)); sh.line((cx + side * (a + 12), yb), (xe + side * 14, yb))
    sh.arrow_dim((xe, yt), (xe, yb))
    sh.text(v["H"], (xe + side * (30 + 14 * len(v["H"])), (yt + yb) / 2))
    yd = v.get("d_y", yt - 0.25 * (yt - yb))
    sh.arrow_dim((cx - a, yd), (cx + a, yd)); sh.text(v["D"], (cx, yd + 40))
    if v.get("R"):
        xr = cx + 0.7 * a
        sh.line((xr, head_t(xr)), (cx + a + 40, yt + v["kt"] * a + 60))
        sh.text(v["R"], (cx + a + 40 + 14 * len(v["R"]), yt + v["kt"] * a + 70))
    if v.get("R_bot"):
        xr = cx - 0.7 * a
        sh.line((xr, head_b(xr)), (cx - a - 40, yb - v["kb"] * a - 60))
        sh.text(v["R_bot"], (cx - a - 40 - 14 * len(v["R_bot"]), yb - v["kb"] * a - 80))
    if v.get("ep"):
        sh.text(v["ep"], (max(120, cx - a - 140), max(80, yb - v["kb"] * a - 60)))


def as_photo(flat: np.ndarray, seed=3) -> np.ndarray:
    """Feuille posée en biais sur une table, éclairage inégal, ombre douce."""
    rng = np.random.default_rng(seed)
    H, W = flat.shape[:2]
    PW, PH = int(W * 1.25), int(H * 1.33)
    table = np.full((PH, PW, 3), (60, 85, 115), np.float32) + rng.normal(0, 6, (PH, PW, 3))
    dst = np.float32([[0.11 * PW, 0.10 * PH], [0.90 * PW, 0.16 * PH], [0.85 * PW, 0.91 * PH], [0.08 * PW, 0.86 * PH]])
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), dst)
    warped = cv2.warpPerspective(flat, M, (PW, PH)).astype(np.float32)
    mask = cv2.warpPerspective(np.full((H, W), 255, np.uint8), M, (PW, PH))
    shadow = cv2.GaussianBlur(np.roll(mask, (25, 25), (0, 1)).astype(np.float32) / 255, (61, 61), 0)
    table *= 1 - 0.45 * shadow[..., None]
    photo = np.where(mask[..., None] > 0, warped, table)
    light = (np.linspace(1.05, 0.78, PW)[None, :] * np.linspace(1.0, 0.9, PH)[:, None])[..., None]
    return np.clip(photo * light, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- jeu de test

L_SHAPE = dict(
    pts=[(0, 0), (100, 0), (100, 30), (40, 30), (40, 70), (0, 70)],
    holes=[(20, 50, 8)], distort=(0.8, 1.25),
    dims=[("len", 0, "100"), ("len", 1, "30"), ("len", 3, "40", 60), ("len", 4, "40"), ("len", 5, "70"),
          ("dia", 0, "Ø8")],
    thickness="ep 6")

FILLETS = dict(
    pts=[(0, 0), (140, 0), (140, 90), (0, 90)], fillets={1: 15, 2: 15},
    holes=[(35, 45, 12), (100, 45, 12)], distort=(1.0, 1.3),
    dims=[("len", 0, "140"), ("len", 3, "90"), ("rad", 1, "R15"), ("rad", 2, "R15"),
          ("dia", 0, "Ø12"), ("dia", 1, "Ø12")],
    thickness="ep 5")

CASES = [
    dict(file="01_plaque_rectangle.png", code="prism", font=0, ink=0, kind="plate",
         spec=dict(pts=[(0, 0), (120, 0), (120, 80), (0, 80)], holes=[(60, 40, 20)], distort=(1.0, 0.75),
                   dims=[("len", 0, "120"), ("len", 1, "80"), ("dia", 0, "Ø20")], thickness="ep 10"),
         expect=dict(shape="pièce plate", size=[120, 80, 10], holes=["Ø20"], volume_mm3=120 * 80 * 10 - math.pi * 100 * 10),
         what="Rectangle simple, un perçage au centre. Le croquis est volontairement trop étroit en hauteur."),
    dict(file="02_plaque_en_L.png", code="prism", font=1, ink=1, kind="plate", spec=L_SHAPE,
         expect=dict(shape="pièce plate", size=[100, 70, 6], vertices=6, holes=["Ø8"],
                     volume_mm3=(100 * 30 + 40 * 40) * 6 - math.pi * 16 * 6),
         what="Équerre en L, 6 sommets, angles droits non cotés (déduits)."),
    dict(file="03_plaque_conges.png", code="prism", font=2, ink=0, kind="plate", spec=FILLETS,
         expect=dict(shape="pièce plate", size=[140, 90, 5], holes=["Ø12", "Ø12"],
                     fillets=["R15", "R15"],
                     volume_mm3=(140 * 90 - 2 * (15 * 15 - math.pi * 225 / 4)) * 5 - 2 * math.pi * 36 * 5),
         what="Deux congés R15, deux perçages Ø12."),
    dict(file="04_plaque_angle.png", code="prism", font=0, ink=2, kind="plate",
         spec=dict(pts=[(0, 0), (120, 0), (90, 51.962), (0, 51.962)], holes=[(40, 26, 10)], distort=(1.15, 0.9),
                   dims=[("len", 0, "120"), ("len", 1, "60"), ("ang", 1, "60°"), ("dia", 0, "Ø10")],
                   thickness="ep 8"),
         expect=dict(shape="pièce plate", size=[120, 51.962, 8], vertices=4, holes=["Ø10"],
                     volume_mm3=(120 + 90) / 2 * 51.962 * 8 - math.pi * 25 * 8),
         what="Trapèze : un côté incliné coté par sa longueur (60) et l'angle (60°)."),
    dict(file="05_plaque_en_cm.png", code="prism", font=1, ink=0, kind="plate",
         spec=dict(pts=[(0, 0), (12, 0), (12, 8), (0, 8)], holes=[(4, 4, 2), (8.5, 4, 2)], distort=(0.85, 1.0),
                   dims=[("len", 0, "12 cm"), ("len", 1, "8 cm"), ("dia", 0, "Ø2 cm"), ("dia", 1, "Ø2 cm")],
                   thickness="ep 1 cm"),
         expect=dict(shape="pièce plate", size=[120, 80, 10], holes=["Ø20", "Ø20"],
                     volume_mm3=120 * 80 * 10 - 2 * math.pi * 100 * 10),
         what="Cotes en centimètres : le résultat doit sortir en millimètres."),
    dict(file="06_cylindre.png", code="cylinder", font=2, ink=1, kind="cylinder",
         expect=dict(shape="cylindre", size=[40, 40, 60], volume_mm3=math.pi * 400 * 60),
         what="Cylindre en perspective : Ø40 dans la base, hauteur 60 à droite."),
    dict(file="07_gelule.png", code="revolution", font=0, ink=0, kind="capsule",
         expect=dict(shape="corps de révolution", size=[40, 40, 80],
                     volume_mm3=math.pi * 400 * 60 + 2 / 3 * math.pi * 8000),
         what="Gélule : base Ø40, flanc 60, dôme R20 (hauteur totale 80)."),
    dict(file="08_tronc_de_cone.png", code="revolution", font=1, ink=2, kind="frustum",
         expect=dict(shape="corps de révolution", size=[60, 60, 50],
                     volume_mm3=math.pi * 50 / 3 * (900 + 900 / 4 + 450)),
         what="Tronc de cône : Ø60 en bas, Ø30 en haut, hauteur 50."),
    dict(file="09_cercle.png", code="choice_interpretation", font=2, ink=0, kind="circle",
         expect=dict(shape="choix demandé : sphère ou disque", size=[50, 50, 50],
                     note="sphère Ø50, ou disque Ø50 × 5 (épaisseur par défaut)"),
         what="Cercle seul : l'interface doit demander « sphère ou disque ? »."),
    dict(file="10_feuille_3_dessins.png", code="choice_drawing", font=0, ink=1, kind="sheet",
         expect=dict(shape="choix demandé : 3 dessins",
                     note="dessin plaque 80×50×4 + Ø10 ; cylindre Ø30 × 45 ; cercle Ø40"),
         what="Trois dessins sur la même feuille : l'interface doit demander lequel traiter."),
    dict(file="11_dessin_ouvert.png", code="open", font=1, ink=0, kind="open",
         expect=dict(shape="dessin ouvert (pas de solide)"),
         what="Un « V » non fermé : le programme doit refuser proprement."),
    dict(file="12_photo_en_biais.jpg", code="prism", font=1, ink=1, kind="photo_of", of="02_plaque_en_L.png",
         expect=dict(shape="pièce plate", size=[100, 70, 6], vertices=6, holes=["Ø8"],
                     volume_mm3=(100 * 30 + 40 * 40) * 6 - math.pi * 16 * 6),
         what="Le croquis 02 photographié en biais sur une table, avec ombre et éclairage inégal."),
    dict(file="13_capture_basse_resolution.png", code="prism", font=2, ink=0, kind="lowres_of", of="03_plaque_conges.png",
         expect=dict(shape="pièce plate", size=[140, 90, 5], holes=["Ø12", "Ø12"], fillets=["R15", "R15"]),
         what="Le croquis 03 en petite capture d'écran (480 px de large)."),
    dict(file="14_un_seul_stylo.png", code="prism", font=0, ink=2, kind="plate", single_pen=True,
         spec=dict(pts=[(0, 0), (100, 0), (100, 60), (0, 60)], holes=[(30, 30, 16)], distort=(1.1, 1.0),
                   dims=[("len", 0, "100"), ("len", 1, "60"), ("dia", 0, "Ø16")], thickness="ep 4"),
         expect=dict(shape="pièce plate", size=[100, 60, 4], holes=["Ø16"],
                     volume_mm3=100 * 60 * 4 - math.pi * 64 * 4),
         what="Contour et cotation au même stylo : séparation contour / cotes plus difficile."),
    dict(file="15_cuve_complete.png", code="vessel", font=1, ink=1, kind="vessel",
         spec=dict(cx=560, a=220, yb=560, yt=1100, kt=0.5, kb=0.5, openings=[(0, 760, 70, 100, "Ø400")],
                   top=[(-105, 50, 50), (0, 36, 40)], legs=(190, 200), D="Ø1200", H="1400", R="R1200", d_y=960),
         expect=dict(shape="réservoir", size=[1200, 1200, round(1400 + 2 * (1200 - math.sqrt(1200 ** 2 - 600 ** 2)), 3)],
                     openings=1, nozzles=2, legs=True,
                     note="corps Ø1200 × 1400, fonds R1200 (flèche 160,8), trou d'homme Ø400 à bride, 2 piquages, pieds"),
         what="Cuve complète : deux fonds bombés R1200, trou d'homme à bride, deux piquages, pieds."),
    dict(file="16_cuve_hemispherique.png", code="vessel", font=0, ink=0, kind="vessel",
         spec=dict(cx=600, a=200, yb=520, yt=980, kt=1.0, kb=1.0, top=[(0, 44, 60)], D="Ø800", H="1000",
                   R="R400", ep="ep 6", h_side="left", d_y=900),
         expect=dict(shape="réservoir", size=[800, 800, 1800], openings=0, nozzles=1, legs=False,
                     note="corps Ø800 × 1000, fonds hémisphériques R400, paroi 6 mm, un piquage"),
         what="Cuve à fonds hémisphériques, paroi cotée « ep 6 », un piquage, hauteur cotée à gauche."),
    dict(file="17_cuve_sur_pieds.png", code="vessel", font=2, ink=2, kind="vessel",
         spec=dict(cx=580, a=210, yb=520, yt=1180, kt=0.45, kb=0.45,
                   openings=[(0, 900, 80, 115, "Ø450"), (110, 640, 32, None, None)], top=[(-90, 40, 45), (80, 40, 45)],
                   bottom=[(0, 40, 40)], legs=(170, 150), D="Ø1000", H="1800", R="R1000", d_y=1080),
         expect=dict(shape="réservoir", size=[1000, 1000, round(1800 + 2 * (1000 - math.sqrt(1000 ** 2 - 500 ** 2)), 3)],
                     openings=2, nozzles=3, legs=True,
                     note="corps Ø1000 × 1800, fonds R1000, trou d'homme Ø450 + petit piquage radial, 2 piquages en haut, 1 vidange, pieds"),
         what="Cuve sur pieds : deux ouvertures en façade, piquages en haut et vidange en bas."),
    dict(file="18_cuve_photo.jpg", code="vessel", font=2, ink=2, kind="photo_of", of="17_cuve_sur_pieds.png",
         expect=dict(shape="réservoir", size=[1000, 1000, round(1800 + 2 * (1000 - math.sqrt(1000 ** 2 - 500 ** 2)), 3)],
                     openings=2, nozzles=3, legs=True),
         what="Le croquis 17 photographié en biais."),
]


def make(case: dict, out: Path, made: dict) -> np.ndarray:
    k = case["kind"]
    seed = int(case["file"][:2])
    if k == "photo_of":
        return as_photo(made[case["of"]], seed)
    if k == "lowres_of":
        src = made[case["of"]]
        s = 480 / src.shape[1]
        return cv2.resize(src, (480, int(src.shape[0] * s)), interpolation=cv2.INTER_AREA)
    pen = 3 if case.get("single_pen") else None
    sh = Sheet(seed=seed, font=case["font"], ink=case["ink"], thick=pen or 4, thin=pen or 2)
    if k == "plate":
        draw_plate(sh, case["spec"])
    elif k == "cylinder":
        draw_cylinder(sh, (800, 600), 380, 520, "Ø40", "60")
    elif k == "capsule":
        draw_capsule(sh, (780, 560), 340, 380, "R20", "60", "Ø40")
    elif k == "frustum":
        draw_frustum(sh, (780, 600), 520, 230, 420, "Ø60", "Ø30", "50")
    elif k == "vessel":
        sh = Sheet(W=1200, H=1600, seed=seed, font=case["font"], ink=case["ink"], thick=pen or 4, thin=pen or 2)
        draw_vessel(sh, case["spec"])
    elif k == "circle":
        sh.ellipse((800, 600), 300, 296, sh.thick)
        sh.text("Ø50", (800, 600))
    elif k == "open":
        sh.stroke([(450, 900), (800, 300), (1150, 900)], sh.thick)
        sh.ellipse((800, 300), 70, 70, sh.thin, 55, 125)
        sh.text("60°", (800, 430))
        sh.text("80", (560, 560))
        sh.text("80", (1040, 560))
    elif k == "sheet":
        sh = Sheet(W=1500, H=2000, seed=seed, font=case["font"], ink=case["ink"])
        draw_plate(sh, dict(pts=[(0, 0), (80, 0), (80, 50), (0, 50)], holes=[(40, 25, 10)], distort=(1.2, 0.8),
                            dims=[("len", 0, "80"), ("len", 1, "50"), ("dia", 0, "Ø10")], thickness="ep 4"),
                   box=(330, 1350, 1150, 1780))
        draw_cylinder(sh, (420, 750), 260, 360, "Ø30", "45")
        sh.ellipse((1080, 760), 190, 186, sh.thick)
        sh.text("Ø40", (1080, 760))
    img = sh.render()
    return img


CSS = """:root{--bg:#f6f5f0;--card:#fff;--ink:#1b2430;--mut:#5f6b78;--line:#dcd8cc;--acc:#1f5fbf;--ok:#1d7a46;--bad:#b3261e;--warn:#9a6200}
@media (prefers-color-scheme:dark){:root{--bg:#12161c;--card:#1a2029;--ink:#e6e9ee;--mut:#9aa5b1;--line:#2c3440;--acc:#6aa0ff;--ok:#5fd18f;--bad:#ff8a80;--warn:#f0b54a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Segoe UI",system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px}h1{font-size:24px;margin:0 0 6px}p.lead{color:var(--mut);margin:0 0 20px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}
.c{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.c img{width:100%;aspect-ratio:4/3;object-fit:contain;background:#fafafa;border-bottom:1px solid var(--line)}
.c .b{padding:12px 14px}.c h2{font-size:15px;margin:0 0 4px}.c .w{color:var(--mut);font-size:13px;margin:0 0 8px}
dl{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin:0;font-size:13px}dt{color:var(--mut)}dd{margin:0;font-family:ui-monospace,Consolas,monospace}
.r{margin-top:8px;font-size:13px;padding:6px 8px;border-radius:6px;border:1px solid var(--line)}
.r.ok{color:var(--ok)}.r.bad{color:var(--bad)}.r.part{color:var(--warn)}
table{border-collapse:collapse;width:100%;font-size:14px;background:var(--card)}td,th{border:1px solid var(--line);padding:6px 8px;text-align:left}
.steps{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 18px;margin-bottom:20px}
a{color:var(--acc)}"""


def _fmt(v):
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def write_corrige(out: Path, key: list[dict], results: dict | None = None, title="Croquis de test — corrigé") -> Path:
    """Page HTML (hors ligne) : chaque croquis, ce qu'il teste, le résultat attendu (et obtenu)."""
    import html
    cards = []
    for k in key:
        size = " × ".join(_fmt(v) for v in k["size"]) + " mm" if k.get("size") else "—"
        rows = [("Forme", k["shape"]), ("Encombrement", size)]
        if k.get("vertices"):
            rows.append(("Sommets", k["vertices"]))
        if k.get("holes"):
            rows.append(("Perçages", ", ".join(k["holes"])))
        if "openings" in k:
            rows.append(("Accessoires", f"{k['openings']} ouverture(s), {k['nozzles']} piquage(s), "
                                        f"{'pieds' if k.get('legs') else 'sans pieds'}"))
        if k.get("fillets"):
            rows.append(("Congés", ", ".join(k["fillets"])))
        if k.get("volume_mm3"):
            rows.append(("Volume", f"{k['volume_mm3']:,.0f} mm³".replace(",", " ")))
        if k.get("note"):
            rows.append(("Note", k["note"]))
        res = ""
        if results and k["file"] in results:
            r = results[k["file"]]
            res = f'<div class="r {r["cls"]}">{html.escape(r["label"])}</div>'
        cards.append(f"""<div class="c"><a href="{k['file']}"><img src="{k['file']}" alt="{html.escape(k['file'])}" loading="lazy"></a>
<div class="b"><h2>{html.escape(k['file'])}</h2><p class="w">{html.escape(k['what'])}</p>
<dl>{''.join(f'<dt>{html.escape(a)}</dt><dd>{html.escape(str(b))}</dd>' for a, b in rows)}</dl>{res}</div></div>""")
    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="lead">Croquis volontairement hors échelle : seules les cotes écrites comptent. Comparez le résultat du programme
(onglet 3D, encombrement, rapport) aux valeurs attendues.</p>
<div class="steps"><b>Comment tester</b><ol>
<li>Lancez <code>web.bat</code>, déposez un croquis, cliquez <b>Lancer</b> (mode Local, puis refaites en mode API pour comparer).</li>
<li>Comparez l'encombrement affiché et les perçages avec la carte du croquis ci-dessous.</li>
<li>Pour tester une vraie photo : imprimez <a href="croquis_a_imprimer.pdf">croquis_a_imprimer.pdf</a>, photographiez une page avec votre téléphone.</li>
<li>Tout d'un coup : <code>python tools\\bench_test_set.py croquis_test</code> produit <code>resultats.html</code>.</li></ol></div>
<div class="grid">{''.join(cards)}</div></main></body></html>"""
    path = out / ("resultats.html" if results else "corrige.html")
    path.write_text(page, encoding="utf-8")
    return path


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    made: dict[str, np.ndarray] = {}
    key = []
    for case in CASES:
        img = make(case, out, made)
        made[case["file"]] = img
        p = out / case["file"]
        if p.suffix == ".jpg":
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 86])
        else:
            cv2.imwrite(str(p), img)
        key.append({"file": case["file"], "code": case["code"], "what": case["what"], **case["expect"]})
        print("  ", case["file"])
    (out / "attendu.json").write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    write_corrige(out, key)
    # version à imprimer : une page A4 par croquis (pour tester avec une vraie photo)
    pages = []
    for case in CASES:
        if case["kind"] in ("photo_of", "lowres_of"):
            continue
        im = Image.fromarray(cv2.cvtColor(made[case["file"]], cv2.COLOR_BGR2RGB))
        A4 = Image.new("RGB", (2480, 3508), "white")
        im.thumbnail((2200, 3100))
        A4.paste(im, ((2480 - im.width) // 2, (3508 - im.height) // 2))
        pages.append(A4)
    pages[0].save(out / "croquis_a_imprimer.pdf", save_all=True, append_images=pages[1:], resolution=300)
    print(f"{len(CASES)} croquis + attendu.json + croquis_a_imprimer.pdf dans {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "croquis_test"))
