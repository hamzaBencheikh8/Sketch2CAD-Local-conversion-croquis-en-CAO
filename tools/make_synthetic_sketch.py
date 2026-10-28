"""Génère un croquis « à main levée » volontairement HORS ÉCHELLE pour tester le pipeline.

Pièce de vérité (mm) : plaque à 6 sommets, encoche à 240° en V4 (cotée « 120° »
côté extérieur, pour tester l'agent critique), congé R10, perçages Ø12 et Ø8, ep 5.

Sorties dans le dossier donné :
  sketch_flat.png   feuille vue de face
  sketch_photo.jpg  même feuille photographiée en biais sur une table
  dims.json         cotes « OCR parfaites » (coordonnées de sketch_flat.png)
  truth.json        géométrie exacte attendue
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RNG = np.random.default_rng(7)
W, H = 1600, 1200
INK = (60, 45, 35)          # stylo bleu-noir (BGR inversé plus bas)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def turtle(lengths, interior_deg, start=(0.0, 0.0)):
    """Polygone CCW (y-haut) à partir des longueurs et angles intérieurs (V2..)."""
    pts = [np.array(start)]
    heading = 0.0
    for i, L in enumerate(lengths):
        p = pts[-1] + L * np.array([math.cos(heading), math.sin(heading)])
        pts.append(p)
        if i < len(interior_deg):
            heading += math.radians(180 - interior_deg[i])
    return pts


def truth_geometry():
    # E1=150, E2=50, E3=40, E4=40 (angle intérieur V4 = 240°), E5 et E6 horizontale/verticale
    p = turtle([150, 50, 40, 40], [90, 90, 240])
    v5 = p[4]
    pts = p[:5] + [np.array([0.0, v5[1]])]
    return [tuple(map(float, q)) for q in pts]


def drawn_geometry():
    # longueurs dessinées volontairement incohérentes (px) + angles approximatifs
    p = turtle([700, 330, 260, 200], [91, 88, 238])
    v5 = p[4]
    pts = p[:5] + [np.array([0.0, v5[1]])]
    off = np.array([330.0, 300.0])   # position sur la page (px, y-haut)
    return [np.array([q[0] + off[0], H - (q[1] + off[1])]) for q in pts]  # -> y-bas


def wobbly(a, b, amp=2.5, step=6):
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = np.linalg.norm(b - a)
    n = max(2, int(L / step))
    t = np.linspace(0, 1, n)
    nrm = np.array([-(b - a)[1], (b - a)[0]]) / (L + 1e-9)
    phase = RNG.uniform(0, 6)
    off = amp * np.sin(t * math.pi * RNG.uniform(1, 3) + phase) + RNG.normal(0, 0.6, n)
    off[0] = off[-1] = 0
    pts = a + np.outer(t, b - a) + np.outer(off, nrm)
    return pts.astype(np.int32)


def arrow(img, a, b, th=2):
    cv2.polylines(img, [wobbly(a, b, amp=1.0)], False, INK, th, cv2.LINE_AA)
    for p, q in ((a, b), (b, a)):
        d = (np.asarray(q) - np.asarray(p)) / (np.linalg.norm(np.asarray(q) - np.asarray(p)) + 1e-9)
        n = np.array([-d[1], d[0]])
        tip = np.asarray(q)
        cv2.line(img, tuple(tip.astype(int)), tuple((tip - 18 * d + 7 * n).astype(int)), INK, th, cv2.LINE_AA)
        cv2.line(img, tuple(tip.astype(int)), tuple((tip - 18 * d - 7 * n).astype(int)), INK, th, cv2.LINE_AA)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    paper = np.full((H, W, 3), 250, np.uint8)
    grad = np.linspace(0, 18, W)[None, :, None]
    paper = np.clip(paper - grad - RNG.normal(0, 2.0, (H, W, 1)), 0, 255).astype(np.uint8)
    img = paper.copy()
    P = drawn_geometry()
    n = len(P)

    # contour, avec congé dessiné en V2
    fil_r = 45
    corner = P[1]
    a = corner + np.array([-fil_r, 0])
    b = corner + np.array([0, -fil_r])
    for i in range(n):
        p, q = P[i], P[(i + 1) % n]
        if i == 0:
            q = a
        if i == 1:
            p = b
        cv2.polylines(img, [wobbly(p, q)], False, INK, 4, cv2.LINE_AA)
    cv2.ellipse(img, tuple((corner + np.array([-fil_r, -fil_r])).astype(int)), (fil_r, fil_r), 0, 0, 90,
                INK, 4, cv2.LINE_AA)

    # perçages
    holes = [((560, 760), 46), ((920, 690), 30)]
    for (cx, cy), r in holes:
        ang = np.linspace(0, 2 * np.pi, 80)
        rr = r + RNG.normal(0, 0.8, 80)
        pts = np.stack([cx + rr * np.cos(ang), cy + rr * np.sin(ang)], 1).astype(np.int32)
        cv2.polylines(img, [pts], True, INK, 4, cv2.LINE_AA)

    # lignes de cote avec flèches + lignes d'attache (petit jour avec le contour)
    def dim_line(i, j, offset):
        p, q = P[i], P[j]
        d = (q - p) / np.linalg.norm(q - p)
        nrm = np.array([d[1], -d[0]])  # extérieur (polygone CCW en y-haut = CW en y-bas)
        nrm = -nrm
        pa, qa = p + nrm * offset, q + nrm * offset
        for s, e in ((p + nrm * 12, pa + nrm * 14), (q + nrm * 12, qa + nrm * 14)):
            cv2.line(img, tuple(s.astype(int)), tuple(e.astype(int)), INK, 2, cv2.LINE_AA)
        arrow(img, pa, qa)
        return (pa + qa) / 2 + nrm * 34

    texts = []
    texts.append(("150", dim_line(0, 1, 70)))
    texts.append(("50", dim_line(1, 2, 70) + np.array([10, 0])))
    texts.append(("40", dim_line(2, 3, 130) + np.array([70, 0])))
    texts.append(("40", dim_line(3, 4, 100)))
    v4 = P[3]
    texts.append(("120°", v4 + np.array([115, -62])))
    cv2.ellipse(img, tuple(v4.astype(int)), (55, 55), 0, -120, 0, INK, 2, cv2.LINE_AA)
    texts.append(("R10", corner + np.array([60, 55])))
    cv2.line(img, tuple((corner + np.array([-14, -14])).astype(int)), tuple((corner + np.array([40, 40])).astype(int)),
             INK, 2, cv2.LINE_AA)
    texts.append(("Ø12", np.array([560, 760]) + np.array([0, 85])))
    texts.append(("Ø8", np.array([920, 690]) + np.array([0, 65])))
    texts.append(("ep 5", np.array([260, 1110])))

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(FONT, 46)
    dims = []
    kinds = {"150": ("length", 150), "50": ("length", 50), "40": ("length", 40), "120°": ("angle", 120),
             "R10": ("radius", 10), "Ø12": ("diameter", 12), "Ø8": ("diameter", 8), "ep 5": ("thickness", 5)}
    for k, (txt, c) in enumerate(texts):
        x0, y0, x1, y1 = draw.textbbox((0, 0), txt, font=font)
        tx, ty = c[0] - (x1 - x0) / 2, c[1] - (y1 - y0) / 2 - y0
        draw.text((tx, ty), txt, font=font, fill=tuple(int(v) for v in INK[::-1]))
        bb = draw.textbbox((tx, ty), txt, font=font)
        kind, val = kinds[txt]
        dims.append({"id": f"T{k + 1}", "raw": txt, "kind": kind, "value": val,
                     "bbox": [float(v) for v in bb], "conf": 1.0})
    flat = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    flat = cv2.GaussianBlur(flat, (3, 3), 0)
    cv2.imwrite(str(out / "sketch_flat.png"), flat)

    # « photo » : feuille en biais sur une table sombre, éclairage inégal
    PW, PH = 2000, 1600
    table = np.zeros((PH, PW, 3), np.uint8)
    table[:] = (70, 90, 110)
    table = np.clip(table + RNG.normal(0, 6, table.shape), 0, 255).astype(np.uint8)
    dst = np.float32([[230, 160], [1780, 260], [1700, 1450], [180, 1380]])
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), dst)
    warped = cv2.warpPerspective(flat, M, (PW, PH))
    mask = cv2.warpPerspective(np.full((H, W), 255, np.uint8), M, (PW, PH))
    photo = np.where(mask[..., None] > 0, warped, table)
    shade = np.linspace(1.0, 0.82, PW)[None, :, None]
    photo = np.clip(photo * shade, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out / "sketch_photo.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 88])

    (out / "dims.json").write_text(json.dumps(dims, indent=2, ensure_ascii=False), encoding="utf-8")
    truth = {"points": truth_geometry(), "fillets": {"1": 10}, "thickness": 5,
             "hole_diameters": [12, 8],
             "notes": "V4 rentrant (240°) ; la cote écrite 120° est mesurée côté extérieur."}
    (out / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"Croquis écrits dans {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "examples"))
