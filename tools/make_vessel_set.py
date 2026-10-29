"""Jeu de 15 croquis d'enceintes sous pression (réservoirs verticaux) pour un benchmark.

    python tools/make_vessel_set.py croquis_cuves
    python tools/bench_test_set.py croquis_cuves

Chaque croquis est différent : dimensions, fonds (hémisphériques ou bombés, deux R différents),
ouvertures, piquages, pieds, unités (mm, cm, m), épaisseur cotée, écriture, encre, et conditions de
prise de vue (photo en biais, capture basse résolution, un seul stylo, feuille tournée).
Les tracés sont volontairement hors échelle : seules les cotes font foi.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_test_set import Sheet, as_photo, draw_vessel, write_corrige  # noqa: E402


def cap(R, a):
    return R - math.sqrt(R * R - a * a)


# (fichier, ce qui est testé, cotes en mm (D, H, R haut, R bas), dessin, accessoires, condition)
V = [
    ("V01_standard.png", "Enceinte standard : trou d'homme à bride, 2 piquages, pieds.",
     dict(D=1000, H=1500, Rt=1000, Rb=1000), dict(cx=580, a=210, yb=540, yt=1160, kt=0.45, kb=0.45,
     openings=[(0, 860, 75, 105, "Ø450")], top=[(-95, 40, 45), (85, 34, 40)], legs=(175, 180),
     D="Ø1000", H="1500", R="R1000", d_y=1060), dict(font=0, ink=0, seed=101), None),
    ("V02_hemispherique_ep8.png", "Fonds hémisphériques, paroi cotée ep 8, piquage haut et vidange.",
     dict(D=800, H=1200, Rt=400, Rb=400, wall=8), dict(cx=600, a=190, yb=500, yt=1020, kt=1.0, kb=1.0,
     top=[(0, 40, 55)], bottom=[(0, 40, 50)], D="Ø800", H="1200", R="R400", ep="ep 8", d_y=930),
     dict(font=1, ink=1, seed=102), None),
    ("V03_grande_3_piquages.png", "Grande enceinte : trou d'homme + petit piquage radial, 3 piquages, pieds.",
     dict(D=1500, H=2000, Rt=1500, Rb=1500), dict(cx=560, a=240, yb=560, yt=1160, kt=0.4, kb=0.4,
     openings=[(-20, 900, 80, 112, "Ø500"), (130, 680, 28, None, None)], top=[(-140, 40, 45), (0, 36, 50), (130, 40, 45)],
     legs=(200, 200), D="Ø1500", H="2000", R="R1500", d_y=1080), dict(font=2, ink=2, seed=103), None),
    ("V04_petite_cote_gauche.png", "Petite enceinte, hauteur cotée à gauche, un piquage, pieds.",
     dict(D=600, H=900, Rt=600, Rb=600), dict(cx=640, a=170, yb=560, yt=1050, kt=0.5, kb=0.5,
     top=[(0, 36, 50)], legs=(140, 220), D="Ø600", H="900", R="R600", h_side="left", d_y=980),
     dict(font=0, ink=1, seed=104), None),
    ("V05_unites_cm.png", "Cotes en centimètres (Ø120 cm, 160 cm, R120 cm).",
     dict(D=1200, H=1600, Rt=1200, Rb=1200), dict(cx=570, a=220, yb=560, yt=1140, kt=0.45, kb=0.45,
     openings=[(0, 820, 70, 100, "Ø40 cm")], top=[(-100, 40, 45)], legs=(185, 200),
     D="Ø120 cm", H="160 cm", R="R120 cm", d_y=1050), dict(font=1, ink=0, seed=105), None),
    ("V06_unites_m.png", "Cotes en mètres (Ø1,4 m, 2,2 m, R1,4 m), deux ouvertures simples, vidange.",
     dict(D=1400, H=2200, Rt=1400, Rb=1400), dict(cx=580, a=200, yb=520, yt=1180, kt=0.4, kb=0.4,
     openings=[(-60, 900, 40, None, None), (70, 700, 34, None, None)], top=[(-90, 38, 45), (90, 38, 45)],
     bottom=[(0, 38, 45)], legs=(165, 170), D="Ø1,4m", H="2,2m", R="R1,4m", d_y=1090),
     dict(font=2, ink=1, seed=106), None),
    ("V07_deux_rayons.png", "Deux fonds différents : R1500 en haut, R600 en bas.",
     dict(D=1000, H=1600, Rt=1500, Rb=600), dict(cx=600, a=200, yb=600, yt=1160, kt=0.3, kb=0.75,
     openings=[(0, 880, 70, 98, "Ø400")], top=[(0, 40, 50)], D="Ø1000", H="1600", R="R1500", R_bot="R600",
     d_y=1080), dict(font=0, ink=2, seed=107), None),
    ("V08_trapue.png", "Enceinte trapue (H < D), 2 piquages, trou d'homme, pieds.",
     dict(D=1600, H=1200, Rt=1600, Rb=1600), dict(cx=600, a=270, yb=620, yt=1000, kt=0.4, kb=0.4,
     openings=[(0, 810, 70, 98, "Ø450")], top=[(-130, 44, 45), (120, 44, 45)], legs=(230, 260),
     D="Ø1600", H="1200", R="R1600", d_y=950), dict(font=1, ink=2, seed=108), None),
    ("V09_elancee.png", "Colonne élancée (H = 5 × D), piquages haut et bas, petite ouverture, pieds.",
     dict(D=500, H=2500, Rt=500, Rb=500), dict(cx=600, a=130, yb=380, yt=1260, kt=0.55, kb=0.55,
     openings=[(0, 820, 36, None, None)], top=[(0, 30, 45)], bottom=[(0, 30, 40)], legs=(105, 120),
     D="Ø500", H="2500", R="R500", d_y=1200), dict(font=2, ink=0, seed=109), None),
    ("V10_complete_ep10.png", "Enceinte complète : ep 10, bride, 2 piquages haut, vidange, pieds.",
     dict(D=1200, H=1800, Rt=1200, Rb=1200, wall=10), dict(cx=580, a=215, yb=560, yt=1160, kt=0.42, kb=0.42,
     openings=[(0, 860, 78, 110, "Ø500")], top=[(-100, 40, 45), (95, 40, 45)], bottom=[(0, 40, 42)],
     legs=(180, 190), D="Ø1200", H="1800", R="R1200", ep="ep 10", d_y=1080), dict(font=0, ink=1, seed=110), None),
    ("V11_photo_en_biais.jpg", "Photo en biais sur une table (ombre, éclairage inégal).",
     dict(D=2000, H=2400, Rt=2000, Rb=2000), dict(cx=580, a=230, yb=560, yt=1150, kt=0.42, kb=0.42,
     openings=[(0, 850, 80, 110, "Ø600")], top=[(-110, 42, 45), (100, 42, 45)], legs=(190, 190),
     D="Ø2000", H="2400", R="R2000", d_y=1070), dict(font=1, ink=0, seed=111), "photo"),
    ("V12_photo_sombre.jpg", "Deuxième photo, cadrage et lumière différents, fonds hémisphériques.",
     dict(D=900, H=1300, Rt=450, Rb=450), dict(cx=600, a=190, yb=520, yt=1000, kt=0.95, kb=0.95,
     openings=[(0, 760, 60, 85, "Ø300")], top=[(0, 38, 50)], D="Ø900", H="1300", R="R450", d_y=920),
     dict(font=2, ink=1, seed=112), "photo_dark"),
    ("V13_capture_basse_res.png", "Capture d'écran de 480 px de large.",
     dict(D=1100, H=1600, Rt=1100, Rb=1100), dict(cx=580, a=215, yb=560, yt=1150, kt=0.45, kb=0.45,
     openings=[(0, 850, 72, 100, "Ø400")], top=[(-95, 42, 45), (95, 42, 45)], legs=(180, 200),
     D="Ø1100", H="1600", R="R1100", d_y=1070), dict(font=0, ink=0, seed=113), "lowres"),
    ("V14_un_seul_stylo.png", "Contour et cotation au même stylo.",
     dict(D=1300, H=1700, Rt=1300, Rb=1300), dict(cx=580, a=210, yb=560, yt=1150, kt=0.45, kb=0.45,
     openings=[(0, 860, 74, 102, "Ø450")], top=[(-95, 40, 45)], legs=(175, 200),
     D="Ø1300", H="1700", R="R1300", d_y=1070), dict(font=1, ink=2, seed=114, pen=3), None),
    ("V15_feuille_tournee.png", "Feuille tournée de 4°, hauteur cotée à gauche.",
     dict(D=700, H=1000, Rt=700, Rb=700), dict(cx=620, a=180, yb=560, yt=1060, kt=0.5, kb=0.5,
     openings=[(0, 810, 55, 78, "Ø250")], top=[(0, 36, 45)], legs=(150, 220), D="Ø700", H="1000", R="R700",
     h_side="left", d_y=990), dict(font=2, ink=2, seed=115), "rotate"),
]


def photo_dark(flat, seed):
    """Autre prise de vue : feuille penchée dans l'autre sens, lumière venant de la droite, plus sombre."""
    rng = np.random.default_rng(seed)
    H, W = flat.shape[:2]
    PW, PH = int(W * 1.3), int(H * 1.25)
    table = np.full((PH, PW, 3), (85, 80, 75), np.float32) + rng.normal(0, 7, (PH, PW, 3))
    dst = np.float32([[0.14 * PW, 0.07 * PH], [0.86 * PW, 0.04 * PH], [0.92 * PW, 0.93 * PH], [0.10 * PW, 0.88 * PH]])
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), dst)
    warped = cv2.warpPerspective(flat, M, (PW, PH)).astype(np.float32)
    mask = cv2.warpPerspective(np.full((H, W), 255, np.uint8), M, (PW, PH))
    photo = np.where(mask[..., None] > 0, warped, table)
    light = (np.linspace(0.72, 1.0, PW)[None, :] * np.linspace(0.85, 1.0, PH)[:, None])[..., None]
    photo = photo * light * 0.88 + rng.normal(0, 3, photo.shape)
    return np.clip(photo, 0, 255).astype(np.uint8)


def rotate(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(246, 246, 246))


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    key, pages = [], []
    for f, what, dims, spec, style, cond in V:
        pen = style.get("pen")
        sh = Sheet(W=1200, H=1600, seed=style["seed"], font=style["font"], ink=style["ink"],
                   thick=pen or 4, thin=pen or 2)
        draw_vessel(sh, spec)
        img = sh.render()
        if cond == "photo":
            img = as_photo(img, style["seed"])
        elif cond == "photo_dark":
            img = photo_dark(img, style["seed"])
        elif cond == "lowres":
            img = cv2.resize(img, (480, 640), interpolation=cv2.INTER_AREA)
        elif cond == "rotate":
            img = rotate(img, 4)
        p = out / f
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 88] if p.suffix == ".jpg" else [])
        a = dims["D"] / 2
        total = dims["H"] + cap(dims["Rt"], a) + cap(dims["Rb"], a)
        key.append({"file": f, "code": "vessel", "what": what, "shape": "réservoir",
                    "size": [dims["D"], dims["D"], round(total, 3)],
                    "openings": len(spec.get("openings", [])),
                    "nozzles": len(spec.get("top", [])) + len(spec.get("bottom", [])),
                    "legs": bool(spec.get("legs")),
                    "note": f"corps Ø{dims['D']} × {dims['H']}, fonds R{dims['Rt']} / R{dims['Rb']}"
                            + (f", paroi {dims['wall']} mm" if dims.get("wall") else "")})
        if cond is None or cond == "rotate":
            im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            A4 = Image.new("RGB", (2480, 3508), "white")
            im.thumbnail((2200, 3100))
            A4.paste(im, ((2480 - im.width) // 2, (3508 - im.height) // 2))
            pages.append(A4)
        print("  ", f)
    (out / "attendu.json").write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    write_corrige(out, key, title="Enceintes sous pression — 15 croquis de test")
    pages[0].save(out / "croquis_a_imprimer.pdf", save_all=True, append_images=pages[1:], resolution=300)
    print(f"{len(V)} croquis + attendu.json + corrige.html + croquis_a_imprimer.pdf dans {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "croquis_cuves"))
