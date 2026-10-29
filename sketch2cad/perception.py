"""Agent 1 — Prétraitement visuel, vectorisation, OCR et marquage Set-of-Mark.

Entièrement déterministe (OpenCV) : aucune dépendance GPU.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .ocr import run_ocr
from .schema import Circle, Dimension, Edge, Perception, Vertex

log = logging.getLogger(__name__)

MAX_SIDE = 2000
MIN_SIDE = 1000   # les seuils sont réglés pour des photos de 1000 à 2000 px : on agrandit les petites


# --------------------------------------------------------------------------- image

def load_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)  # supporte les chemins Windows accentués
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image illisible : {path}")
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    elif max(h, w) < MIN_SIDE:
        # capture d'écran ou photo réduite : traits de 1-2 px, chiffres minuscules pour l'OCR
        s = MIN_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    return img


def rectify_page(img: np.ndarray) -> np.ndarray:
    """Redresse la feuille si un grand quadrilatère clair est détecté."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    H, W = gray.shape
    if area < 0.3 * H * W or area > 0.98 * H * W:
        return img  # la feuille remplit déjà l'image (ou n'est pas trouvée)
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(approx) != 4:
        return img
    pts = approx.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    tl, br, tr, bl = pts[np.argmin(s)], pts[np.argmax(s)], pts[np.argmin(d)], pts[np.argmax(d)]
    wd = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    ht = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]),
                                    np.array([[0, 0], [wd, 0], [wd, ht], [0, ht]], np.float32))
    log.info("Feuille redressée (%dx%d)", wd, ht)
    return cv2.warpPerspective(img, M, (wd, ht))


def normalize(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (gris normalisé fond blanc, binaire traits=255)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(gray.shape) / 30)
    norm = cv2.divide(gray, bg, scale=255)
    binary = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return norm, binary


def _ridge_widths(binary: np.ndarray) -> np.ndarray:
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    ridge = (dist > 0) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)))
    return 2 * dist[ridge]


def stroke_width(binary: np.ndarray) -> float:
    w = _ridge_widths(binary)
    if w.size == 0:
        return 3.0
    return float(np.clip(np.percentile(w, 75), 2, 15))


def _split_by_darkness(binary: np.ndarray, gray: np.ndarray) -> Optional[tuple[np.ndarray, float, float]]:
    """Repli quand les épaisseurs ne se séparent pas (capture d'écran basse résolution agrandie :
    tous les traits font 1–2 px à l'origine). Le contour au feutre reste noir saturé, la cotation
    au stylo fin devient grise par anticrénelage : on sépare alors par le noir du trait."""
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    ridge = (dist > 0) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)))
    dark = 255 - cv2.erode(gray, np.ones((5, 5), np.uint8))
    v = dark[ridge].astype(np.uint8)
    if v.size < 200 or v.var() < 1:
        return None
    t, _ = cv2.threshold(v.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo, hi = v[v <= t].astype(float), v[v > t].astype(float)
    f = lo.size / v.size
    if not 0.1 <= f <= 0.9:
        return None
    eta = f * (1 - f) * (hi.mean() - lo.mean()) ** 2 / v.var()
    # contour franchement noir, cotation nettement plus claire, deux populations bien séparées
    if eta < 0.75 or np.median(hi) < 170 or np.median(hi) - np.median(lo) < 40:
        return None
    strong = np.where(dark > t, binary, 0).astype(np.uint8)
    # miettes (pointes de flèche, croisements) : on ne garde que les traits d'une certaine longueur
    n, lab, st, _ = cv2.connectedComponentsWithStats(strong, 8)
    big = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]) >= 0.03 * max(binary.shape)
    big[0] = False
    strong = np.where(big[lab], strong, 0).astype(np.uint8)
    if cv2.countNonZero(strong) < 0.1 * cv2.countNonZero(binary):
        return None
    return strong, float(np.median(2 * dist[ridge & (strong > 0)])), float(t)


def split_strokes(binary: np.ndarray, gray: Optional[np.ndarray] = None) -> tuple[np.ndarray, float, Optional[float]]:
    """Sépare traits forts (contour, ISO 128) et traits fins (cotation).

    Retourne (masque des traits forts, épaisseur forte, seuil) ; si les épaisseurs ne sont
    pas bimodales, renvoie le binaire complet et seuil None.
    """
    w = _ridge_widths(binary)
    w = w[(w > 1) & (w < 20)]
    if w.size < 100:
        return binary, stroke_width(binary), None
    hist = np.round(w * 2).astype(np.uint8).reshape(-1, 1)
    t2, _ = cv2.threshold(hist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = t2 / 2.0
    thin, thick = w[w <= t], w[w > t]
    if thin.size < 0.1 * w.size or thick.size < 0.1 * w.size or np.median(thick) < 1.3 * np.median(thin):
        alt = _split_by_darkness(binary, gray) if gray is not None else None
        if alt is not None:
            return alt
        return binary, stroke_width(binary), None
    k = int(np.ceil(t)) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    strong = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ker)
    strong = cv2.bitwise_and(cv2.dilate(strong, ker), binary)
    return strong, float(np.median(thick)), t


# --------------------------------------------------------------------------- géométrie

def _mask_text(binary: np.ndarray, dims: list[Dimension], pad: float) -> np.ndarray:
    """Efface les traits des textes sans couper le dessin voisin.

    Un trait (composante connexe) est retiré s'il est surtout dans une boîte de texte ;
    un trait du contour qui ne fait que traverser la boîte est conservé.
    """
    if not dims:
        return binary.copy()
    tmask = np.zeros_like(binary)
    for d in dims:
        x0, y0, x1, y1 = d.bbox
        cv2.rectangle(tmask, (int(x0 - pad), int(y0 - pad)), (int(x1 + pad), int(y1 + pad)), 255, -1)
    n, lab, st, _ = cv2.connectedComponentsWithStats(binary, 8)
    inside = np.bincount(lab[tmask > 0].ravel(), minlength=n)
    total = st[:, cv2.CC_STAT_AREA]
    drop = (inside >= 0.7 * np.maximum(total, 1))
    drop[0] = False
    out = binary.copy()
    out[drop[lab]] = 0
    return out


def extract_outline(binary: np.ndarray, sw: float) -> tuple[np.ndarray, np.ndarray]:
    """Plus grande région fermée = la pièce. Retourne (polygone px Nx2, masque rempli)."""
    k = max(3, int(round(sw * 1.5)) | 1)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise RuntimeError("Aucun contour trouvé dans l'image.")
    c = max(cnts, key=cv2.contourArea)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, [c], -1, 255, -1)
    # l'ouverture retire les lignes d'attache / flèches collées au contour
    ko = max(5, int(round(sw * 4)) | 1)
    opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
    cnts2, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts2, key=cv2.contourArea) if cnts2 else c
    peri = cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, 0.012 * peri, True).reshape(-1, 2).astype(float)
    poly = _clean_polygon(poly, peri)
    return poly, filled


def extract_outline_faces(binary: np.ndarray, sw: float) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Variante quand contour et cotation ont la même épaisseur (un seul stylo).

    Les lignes de cote et leurs lignes d'attache forment avec le bord de la pièce des bandes
    fermées étroites ; la pièce est la plus grande « face » (zone blanche enclose par les traits),
    pas l'enveloppe extérieure de tous les traits. Retourne None si aucune face n'est trouvée.
    """
    k = max(3, int(round(sw * 1.5)) | 1)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(closed), 4)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist())
    faces = [i for i in range(1, n) if i not in border]
    if not faces:
        return None
    f = max(faces, key=lambda i: st[i, cv2.CC_STAT_AREA])
    face = (lab == f).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(face, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 255, -1)   # perçages rebouchés
    g = max(3, int(round(sw)) | 1)                                            # jusqu'au milieu du trait
    filled = cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g)))
    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, 0.012 * peri, True).reshape(-1, 2).astype(float)
    return _clean_polygon(poly, peri), filled


def _clean_polygon(poly: np.ndarray, peri: float) -> np.ndarray:
    """Fusionne les sommets trop proches (congés dessinés) et supprime les sommets plats."""
    changed = True
    while changed and len(poly) > 3:
        changed = False
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            if np.linalg.norm(b - a) < 0.03 * peri:
                # remplace la paire par l'intersection des arêtes voisines (coin vif)
                p_prev, p_next = poly[i - 1], poly[(i + 2) % n]
                x = _line_intersection(p_prev, a, b, p_next)
                newpt = x if x is not None and np.linalg.norm(x - (a + b) / 2) < 0.08 * peri else (a + b) / 2
                poly = np.delete(poly, (i + 1) % n, axis=0)
                poly[i if (i + 1) % n != 0 else i - 1] = newpt
                changed = True
                break
        if changed:
            continue
        n = len(poly)
        for i in range(n):
            u = poly[i - 1] - poly[i]
            v = poly[(i + 1) % n] - poly[i]
            cosang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
            if cosang < -0.985:  # angle > ~170° : sommet plat
                poly = np.delete(poly, i, axis=0)
                changed = True
                break
    return poly


def _line_intersection(p1, p2, p3, p4) -> Optional[np.ndarray]:
    d1, d2 = p2 - p1, p4 - p3
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den
    return p1 + t * d1


def _char_height(d: Dimension) -> float:
    """Hauteur d'un caractère de la cote (la boîte peut englober un perçage voisin)."""
    x0, y0, x1, y1 = d.bbox
    n = max(1, len(d.raw.replace(" ", "")))
    return min(y1 - y0, 1.6 * (x1 - x0) / n)


def extract_circles(binary: np.ndarray, filled: np.ndarray, sw: float,
                    dims: list[Dimension], part_diag: float) -> list[tuple[float, float, float]]:
    """Perçages = régions intérieures quasi circulaires délimitées par un trait fermé."""
    k = max(3, int(round(sw)) | 1)
    strokes = cv2.dilate(binary, np.ones((k, k), np.uint8))
    interior = cv2.bitwise_and(filled, cv2.bitwise_not(strokes))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(interior, 8)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        r_area = np.sqrt(area / np.pi)
        if r_area < 0.012 * part_diag or r_area > 0.35 * part_diag:
            continue
        comp = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = cnts[0]
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (peri * peri + 1e-9)
        (cx, cy), r_enc = cv2.minEnclosingCircle(c)
        if circ < 0.75 or area < 0.6 * np.pi * r_enc ** 2:
            continue
        if any(d.bbox[0] <= cx <= d.bbox[2] and d.bbox[1] <= cy <= d.bbox[3]
               and r_area < 0.35 * _char_height(d) for d in dims):
            continue  # boucle d'un chiffre (0, 8, 9...) ; un perçage collé à sa cote reste un perçage
        out.append((float(cx), float(cy), float(r_area + sw / 2)))
    return out


# --------------------------------------------------------------------------- Set-of-Mark

def draw_som(img: np.ndarray, p: Perception) -> np.ndarray:
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    sc = max(0.5, max(out.shape[:2]) / 1600)
    th = max(1, int(round(2 * sc)))
    vmap = {v.id: v.px for v in p.vertices}

    def label(txt, pos, color):
        (tw, tht), _ = cv2.getTextSize(txt, font, sc, th)
        x, y = int(pos[0]), int(pos[1])
        cv2.rectangle(out, (x - 2, y - tht - 4), (x + tw + 2, y + 4), (255, 255, 255), -1)
        cv2.rectangle(out, (x - 2, y - tht - 4), (x + tw + 2, y + 4), color, 1)
        cv2.putText(out, txt, (x, y), font, sc, color, th, cv2.LINE_AA)

    for e in p.edges:
        a, b = np.array(vmap[e.a]), np.array(vmap[e.b])
        cv2.line(out, tuple(a.astype(int)), tuple(b.astype(int)), (0, 160, 0), th)
        label(e.id, (a + b) / 2, (0, 130, 0))
    for v in p.vertices:
        cv2.circle(out, tuple(np.array(v.px).astype(int)), int(6 * sc), (0, 0, 220), -1)
        label(v.id, (v.px[0] + 8 * sc, v.px[1] - 8 * sc), (0, 0, 200))
    for c in p.circles:
        cv2.circle(out, tuple(np.array(c.px).astype(int)), int(c.r_px), (200, 0, 200), th)
        label(c.id, (c.px[0] + c.r_px, c.px[1]), (180, 0, 180))
    for d in p.dimensions:
        x0, y0, x1, y1 = (int(v) for v in d.bbox)
        cv2.rectangle(out, (x0, y0), (x1, y1), (200, 120, 0), th)
        label(d.id, (x0, y0 - 4), (200, 120, 0))
    return out


# --------------------------------------------------------------------------- agent

def perceive(image_path: str | Path, out_dir: Path, ocr_backends: list[str],
             dims_override: Optional[list[Dimension]] = None, drawing: Optional[int] = None,
             ask: bool = False, text_reader=None) -> tuple[Perception, np.ndarray]:
    """Exécute l'agent 1.

    ``text_reader`` (profil « api ») relit les textes de l'OCR avec un modèle de vision distant.

    ``dims_override`` remplace l'OCR (textes corrigés à la main). ``drawing`` choisit le dessin
    quand la feuille en contient plusieurs ; avec ``ask=True`` on s'arrête pour demander.
    """
    from .primitives import detect_cylinder, draw_cylinder_som
    from .revolution import detect_revolution, draw_revolution_som
    from .sheet import draw_sheet, find_drawings, region_mask

    img = rectify_page(load_image(image_path))
    norm, binary = normalize(img)
    sw = stroke_width(binary)
    warnings: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", img)[1].tofile(str(out_dir / "rectified.png"))

    if dims_override is not None:
        all_dims = dims_override
    else:
        all_dims, w = run_ocr(norm, ocr_backends)
        warnings += w
        texts = [{"id": d.id, "raw": d.raw, "bbox": list(d.bbox), "conf": d.conf} for d in all_dims]
        import json as _json
        if text_reader is not None:
            from .ocr import dims_from_texts
            texts_api, note = text_reader(norm, all_dims)
            (out_dir / "api_read.json").write_text(_json.dumps(note, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
            if note.get("ok"):
                texts = texts_api
                all_dims = dims_from_texts(texts)
            else:
                warnings.append(f"API indisponible, lecture par l'OCR local : {note.get('error', '')[:160]}")
        # textes bruts conservés : l'utilisateur pourra les corriger puis relancer sans OCR
        (out_dir / "texts_all.json").write_text(_json.dumps(texts, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    log.info("%d textes lus", len(all_dims))

    # ---- plusieurs dessins sur la feuille ?
    drawings = find_drawings(binary, all_dims)
    for d in drawings:
        d["cut"] = bool(d["cut"])
    cv2.imencode(".png", draw_sheet(img, drawings))[1].tofile(str(out_dir / "sheet.png"))
    base = dict(image_size=(img.shape[1], img.shape[0]), stroke_px=sw, vertices=[], edges=[], circles=[],
                warnings=warnings, drawings=drawings)

    def save(p: Perception) -> Perception:
        (out_dir / "perception.json").write_text(p.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
        return p

    whole = [d for d in drawings if not d["cut"]]
    if len(whole) > 1 and drawing is None and ask:
        return save(Perception(**base, dimensions=all_dims, shape="choose_drawing")), img
    if drawings:
        if drawing is None:
            drawing = 0
            if len(whole) > 1:
                warnings.append(f"{len(drawings)} dessins sur la feuille : le n°1 (le plus grand) est traité ; "
                                "choisissez-en un autre avec --drawing ou dans l'interface.")
        drawing = max(0, min(drawing, len(drawings) - 1))
        mask, _r = region_mask(binary.shape, drawings[drawing]["bbox"])
        keep = set(drawings[drawing]["texts"])
        dims = [d for d in all_dims if d.id in keep]
        binary = cv2.bitwise_and(binary, mask)
    else:
        dims = all_dims
    base["drawing"] = drawing
    clean = _mask_text(binary, dims, pad=sw)

    # ---- cylindre en perspective
    cyl = detect_cylinder(clean)
    if cyl is not None:
        cv2.imencode(".png", draw_cylinder_som(img, cyl, dims))[1].tofile(str(out_dir / "som.png"))
        log.info("Perception : cylindre")
        return save(Perception(**base, dimensions=dims, shape="cylinder", cylinder=cyl)), img

    # ---- cercle seul (sphère ou disque ?)
    circ = detect_circle(clean)
    if circ is not None:
        som = img.copy()
        cv2.circle(som, (int(circ["cx"]), int(circ["cy"])), int(circ["r_px"]), (200, 0, 200), 3)
        _draw_texts(som, dims)
        cv2.imencode(".png", som)[1].tofile(str(out_dir / "som.png"))
        return save(Perception(**base, dimensions=dims, shape="circle", circle=circ,
                               choices=["sphere", "disc"])), img

    # ---- corps de révolution
    # contour au trait fort et cotation au trait fin : la silhouette ne doit pas englober les lignes de cote
    strong_r, _sw_r, thr_r = split_strokes(clean, norm)
    # silhouette au trait fort d'abord ; si le croquis est tracé d'un seul stylo, la séparation
    # forte/fine peut morceler le contour : on retente alors avec tous les traits
    rev = detect_revolution(strong_r) if thr_r is not None else None
    if rev is None:
        rev = detect_revolution(clean)
    if rev is not None:
        cv2.imencode(".png", draw_revolution_som(img, rev, dims))[1].tofile(str(out_dir / "som.png"))
        log.info("Perception : corps de révolution")
        return save(Perception(**base, dimensions=dims, shape="revolution", revolution=rev)), img

    # ---- contour ouvert (un « V », un arc seul) : pas de solide possible
    if _is_open(clean):
        som = img.copy()
        _draw_texts(som, dims)
        cv2.imencode(".png", som)[1].tofile(str(out_dir / "som.png"))
        return save(Perception(**base, dimensions=dims, shape="open")), img

    strong, sw_strong, thr = split_strokes(clean, norm)
    if thr is not None:
        sw = sw_strong
        log.info("Traits forts > %.1f px (contour), traits fins (cotation) écartés", thr)
        poly, filled = extract_outline(strong, sw)
        # contour morcelé par la séparation fort/fin (petite capture agrandie : toutes les
        # épaisseurs se ressemblent) : la « pièce » trouvée n'est qu'un fragment -> tous les traits
        poly_all, filled_all = extract_outline(clean, stroke_width(clean))
        if len(poly) < 3 or cv2.countNonZero(filled) < 0.4 * cv2.countNonZero(filled_all):
            log.info("Séparation fort/fin inexploitable (contour morcelé) : tous les traits sont gardés")
            thr, sw = None, stroke_width(clean)
    if thr is None:
        # un seul stylo : la pièce est la plus grande zone fermée, pas l'enveloppe des traits
        poly, filled = extract_outline(clean, sw)
        fac = extract_outline_faces(clean, sw)
        if fac is not None and len(fac[0]) >= 3 and cv2.countNonZero(fac[1]) >= 0.5 * cv2.countNonZero(filled):
            poly, filled = fac
        else:
            warnings.append("Traits forts/fins non distingués : repassez le contour au trait épais "
                            "pour séparer le contour de la cotation.")
    # orientation CCW dans le repère y-haut  <=>  aire signée négative en pixels (y-bas)
    area_px = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    if area_px > 0:
        poly = poly[::-1]
    # V1 = sommet le plus en bas à gauche (lecture stable)
    start = int(np.argmax(poly[:, 1] - poly[:, 0]))
    poly = np.roll(poly, -start, axis=0)

    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    diag = float(np.hypot(x1 - x0, y1 - y0))
    # traits non masqués : un texte collé à un perçage ne doit pas ouvrir son cercle
    circles = extract_circles(binary, filled, sw, dims, diag)

    vertices = [Vertex(id=f"V{i + 1}", px=(float(x), float(y))) for i, (x, y) in enumerate(poly)]
    n = len(vertices)
    edges = [Edge(id=f"E{i + 1}", a=vertices[i].id, b=vertices[(i + 1) % n].id) for i in range(n)]
    circ = [Circle(id=f"C{i + 1}", px=(cx, cy), r_px=r) for i, (cx, cy, r) in enumerate(circles)]

    p = Perception(image_size=(img.shape[1], img.shape[0]), stroke_px=sw, vertices=vertices,
                   edges=edges, circles=circ, dimensions=dims, warnings=warnings, drawings=drawings,
                   drawing=drawing)
    som = draw_som(img, p)
    cv2.imencode(".png", som)[1].tofile(str(out_dir / "som.png"))
    cv2.imencode(".png", img)[1].tofile(str(out_dir / "rectified.png"))
    (out_dir / "perception.json").write_text(p.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
    log.info("Perception : %d sommets, %d perçages", n, len(circ))
    return p, img


def detect_circle(strokes: np.ndarray) -> Optional[dict]:
    """Un cercle seul (sans ellipse ni génératrice) : sphère ou disque, à faire préciser."""
    from .primitives import _clean
    from .revolution import _silhouette
    fill = _silhouette(_clean(strokes))
    if fill is None:
        return None
    cnts, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    area, peri = cv2.contourArea(c), cv2.arcLength(c, True)
    circ = 4 * np.pi * area / (peri * peri + 1e-9)
    if len(c) < 5:
        return None
    (cx, cy), (a1, a2), _ang = cv2.fitEllipse(c)
    ratio = min(a1, a2) / max(a1, a2)
    if circ < 0.82 or ratio < 0.85:
        return None
    return {"cx": float(cx), "cy": float(cy), "r_px": float((a1 + a2) / 4), "circularity": float(circ)}


def _draw_texts(img: np.ndarray, dims: list[Dimension]) -> None:
    sc = max(0.5, max(img.shape[:2]) / 1600)
    th = max(2, int(round(3 * sc)))
    for d in dims:
        x0, y0, x1, y1 = (int(v) for v in d.bbox)
        col = (0, 0, 220) if d.kind == "unknown" else (200, 120, 0)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, th)
        cv2.putText(img, d.id, (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * sc + 0.3, col, th, cv2.LINE_AA)


def _is_open(strokes: np.ndarray) -> bool:
    from .primitives import _clean
    st = _clean(strokes)
    if cv2.countNonZero(st) == 0:
        return True
    k = max(5, int(0.012 * max(st.shape))) | 1
    closed = cv2.morphologyEx(st, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return cv2.contourArea(c) < 0.2 * w * h
