"""OCR des cotes manuscrites + typage (longueur, angle, rayon, diamètre, épaisseur).

Backends essayés dans l'ordre de ``config.ocr.backends`` : paddle, easyocr, tesseract.
Tous tournent sur CPU. Chaque backend renvoie des tuples (texte, bbox, confiance).
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

from .schema import Dimension

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- parsing

_NUM = r"(\d+(?:[.,]\d+)?)"
_DIAM_PREFIX = r"(?:Ø|ø|⌀|∅|Φ|φ|phi|PHI|diam|D(?=\s*\d)|@)"
_THICK_PREFIX = r"(?:ep|EP|Ep|ép|e\s*=|t\s*=|th|TH|épaisseur|epaisseur|thk)"
_DEG_SUFFIX = r"(?:°|º|˚|deg|o\b|\*)"

_FIXES = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8"})


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def parse_dimension(text: str) -> Optional[tuple[str, float]]:
    """Retourne (kind, valeur) ou None si le texte n'est pas une cote.

    >>> parse_dimension("R10")
    ('radius', 10.0)
    >>> parse_dimension("Ø 12,5")
    ('diameter', 12.5)
    >>> parse_dimension("120°")
    ('angle', 120.0)
    >>> parse_dimension("ep=5")
    ('thickness', 5.0)
    >>> parse_dimension("150")
    ('length', 150.0)
    """
    r = parse_dimension_ex(text)
    return None if r is None else (r[0], r[1])


# facteurs vers le millimètre
UNITS = {"mm": 1.0, "cm": 10.0, "m": 1000.0}

# notation « lettre = valeur » (ex. « r = 0,5 », « d = 2m », « h=40 ») ; un r manuscrit est
# souvent lu « F » ou « Γ » par l'OCR
_LABELS = {
    "r": "radius", "f": "radius", "γ": "radius", "ŕ": "radius", "ray": "radius", "rayon": "radius",
    "d": "ambiguous",            # diamètre OU longueur : tranché par la géométrie
    "ø": "diameter", "⌀": "diameter", "∅": "diameter", "φ": "diameter", "phi": "diameter",
    "diam": "diameter",
    "h": "length", "l": "length", "lg": "length", "ht": "length", "hauteur": "length",
    "longueur": "length", "largeur": "length", "w": "length", "b": "length", "a": "length",
    "e": "thickness", "ep": "thickness", "ép": "thickness", "t": "thickness", "th": "thickness",
}


def parse_dimension_ex(text: str) -> Optional[tuple[str, float, Optional[str]]]:
    """Comme ``parse_dimension`` mais renvoie aussi l'unité écrite (mm, cm, m) ou None.

    >>> parse_dimension_ex("d= 2m")
    ('ambiguous', 2.0, 'm')
    >>> parse_dimension_ex("F=0.5")
    ('radius', 0.5, None)
    """
    t = text.strip().replace(" ", "")
    if not t:
        return None
    unit = None
    mu = re.search(r"(?<=\d)(mm|cm|m|MM|CM|M)\.?$", t)
    if mu:
        unit = mu.group(1).lower()
        t = t[: mu.start()]
    m = re.fullmatch(r"([^\d=:]{1,10})[:=]" + _NUM, t)
    if m and m.group(1).lower() in _LABELS:
        return _LABELS[m.group(1).lower()], _num(m.group(2)), unit
    r = _parse_plain(t)
    return None if r is None else (r[0], r[1], unit)


def _parse_plain(t: str) -> Optional[tuple[str, float]]:
    m = re.fullmatch(_THICK_PREFIX + r"[:=]?" + _NUM, t, flags=re.IGNORECASE)
    if m:
        return "thickness", _num(m.group(1))
    m = re.fullmatch(_DIAM_PREFIX + r"[:=]?" + _NUM, t)
    if m:
        return "diameter", _num(m.group(1))
    m = re.fullmatch(r"[Rr][:=]?" + _NUM, t)
    if m:
        return "radius", _num(m.group(1))
    m = re.fullmatch(_NUM + _DEG_SUFFIX, t)
    if m:
        v = _num(m.group(1))
        if 0 < v < 360:
            return "angle", v
    # corrections OCR usuelles (O->0, l->1...) puis nombre seul
    fixed = t.translate(_FIXES)
    m = re.fullmatch(_NUM, fixed)
    if m:
        s = m.group(1)
        v = _num(s)
        if v <= 0:
            return None
        # « 012 » : un Ø lu comme un zéro -> ambigu (tranché par la proximité d'un cercle)
        if len(s) >= 2 and s[0] == "0" and s[1].isdigit():
            return "ambiguous", _num(s[1:])
        return "length", v
    return None


# --------------------------------------------------------------------------- backends

def _paddle(img: np.ndarray):
    from paddleocr import PaddleOCR  # type: ignore

    global _PADDLE
    try:
        _PADDLE
    except NameError:
        try:  # PaddleOCR 3.x ; oneDNN désactivé (bug de certaines versions CPU)
            _PADDLE = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                                use_textline_orientation=True, lang="en", enable_mkldnn=False)
        except (TypeError, ValueError):  # PaddleOCR 2.x
            _PADDLE = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    if img.ndim == 2:
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out = []
    if hasattr(_PADDLE, "predict"):
        for res in _PADDLE.predict(img):
            r = res.json.get("res", res) if hasattr(res, "json") else res
            for txt, poly, sc in zip(r["rec_texts"], r["rec_polys"], r["rec_scores"]):
                p = np.asarray(poly)
                out.append((txt, (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()), float(sc)))
    else:
        for line in (_PADDLE.ocr(img, cls=True) or [[]])[0] or []:
            p = np.asarray(line[0])
            out.append((line[1][0], (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()),
                        float(line[1][1])))
    return out


def _easyocr(img: np.ndarray):
    import easyocr  # type: ignore

    global _EASY
    try:
        _EASY
    except NameError:
        _EASY = easyocr.Reader(["en"], gpu=False, verbose=False)
    out = []
    for box, txt, sc in _EASY.readtext(img, allowlist="0123456789.,°RØøDdepth=Ee:"):
        p = np.asarray(box)
        out.append((txt, (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()), float(sc)))
    return out


def _tesseract(img: np.ndarray):
    import pytesseract  # type: ignore
    from pytesseract import Output

    out = []
    for rot in (0, 90):  # 2e passe pour les cotes écrites verticalement
        im = img if rot == 0 else np.rot90(img, k=-1).copy()
        d = pytesseract.image_to_data(im, config="--psm 11", output_type=Output.DICT)
        h = img.shape[0]
        for i, txt in enumerate(d["text"]):
            txt = txt.strip()
            conf = float(d["conf"][i])
            if not txt or conf < 30:
                continue
            x, y, w, hh = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            if rot == 0:
                bbox = (x, y, x + w, y + hh)
            else:  # rot90 horaire : (x', y') -> (y', H - x')
                bbox = (y, h - (x + w), y + hh, h - x)
                if hh > w:  # texte horizontal lu de travers : déjà couvert à 0°
                    continue
            out.append((txt, bbox, conf / 100))
    return out


_BACKENDS = {"paddle": _paddle, "easyocr": _easyocr, "tesseract": _tesseract}


def run_ocr(gray: np.ndarray, backends: list[str]) -> tuple[list[Dimension], list[str]]:
    """Exécute le premier backend disponible et retourne les cotes typées."""
    warnings: list[str] = []
    raw = None
    for name in backends:
        fn = _BACKENDS.get(name)
        if fn is None:
            continue
        try:
            raw = fn(gray)
            log.info("OCR : backend %s, %d textes", name, len(raw))
            break
        except ImportError:
            log.debug("OCR %s non installé", name)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"OCR {name} a échoué : {exc}")
    if raw is None:
        warnings.append("Aucun backend OCR disponible : installez paddleocr, easyocr ou tesseract.")
        return [], warnings

    dims: list[Dimension] = []
    for txt, bbox, conf in _merge_tokens(raw):
        parsed = parse_dimension_ex(txt)
        if parsed is None:
            # texte illisible conservé pour correction manuelle (« 0r », « d=1~ »...)
            t = txt.strip()
            if conf >= 0.2 and (any(c.isdigit() for c in t) or "=" in t or len(t) >= 2):
                dims.append(Dimension(id=f"T{len(dims) + 1}", raw=t, kind="unknown", value=0.0,
                                      bbox=tuple(float(b) for b in bbox), conf=conf))
            continue
        kind, val, unit = parsed
        dims.append(Dimension(id=f"T{len(dims) + 1}", raw=txt, kind=kind, value=val, unit=unit,
                              bbox=tuple(float(b) for b in bbox), conf=conf))
    dims = _dedupe(dims)
    warnings += apply_units(dims)
    return dims, warnings


def apply_units(dims: list[Dimension]) -> list[str]:
    """Convertit en mm. Une cote sans unité prend l'unité écrite sur les autres cotes
    (ex. « r = 0,5 » à côté de « d = 2m » -> 0,5 m)."""
    explicit = {d.unit for d in dims if d.unit and d.kind not in ("angle", "unknown")}
    default = explicit.pop() if len(explicit) == 1 else "mm"
    warnings = []
    for d in dims:
        if d.kind in ("angle", "unknown"):
            continue
        u = d.unit or default
        if d.unit is None and u != "mm":
            warnings.append(f"{d.id} « {d.raw} » sans unité : lue en {u} comme les autres cotes.")
        d.value = d.value * UNITS.get(u, 1.0)
        d.unit = "mm"
    return warnings


def _merge_tokens(raw):
    """Fusionne les jetons voisins sur une même ligne (« ep » + « 5 », « R » + « 10 »)."""
    items = sorted(raw, key=lambda r: (r[1][1], r[1][0]))
    merged: list = []
    used = [False] * len(items)
    for i, (t, b, c) in enumerate(items):
        if used[i]:
            continue
        x0, y0, x1, y1 = b
        h = y1 - y0
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            t2, b2, c2 = items[j]
            same_line = abs((b2[1] + b2[3]) / 2 - (y0 + y1) / 2) < 0.5 * h
            close = 0 <= b2[0] - x1 < 1.2 * h
            if same_line and close and parse_dimension(t) is None:
                t, x1, y0, y1, c = t + t2, b2[2], min(y0, b2[1]), max(y1, b2[3]), min(c, c2)
                used[j] = True
        merged.append((t, (x0, y0, x1, y1), c))
    return merged


def _dedupe(dims: list[Dimension]) -> list[Dimension]:
    """Supprime les doublons (même texte lu par deux passes, bbox qui se chevauchent)."""
    keep: list[Dimension] = []
    for d in sorted(dims, key=lambda d: -d.conf):
        dup = False
        for k in keep:
            ax0, ay0, ax1, ay1 = d.bbox
            bx0, by0, bx1, by1 = k.bbox
            ix = max(0, min(ax1, bx1) - max(ax0, bx0))
            iy = max(0, min(ay1, by1) - max(ay0, by0))
            if ix * iy > 0.3 * min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0)):
                dup = True
                break
        if not dup:
            keep.append(d)
    for i, d in enumerate(sorted(keep, key=lambda d: (d.bbox[1], d.bbox[0]))):
        d.id = f"T{i + 1}"
    return sorted(keep, key=lambda d: int(d.id[1:]))


def dims_from_texts(texts: list[dict]) -> list[Dimension]:
    """Re-analyse une liste de textes {id, raw, bbox, conf} (après correction manuelle)."""
    dims = []
    for t in texts:
        raw = (t.get("raw") or "").strip()
        if not raw:
            continue  # texte effacé par l'utilisateur
        parsed = parse_dimension_ex(raw)
        kind, val, unit = parsed if parsed else ("unknown", 0.0, None)
        dims.append(Dimension(id=t["id"], raw=raw, kind=kind, value=val, unit=unit,
                              bbox=tuple(float(b) for b in t["bbox"]), conf=float(t.get("conf", 1.0))))
    apply_units(dims)
    return dims
