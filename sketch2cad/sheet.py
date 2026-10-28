"""Découpage d'une feuille en dessins indépendants.

Une photo peut contenir plusieurs croquis. Les traits sont regroupés par proximité
(dilatation), chaque groupe assez grand devient un « dessin » ; les textes OCR sont
rattachés au dessin le plus proche. L'utilisateur choisit ensuite le dessin à traiter.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .primitives import _clean
from .schema import Dimension


def find_drawings(binary: np.ndarray, dims: list[Dimension]) -> list[dict]:
    H, W = binary.shape
    S = max(H, W)
    strokes = _clean(binary)
    k = max(5, int(0.025 * S)) | 1
    dil = cv2.dilate(strokes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dil, 8)
    text_mask = np.zeros_like(binary)
    for d in dims:
        x0, y0, x1, y1 = (int(v) for v in d.bbox)
        cv2.rectangle(text_mask, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), 255, -1)
    out = []
    for i in range(1, n):
        x, y, w, h, _a = st[i]
        if max(w, h) < 0.08 * S or min(w, h) < 0.05 * S:
            continue  # poussière, bord de table, trait isolé
        comp = (lab == i) & (strokes > 0)
        ink = int(comp.sum())
        if ink == 0:
            continue
        if (comp & (text_mask > 0)).sum() > 0.7 * ink:
            continue  # uniquement du texte
        touches = int(x <= 1) + int(y <= 1) + int(x + w >= W - 1) + int(y + h >= H - 1)
        out.append({"bbox": [int(x), int(y), int(x + w), int(y + h)], "ink": ink,
                    "cut": touches >= 2 or (touches == 1 and max(w, h) < 0.15 * S)})
    # un dessin au trait fragmenté donne plusieurs morceaux dont les rectangles se chevauchent
    merged = True
    while merged:
        merged = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i]["bbox"], out[j]["bbox"]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    out[i] = {"bbox": [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])],
                              "ink": out[i]["ink"] + out[j]["ink"], "cut": out[i]["cut"] and out[j]["cut"]}
                    out.pop(j)
                    merged = True
                    break
            if merged:
                break
    # les textes : au dessin le plus proche (distance au rectangle)
    for d in out:
        d["texts"] = []
    for t in dims:
        cx, cy = t.center
        best, bd = None, 1e18
        for j, d in enumerate(out):
            x0, y0, x1, y1 = d["bbox"]
            dist = math.hypot(max(x0 - cx, 0, cx - x1), max(y0 - cy, 0, cy - y1))
            if dist < bd:
                best, bd = j, dist
        if best is not None and bd < 0.12 * S:
            out[best]["texts"].append(t.id)
    # les dessins coupés par le bord de la photo passent en dernier
    out.sort(key=lambda d: (d["cut"], -d["ink"]))
    for i, d in enumerate(out):
        d["index"] = i
    return out


def draw_sheet(img: np.ndarray, drawings: list[dict]) -> np.ndarray:
    vis = img.copy()
    sc = max(0.6, max(img.shape[:2]) / 1200)
    for d in drawings:
        x0, y0, x1, y1 = d["bbox"]
        col = (160, 160, 160) if d["cut"] else (0, 90, 230)
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, max(2, int(3 * sc)))
        label = f"{d['index'] + 1}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1 * sc, max(2, int(3 * sc)))
        cv2.rectangle(vis, (x0, y0), (x0 + tw + 16, y0 + th + 16), col, -1)
        cv2.putText(vis, label, (x0 + 8, y0 + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.1 * sc, (255, 255, 255),
                    max(2, int(3 * sc)), cv2.LINE_AA)
    return vis


def region_mask(shape, bbox, pad_frac: float = 0.0) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    H, W = shape
    x0, y0, x1, y1 = bbox
    p = int(pad_frac * max(H, W))
    r = (max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p))
    m = np.zeros((H, W), np.uint8)
    m[r[1]:r[3], r[0]:r[2]] = 255
    return m, r
