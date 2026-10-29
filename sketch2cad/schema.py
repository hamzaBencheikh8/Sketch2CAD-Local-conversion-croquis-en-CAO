"""Structures de données partagées entre les agents (validées par Pydantic).

Convention de coordonnées :
- ``px`` : pixels de l'image redressée (origine en haut à gauche, y vers le bas) ;
- ``xy`` : millimètres dans le repère pièce (y vers le haut), produit par le solveur.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

DimKind = Literal["length", "angle", "radius", "diameter", "thickness", "ambiguous", "unknown"]


class Dimension(BaseModel):
    """Une cote manuscrite lue par l'OCR."""

    id: str                                   # T1, T2...
    raw: str                                  # texte brut lu
    kind: DimKind
    value: float
    bbox: tuple[float, float, float, float]   # x0, y0, x1, y1 (px)
    conf: float = 1.0
    unit: Optional[str] = None                # unité écrite (mm, cm, m) ; valeurs converties en mm

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)


class Vertex(BaseModel):
    id: str
    px: tuple[float, float]


class Edge(BaseModel):
    id: str
    a: str = Field(alias="from")
    b: str = Field(alias="to")

    model_config = {"populate_by_name": True}


class Circle(BaseModel):
    id: str
    px: tuple[float, float]
    r_px: float


class Perception(BaseModel):
    """Sortie de l'agent 1."""

    image_size: tuple[int, int]               # largeur, hauteur
    stroke_px: float
    vertices: list[Vertex]
    edges: list[Edge]
    circles: list[Circle]
    dimensions: list[Dimension]
    warnings: list[str] = []
    shape: str = "prism"                      # prism | cylinder | revolution | vessel | circle | choose_drawing
    cylinder: Optional[dict] = None           # mesures px si shape == cylinder
    revolution: Optional[dict] = None         # demi-profil px si shape == revolution
    circle: Optional[dict] = None             # cercle px si shape == circle
    vessel: Optional[dict] = None             # réservoir px si shape == vessel
    drawings: list[dict] = []                 # dessins trouvés sur la feuille
    drawing: Optional[int] = None             # dessin traité
    choices: list[str] = []                   # interprétations possibles si un choix est requis


ConstraintType = Literal[
    "length", "angle", "horizontal", "vertical", "perpendicular", "parallel",
    "fillet", "diameter", "distance", "arc_radius", "arc_tangent", "radius_at", "height",
]


class Constraint(BaseModel):
    type: ConstraintType
    value: Optional[float] = None
    edge: Optional[str] = None
    edges: Optional[list[str]] = None
    vertex: Optional[str] = None
    circle: Optional[str] = None
    target: Optional[str] = None             # pour "distance" : arête de référence
    implicit: bool = False
    source: Optional[str] = None             # id de la cote OCR (T*) d'origine
    note: Optional[str] = None


class Graph(BaseModel):
    """Graphe de contraintes géométriques (sortie de l'agent 2, entrée du solveur)."""

    vertices: list[Vertex]
    edges: list[Edge]
    circles: list[Circle] = []
    constraints: list[Constraint] = []
    thickness: Optional[float] = None
    units: str = "mm"
    unassigned: list[str] = []               # cotes OCR non rattachées
    warnings: list[str] = []
    method: str = "heuristic"                 # heuristic | vlm

    def edge(self, eid: str) -> Edge:
        return next(e for e in self.edges if e.id == eid)

    def vidx(self, vid: str) -> int:
        return next(i for i, v in enumerate(self.vertices) if v.id == vid)


class Solution(BaseModel):
    """Sortie de l'agent 3."""

    points: list[tuple[float, float]]                    # mm, ordre des sommets, CCW
    fillets: dict[int, float] = {}                       # index sommet -> rayon
    holes: list[tuple[float, float, float]] = []         # x, y, diamètre
    thickness: float
    dof: int = 0
    max_residual_mm: float = 0.0
    conflicts: list[str] = []
    relaxed: list[str] = []
    warnings: list[str] = []
    iterations: int = 0
