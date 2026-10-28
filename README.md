# Sketch2CAD-Local

Photo d'un croquis mécanique coté à la main → plan 2D (`.dxf`, `.svg`) et solide 3D (`.step`, `.stl`), **100 % en local** par défaut — avec, en option, une lecture des cotes par une **API gratuite** (Gemini, Groq ou OpenRouter).

Les cotes manuscrites font foi : le croquis peut être complètement hors échelle, le solveur de contraintes recalcule la géométrie exacte au millimètre.

```
photo ──► 1 Perception ──► 2 Graphe ──► 3 Solveur ◄──► 5 Critique ──► 4 Code-as-CAD ──► .dxf .svg .step .stl
          OpenCV + OCR     VLM ou        moindres     convexité,       part.py
          Set-of-Mark      proximité     carrés       IoU, perçages    build123d + ezdxf
```

## Installation (Windows)

1. Installer **Python 3.12** (python.org, cocher « Add to PATH »).
2. Double-cliquer `install.bat` (crée `.venv`, installe les dépendances, lance un diagnostic).
3. *(Optionnel, pour les profils avec IA)* installer [Ollama](https://ollama.com) puis :
   ```
   ollama pull qwen2.5vl:3b        # VLM, profils gpu4 / cpu
   ollama pull qwen2.5-coder:3b    # codeur, profil gpu4
   ollama pull qwen2.5vl:7b        # profil gpu8
   ollama pull qwen2.5-coder:7b    # profil gpu8
   ```

Au premier lancement, PaddleOCR télécharge ses modèles (~100 Mo) ; ensuite tout fonctionne sans réseau.

## Interface web de test

Double-cliquer `web.bat` (ou `python -m sketch2cad web`) : le navigateur s'ouvre sur http://127.0.0.1:8765.

- déposer une photo (ou cliquer **Exemple**), choisir le profil, **Lancer** ;
- onglets **3D** (rotation à la souris), **Plan 2D**, **Croquis annoté**, **Comparaison**, **Rapport** ;
- onglet **Cotes** : corriger une valeur ou l'élément coté, puis **Recalculer** (quelques secondes, sans refaire l'OCR) ;
- boutons de téléchargement : `.step`, `.stl`, `.dxf`, `.svg`, `part.py`.

Le serveur n'écoute que sur 127.0.0.1 ; les essais sont rangés dans `sortie_web\`. La visionneuse 3D (three.js) est embarquée dans le projet : aucune ressource n'est chargée depuis Internet.

## Utilisation en ligne de commande

```
run.bat C:\chemin\croquis.jpg             :: profil offline (aucun modèle de langage)
run.bat C:\chemin\croquis.jpg gpu4        :: avec Qwen2.5-VL 3B
```

ou directement :

```
python -m sketch2cad --profile offline run croquis.jpg -o sortie\piece1
python -m sketch2cad solve sortie\piece1\graph.json -o sortie\piece1b   # après correction manuelle
python -m sketch2cad --profile gpu4 doctor                               # vérifier l'installation
```

| Profil    | Matériel              | VLM (agent 2)   | Codeur (agent 4)   |
|-----------|-----------------------|-----------------|--------------------|
| `gpu8`    | GPU 8–12 Go           | qwen2.5vl:7b    | qwen2.5-coder:7b   |
| `gpu4`    | GPU 4–8 Go            | qwen2.5vl:3b    | qwen2.5-coder:3b   |
| `cpu`     | CPU, ≥ 16 Go RAM      | qwen2.5vl:3b    | gabarit seul       |
| `offline` | n'importe quel PC     | aucun           | gabarit seul       |
| `api`     | n'importe quel PC + Internet | API gratuite (lecture des cotes) | gabarit seul |

## Profil « api » : lecture des cotes par une API gratuite

L'écriture manuscrite est le point faible de l'OCR local. Le profil `api` fait relire les cotes par
un grand modèle de vision **gratuit** ; le solveur, la critique et la CAO restent locaux et exacts.

| Fournisseur | Clé gratuite | Modèle par défaut |
|---|---|---|
| Google Gemini | https://aistudio.google.com/apikey | `gemini-2.5-flash` |
| Groq | https://console.groq.com/keys | `qwen/qwen3.8-27b` |
| OpenRouter | https://openrouter.ai/keys | `openrouter/free` (routeur vers les modèles gratuits) |

Les modèles proposés changent souvent : le bouton **Lister** de l'interface (ou `api --list-models`) donne la liste à jour.

**Dans l'interface web** : « Lecture des cotes » → **API gratuite**, choisir le fournisseur, coller la clé,
**Enregistrer** (la connexion est testée), puis **Lancer**. Deux portées :

- **Zones de texte seulement** (par défaut) : seules les petites zones de texte (« 150 », « R10 »…) sont
  envoyées, rassemblées dans une seule image ; **le dessin de la pièce ne quitte pas le PC** ;
- **Image entière** : meilleure lecture, l'API retrouve aussi les cotes oubliées par l'OCR et rattache les cotes aux éléments.

**En ligne de commande** :

```
python -m sketch2cad api --provider gemini --key VOTRE_CLE --test
python -m sketch2cad --profile api run croquis.jpg -o sortie\piece1
```

La clé est enregistrée dans `api_settings.json` (sur ce PC, ignoré par git) ou lue dans la variable
d'environnement `GEMINI_API_KEY`, `GROQ_API_KEY` ou `OPENROUTER_API_KEY`. Une seule requête par croquis
(portée « textes »), ce qui tient largement dans les quotas gratuits. Sans clé, sans réseau ou quota
épuisé : avertissement, et l'OCR local prend le relais. Les cotes relues sont marquées **API** dans
l'onglet Textes, et le rapport détaille chaque correction (`report.json` → `api`).

> Les offres gratuites peuvent utiliser les données envoyées pour améliorer les modèles du fournisseur :
> pour des plans confidentiels, restez en mode **Local** ou en portée « zones de texte ».

## Fichiers produits (dans le dossier de sortie)

| Fichier            | Contenu |
|--------------------|---------|
| `part.step`, `part.stl` | solide 3D extrudé (perçages, congés) |
| `part.dxf`, `part.svg`  | mise en plan cotée (calques CONTOUR, PERCAGES, COTATION, AXES) |
| `part.py`          | script CAO autonome et lisible : modifiez-le et relancez `python part.py` |
| `som.png`          | croquis annoté V*/E*/C*/T* : à regarder en premier en cas de doute |
| `comparison.png`   | croquis original à côté du profil reconstruit |
| `graph.json`       | graphe de contraintes, **éditable** puis rejouable avec `solve` |
| `report.json`      | résidus, degrés de liberté, conflits, corrections de l'agent critique, lecture API |
| `api_read.json`    | (profil api) ce qui a été envoyé et chaque cote corrigée ou ajoutée par l'API |

## Bien dessiner pour la machine

- **Contour au trait épais** (feutre), **cotation au trait fin** (stylo) : c'est ainsi que l'agent 1 sépare la pièce des lignes de cote (convention ISO 128).
- Une cote par élément, écrite près de sa ligne de cote ; `R` pour les congés, `Ø` pour les perçages, `ep 5` (ou `e=5`) pour l'épaisseur.
- Un angle rentrant peut être coté côté extérieur (ex. `120°` pour un coin à 240°) : l'agent critique corrige le sens.
- Feuille blanche entière dans la photo, éclairage homogène, sans ombre portée sur le dessin.
- Une petite capture d'écran (< 1000 px) est agrandie automatiquement ; le contour et la cotation sont alors séparés par le noir du trait (contour plus foncé). Une photo nette d'au moins 1500 px reste préférable.

## Corriger un résultat

1. Ouvrir `som.png` : chaque cote lue (T*) et chaque élément (V*, E*, C*) y est étiqueté.
2. Corriger `graph.json` (valeur, ou arête `edge` à laquelle la cote est rattachée).
3. `python -m sketch2cad solve sortie\piece1\graph.json -o sortie\piece1b`.

Pour court-circuiter l'OCR, on peut aussi fournir les cotes à la main : `run croquis.jpg --dims cotes.json` (format : voir `examples/dims.json`).

## Tests

```
python tools\make_synthetic_sketch.py examples   # croquis hors échelle + vérité terrain
python -m pytest -q tests
```

Le croquis d'exemple (`examples/sketch_photo.jpg`, feuille photographiée en biais) est reconstruit à 10⁻¹³ mm près en mode `offline` avec PaddleOCR.

## Formes reconnues

| Forme | Comment elle est reconnue | Cotes utiles |
|---|---|---|
| Pièce plate (extrusion) | contour fermé vu de dessus | longueurs, angles, `R`, `Ø` des perçages, `ep` |
| Cylindre | deux génératrices + ellipses des bases | `r=` / `Ø` dans une base, hauteur écrite à côté |
| Corps de révolution (gélule, cône, tronc, hémisphère, bouteille…) | silhouette symétrique + ellipse de base ou dôme | `r=` du dôme, longueur à côté d'un flanc, angle, `Ø` d'une base |
| Cercle seul | contour circulaire | `r=` ou `Ø` ; **l'interface demande : sphère ou disque ?** |
| Dessin ouvert (un « V », un arc) | contour non fermé | signalé : pas de solide possible |

Unités m, cm, mm : une cote sans unité prend celle des autres cotes du dessin.

## Feuille avec plusieurs dessins

Les dessins sont détectés et numérotés (image `sheet.png`) ; l'interface demande lequel traiter,
puis permet d'en changer (« Changer de dessin »). En ligne de commande : `--drawing 2`.

## Quand l'OCR lit mal

L'écriture manuscrite est souvent mal lue (« r=0,5 » lu « r= 0r »). Onglet **Textes** de l'interface :
corriger le texte tel qu'il est écrit, en vider un pour l'ignorer, ou **ajouter** une cote manquante en
indiquant où elle est écrite (au centre, à droite…), puis « Appliquer et relancer » (sans refaire l'OCR).
Un texte illisible ou l'absence de cote donne toujours le statut « À VÉRIFIER ».

## Limites de la v1

Pas encore : pièces combinant plusieurs formes (perçage dans un cylindre, assemblages), poches, chanfreins, tôlerie pliée, vues multiples. Les angles de révolution sont pris comme angles intérieurs du profil (un avertissement signale un écart > 30° avec le dessin). Cotes de position des perçages par l'heuristique (le VLM peut les fournir ; sinon position d'après le dessin, signalée dans le rapport).
