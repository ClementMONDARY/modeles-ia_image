# Pipeline Détection + Segmentation d'Images

Pipeline complet qui, à partir d'une image :
1. Détecte les objets présents avec **YOLOS-small**
2. Extrait chaque objet détecté (crop)
3. Applique la segmentation d'instances avec **MaskFormer**
4. Reprojette les masques sur l'image originale

---

## Structure du projet

```
modeles-ia_image/
├── main.py                 # CLI : lancer le pipeline sur une image
├── requirements.txt        # Dépendances Python
├── src/
│   ├── detection.py        # Détection via YOLOS-small (HuggingFace)
│   ├── segmentation.py     # Segmentation via MaskFormer (HuggingFace)
│   ├── pipeline.py         # Pipeline combiné détection → crop → segmentation
│   └── visualization.py    # Superposition des masques sur l'image originale
├── api/
│   └── main.py             # API FastAPI (optionnel)
├── images/                 # Dossier pour les images d'entrée
└── outputs/                # Dossier pour les résultats générés
```

---

## Installation

```bash
# 1. Créer un environnement virtuel (recommandé)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 2. Installer les dépendances
pip install -r requirements.txt
```

> **Note GPU** : si vous avez un GPU NVIDIA avec CUDA, installez d'abord PyTorch
> avec support CUDA depuis [pytorch.org](https://pytorch.org/get-started/locally/).
> Le pipeline détecte automatiquement le GPU disponible.

---

## Utilisation

Modifiez les variables en haut de `main.py` :

```python
IMAGE_PATH  = "images/cat.jpg"      # image à analyser
OUTPUT_PATH = "outputs/result.jpg"  # résultat annoté
SAVE_CROPS  = True                  # sauvegarder les crops individuels
```

Puis lancez :

```bash
python main.py
```

### En Python

```python
from PIL import Image
from src.pipeline import DetectionSegmentationPipeline
from src.visualization import draw_results, print_summary

pipeline = DetectionSegmentationPipeline(detection_threshold=0.5)
image = Image.open("images/photo.jpg").convert("RGB")

results = pipeline.run(image)
print_summary(results)

annotated = draw_results(image, results)
annotated.save("outputs/result.jpg")
```

---

## API FastAPI (optionnel)

```bash
# Lancer le serveur
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Documentation interactive : http://localhost:8000/docs
```

### Endpoints

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/health` | Statut de l'API |
| `POST` | `/analyze` | JSON + image annotée en base64 |
| `POST` | `/analyze/image` | Image annotée directement (JPEG) |

### Exemples cURL

```bash
# Récupérer l'image annotée
curl -X POST "http://localhost:8000/analyze/image" \
     -F "file=@images/photo.jpg" \
     --output result.jpg

# Obtenir les détections en JSON
curl -X POST "http://localhost:8000/analyze" \
     -F "file=@images/photo.jpg"
```

---

## Modèles utilisés

### Détection : `hustvl/yolos-small`

- Architecture transformeur (YOLOS), fine-tuné sur **COCO** (80 classes)
- Entrée : image de taille quelconque → Sortie : boîtes englobantes + labels + scores
- ~30 M paramètres, rapide à l'inférence

### Segmentation : `facebook/maskformer-swin-small-coco`

- Architecture MaskFormer avec backbone Swin-Small, fine-tuné sur **COCO Panoptic** (133 classes)
- Entrée : crop d'un objet → Sortie : carte de segmentation panoptique pixel-à-pixel
- ~70 M paramètres

### Stratégie de sélection du masque

Lors de la segmentation d'un crop, le modèle peut renvoyer plusieurs segments. La sélection suit la priorité suivante :

1. **Correspondance par label** : segment dont le label correspond à celui détecté par YOLOS
2. **Plus grande instance** : segment "thing" (pas de la classe "stuff") avec la plus grande aire
3. **Segment central** : segment présent au centre géométrique du crop

---

## Dépendances principales

| Paquet | Rôle |
|---|---|
| `torch` / `torchvision` | Inférence des modèles |
| `transformers` | Chargement des modèles HuggingFace |
| `Pillow` | Manipulation d'images |
| `numpy` | Opérations sur les masques |
| `fastapi` + `uvicorn` | API REST (optionnel) |

---

## Obtenir des images de test

Vous pouvez tester le pipeline avec des images issues de :

- **MS COCO** : [cocodataset.org](https://cocodataset.org/#download) — base de données généraliste
- **HuggingFace Datasets** :
  ```python
  from datasets import load_dataset
  ds = load_dataset("rafaelpadilla/coco2017", split="validation", streaming=True)
  sample = next(iter(ds))
  sample["image"].save("images/coco_sample.jpg")
  ```
- **Vos propres photos** : placez-les dans le dossier `images/`
