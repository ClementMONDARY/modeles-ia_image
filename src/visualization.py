"""
Utilitaires de visualisation des résultats détection + segmentation.

Fonctions principales :
  - draw_results : superpose masques et boîtes sur l'image originale.
  - save_crops   : enregistre les crops avec leur masque de segmentation.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw

from .pipeline import PipelineResult

# Palette de couleurs vives pour différencier les objets
_PALETTE = [
    (220,  50,  47),   # rouge
    ( 38, 139, 210),   # bleu
    (133, 153,   0),   # vert
    (211,  54, 130),   # magenta
    ( 42, 161, 152),   # cyan
    (203,  75,  22),   # orange
    (108, 113, 196),   # violet
    (181, 137,   0),   # jaune
]


def _color(index: int) -> tuple:
    return _PALETTE[index % len(_PALETTE)]


def draw_results(
    image: Image.Image,
    results: List[PipelineResult],
    mask_alpha: float = 0.45,
) -> Image.Image:
    """
    Superpose les masques de segmentation et les boîtes englobantes sur l'image.

    Args:
        image: Image originale PIL (RGB).
        results: Liste de PipelineResult issus du pipeline.
        mask_alpha: Opacité des masques (0 = transparent, 1 = opaque).

    Returns:
        Nouvelle image PIL (RGB) avec les annotations.
    """
    output = image.convert("RGBA").copy()

    # --- Couche des masques de segmentation ---
    for i, result in enumerate(results):
        if result.mask_on_original is None:
            continue
        color = _color(i)
        alpha_val = int(255 * mask_alpha)

        # Création d'une couche RGBA colorée là où le masque est actif
        mask_layer = np.zeros((*result.mask_on_original.shape, 4), dtype=np.uint8)
        mask_layer[result.mask_on_original > 0] = (*color, alpha_val)
        mask_img = Image.fromarray(mask_layer, "RGBA")
        output = Image.alpha_composite(output, mask_img)

    # --- Couche des boîtes englobantes et labels ---
    draw = ImageDraw.Draw(output)
    for i, result in enumerate(results):
        color = _color(i)
        xmin, ymin, xmax, ymax = result.detection.box

        # Boîte englobante
        draw.rectangle(
            [xmin, ymin, xmax, ymax],
            outline=(*color, 255),
            width=3,
        )

        # Étiquette : label du détecteur + score
        det_label = f"{result.detection.label} {result.detection.score:.0%}"
        # Label du segmenteur si disponible et différent
        if result.segmentation and result.segmentation.label.lower() != result.detection.label.lower():
            det_label += f" → {result.segmentation.label}"

        text_w = len(det_label) * 7 + 6
        text_h = 18
        ty = max(0, ymin - text_h)
        draw.rectangle([xmin, ty, xmin + text_w, ty + text_h], fill=(*color, 210))
        draw.text((xmin + 3, ty + 2), det_label, fill=(255, 255, 255, 255))

    return output.convert("RGB")


def save_crops(
    results: List[PipelineResult],
    output_dir: str,
    draw_mask: bool = True,
) -> List[Path]:
    """
    Enregistre chaque crop (optionnellement avec son masque) dans un répertoire.

    Args:
        results: Résultats du pipeline.
        output_dir: Dossier de destination (créé si absent).
        draw_mask: Si True, superpose le masque sur chaque crop.

    Returns:
        Liste des chemins de fichiers créés.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    for i, result in enumerate(results):
        label = result.detection.label.replace(" ", "_")
        filename = out / f"crop_{i:02d}_{label}.jpg"

        crop_img = result.crop.copy()

        if draw_mask and result.segmentation is not None:
            color = _color(i)
            mask = result.segmentation.mask
            # Redimensionne le masque à la taille du crop si nécessaire
            cw, ch = crop_img.size
            if mask.shape != (ch, cw):
                mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((cw, ch), Image.NEAREST)
                mask = (np.array(mask_pil) > 127).astype(np.uint8)

            overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
            overlay[mask > 0] = (*color, 100)
            crop_rgba = crop_img.convert("RGBA")
            crop_rgba = Image.alpha_composite(crop_rgba, Image.fromarray(overlay, "RGBA"))
            crop_img = crop_rgba.convert("RGB")

        crop_img.save(filename)
        paths.append(filename)

    return paths


def print_summary(results: List[PipelineResult]) -> None:
    """Affiche un récapitulatif textuel des résultats dans le terminal."""
    if not results:
        print("Aucun objet détecté.")
        return

    print(f"\n{'─' * 55}")
    print(f"  {'N°':<4} {'Détection':<18} {'Score':>6}  {'Segmentation':<18}")
    print(f"{'─' * 55}")
    for i, r in enumerate(results):
        seg_label = r.segmentation.label if r.segmentation else "—"
        seg_score = f"{r.segmentation.score:.0%}" if r.segmentation else ""
        print(
            f"  {i + 1:<4} {r.detection.label:<18} {r.detection.score:>5.0%}  "
            f"{seg_label:<18} {seg_score}"
        )
    print(f"{'─' * 55}\n")
