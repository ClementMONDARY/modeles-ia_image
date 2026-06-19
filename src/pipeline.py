from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PIL import Image

from .detection import Detection, ObjectDetector
from .segmentation import ImageSegmentor, SegmentationResult

DEFAULT_DETECTION_MODEL = "hustvl/yolos-small"
DEFAULT_SEGMENTATION_MODEL = "facebook/maskformer-swin-small-coco"


@dataclass
class PipelineResult:
    """Résultat complet pour un objet détecté."""

    detection: Detection
    crop: Image.Image                               # Portion de l'image originale
    segmentation: Optional[SegmentationResult]     # Masque local sur le crop
    mask_on_original: Optional[np.ndarray]         # Masque reprojeté sur l'image originale
    crop_box: tuple = field(default_factory=tuple) # (px1, py1, px2, py2) avec padding


class DetectionSegmentationPipeline:
    def __init__(
        self,
        detection_model: str = DEFAULT_DETECTION_MODEL,
        segmentation_model: str = DEFAULT_SEGMENTATION_MODEL,
        detection_threshold: float = 0.5,
        crop_padding: int = 10,
        device: str = None,
    ):
        self.crop_padding = crop_padding

        self.detector = ObjectDetector(
            model_name=detection_model,
            threshold=detection_threshold,
            device=device,
        )
        self.segmentor = ImageSegmentor(
            model_name=segmentation_model,
            device=device,
        )

    def run(self, image: Image.Image) -> List[PipelineResult]:
        image = image.convert("RGB")
        img_w, img_h = image.size

        print(f"\n[Pipeline] Détection sur image {img_w}×{img_h}...")
        detections = self.detector.detect(image)
        print(f"[Pipeline] {len(detections)} objet(s) détecté(s).")

        results = []
        for i, det in enumerate(detections):
            print(f"[Pipeline] Segmentation {i + 1}/{len(detections)} : '{det.label}' ({det.score:.2f})")

            # Crop avec padding, borné aux dimensions de l'image
            xmin, ymin, xmax, ymax = det.box
            px1 = max(0, xmin - self.crop_padding)
            py1 = max(0, ymin - self.crop_padding)
            px2 = min(img_w, xmax + self.crop_padding)
            py2 = min(img_h, ymax + self.crop_padding)

            crop = image.crop((px1, py1, px2, py2))
            segmentation = self.segmentor.segment(crop, expected_label=det.label)

            mask_on_original = self._project_mask(
                segmentation, px1, py1, px2, py2, img_h, img_w
            )

            results.append(
                PipelineResult(
                    detection=det,
                    crop=crop,
                    segmentation=segmentation,
                    mask_on_original=mask_on_original,
                    crop_box=(px1, py1, px2, py2),
                )
            )

        return results

    def _project_mask(
        self,
        segmentation: Optional[SegmentationResult],
        px1: int,
        py1: int,
        px2: int,
        py2: int,
        img_h: int,
        img_w: int,
    ) -> Optional[np.ndarray]:
        if segmentation is None:
            return None

        crop_w = px2 - px1
        crop_h = py2 - py1

        # Redimensionnement du masque à la taille exacte du crop
        mask_img = Image.fromarray(segmentation.mask * 255, mode="L")
        mask_img = mask_img.resize((crop_w, crop_h), Image.NEAREST)
        mask_resized = (np.array(mask_img) > 127).astype(np.uint8)

        # Projection dans un tableau de la taille de l'image originale
        mask_full = np.zeros((img_h, img_w), dtype=np.uint8)
        mask_full[py1:py2, px1:px2] = mask_resized

        return mask_full
