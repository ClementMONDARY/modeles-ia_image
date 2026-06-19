from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor

DEFAULT_MODEL = "facebook/maskformer-swin-small-coco"


@dataclass
class SegmentationResult:
    """Résultat de segmentation : masque binaire, label et score."""

    mask: np.ndarray   # Tableau uint8 de même taille que le crop (0 ou 1)
    label: str
    score: float


class ImageSegmentor:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Segmentation] Chargement du modèle '{model_name}' sur {self.device}...")
        self.processor = MaskFormerImageProcessor.from_pretrained(model_name)
        self.model = MaskFormerForInstanceSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("[Segmentation] Modèle prêt.")

    def segment(
        self,
        crop: Image.Image,
        expected_label: Optional[str] = None,
    ) -> Optional[SegmentationResult]:
        inputs = self.processor(images=crop, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # target_sizes attend (height, width)
        h, w = crop.size[1], crop.size[0]
        panoptic_result = self.processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[(h, w)],
        )[0]

        segmentation_map = panoptic_result.get("segmentation")
        segments_info = panoptic_result.get("segments_info", [])

        if segmentation_map is None or not segments_info:
            return None

        seg_map = segmentation_map.cpu().numpy()

        # --- Stratégie 1 : correspondance par label ---
        if expected_label:
            candidate = self._find_by_label(seg_map, segments_info, expected_label)
            if candidate:
                return candidate

        # --- Stratégie 2 : plus grand segment "thing" (was_fused=False) ---
        candidate = self._find_largest_thing(seg_map, segments_info)
        if candidate:
            return candidate

        # --- Stratégie 3 : segment couvrant le centre du crop ---
        return self._find_center_segment(seg_map, segments_info, h, w)

    # ------------------------------------------------------------------ #
    #  Méthodes de sélection internes                                     #
    # ------------------------------------------------------------------ #

    def _label_name(self, label_id: int) -> str:
        return self.model.config.id2label.get(label_id, f"class_{label_id}")

    def _build_result(self, seg_map: np.ndarray, seg: dict) -> SegmentationResult:
        mask = (seg_map == seg["id"]).astype(np.uint8)
        label = self._label_name(seg.get("label_id", 0))
        score = float(seg.get("score", 1.0))
        return SegmentationResult(mask=mask, label=label, score=score)

    def _find_by_label(
        self, seg_map: np.ndarray, segments_info: list, expected_label: str
    ) -> Optional[SegmentationResult]:
        """Retourne le segment dont le label correspond, en prenant le plus grand si plusieurs."""
        matches = [
            s for s in segments_info
            if self._label_name(s.get("label_id", 0)).lower() == expected_label.lower()
        ]
        if not matches:
            return None
        best = max(matches, key=lambda s: int((seg_map == s["id"]).sum()))
        return self._build_result(seg_map, best)

    def _find_largest_thing(
        self, seg_map: np.ndarray, segments_info: list
    ) -> Optional[SegmentationResult]:
        """Retourne l'instance (was_fused=False) avec la plus grande aire."""
        things = [s for s in segments_info if not s.get("was_fused", False)]
        if not things:
            return None
        best = max(things, key=lambda s: int((seg_map == s["id"]).sum()))
        return self._build_result(seg_map, best)

    def _find_center_segment(
        self, seg_map: np.ndarray, segments_info: list, h: int, w: int
    ) -> Optional[SegmentationResult]:
        """Retourne le segment présent au centre du crop."""
        cy, cx = h // 2, w // 2
        center_id = int(seg_map[cy, cx])
        for seg in segments_info:
            if seg["id"] == center_id:
                return self._build_result(seg_map, seg)
        return None
