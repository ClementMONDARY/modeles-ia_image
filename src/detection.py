from dataclasses import dataclass
from typing import List

import torch
from PIL import Image
from transformers import YolosForObjectDetection, YolosImageProcessor

DEFAULT_MODEL = "hustvl/yolos-small"


@dataclass
class Detection:
    """Résultat d'une détection : label, score et boîte englobante en pixels."""

    label: str
    score: float
    box: tuple  # (xmin, ymin, xmax, ymax)


class ObjectDetector:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: float = 0.5,
        device: str = None,
    ):
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Détection] Chargement du modèle '{model_name}' sur {self.device}...")
        self.processor = YolosImageProcessor.from_pretrained(model_name)
        self.model = YolosForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("[Détection] Modèle prêt.")

    def detect(self, image: Image.Image) -> List[Detection]:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-traitement : conversion logits -> boîtes en pixels
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.threshold,
            target_sizes=target_sizes,
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box_px = box.cpu().numpy().astype(int)
            detections.append(
                Detection(
                    label=self.model.config.id2label[label.item()],
                    score=float(score),
                    box=(int(box_px[0]), int(box_px[1]), int(box_px[2]), int(box_px[3])),
                )
            )

        # Tri par score décroissant
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections
