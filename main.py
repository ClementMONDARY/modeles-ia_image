from pathlib import Path

from PIL import Image

from src.pipeline import DetectionSegmentationPipeline
from src.visualization import draw_results, print_summary, save_crops

# ------------------------------------------------------------------ #
#  Configuration                                                      #
# ------------------------------------------------------------------ #

IMAGE_PATH  = "images/cat.jpg"   # Image à analyser
OUTPUT_PATH = "outputs/result.jpg"  # Image annotée en sortie
SAVE_CROPS  = True               # Sauvegarder les crops dans outputs/crops/

# ------------------------------------------------------------------ #

image = Image.open(IMAGE_PATH).convert("RGB")
print(f"Image chargée : {IMAGE_PATH} ({image.size[0]}×{image.size[1]} pixels)")

pipeline = DetectionSegmentationPipeline()
results = pipeline.run(image)

print_summary(results)

if results:
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    annotated = draw_results(image, results)
    annotated.save(OUTPUT_PATH)
    print(f"Image annotée sauvegardée : {OUTPUT_PATH}")

    if SAVE_CROPS:
        saved = save_crops(results, "outputs/crops")
        print(f"{len(saved)} crop(s) sauvegardé(s) dans outputs/crops/")
else:
    print("Aucun objet détecté.")
