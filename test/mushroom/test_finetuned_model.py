from pathlib import Path
import mediapipe as mp
import numpy as np
from PIL import Image

BaseOptions = mp.tasks.BaseOptions
ImageClassifier = mp.tasks.vision.ImageClassifier
ImageClassifierOptions = mp.tasks.vision.ImageClassifierOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "test/mushroom/mushroom.tflite"
image_path = "test/mushroom/mushroom.jpg"

options = ImageClassifierOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    max_results=5,
    running_mode=VisionRunningMode.IMAGE,
)


with ImageClassifier.create_from_options(options) as classifier:
    image = Image.open(image_path).convert("RGB")
    numpy_image = np.array(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
    classification_result = classifier.classify(mp_image)

    if classification_result.classifications:
        print(f"\n{'SPECIES':<25} | {'SCORE':<10} | {'PERCENT':<10} | {'ID'}")
        print("-" * 65)

        top_categories = classification_result.classifications[0].categories

        for category in top_categories:
            name = category.display_name
            raw_score = category.score
            percentage = category.score * 100
            cat_id = category.category_name

            print(f"{name:<25} | {raw_score:<10.4f} | {percentage:>8.2f}% | {cat_id}")

        # Should return
        # SPECIES                   | SCORE      | PERCENT    | ID
        # -----------------------------------------------------------------
        # Amanita muscaria          | 0.9961     |    99.61% | 10252
        # Agaricus altipes          | 0.0000     |     0.00% | 10052
        # Agaricus augustus         | 0.0000     |     0.00% | 10057
        # Abortiporus biennis       | 0.0000     |     0.00% | 10000
        # Achroomyces disciformis   | 0.0000     |     0.00% | 10025
