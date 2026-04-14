# Fine-tune EfficientNet

### 1. Fine-tune a pre-trained **TF-EfficientNet-Lite4** model on a custom image dataset.

> [!NOTE]
> It's recommended to use Python 3.11.x

1. Create a `dataset` folder.
2. Create subfolders (example: dog, cat, fish, bird) and populate them.
3. Run `python src/fine_tune_from_base.py`.

When its finished it will return a checkpoint file.

### 2. Create a tflite model
1. Run `python src/convert_to_tflite.py`.

This will return a tflite and labels.txt file.

### 3. Add metadata to tflite model
1. Run `python src/add_metadata_to_tflite.py`.

> [!TIP]
> Fine-tuned models are used in conjunction with the [Vision Classification](https://github.com/doppeltilde/vision_classification) project, utilizing its Mediapipe Image Classification task.

---
_Notice:_ _This project was initally created to be used in-house, as such the
development is first and foremost aligned with the internal requirements._