import os
import torch
import timm
import litert_torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import traceback
from pathlib import Path
import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb

script_dir = Path(__file__).parent.resolve()


class NormalizedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(NORM_MEAN).view(1, 1, 1, 3))
        self.register_buffer("std", torch.tensor(NORM_STD).view(1, 1, 1, 3))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)


class LogitNormalizer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.model(x)
        return self.softmax(logits)


def main():
    print("Loading fine-tuned model for conversion...")

    transform = transforms.Compose(
        [
            transforms.Resize(456),
            transforms.CenterCrop(380),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageFolder(root="dataset", transform=transform)
    class_names = dataset.classes
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")

    base_model = timm.create_model(
        "tf_efficientnet_lite4", pretrained=False, num_classes=num_classes
    )

    checkpoint_path = script_dir / "tflite/finetuned.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    base_model.load_state_dict(checkpoint["model_state_dict"])

    print("Finetuned model loaded successfully.")

    labels_path = Path(script_dir / "tflite/labels.txt")
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    with open(labels_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")

    print(f"labels.txt created successfully at: {labels_path}")
    print(f"Total labels written: {len(class_names)}")

    model = LogitNormalizer(base_model)
    model.eval()

    print("\nConverting model to TensorFlow Lite format...")

    try:
        sample_input_nhwc = torch.randn(1, 380, 380, 3, dtype=torch.float32)

        model_for_conversion = litert_torch.to_channel_last_io(model, args=[0])

        edge_model = litert_torch.convert(model_for_conversion, (sample_input_nhwc,))

        os.makedirs("tflite", exist_ok=True)
        tflite_path = script_dir / "tflite/model.tflite"

        edge_model.export(tflite_path)

        print(f"Conversion successful!")
        print(f"TensorFlow Lite model saved to: {tflite_path}")

    except Exception as e:
        print(f"\nLiteRT conversion failed: {e}")
        traceback.print_exc()
        return

    print("\nAttaching metadata...")
    try:
        attach_metadata(tflite_path, labels_path)
    except Exception as e:
        print(f"Metadata step failed: {e}")
        traceback.print_exc()


NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

SCALAR_MEAN = [sum(NORM_MEAN) / len(NORM_MEAN)]
SCALAR_STD = [sum(NORM_STD) / len(NORM_STD)]


def attach_metadata(tflite_path: Path, labels_path: Path):
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "EfficientNet Lite4 Classifier"
    model_meta.version = "1.0"

    input_meta = _metadata_fb.TensorMetadataT()
    input_meta.name = "image"
    input_meta.description = "Input RGB image (H x W x 3, float32)"

    norm_unit = _metadata_fb.ProcessUnitT()
    norm_unit.optionsType = _metadata_fb.ProcessUnitOptions.NormalizationOptions
    norm_opts = _metadata_fb.NormalizationOptionsT()
    norm_opts.mean = SCALAR_MEAN
    norm_opts.std = SCALAR_STD
    norm_unit.options = norm_opts
    input_meta.processUnits = [norm_unit]

    output_meta = _metadata_fb.TensorMetadataT()
    output_meta.name = "probability"
    output_meta.description = "Per-class softmax scores"

    label_file = _metadata_fb.AssociatedFileT()
    label_file.name = labels_path.name
    label_file.description = "Class labels"
    label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
    output_meta.associatedFiles = [label_file]

    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = [output_meta]
    model_meta.subgraphMetadata = [subgraph]

    builder = flatbuffers.Builder(0)
    builder.Finish(
        model_meta.Pack(builder), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER
    )
    metadata_buf = builder.Output()

    populator = _metadata.MetadataPopulator.with_model_file(str(tflite_path))
    populator.load_metadata_buffer(metadata_buf)
    populator.load_associated_files([str(labels_path)])
    populator.populate()

    print(f"Metadata attached successfully to: {tflite_path}")


if __name__ == "__main__":
    main()
