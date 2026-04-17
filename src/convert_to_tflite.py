import os
import torch
import timm
import litert_torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import traceback
from pathlib import Path

script_dir = Path(__file__).parent.resolve()


class NormalizedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = x.float() / 255.0
        x = (x - torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)) / torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 1, 1, 3)
        x = x.permute(0, 3, 1, 2)
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

    checkpoint_path = script_dir / "tflite/efficientnet_lite4_final_finetuned.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)
    print(f"Loading model with {num_classes} classes from checkpoint.")

    base_model = timm.create_model(
        "tf_efficientnet_lite4",
        pretrained=False,
        num_classes=num_classes,
    )

    base_model.load_state_dict(checkpoint["model_state_dict"])

    print("Finetuned model loaded successfully.")

    labels_path = Path(script_dir / "tflite/labels.txt")
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    with open(labels_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")

    print(f"labels.txt created successfully at: {labels_path}")
    print(f"Total labels written: {len(class_names)}")

    model = LogitNormalizer(NormalizedModel(base_model))
    model.eval()

    print("\nConverting model to TensorFlow Lite format...")

    try:
        sample_input = torch.randn(1, 380, 380, 3)

        edge_model = litert_torch.convert(model, (sample_input,))

        os.makedirs("tflite", exist_ok=True)
        tflite_path = script_dir / "tflite/model.tflite"

        edge_model.export(tflite_path)

        print(f"Conversion successful!")
        print(f"TensorFlow Lite model saved to: {tflite_path}")

    except Exception as e:
        print(f"\nLiteRT conversion failed: {e}")
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
