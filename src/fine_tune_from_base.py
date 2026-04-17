import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
import torch.multiprocessing as mp
import datetime
from PIL import Image, UnidentifiedImageError


def is_valid_image_file(filepath: str) -> bool:
    try:
        with Image.open(filepath) as img:
            img.load()
            print(f"Valid Image: {filepath}")
        return True
    except (UnidentifiedImageError, OSError, IOError):
        print(f"Skipping invalid/corrupted image: {filepath}")
        return False
    except Exception as e:
        print(f"Unexpected error with file {filepath}: {type(e).__name__}")
        return False


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} - {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.Resize(456),
            transforms.CenterCrop(380),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(
        root="dataset",
        transform=transform,
        is_valid_file=is_valid_image_file,
    )

    print(
        f"Dataset successfully loaded: {len(train_dataset)} valid images, "
        f"{len(train_dataset.classes)} classes."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=6,
        pin_memory=True if device.type in ["cuda", "mps"] else False,
        persistent_workers=False,
    )

    model = timm.create_model(
        "tf_efficientnet_lite4", pretrained=True, num_classes=len(train_dataset.classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
    )

    use_amp = device.type in ["cuda", "mps"]
    scaler = torch.amp.GradScaler(enabled=use_amp)

    os.makedirs("checkpoints", exist_ok=True)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        checkpoint_name = f"checkpoints/checkpoint_epoch_{epoch+1}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "classes": train_dataset.classes,
            },
            checkpoint_name,
        )
        print(f"Saved checkpoint: {checkpoint_name}")

    print("Training completed.")

    model_filename = f"efficientnet_lite4_final_finetuned.pth"

    os.makedirs("tflite", exist_ok=True)
    checkpoint_path = f"tflite/{model_filename}"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": train_dataset.classes,
            "num_classes": len(train_dataset.classes),
            "timestamp": timestamp,
        },
        checkpoint_path,
    )

    print(f"PyTorch model saved to: {checkpoint_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    main()
