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


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.Resize(456),
            transforms.CenterCrop(380),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(root="dataset", transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    model = timm.create_model(
        "tf_efficientnet_lite4", pretrained=True, num_classes=len(train_dataset.classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    scaler = torch.amp.GradScaler(enabled=(device.type == "mps"))

    os.makedirs("checkpoints", exist_ok=True)

    num_epochs = 10

    print("Starting fine-tuning from pretrained weights...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        checkpoint_name = f"checkpoints/checkpoint_epoch_{epoch+1}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "classes": train_dataset.classes,
            },
            checkpoint_name,
        )
        print(f"Saved checkpoint: {checkpoint_name}")

    print("\nTraining completed successfully.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_filename = f"efficientnet_lite4_final_finetuned_{timestamp}.pth"

    os.makedirs("tflite", exist_ok=True)
    checkpoint_path = f"tflite/{model_filename}"

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
