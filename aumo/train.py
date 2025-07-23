import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .model import ResNetModel
from .config import CLASSIFY_IMAGE_SIZE

def train_resnet50(size=(CLASSIFY_IMAGE_SIZE, CLASSIFY_IMAGE_SIZE), batch_size=32, epochs=10):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    train_dataset = datasets.ImageFolder('datasets/train', transform=transform)
    print(train_dataset.class_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    num_classes = len(train_dataset.classes)
    model = ResNetModel(num_classes=num_classes, device='cuda:0' if torch.cuda.is_available() else 'cpu', pretrained=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(model.device), labels.to(model.device)
            optimizer.zero_grad()
            outputs = model.model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")
        if (epoch+1) % 10 == 0 and (epoch+1) != epochs:
            print(f"Epoch {epoch+1} model save")
            model.save(f"models/resnet50_middle_{size[0]}_{epoch+1}.pt")

    os.makedirs('models', exist_ok=True)
    save_path = f"models/resnet50_{size[0]}_{epochs}.pth"
    model.save(save_path)
    return save_path
