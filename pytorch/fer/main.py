import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter   # NEW

def main():

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48,48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1,0.1),
            scale=(0.9,1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48,48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Dataset
    train_dataset = datasets.ImageFolder("../archive/train", transform=train_transform)
    test_dataset = datasets.ImageFolder("../archive/test", transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0
    )

    print("Classes:", train_dataset.classes)

    
    # CLASS WEIGHT CALCULATION
   
    targets = train_dataset.targets
    class_counts = Counter(targets)

    print("Class counts:", class_counts)

    num_samples = sum(class_counts.values())

    weights = []
    for i in range(len(train_dataset.classes)):
        w = (num_samples / class_counts[i]) ** 0.5
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print("Class weights:", weights)
    # ==========================

    # Model
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv = nn.Sequential(
                nn.Conv2d(1,32,3,padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32,64,3,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64,128,3,padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(128,256,3,padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256*3*3,128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128,7)
            )

        def forward(self,x):
            return self.fc(self.conv(x))

    model = CNN().to(device)

    # ✅ USE WEIGHTED LOSS HERE
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    epochs = 40

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), "emotion_model.pth")
    print("Model saved")

    # ==================
    # EVALUATION
    # ==================
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"\nTest Accuracy: {100 * correct / total:.2f}%")

    # Confusion Matrix
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    print("\nConfusion Matrix:\n", cm)

    classes = train_dataset.classes

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=classes,
                yticklabels=classes,
                cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    #perclass accuracy
    cm = np.array(cm)  

    print("\nPer-Class Accuracy:\n")

    for i, class_name in enumerate(classes):
        correct = cm[i,i]
        total = cm[i].sum()
        acc = 100 * correct / total
        
        print(f"{class_name}: {acc:.2f}%")
if __name__ == "__main__":
    main()
