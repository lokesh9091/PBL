import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model (same as training)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*6*6,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,7)
        )

    def forward(self,x):
        return self.fc(self.conv(x))

# Load model
model = CNN().to(device)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.eval()

print("✅ Model loaded!")

# Image transform (same as test)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load image
img_path = "fer/test_1.jpg"   
image = Image.open(img_path)

image = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

classes = ['angry','disgust','fear','happy','neutral','sad','surprise']

print("🎯 Predicted Emotion:", classes[predicted.item()])
