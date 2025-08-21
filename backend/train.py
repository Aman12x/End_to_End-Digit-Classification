# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# ----------------------------
# 1. Define CNN Model
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),  # 28x28 -> 26x26
            nn.ReLU(),
            nn.MaxPool2d(2),  # 26x26 -> 13x13
            nn.Conv2d(16, 32, 3, 1),  # 13x13 -> 11x11
            nn.ReLU(),
            nn.MaxPool2d(2),  # 11x11 -> 5x5
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


# ----------------------------
# 2. Training Function
# ----------------------------
def train_model():
    transform = transforms.Compose([transforms.ToTensor()])

    # MNIST Dataset
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🚀 Training started...")
    for epoch in range(10):  # train for 10 epochs
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/10 - Loss: {running_loss/len(trainloader):.4f}")

    # Save trained CNN model
    # Save only weights
    torch.save(model.state_dict(), "mnist_model.pth")

    print("✅ Model trained and saved as mnist_model.pkl")


if __name__ == "__main__":
    train_model()
