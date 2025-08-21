from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict


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
# 2. Load trained CNN model
# ----------------------------
def load_model(path="mnist_model.pth"):
    model = SimpleCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


model = load_model()

# ----------------------------
# 3. FastAPI App Setup
# ----------------------------
app = FastAPI(title="MNIST CNN Digit Classifier API")


class DigitInput(BaseModel):
    pixels: List[float]  # flattened 28x28 grayscale image


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    probabilities: Dict[str, float]  # keep keys as strings for frontend


# ----------------------------
# 4. Prediction Endpoint
# ----------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(data: DigitInput):
    # Convert input list -> tensor shape [1, 1, 28, 28]
    x = torch.tensor(np.array(data.pixels).reshape(1, 1, 28, 28), dtype=torch.float32)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).numpy()[0]
        pred = int(np.argmax(probs))
        confidence = float(np.max(probs))

    # Return dict with string keys (frontend expects this)
    prob_dict = {str(i): float(probs[i]) for i in range(10)}

    return PredictionResponse(
        prediction=pred, confidence=confidence, probabilities=prob_dict
    )
