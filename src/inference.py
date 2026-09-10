import json

import numpy as np
import torch

from src.model import GestureCNN


def main():
    # Load class names
    with open("models/class_names.json", "r") as file:
        class_names = json.load(file)

    # Create the model and initialise LazyLinear
    model = GestureCNN(num_classes=len(class_names))
    model(torch.randn(1, 16, 50))

    # Load trained weights
    model.load_state_dict(
        torch.load(
            "models/gesture_cnn.pth",
            map_location="cpu",
        )
    )

    model.eval()

    # Load one test window
    X = np.load("data/simulated/X.npy")
    test_window = torch.tensor(X[0], dtype=torch.float32)

    # Convert from 50 × 16 to 1 × 16 × 50
    test_window = test_window.permute(1, 0).unsqueeze(0)

    with torch.no_grad():
        output = model(test_window)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    result = {
        "gesture": class_names[predicted_class],
        "confidence": round(confidence, 4),
        "status": "VALID" if confidence >= 0.70 else "LOW_CONFIDENCE",
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()