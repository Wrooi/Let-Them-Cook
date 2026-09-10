import json

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.model import GestureCNN


def main():
    with open("models/class_names.json", "r") as file:
        class_names = json.load(file)

    X = np.load("data/simulated/X.npy")
    y = np.load("data/simulated/y.npy")

    # Convert from:
    # (number of windows, 50 samples, 16 features)
    # to:
    # (number of windows, 16 features, 50 samples)
    X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    _, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
    )

    model = GestureCNN(num_classes=len(class_names))

    # Initialise LazyLinear before loading the trained weights
    with torch.no_grad():
        model(torch.randn(1, 16, 50))

    model.load_state_dict(
        torch.load(
            "models/gesture_cnn.pth",
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)

            true_labels.extend(labels.numpy())
            predicted_labels.extend(predictions.numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)

    print(f"Test accuracy: {accuracy:.2%}")

    print("\nClassification report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=range(len(class_names)),
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            true_labels,
            predicted_labels,
            labels=range(len(class_names)),
        )
    )


if __name__ == "__main__":
    main()