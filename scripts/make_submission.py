import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

from src.classifiers import build_classifier
from cpr_data import data_file, check_submission


def predict(model_name, features, device):

    checkpoint = torch.load(
        ROOT / "checkpoints" / f"{model_name}.pt",
        map_location=device
    )

    model = build_classifier(model_name).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    loader = DataLoader(
        features,
        batch_size=128,
        shuffle=False
    )

    probabilities = []

    with torch.inference_mode():

        for x in loader:

            x = x.to(device)

            logits = model(x)

            probs = torch.sigmoid(logits)

            probabilities.append(probs.cpu())

    probabilities = torch.cat(
        probabilities, dim=0
    ).numpy()

    print(
        f"{model_name}:",
        probabilities.shape
    )

    return probabilities


def save_submission(
    sample,
    probabilities,
    threshold,
    name
):

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    submission = sample.copy()

    submission.iloc[:, 1:] = predictions

    check_submission(submission, sample)

    output_path = ROOT / f"submission_{name}.csv"

    submission.to_csv(output_path, index=False)

    counts = predictions.sum(axis=1)

    print("\nSaved:", output_path)
    print("Threshold:", threshold)
    print("Rows:", len(submission))
    print("Mean errors per video:", counts.mean())
    print("Zero-error predictions:", (counts == 0).sum())
    print("Single-error predictions:", (counts == 1).sum())
    print("Multi-error predictions:", (counts > 1).sum())

    print("\nPredicted positives per class:")

    for col, count in zip(
        sample.columns[1:],
        predictions.sum(axis=0)
    ):
        print(col, count)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    sample = pd.read_csv(
        data_file("sample_submission.csv"),
        dtype={"video_id": str}
    )

    train_df = pd.read_csv(
        ROOT / "data/train_split.csv",
        nrows=1
    )

    train_columns = [
        col for col in train_df.columns
        if col.startswith("e") and col[1:3].isdigit()
    ]

    assert train_columns == sample.columns[1:].tolist(), \
        "Training and submission label order mismatch!"

    data = torch.load(
        ROOT / "features/test_features.pt",
        map_location="cpu"
    )

    features = data["features"].float()

    assert data["video_ids"] == sample["video_id"].tolist()
    assert features.shape == (len(sample), 32, 512)

    cnn_probs = predict(
        "temporal_cnn", features, device
    )

    statistical_probs = predict(
        "statistical", features, device
    )

    mlp_probs = predict(
        "mlp", features, device
    )

    ensemble_probs = (
        0.5 * cnn_probs
        + 0.3 * statistical_probs
        + 0.2 * mlp_probs
    )

    np.savez_compressed(
        ROOT / "results/test_probabilities.npz",
        temporal_cnn=cnn_probs,
        statistical=statistical_probs,
        mlp=mlp_probs,
        ensemble=ensemble_probs,
        video_ids=np.array(data["video_ids"], dtype=str)
    )

    save_submission(
        sample, cnn_probs, 0.5, "temporal_cnn"
    )

    save_submission(
        sample, ensemble_probs, 0.5, "ensemble"
    )

    save_submission(
        sample, ensemble_probs, 0.35, "ensemble_recall"
    )

    print("\nALL SUBMISSIONS COMPLETE!")


if __name__ == "__main__":
    main()
