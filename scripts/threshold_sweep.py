from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

data = np.load(
    ROOT / "results/test_probabilities.npz"
)

sample = pd.read_csv(
    ROOT / "submission_baseline.csv",
    dtype={"video_id": str}
)

assert data["video_ids"].tolist() == sample["video_id"].tolist()

output_dir = ROOT / "submissions"
output_dir.mkdir(exist_ok=True)

for model in ["temporal_cnn", "ensemble"]:

    probs = data[model]

    assert probs.shape == (len(sample), 13)

    print(f"\n===== {model.upper()} =====")

    for threshold in [0.50, 0.35, 0.25, 0.15, 0.10, 0.05]:

        preds = (probs >= threshold).astype(int)

        submission = sample.copy()
        submission.iloc[:, 1:] = preds

        counts = preds.sum(axis=1)

        filename = (
            f"{model}_threshold_{threshold:.2f}.csv"
        )

        submission.to_csv(
            output_dir / filename,
            index=False
        )

        print(
            f"Threshold: {threshold:.2f} | "
            f"Mean errors: {counts.mean():.3f} | "
            f"Zero: {(counts == 0).sum()} | "
            f"Single: {(counts == 1).sum()} | "
            f"Multi: {(counts > 1).sum()}"
        )

print("\nTHRESHOLD SWEEP COMPLETE!")
