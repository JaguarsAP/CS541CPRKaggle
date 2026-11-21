
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))

from cpr_data import data_file, check_submission


def main():
    # Use the local sample CSV when available,
    # otherwise read it from the shared SCC dataset.
    local_sample = ROOT / "data" / "sample_submission.csv"

    if local_sample.exists():
        sample_path = local_sample
    else:
        sample_path = data_file("sample_submission.csv")

    sample = pd.read_csv(sample_path)

    submission = sample.copy()

    # Predict every error for every test video.
    submission.iloc[:, 1:] = 1

    output_path = ROOT / "submission_baseline.csv"
    submission.to_csv(output_path, index=False)

    check_submission(submission, sample)

    print("Submission saved:", output_path)
    print("Rows:", len(submission))
    print("Columns:", len(submission.columns))


if __name__ == "__main__":
    main()
