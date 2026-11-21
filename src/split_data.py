
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
STARTER_DIR = PROJECT_ROOT / "starter"

sys.path.insert(0, str(STARTER_DIR))
from cpr_data import data_file


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer the local CSV, otherwise use the shared SCC dataset.
    local_csv = DATA_DIR / "train.csv"

    if local_csv.exists():
        csv_path = local_csv
    else:
        csv_path = data_file("train.csv")

    print(f"Reading training data from: {csv_path}")
    train = pd.read_csv(csv_path)

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    train_idx, val_idx = next(
        splitter.split(
            train,
            y=train["class_id"],
            groups=train["take"]
        )
    )

    train_split = train.iloc[train_idx].reset_index(drop=True)
    val_split = train.iloc[val_idx].reset_index(drop=True)

    assert set(train_split["take"]).isdisjoint(
        set(val_split["take"])
    ), "Data leakage: overlapping takes!"

    train_split.to_csv(DATA_DIR / "train_split.csv", index=False)
    val_split.to_csv(DATA_DIR / "val_split.csv", index=False)

    print(f"Training videos: {len(train_split)}")
    print(f"Validation videos: {len(val_split)}")
    print(f"Training takes: {train_split['take'].nunique()}")
    print(f"Validation takes: {val_split['take'].nunique()}")
    print("No take overlap detected.")


if __name__ == "__main__":
    main()
