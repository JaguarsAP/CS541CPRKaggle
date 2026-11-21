
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

def create_split():
    train = pd.read_csv(DATA_DIR / "train.csv")

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    train_idx, val_idx = next(
        splitter.split(
            X=train,
            y=train["class_id"],
            groups=train["take"]
        )
    )

    train_df = train.iloc[train_idx].copy()
    val_df = train.iloc[val_idx].copy()

    # Ensure no performance appears in both sets
    assert set(train_df["take"]).isdisjoint(
        set(val_df["take"])
    )

    print("Training videos:", len(train_df))
    print("Validation videos:", len(val_df))

    print("\nTraining class distribution:")
    print(train_df["class_name"].value_counts())

    print("\nValidation class distribution:")
    print(val_df["class_name"].value_counts())

    # Save splits for reproducibility
    train_df.to_csv(DATA_DIR / "train_split.csv", index=False)
    val_df.to_csv(DATA_DIR / "val_split.csv", index=False)

    print("\nSplits saved successfully!")

if __name__ == "__main__":
    create_split()
