
import pandas as pd
from pathlib import Path

# Locate the project's data directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Load the CSV files
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test_ids.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

# The sample submission defines the 13 target labels
error_cols = list(sample.columns[1:])

print("===== DATASET OVERVIEW =====")
print(f"Training videos: {len(train)}")
print(f"Test videos: {len(test)}")
print(f"Number of error classes: {len(error_cols)}")

print("\n===== TRAINING DATA =====")
print(train.head())

print("\n===== ERROR DISTRIBUTION =====")
error_counts = train[error_cols].sum().sort_values(ascending=False)

for error, count in error_counts.items():
    print(f"{error}: {count}")

print("\n===== CORRECT VS INCORRECT =====")
errors_per_video = train[error_cols].sum(axis=1)
print(errors_per_video.value_counts().sort_index())

print("\n===== CAMERA DISTRIBUTION =====")
print(train["view"].value_counts())

print("\n===== TAKE INFORMATION =====")
print(f"Unique takes: {train['take'].nunique()}")
print(f"Average videos per take: {len(train) / train['take'].nunique():.2f}")

print("\n===== MISSING VALUES =====")
print(train.isnull().sum())
