#!/usr/bin/env python3
"""CS541 CPR-Coach challenge: minimal starter.

Shows how to (1) read the label files, (2) load a video as a frame array, and
(3) write a correctly formatted submission. The "model" here is the trivial
"flag every error on every video" baseline (macro-F1 ~= 0.25). Beat it.

Data access lives in cpr_data.py (next to this file): it reads the shared copy on the SCC or
your own unzipped copy, and otherwise downloads from the course's Hugging Face dataset.
See its docstring.

On the SCC:
    module load miniconda academic-ml/fall-2026
    conda activate fall-2026-pyt
    python starter.py --out ~/my_submission.csv

With your own copy (the zips from the SCC's downloads/ folder, unzipped into one folder):
    python starter.py --data ~/cpr_challenge --out my_submission.csv

On Colab or a laptop without a copy (downloads from Hugging Face, once staff announce it is open;
no login needed):
    python starter.py --out my_submission.csv
"""
import argparse

import pandas as pd

import cpr_data
from cpr_data import check_submission, data_file, load_video, video_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--data", default=None, help="your own unzipped copy (default: $CPR_DATA, else the SCC copy, else HF)")
    a = ap.parse_args()
    if a.data:
        cpr_data.LOCAL_ROOT = a.data

    train = pd.read_csv(data_file("train.csv"))
    sample = pd.read_csv(data_file("sample_submission.csv"))
    labels = list(sample.columns[1:])                     # the 13 error columns
    print("train videos: %d   test videos: %d   labels: %d" % (len(train), len(sample), len(labels)))
    print("training labels per video (0 = 'Correct'):", train[labels].sum(axis=1).value_counts().to_dict())

    first = train.iloc[0]
    clip = load_video(video_file(first.video_id, "train"))
    print("one training clip -> frames array", clip.shape, clip.dtype, "| class:", first.class_name)

    sub = sample.copy()
    sub[labels] = 1                                       # trivial baseline: flag every error
    check_submission(sub, sample)
    sub.to_csv(a.out, index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
