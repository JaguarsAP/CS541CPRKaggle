"""CS541 CPR-Coach challenge: find and read the data, wherever you run.

    import pandas as pd
    from cpr_data import data_file, video_file, load_video

    train = pd.read_csv(data_file("train.csv"))
    clip = load_video(video_file(train.video_id[0], "train"))   # (T, H, W, 3) uint8, RGB

Where the files come from
-------------------------
* On the SCC: the shared course copy in /projectnb/cs541/materials/cpr_challenge is read in
  place. Nothing is downloaded, and nothing is written to your home directory.
* Your own copy (laptop or Colab): download the zips in /projectnb/cs541/materials/cpr_challenge/
  downloads, unzip them all into one folder, and set the CPR_DATA environment variable to it,
  e.g. os.environ["CPR_DATA"] = "/content/cpr_challenge" before importing this module.
* Otherwise (Hugging Face): each file is downloaded from the course's Hugging Face
  dataset the first time you ask for it, then read from the local cache. No login is needed once
  staff announce that the Hugging Face copy is open.
  - To fetch everything at once instead (faster than one file at a time):
    download_all("train"), then download_all("test") when you need to predict.
  - The report subset is one zip there: subset_dir() downloads and unzips it for you.
  - Colab's disk is wiped when a session ends; fetching the training videos again (~0.6 GB) is
    quick. Keep the data on Colab's own disk, not Google Drive, which is slow with many files.

Don't stream from Hugging Face inside a training loop. Each file is fetched once; every later
call reads the local copy without touching the network.
"""
import os

import cv2
import numpy as np

HF_REPO = "dghadiya/cs541-cpr-challenge"                  # the course's Hugging Face dataset
SCC_ROOT = "/projectnb/cs541/materials/cpr_challenge"     # the shared course copy on the SCC
# Read from this folder when it exists: your own copy if CPR_DATA is set, else the SCC copy.
LOCAL_ROOT = os.path.expanduser(os.environ.get("CPR_DATA", SCC_ROOT))


def data_file(relpath):
    """Return a local path to a challenge file, e.g. "train.csv" or "videos/test/1a2b3c4d.mp4"."""
    local = os.path.join(LOCAL_ROOT, relpath)
    if os.path.exists(local):
        return local
    if LOCAL_ROOT != SCC_ROOT:   # you pointed us at your own copy: say what's missing from it
        z = ("cpr_test.zip" if relpath.startswith("videos/test") else
             "flow_pose_subset.zip" if relpath.startswith("flow_pose_subset") else "cpr_train.zip")
        raise FileNotFoundError("%s is not in your copy at %s. Unzip %s into that folder (see README.md)."
                                % (relpath, LOCAL_ROOT, z))
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    cached = try_to_load_from_cache(HF_REPO, relpath, repo_type="dataset")   # no network
    if isinstance(cached, str):
        return cached
    return hf_hub_download(HF_REPO, relpath, repo_type="dataset")


def video_file(video_id, split):
    """Local path to one video. `split` is "train" or "test"."""
    if split not in ("train", "test"):
        raise ValueError('split must be "train" or "test", got %r' % (split,))
    return data_file("videos/%s/%s.mp4" % (split, video_id))


def download_all(split=None):
    """Download the CSVs plus every video of `split` ("train", "test", or None for both).

    Returns the folder that holds them. On the SCC this downloads nothing.
    The 100-video report subset is not included; use subset_dir() for it.
    """
    if os.path.isdir(LOCAL_ROOT):
        return LOCAL_ROOT
    if split not in (None, "train", "test"):
        raise ValueError('split must be "train", "test" or None, got %r' % (split,))
    from huggingface_hub import snapshot_download
    patterns = ["*.csv", "videos/%s/*" % split if split else "videos/*"]
    return snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=patterns)


def subset_dir(dest="."):
    """Folder holding the 100-video report subset: frames, reference flow, keypoints.pkl, README.md.

    On the SCC the course copy is read in place. Anywhere else, flow_pose_subset.zip (~5 GB) is
    downloaded once and unzipped into `dest`/flow_pose_subset (another ~5 GB); later calls reuse it.
    """
    local = os.path.join(LOCAL_ROOT, "flow_pose_subset")
    if os.path.isdir(local):
        return local
    out = os.path.join(dest, "flow_pose_subset")
    if os.path.isdir(out):
        return out
    import shutil
    import zipfile
    part = os.path.join(dest, ".flow_pose_subset.part")      # unzip here, then rename: an
    shutil.rmtree(part, ignore_errors=True)                  # interrupted unzip never looks done
    with zipfile.ZipFile(data_file("flow_pose_subset.zip")) as z:
        z.extractall(part)
    os.replace(os.path.join(part, "flow_pose_subset"), out)
    os.rmdir(part)
    return out


def load_video(path, stride=4, max_frames=64, size=None):
    """Return frames as a uint8 array of shape (T, H, W, 3), RGB.

    stride: keep every `stride`-th frame. Videos are 25 fps and one chest compression takes
    roughly 12-15 frames, so a stride above ~6 can no longer "see" the pumping motion
    (it aliases away), which matters for errors about pressing rate and depth.
    size: optional (width, height) to resize every frame to, e.g. (224, 224).
    """
    cap = cv2.VideoCapture(path)
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame if size is None else cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
        i += 1
    cap.release()
    if not frames:
        raise IOError("could not read any frames from %s" % path)
    return np.stack(frames)


def check_submission(sub, sample):
    """Catch the format mistakes Kaggle would reject, before you upload."""
    assert list(sub.columns) == list(sample.columns), "columns must match sample_submission.csv exactly"
    assert len(sub) == len(sample) and set(sub.video_id) == set(sample.video_id), "need one row per test video"
    assert not sub.video_id.duplicated().any(), "duplicate video_id rows"
    labels = sub.columns[1:]
    assert sub[labels].isin([0, 1]).all().all(), "every label cell must be 0 or 1"
