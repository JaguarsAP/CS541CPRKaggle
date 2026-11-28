import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifiers import build_classifier


def train(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Model:", args.model)
    print("Device:", device)

    train_data = torch.load(
        ROOT / "features/train_features.pt",
        map_location="cpu"
    )

    val_data = torch.load(
        ROOT / "features/val_features.pt",
        map_location="cpu"
    )

    train_dataset = TensorDataset(
        train_data["features"].float(),
        train_data["labels"].float()
    )

    val_dataset = TensorDataset(
        val_data["features"].float(),
        val_data["labels"].float()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = build_classifier(args.model).to(device)

    # Balance positive and negative examples per error.
    positives = train_data["labels"].sum(dim=0)
    negatives = len(train_dataset) - positives

    pos_weight = negatives / positives.clamp(min=1)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):

        # Training
        model.train()
        train_loss = 0.0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()

        val_loss = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item() * x.size(0)

                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu())
                all_labels.append(y.cpu())

        val_loss /= len(val_dataset)

        probs = torch.cat(all_probs).numpy()
        labels = torch.cat(all_labels).numpy()

        predictions = (probs >= args.threshold).astype(int)

        macro_f1 = f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "macro_f1": float(macro_f1)
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Macro-F1: {macro_f1:.4f}",
            flush=True
        )

        if macro_f1 > best_f1:

            best_f1 = macro_f1
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "macro_f1": float(macro_f1),
                "threshold": args.threshold
            }, checkpoint_dir / f"{args.model}.pt")

            np.savez(
                results_dir / f"{args.model}_val_predictions.npz",
                probabilities=probs,
                labels=labels
            )

            print("Saved best checkpoint!", flush=True)

        else:

            patience_counter += 1

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    results = {
        "model": args.model,
        "best_epoch": best_epoch,
        "best_macro_f1": float(best_f1),
        "config": vars(args),
        "history": history
    }

    with open(
        results_dir / f"{args.model}.json", "w"
    ) as f:
        json.dump(results, f, indent=4)

    print("\nTraining complete!")
    print("Best epoch:", best_epoch)
    print("Best validation macro-F1:", best_f1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "linear",
            "mlp",
            "statistical",
            "lstm",
            "temporal_cnn"
        ],
        required=True
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    train(parser.parse_args())
