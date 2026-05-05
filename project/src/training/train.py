# src/training/train.py

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.nyu_dataset import NYUDepthDataset, read_existing_pairs
from src.models import DEFAULT_CHECKPOINTS, MODEL_NAMES, build_depth_model
from src.training.losses import masked_l1_loss
from src.training.metrics import depth_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train a monocular depth model.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="baseline")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--train-limit", type=int, default=10000)
    parser.add_argument("--test-limit", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
        help="Train MobileNetV3 from random weights instead of ImageNet weights.",
    )
    return parser.parse_args()


def limit_pairs(pairs, limit):
    if limit is None or limit < 0:
        return pairs

    return pairs[:limit]


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or Path(DEFAULT_CHECKPOINTS[args.model])

    nyu_root = Path("nyu_data")
    train_csv = nyu_root / "data" / "nyu2_train.csv"
    test_csv = nyu_root / "data" / "nyu2_test.csv"

    train_pairs = read_existing_pairs(train_csv, nyu_root)
    test_pairs = read_existing_pairs(test_csv, nyu_root)
    train_pairs = limit_pairs(train_pairs, args.train_limit)
    test_pairs = limit_pairs(test_pairs, args.test_limit)

    print("Model:", args.model)
    print("Train pairs:", len(train_pairs))
    print("Test pairs:", len(test_pairs))
    print("Device:", device)
    print("Checkpoint:", checkpoint_path)
    if args.model == "mobilenet_v3":
        print("Pretrained backbone:", not args.no_pretrained_backbone)

    train_dataset = NYUDepthDataset(train_pairs, image_size=(args.height, args.width))
    test_dataset = NYUDepthDataset(test_pairs, image_size=(args.height, args.width))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_depth_model(
        args.model,
        max_depth=10.0,
        pretrained_backbone=not args.no_pretrained_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            leave=True,
        )

        for batch in train_bar:
            image = batch["image"].to(device)
            depth = batch["depth"].to(device)
            mask = batch["mask"].to(device)

            pred = model(image)
            loss = masked_l1_loss(pred, depth, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        model.eval()
        metric_sums = {"abs_rel": 0.0, "rmse": 0.0, "delta1": 0.0}

        with torch.no_grad():
            eval_bar = tqdm(
                test_loader,
                desc="Evaluating",
                leave=False,
            )

            for batch in eval_bar:
                image = batch["image"].to(device)
                depth = batch["depth"].to(device)
                mask = batch["mask"].to(device)

                pred = model(image)
                metrics = depth_metrics(pred, depth, mask)

                for key in metric_sums:
                    metric_sums[key] += metrics[key]

        metric_avgs = {
            key: value / len(test_loader)
            for key, value in metric_sums.items()
        }

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss={avg_loss:.4f} | "
            f"AbsRel={metric_avgs['abs_rel']:.4f} | "
            f"RMSE={metric_avgs['rmse']:.4f} | "
            f"delta1={metric_avgs['delta1']:.4f}"
        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    train()
