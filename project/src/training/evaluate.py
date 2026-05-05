import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.nyu_dataset import NYUDepthDataset, read_existing_pairs
from src.models import DEFAULT_CHECKPOINTS, MODEL_NAMES, build_depth_model
from src.training.metrics import depth_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained depth model.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="baseline")
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="Use pairs from nyu2_train.csv or nyu2_test.csv.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First pair index to evaluate, inclusive.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last pair index to evaluate, exclusive. Uses all remaining pairs if omitted.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the saved model checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "prediction_samples",
        help="Directory where individual prediction images are saved.",
    )
    parser.add_argument(
        "--num-visuals",
        type=int,
        default=4,
        help="Number of individual prediction images to save. Use -1 to save all evaluated images.",
    )
    return parser.parse_args()


def get_pairs(split, start, end):
    nyu_root = Path("nyu_data")
    csv_name = "nyu2_train.csv" if split == "train" else "nyu2_test.csv"
    csv_path = nyu_root / "data" / csv_name

    pairs = read_existing_pairs(csv_path, nyu_root)
    return pairs[start:end], csv_path, len(pairs)


def save_prediction_images(
    model,
    dataset,
    device,
    output_dir,
    max_images=4,
    depth_vmax=10,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    if max_images < 0:
        max_images = len(dataset)
    else:
        max_images = min(max_images, len(dataset))

    with torch.no_grad():
        for i in tqdm(range(max_images), desc="Saving visuals"):
            sample = dataset[i]

            image = sample["image"].unsqueeze(0).to(device)
            depth = sample["depth"].squeeze(0).cpu().numpy()
            pred = model(image).squeeze().cpu().numpy()
            rgb = sample["image"].permute(1, 2, 0).cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(rgb)
            axes[0].set_title("RGB")
            axes[0].axis("off")

            gt_plot = axes[1].imshow(depth, cmap="plasma", vmin=0, vmax=depth_vmax)
            axes[1].set_title(
                f"Ground Truth\nmin={depth.min():.2f}m max={depth.max():.2f}m"
            )
            axes[1].axis("off")
            fig.colorbar(gt_plot, ax=axes[1], fraction=0.046, pad=0.04, label="m")

            pred_plot = axes[2].imshow(pred, cmap="plasma", vmin=0, vmax=depth_vmax)
            axes[2].set_title(
                f"Prediction\nmin={pred.min():.2f}m max={pred.max():.2f}m"
            )
            axes[2].axis("off")
            fig.colorbar(pred_plot, ax=axes[2], fraction=0.046, pad=0.04, label="m")

            fig.tight_layout()
            fig.savefig(output_dir / f"sample_{i:04d}.png", dpi=150)
            plt.close(fig)

    print("Saved visualizations to:", output_dir)


def evaluate():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or Path(DEFAULT_CHECKPOINTS[args.model])

    pairs, csv_path, total_pairs = get_pairs(args.split, args.start, args.end)

    print("Model:", args.model)
    print("Split:", args.split)
    print("CSV:", csv_path)
    print("Total existing pairs in split:", total_pairs)
    print("Evaluating range:", args.start, "to", args.end)
    print("Pairs evaluated:", len(pairs))
    print("Device:", device)
    print("Checkpoint:", checkpoint_path)

    if len(pairs) == 0:
        raise ValueError("No pairs selected. Check --split, --start, and --end.")

    dataset = NYUDepthDataset(pairs, image_size=(args.height, args.width))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = build_depth_model(args.model, max_depth=10.0).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metric_sums = {"abs_rel": 0.0, "rmse": 0.0, "delta1": 0.0}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            image = batch["image"].to(device)
            depth = batch["depth"].to(device)
            mask = batch["mask"].to(device)

            pred = model(image)
            metrics = depth_metrics(pred, depth, mask)

            for key in metric_sums:
                metric_sums[key] += metrics[key]

    metric_avgs = {
        key: value / len(loader)
        for key, value in metric_sums.items()
    }

    print("Evaluation results:")
    print(f"AbsRel: {metric_avgs['abs_rel']:.4f}")
    print(f"RMSE:   {metric_avgs['rmse']:.4f}")
    print(f"delta1: {metric_avgs['delta1']:.4f}")

    save_prediction_images(
        model,
        dataset,
        device,
        output_dir=args.output_dir,
        max_images=args.num_visuals,
    )


if __name__ == "__main__":
    evaluate()
