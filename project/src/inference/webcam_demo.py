import argparse
import time

import cv2
import numpy as np
import torch

from src.models import DEFAULT_CHECKPOINTS, MODEL_NAMES, build_depth_model


def preprocess_frame(frame, height=240, width=320):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (width, height))

    image = frame_rgb.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    return image, frame_rgb


def colorize_depth(depth, max_depth=10.0, dynamic=False):
    if dynamic:
        depth_min = np.percentile(depth, 2)
        depth_max = np.percentile(depth, 98)
        depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-6)
    else:
        depth_norm = depth / max_depth

    depth_norm = np.clip(depth_norm, 0, 1)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
    return depth_color


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, default="baseline")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument(
        "--dynamic-vis",
        action="store_true",
        help="Normalize depth colors per frame for easier visual inspection.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or DEFAULT_CHECKPOINTS[args.model]

    model = build_depth_model(args.model, max_depth=10.0).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Model:", args.model)
    print("Device:", device)
    print("Checkpoint:", checkpoint_path)
    print("Press q to quit.")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            print("frame shape:", frame.shape)

            if not ret:
                break

            image, frame_rgb = preprocess_frame(frame, args.height, args.width)
            image = image.to(device)

            start = time.perf_counter()

            pred = model(image)

            if device.type == "cuda":
                torch.cuda.synchronize()

            inference_time = time.perf_counter() - start
            fps = 1.0 / inference_time

            depth = pred.squeeze().cpu().numpy()
            depth_color = colorize_depth(depth, dynamic=args.dynamic_vis)

            rgb_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            combined = np.hstack([rgb_bgr, depth_color])

            text = f"Inference FPS: {fps:.1f} | min={depth.min():.2f}m max={depth.max():.2f}m"
            cv2.putText(
                combined,
                text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("RGB | Predicted Depth", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
