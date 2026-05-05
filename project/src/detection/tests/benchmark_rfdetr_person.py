import argparse
import time
import cv2
import numpy as np

from rfdetr import (
    RFDETRNano,
    RFDETRSmall,
    RFDETRMedium,
    RFDETRBase,
    RFDETRLarge,
)

PERSON_CLASS_ID = 0

MODELS = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
}


def draw_person(frame, box, conf, close_enough=False):
    x1, y1, x2, y2 = box.astype(int)
    label = f"person {conf:.2f}"

    if close_enough:
        label += " CLOSE"

    color = (0, 255, 0) if close_enough else (255, 200, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def run_model(model_name, source, threshold, close_ratio, warmup, max_frames):
    print(f"\n=== Testing RF-DETR {model_name.upper()} ===")

    model = MODELS[model_name]()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    frame_count = 0
    measured_frames = 0
    total_time = 0.0
    total_persons = 0
    total_conf = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]

        start = time.perf_counter()
        detections = model.predict(frame, threshold=threshold)
        elapsed = time.perf_counter() - start

        if frame_count >= warmup:
            measured_frames += 1
            total_time += elapsed

        persons_this_frame = 0

        for box, class_id, conf in zip(
            detections.xyxy,
            detections.class_id,
            detections.confidence,
        ):
            if int(class_id) != PERSON_CLASS_ID:
                continue

            persons_this_frame += 1
            total_conf.append(float(conf))

            box_h = box[3] - box[1]
            height_ratio = box_h / h
            close_enough = height_ratio >= close_ratio

            draw_person(frame, box, conf, close_enough)

        total_persons += persons_this_frame

        fps = 1.0 / elapsed if elapsed > 0 else 0
        cv2.putText(
            frame,
            f"{model_name.upper()} | FPS: {fps:.1f} | persons: {persons_this_frame}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.imshow("RF-DETR person detection benchmark", frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == 27:
            break

        if max_frames and frame_count >= max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()

    avg_fps = measured_frames / total_time if total_time > 0 else 0
    avg_persons = total_persons / max(1, frame_count)
    avg_conf = np.mean(total_conf) if total_conf else 0

    print(f"Frames: {frame_count}")
    print(f"Measured FPS after warmup: {avg_fps:.2f}")
    print(f"Average persons/frame: {avg_persons:.2f}")
    print(f"Average person confidence: {avg_conf:.3f}")

    return {
        "model": model_name,
        "fps": avg_fps,
        "avg_persons_per_frame": avg_persons,
        "avg_person_confidence": avg_conf,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["nano", "small", "medium"],
        choices=list(MODELS.keys()),
    )
    parser.add_argument("--source", default=0, help="Webcam index or video path")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--close-ratio",
        type=float,
        default=0.35,
        help="Person bbox height / frame height threshold for close-enough proxy",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=300)

    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    results = []

    for model_name in args.models:
        result = run_model(
            model_name=model_name,
            source=source,
            threshold=args.threshold,
            close_ratio=args.close_ratio,
            warmup=args.warmup,
            max_frames=args.max_frames,
        )
        results.append(result)

    print("\n=== SUMMARY ===")
    for r in results:
        print(
            f"{r['model']:>8} | "
            f"FPS: {r['fps']:.2f} | "
            f"persons/frame: {r['avg_persons_per_frame']:.2f} | "
            f"avg conf: {r['avg_person_confidence']:.3f}"
        )


if __name__ == "__main__":
    main()