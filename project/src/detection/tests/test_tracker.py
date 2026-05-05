import cv2
import time
from collections import defaultdict
from ultralytics import YOLO

PERSON_CLASS_ID = 0

model = YOLO("yolo26s.pt")  # or yolo26n.pt

cap = cv2.VideoCapture(0)

track_history = defaultdict(list)
last_seen = {}
frame_idx = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_idx += 1

    results = model.track(
        frame,
        conf=0.25,
        classes=[PERSON_CLASS_ID],
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    active_ids = []

    if results[0].boxes is not None:
        for box in results[0].boxes:
            if box.id is None:
                continue

            track_id = int(box.id[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            active_ids.append(track_id)
            last_seen[track_id] = frame_idx
            track_history[track_id].append((cx, cy))

            # keep trail short
            # keep the last 30 center points of each tracked person
            track_history[track_id] = track_history[track_id][-30:]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id} conf {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # draw movement trail
            points = track_history[track_id]
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], (255, 0, 0), 2)

    print(f"Frame {frame_idx} active IDs: {active_ids}")

    cv2.putText(
        frame,
        f"Active IDs: {active_ids}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    cv2.imshow("YOLO ByteTrack ID persistence test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()