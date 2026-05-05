import cv2
import supervision as sv
from ultralytics import YOLO

PERSON_CLASS_ID = 0

model = YOLO("yolo26n.pt")

cap = cv2.VideoCapture(0)

box_annotator = sv.BoxAnnotator()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(frame, conf=0.25, verbose=False)

    detections = sv.Detections.from_ultralytics(results[0])

    # filter person
    person_detections = detections[detections.class_id == PERSON_CLASS_ID]

    annotated = frame.copy()

    # assign FAKE IDs (just index per frame)
    for i, (box, conf) in enumerate(
        zip(person_detections.xyxy, person_detections.confidence)
    ):
        x1, y1, x2, y2 = map(int, box)

        label = f"fakeID {i} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )

    cv2.imshow("YOLO WITHOUT tracking (fake IDs)", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()