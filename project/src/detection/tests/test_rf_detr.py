import cv2
import supervision as sv
from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES

PERSON_CLASS_ID = 1

model = RFDETRNano()

# other variants
#RFDETRSmall()   # rf-detr-small
#RFDETRMedium()  # rf-detr-medium
#RFDETRBase()    # rf-detr-base
#RFDETRLarge()   # rf-detr-large
#RFDETRXLarge()  # rf-detr-xlarge
#RFDETR2XLarge() # rf-detr-2xlarge

model.optimize_for_inference()

cap = cv2.VideoCapture(0)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    ok, frame_bgr = cap.read()
    if not ok:
        print("Could not read webcam frame")
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    detections = model.predict(frame_rgb, threshold=0.15)

    # Filter detections for person class
    person_detections = detections[detections.class_id == PERSON_CLASS_ID]

    labels = [f"person {conf:.2f}" for conf in person_detections.confidence]
    
    annotated = frame_bgr.copy()
    annotated = box_annotator.annotate(annotated, person_detections)
    annotated = label_annotator.annotate(annotated, person_detections, labels)

    cv2.imshow("RF-DETR person only", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()