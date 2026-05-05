import cv2

from config import (
    MODEL_NAME,
    PERSON_CLASS_ID,
    CONF_THRESHOLD,
    CAMERA_ID,
    CLOSE_HEIGHT_RATIO,
    MAX_LOST_FRAMES,
)

from detector import PersonDetector
from person_manager import (
    add_person_geometry,
    mark_close_persons,
    select_closest_person,
)
from lock_manager import LockManager
from utils import draw_person, draw_status


def main():
    detector = PersonDetector(
        model_name=MODEL_NAME,
        person_class_id=PERSON_CLASS_ID,
        conf_threshold=CONF_THRESHOLD,
    )

    lock_manager = LockManager(max_lost_frames=MAX_LOST_FRAMES)

    cap = cv2.VideoCapture(CAMERA_ID)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read webcam frame")
            break

        persons = detector.detect_and_track(frame)

        persons = [
            add_person_geometry(person, frame.shape)
            for person in persons
        ]

        persons = mark_close_persons(
            persons,
            close_height_ratio=CLOSE_HEIGHT_RATIO,
        )

        candidate = select_closest_person(persons)

        locked_track_id = lock_manager.update(
            persons,
            candidate_person=candidate,
        )

        for person in persons:
            locked = lock_manager.is_locked(person)
            draw_person(frame, person, locked=locked)

        draw_status(frame, locked_track_id)

        cv2.imshow("Person lock system", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()