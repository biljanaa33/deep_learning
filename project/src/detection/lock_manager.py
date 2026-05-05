class LockManager:
    def __init__(self, max_lost_frames=15):
        self.locked_track_id = None
        self.lost_frames = 0 # in how many consecutive frames the locked person was not detected
        # why consecutive?? robust to motion blur, partial occlusion, lighiting changes, model misses, etc..
        self.max_lost_frames = max_lost_frames

    def update(self, persons, candidate_person=None):
        active_ids = [p["track_id"] for p in persons]

        if self.locked_track_id is None:
            if candidate_person is not None:
                self.locked_track_id = candidate_person["track_id"]
                self.lost_frames = 0
            return self.locked_track_id

        if self.locked_track_id in active_ids:
            self.lost_frames = 0
            return self.locked_track_id

        self.lost_frames += 1

        if self.lost_frames > self.max_lost_frames:
            self.locked_track_id = None
            self.lost_frames = 0

        return self.locked_track_id

    def is_locked(self, person):
        return person["track_id"] == self.locked_track_id