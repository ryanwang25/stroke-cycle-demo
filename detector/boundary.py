class StrokeCycleBoundaryPolicy:
    def __init__(self, debounce_frames=15):
        self.debounce_frames = debounce_frames
        self.last_fired = -999

    def is_boundary(self, prev, curr, frame_idx):
        if prev is None or curr is None:
            return False
        if frame_idx - self.last_fired < self.debounce_frames:
            return False
        if (prev['wrist_x'] > prev['shoulder_x']) and (curr['wrist_x'] < curr['shoulder_x']):
            self.last_fired = frame_idx
            return True
        return False