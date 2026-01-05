import numpy as np
import math

class BlinkDetector:
    def __init__(self, threshold=0.21, min_frames=2):
        self.threshold = threshold
        self.min_frames = min_frames
        self.counter = 0
        self.total_blinks = 0

    def dist(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def eye_aspect_ratio(self, eye):
        A = self.dist(eye[1], eye[5])
        B = self.dist(eye[2], eye[4])
        C = self.dist(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def update(self, left_eye, right_eye):
        ear = (self.eye_aspect_ratio(left_eye) +
               self.eye_aspect_ratio(right_eye)) / 2.0

        if ear < self.threshold:
            self.counter += 1
        else:
            if self.counter >= self.min_frames:
                self.total_blinks += 1
            self.counter = 0

        return ear, self.total_blinks
