import cv2
import numpy as np
import dlib
from sys import argv
from typing import NamedTuple, List


class Face(NamedTuple):

    top: int = None
    left: int = None
    right: int = None
    bottom: int = None

    def __str__(self):
        return "(%d, %d),(%d,%d)" % (self.left, self.top, self.right, self.bottom)

    def __repr__(self):
        return str(self)


class Landmark(NamedTuple):
    x: int = None
    y: int = None

    def __str__(self):
        return "(%d, %d)" % (self.x, self.y)

    def __repr__(self):
        return str(self)


class FrameInfo:

    def __init__(self):
        self.faces = []  # typing: List[Face]
        self.landmarks = {}  # typing: Dict[int, List[Landmark]

    def add_face(self, face : Face):
        self.faces.append(face)

    def add_landmarks(self, landmarks: List[Landmark]):
        """
        Adds landmarks to current face on frame
        :param landmarks:
        :return:
        """

        self.landmarks[len(self.faces) - 1] = landmarks

    def __str__(self):
        return "face: %s, landmarks: %s" % (self.faces[0], self.landmarks[0])


def generate_landmarks(video_file: str):
    cap = cv2.VideoCapture(video_file)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
    frames = []
    while True:

        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_info = FrameInfo()
        faces = detector(gray)
        for i, face in enumerate(faces):
            face_info = Face(left=face.left(),
                        top=face.top(),
                        right=face.right(),
                        bottom=face.bottom())

            landmarks = predictor(gray, face)
            landmarks = [Landmark(x=landmarks.part(n).x, y=landmarks.part(n).y) for n in range(0, 68)]
            frame_info.add_face(face_info)
            frame_info.add_landmarks(landmarks)
            frames.append(frame_info)
    return frames


if __name__ == '__main__':
    frames = generate_landmarks('data/lombargrid/front/s2_l_bbim3a.mov')
    for frame in frames:
        print(frame)
