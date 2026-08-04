import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("downloading hand model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    running_mode=vision.RunningMode.VIDEO,
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def count_fingers(lm):
    wrist = lm[0]
    count = 0

    # thumb - check how far the tip is from the pinky side of the palm
    tip_to_pinky = dist(lm[4], lm[17])
    mcp_to_pinky = dist(lm[2], lm[17])
    if tip_to_pinky > mcp_to_pinky * 1.1:
        count += 1

    # other 4 fingers - tip further from wrist than the pip joint = finger up
    for tip_id, pip_id in zip(TIP_IDS[1:], PIP_IDS[1:]):
        if dist(lm[tip_id], wrist) > dist(lm[pip_id], wrist) * 1.1:
            count += 1

    return count


def draw_hand(frame, lm):
    h, w, _ = frame.shape
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("could not open webcam")
    exit()

print("press q to quit")

ts = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # object detection + tracking
    results = model.track(frame, persist=True, verbose=False)
    annotated = results[0].plot()

    # hand + finger counting
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts += 33
    hand_result = hand_landmarker.detect_for_video(mp_img, ts)

    if hand_result.hand_landmarks:
        h, w, _ = annotated.shape
        for lm, handed in zip(hand_result.hand_landmarks, hand_result.handedness):
            label = handed[0].category_name
            fingers = count_fingers(lm)

            draw_hand(annotated, lm)

            x, y = int(lm[0].x * w), int(lm[0].y * h)
            cv2.putText(annotated, f"{label}: {fingers} fingers", (x - 50, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Object Detection and Tracking - CodeAlpha", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()