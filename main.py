import os
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading hand model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO,
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# Multiple face mesh — max 3 faces
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

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

COLOR_RANGES = [
    ("Red",    (0, 70, 50),   (10, 255, 255),  (0, 0, 255)),
    ("Red",    (170, 70, 50), (180, 255, 255),  (0, 0, 255)),
    ("Orange", (11, 70, 50),  (25, 255, 255),   (0, 165, 255)),
    ("Yellow", (26, 70, 50),  (34, 255, 255),   (0, 255, 255)),
    ("Green",  (35, 50, 50),  (85, 255, 255),   (0, 200, 0)),
    ("Cyan",   (86, 50, 50),  (100, 255, 255),  (255, 255, 0)),
    ("Blue",   (101, 50, 50), (130, 255, 255),  (255, 0, 0)),
    ("Purple", (131, 50, 50), (155, 255, 255),  (200, 0, 200)),
    ("Pink",   (156, 50, 50), (169, 255, 255),  (180, 105, 255)),
]

prev_time = 0
os.makedirs("screenshots", exist_ok=True)

last_emotion_time = 0
emotion_results = []
emotion_interval = 1.0


def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def count_fingers(lm):
    wrist = lm[0]
    count = 0
    tip_to_pinky = dist(lm[4], lm[17])
    mcp_to_pinky = dist(lm[2], lm[17])
    if tip_to_pinky > mcp_to_pinky * 1.1:
        count += 1
    for tip_id, pip_id in zip(TIP_IDS[1:], PIP_IDS[1:]):
        if dist(lm[tip_id], wrist) > dist(lm[pip_id], wrist) * 1.1:
            count += 1
    return count


def detect_gesture(lm):
    wrist = lm[0]
    fingers_up = []
    tip_to_pinky = dist(lm[4], lm[17])
    mcp_to_pinky = dist(lm[2], lm[17])
    fingers_up.append(tip_to_pinky > mcp_to_pinky * 1.1)
    for tip_id, pip_id in zip(TIP_IDS[1:], PIP_IDS[1:]):
        fingers_up.append(dist(lm[tip_id], wrist) > dist(lm[pip_id], wrist) * 1.1)
    thumb, index, middle, ring, pinky = fingers_up
    if thumb and not index and not middle and not ring and not pinky:
        return "thumbs_up"
    if not thumb and index and middle and not ring and not pinky:
        return "peace"
    if not any(fingers_up):
        return "fist"
    return None


def draw_hand(frame, lm):
    h, w, _ = frame.shape
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def get_dominant_color(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "Unknown", (128, 128, 128)
    roi_small = cv2.resize(roi, (60, 60))
    hsv_roi = cv2.cvtColor(roi_small, cv2.COLOR_BGR2HSV)
    v_mean = np.mean(hsv_roi[:, :, 2])
    s_mean = np.mean(hsv_roi[:, :, 1])
    if v_mean < 40:
        return "Black", (20, 20, 20)
    if s_mean < 40 and v_mean > 200:
        return "White", (255, 255, 255)
    if s_mean < 45:
        return "Gray", (128, 128, 128)
    best_color = "Unknown"
    best_count = 0
    best_bgr = (128, 128, 128)
    for name, lower, upper, bgr in COLOR_RANGES:
        mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
        count = cv2.countNonZero(mask)
        if count > best_count:
            best_count = count
            best_color = name
            best_bgr = bgr
    return best_color, best_bgr


def get_emotion_from_landmarks(lm):
    upper_lip = lm[13]
    lower_lip = lm[14]
    left_corner = lm[61]
    right_corner = lm[291]

    mouth_height = abs(upper_lip.y - lower_lip.y)
    mouth_width = abs(left_corner.x - right_corner.x)
    mouth_ratio = mouth_height / (mouth_width + 1e-6)

    mouth_center_y = (left_corner.y + right_corner.y) / 2
    upper_lip_y = upper_lip.y
    smile_score = mouth_center_y - upper_lip_y

    left_brow = lm[105]
    right_brow = lm[334]
    left_eye = lm[159]
    right_eye = lm[386]
    brow_raise = ((left_eye.y - left_brow.y) + (right_eye.y - right_brow.y)) / 2

    if mouth_ratio > 0.15:
        return "Surprised 😮"
    elif smile_score > 0.02 and mouth_ratio > 0.04:
        return "Happy 😊"
    elif smile_score < 0.005:
        return "Sad 😔"
    elif brow_raise < 0.04:
        return "Angry 😠"
    else:
        return "Neutral 😐"


def detect_emotions_multi(frame):
    global last_emotion_time, emotion_results
    current_time = time.time()
    if current_time - last_emotion_time < emotion_interval:
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    emotion_results = []

    if not results.multi_face_landmarks:
        last_emotion_time = current_time
        return

    h, w = frame.shape[:2]

    for i, face_landmarks in enumerate(results.multi_face_landmarks):
        lm = face_landmarks.landmark
        emotion = get_emotion_from_landmarks(lm)

        # Face position — nose tip
        nose = lm[1]
        fx = int(nose.x * w)
        fy = int(nose.y * h)

        emotion_results.append({
            "face": i + 1,
            "emotion": emotion,
            "x": fx,
            "y": fy
        })

    last_emotion_time = current_time


def draw_ui(frame, fps, obj_count, gesture_msg="", screenshot_msg=""):
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"FPS: {int(fps)}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Objects: {obj_count}", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Show emotion for each face
    if emotion_results:
        for er in emotion_results:
            label = f"Face {er['face']}: {er['emotion']}"
            cv2.putText(frame, label, (15, 90 + (er['face'] - 1) * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            # Show on face too
            cv2.putText(frame, er['emotion'], (er['x'] - 40, er['y'] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "Emotion: No face", (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    cv2.putText(frame, "S/Q: key | Thumbs Up: Screenshot | Peace: Quit",
                (15, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    if gesture_msg:
        cv2.putText(frame, gesture_msg, (15, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    if screenshot_msg:
        cv2.putText(frame, screenshot_msg, (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Press Q to quit | Press S to screenshot | Thumbs Up = screenshot | Peace = quit")

ts = 0
screenshot_msg = ""
screenshot_timer = 0
gesture_msg = ""
gesture_cooldown = 2
last_gesture_time = 0
quit_flag = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    results = model.predict(frame, verbose=False, conf=0.3)
    annotated = results[0].plot()

    if results[0].boxes is not None:
        for box in results[0].boxes.xyxy:
            color_name, bgr = get_dominant_color(frame, box)
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = y2 + 25
            cv2.circle(annotated, (cx - 45, cy), 12, bgr, -1)
            cv2.circle(annotated, (cx - 45, cy), 12, (255, 255, 255), 2)
            cv2.putText(annotated, color_name, (cx - 28, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    detect_emotions_multi(frame)

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
            gesture = detect_gesture(lm)

            if gesture and (curr_time - last_gesture_time > gesture_cooldown):
                last_gesture_time = curr_time
                if gesture == "thumbs_up":
                    filename = f"screenshots/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(filename, annotated)
                    screenshot_msg = "Thumbs Up! Screenshot Saved!"
                    screenshot_timer = curr_time
                    print(f"Screenshot saved: {filename}")
                elif gesture == "peace":
                    gesture_msg = "Peace! Quitting..."
                    quit_flag = True

            gesture_label = {
                "thumbs_up": "Thumbs Up",
                "peace": "Peace",
                "fist": "Fist",
            }.get(gesture, "")

            cv2.putText(annotated, f"{label}: {fingers}f {gesture_label}",
                        (x - 50, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    obj_count = len(results[0].boxes) if results[0].boxes is not None else 0

    if curr_time - screenshot_timer > 3:
        screenshot_msg = ""

    annotated = draw_ui(annotated, fps, obj_count, gesture_msg, screenshot_msg)

    cv2.imshow("Object Detection and Tracking - CodeAlpha", annotated)

    if quit_flag:
        time.sleep(1)
        break

    key = cv2.waitKey(30) & 0xFF

    if key == ord("s"):
        filename = f"screenshots/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, annotated)
        screenshot_msg = "Screenshot Saved!"
        screenshot_timer = curr_time
        print(f"Screenshot saved: {filename}")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()