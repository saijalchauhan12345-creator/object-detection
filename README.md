# Object Detection and Tracking

Real-time object detection and tracking tool built for the CodeAlpha AI Internship (Task 4). Uses a webcam or video file as input, detects objects in each frame, and tracks them across frames with a consistent ID.

## Features

- Real-time detection from webcam or a video file
- Uses YOLOv8 (pretrained on the COCO dataset, 80 object classes: person, car, dog, laptop, etc.)
- Object tracking with ByteTrack, assigning a persistent ID to each detected object as it moves across frames
- Bounding boxes, class labels, and track IDs drawn directly on the video feed

## Tech Stack

- Python
- OpenCV - video capture and display
- Ultralytics YOLOv8 - object detection and tracking

## How to Run

1. Install dependencies:
   python -m pip install -r requirements.txt

2. Run the script:
   python main.py

3. Your webcam will open in a new window with live detection and tracking. Press q to quit.

To use a video file instead of the webcam, open main.py and change SOURCE = 0 to SOURCE = "path/to/your/video.mp4"

## How It Works

1. OpenCV captures frames from the webcam one at a time.
2. Each frame is passed to a pretrained YOLOv8 model, which detects objects and their bounding boxes.
3. The track() method (ByteTrack algorithm) links detections across frames, giving each object a consistent ID.
4. The annotated frame is displayed in real time.

Note: the first run downloads the YOLOv8 weights file (about 6 MB).

---
Built as part of the CodeAlpha Artificial Intelligence Internship.
