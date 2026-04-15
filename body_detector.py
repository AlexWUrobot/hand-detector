import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_heavy.task")


def ensure_model(model_path: str) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        return
    print(f"Downloading pose landmarker model to {model_path} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(
            "Failed to download the MediaPipe Pose Landmarker model. "
            f"Download it manually from {MODEL_URL} and save it to {model_path}."
        ) from exc


ensure_model(MODEL_PATH)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam (index 0)")

options = vision.PoseLandmarkerOptions(
    base_options=base_options.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.6,
    min_pose_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

start_time = time.monotonic()

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.monotonic() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            for pose_landmarks in result.pose_landmarks:
                vision.drawing_utils.draw_landmarks(
                    frame,
                    pose_landmarks,
                    vision.PoseLandmarksConnections.POSE_LANDMARKS,
                )

        cv2.imshow("Body Skeleton Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
