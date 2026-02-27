import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "yolov8s.pt"  # Use larger model for better accuracy
VIDEO_SOURCE = 0            # 0 = default webcam
CONFIDENCE = 0.2            # Lower confidence for more detections
IOU_THRESHOLD = 0.3         # Adjust NMS
IMG_SIZE = 768              # Higher resolution for small objects
# ----------------------------------------

# Load YOLO model
model = YOLO(MODEL_PATH)

# Open webcam
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Initialize tracker dictionary
trackers = {}

print("Press ESC to exit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    # Run detection
    results = model.predict(
        frame,
        conf=CONFIDENCE,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        device='cpu',
        verbose=False
    )[0]

    # Update trackers
    new_trackers = {}
    for i, box in enumerate(results.boxes.xyxy):
        cls = int(results.boxes.cls[i])
        conf = float(results.boxes.conf[i])
        label = model.names[cls]

        x1, y1, x2, y2 = map(int, box.cpu().numpy())

        # Only track knives
        if label.lower() == "knife":
            # Create or update tracker
            if i not in trackers:
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                new_trackers[i] = tracker
            else:
                new_trackers[i] = trackers[i]

        # Choose color
        if label.lower() == "knife":
            color = (0, 0, 255)  # Red
        elif "hand" in label.lower():
            color = (255, 0, 0)  # Blue
        else:
            color = (0, 255, 0)  # Green

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Update tracker dictionary
    trackers = new_trackers

    # Show frame
    cv2.imshow("Hand & Knife Detection", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
