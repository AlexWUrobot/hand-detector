import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "yolov8n.pt"  # Replace with your custom knife+hand model
VIDEO_SOURCE = 0           # 0 = default laptop camera
CONFIDENCE = 0.25          # Lowered for testing
# ----------------------------------------

# Load YOLO model
model = YOLO(MODEL_PATH)

# Open webcam
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press ESC to exit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    # Run detection
    results = model.predict(frame, conf=CONFIDENCE, device='cpu', verbose=False)[0]

    # Loop through detections
    for i, box in enumerate(results.boxes.xyxy):
        cls = int(results.boxes.cls[i])
        conf = float(results.boxes.conf[i])
        label = model.names[cls]

        # Print detected labels and confidence
        print(f"Detected: {label} ({conf:.2f})")

        x1, y1, x2, y2 = map(int, box.cpu().numpy())

        # Choose color based on label
        if label.lower() == "knife":
            color = (0, 0, 255)  # Red for knife
        elif "hand" in label.lower():
            color = (255, 0, 0)  # Blue for hand
        else:
            color = (0, 255, 0)  # Green for everything else

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show frame
    cv2.imshow("Hand & Knife Detection", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()