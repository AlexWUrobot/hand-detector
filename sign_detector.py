import os
import time
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")


def ensure_model(model_path: str) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        return
    print(f"Downloading hand landmarker model to {model_path} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(
            "Failed to download the MediaPipe Hand Landmarker model. "
            f"Download it manually from {MODEL_URL} and save it to {model_path}."
        ) from exc


# MediaPipe landmark indices
WRIST = 0
THUMB_TIP = 4
THUMB_IP = 3
INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18

INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17


@dataclass(frozen=True)
class FrameFeatures:
    t_s: float
    palm_x: float
    palm_y: float
    palm_z: float
    finger_mask: int


class GestureRecognizer:
    def __init__(
        self,
        history_size: int = 8,
        stable_frames: int = 3,
        cooldown_s: float = 0.8,
    ) -> None:
        self._hist: Deque[FrameFeatures] = deque(maxlen=history_size)
        self._action_hist: Deque[str] = deque(maxlen=stable_frames)
        self._last_emitted: Optional[str] = None
        self._last_emit_time_s: float = 0.0
        self._cooldown_s = cooldown_s

    @staticmethod
    def _palm_center(hand_landmarks) -> Tuple[float, float, float]:
        # Palm-ish center: wrist + MCPs
        pts = [
            hand_landmarks[WRIST],
            hand_landmarks[INDEX_MCP],
            hand_landmarks[MIDDLE_MCP],
            hand_landmarks[RING_MCP],
            hand_landmarks[PINKY_MCP],
        ]
        x = sum(p.x for p in pts) / len(pts)
        y = sum(p.y for p in pts) / len(pts)
        z = sum(p.z for p in pts) / len(pts)
        return x, y, z

    @staticmethod
    def _finger_mask(hand_landmarks) -> int:
        # Bitmask: 1 thumb, 2 index, 4 middle, 8 ring, 16 pinky
        mask = 0

        # Thumb: use x direction relative to IP (works best when palm faces camera)
        if abs(hand_landmarks[THUMB_TIP].x - hand_landmarks[THUMB_IP].x) > 0.04:
            mask |= 1

        # Other fingers: tip above PIP in image coordinates => extended
        if hand_landmarks[INDEX_TIP].y < hand_landmarks[INDEX_PIP].y:
            mask |= 2
        if hand_landmarks[MIDDLE_TIP].y < hand_landmarks[MIDDLE_PIP].y:
            mask |= 4
        if hand_landmarks[RING_TIP].y < hand_landmarks[RING_PIP].y:
            mask |= 8
        if hand_landmarks[PINKY_TIP].y < hand_landmarks[PINKY_PIP].y:
            mask |= 16

        return mask

    @staticmethod
    def _count_fingers(mask: int) -> int:
        return int(bool(mask & 1)) + int(bool(mask & 2)) + int(bool(mask & 4)) + int(bool(mask & 8)) + int(bool(mask & 16))

    def update(self, hand_landmarks, now_s: float) -> Optional[str]:
        palm_x, palm_y, palm_z = self._palm_center(hand_landmarks)
        finger_mask = self._finger_mask(hand_landmarks)

        self._hist.append(
            FrameFeatures(
                t_s=now_s,
                palm_x=palm_x,
                palm_y=palm_y,
                palm_z=palm_z,
                finger_mask=finger_mask,
            )
        )

        action = self._detect_action(hand_landmarks)
        if action is None:
            self._action_hist.clear()
            return None

        self._action_hist.append(action)
        if len(self._action_hist) < self._action_hist.maxlen:
            return None

        most_common, count = Counter(self._action_hist).most_common(1)[0]
        if count < self._action_hist.maxlen:
            return None

        if most_common == self._last_emitted and (now_s - self._last_emit_time_s) < self._cooldown_s:
            return None

        if (now_s - self._last_emit_time_s) < self._cooldown_s and most_common != self._last_emitted:
            return None

        self._last_emitted = most_common
        self._last_emit_time_s = now_s
        return most_common

    def reset(self) -> None:
        self._hist.clear()
        self._action_hist.clear()

    def _motion_delta(self) -> Tuple[float, float]:
        if len(self._hist) < 2:
            return 0.0, 0.0
        first = self._hist[0]
        last = self._hist[-1]
        return last.palm_x - first.palm_x, last.palm_z - first.palm_z

    def _latest_finger_count(self) -> int:
        if not self._hist:
            return 0
        return self._count_fingers(self._hist[-1].finger_mask)

    @staticmethod
    def _point_dir(hand_landmarks, tip_indices, mcp_indices) -> int:
        tip_x = sum(hand_landmarks[i].x for i in tip_indices) / len(tip_indices)
        tip_y = sum(hand_landmarks[i].y for i in tip_indices) / len(tip_indices)
        mcp_x = sum(hand_landmarks[i].x for i in mcp_indices) / len(mcp_indices)
        mcp_y = sum(hand_landmarks[i].y for i in mcp_indices) / len(mcp_indices)

        vx = tip_x - mcp_x
        vy = tip_y - mcp_y

        # Require the pointing vector to be mostly horizontal
        if abs(vx) > 0.06 and abs(vx) > abs(vy):
            return -1 if vx < 0 else 1
        return 0

    def _detect_action(self, hand_landmarks) -> Optional[str]:
        fingers = self._latest_finger_count()
        mask = self._hist[-1].finger_mask if self._hist else 0

        # Clockwise / Counterclockwise: 1 or 2 fingers pointing left/right
        # (Uses the current frame only, so it reacts immediately.)
        index_up = bool(mask & 2)
        middle_up = bool(mask & 4)
        ring_up = bool(mask & 8)
        thumb_up = bool(mask & 1)
        pinky_up = bool(mask & 16)

        if fingers == 1 and index_up and not (thumb_up or middle_up or ring_up or pinky_up):
            d = self._point_dir(hand_landmarks, [INDEX_TIP], [INDEX_MCP])
            if d != 0:
                return "move clockwise" if d < 0 else "move counterclockwise"

        if fingers == 2 and index_up and middle_up and not (thumb_up or ring_up or pinky_up):
            d = self._point_dir(hand_landmarks, [INDEX_TIP, MIDDLE_TIP], [INDEX_MCP, MIDDLE_MCP])
            if d != 0:
                return "move clockwise" if d < 0 else "move counterclockwise"

        # 3 fingers (index+middle+ring) pointing + swipe left/right
        # - Point left + swipe left => clockwise
        # - Point right + swipe right => counterclockwise
        if fingers == 3 and index_up and middle_up and ring_up and not (thumb_up or pinky_up):
            d = self._point_dir(
                hand_landmarks,
                [INDEX_TIP, MIDDLE_TIP, RING_TIP],
                [INDEX_MCP, MIDDLE_MCP, RING_MCP],
            )
            if d != 0 and len(self._hist) >= self._hist.maxlen:
                dx, dz = self._motion_delta()
                if abs(dx) > 0.12 and abs(dx) > abs(dz):
                    if d < 0 and dx < 0:
                        return "move clockwise"
                    if d > 0 and dx > 0:
                        return "move counterclockwise"

        # Motion-based gestures need some history
        if len(self._hist) < self._hist.maxlen:
            return None

        dx, dz = self._motion_delta()
        dy = self._hist[-1].palm_y - self._hist[0].palm_y

        # Thresholds in normalized landmark space
        z_wave_thresh = 0.10
        y_wave_thresh = 0.12
        stop_stable_x = 0.03
        stop_stable_z = 0.02

        # Move front: five fingers down (fist) waving up/down
        # (Interpretation: 0 fingers extended and dominant vertical motion.)
        if fingers == 0 and abs(dy) > y_wave_thresh and abs(dy) > abs(dx) and abs(dy) > abs(dz):
            return "move front"

        # Move back: five fingers up and moving closer to camera
        # In MediaPipe, z tends to be more negative when closer to camera.
        if fingers == 5 and dz < -0.02 and abs(dz) > abs(dx):
            return "move back"

        # Stop: five fingers up and stable
        if fingers == 5 and abs(dx) < stop_stable_x and abs(dz) < stop_stable_z:
            return "stop"

        # Move back/front: waving toward/away from camera (3+ fingers)
        if fingers >= 3 and abs(dz) > z_wave_thresh and abs(dz) > abs(dx):
            # In MediaPipe, z tends to be more negative when closer to camera.
            return "move back" if dz < 0 else "move front"

        return None


def main() -> None:
    ensure_model(MODEL_PATH)

    cap = cv2.VideoCapture(0)

    options = vision.HandLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    recognizer = GestureRecognizer(history_size=8, stable_frames=3, cooldown_s=0.8)

    start_time = time.monotonic()

    last_active_palm: Optional[Tuple[float, float, float]] = None

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            now_s = time.monotonic()

            if result.hand_landmarks:
                # Always choose the closest hand (most negative z in MediaPipe coords).
                candidates = []
                for idx, hand_landmarks in enumerate(result.hand_landmarks):
                    palm_x, palm_y, palm_z = GestureRecognizer._palm_center(hand_landmarks)
                    candidates.append((palm_z, idx, palm_x, palm_y))

                candidates.sort(key=lambda t: t[0])
                _, best_idx, best_x, best_y = candidates[0]
                hand_landmarks = result.hand_landmarks[best_idx]

                current_palm = (best_x, best_y, GestureRecognizer._palm_center(hand_landmarks)[2])
                if last_active_palm is not None:
                    dx = current_palm[0] - last_active_palm[0]
                    dy = current_palm[1] - last_active_palm[1]
                    dz = current_palm[2] - last_active_palm[2]
                    # If the active hand jumps significantly, treat it as a hand switch.
                    if (dx * dx + dy * dy + dz * dz) ** 0.5 > 0.35:
                        recognizer.reset()
                last_active_palm = current_palm

                vision.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    vision.HandLandmarksConnections.HAND_CONNECTIONS,
                )

                action = recognizer.update(hand_landmarks, now_s)
                if action:
                    print(action)
            else:
                recognizer.reset()
                last_active_palm = None

            cv2.imshow("Sign Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
