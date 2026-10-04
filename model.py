import cv2
import numpy as np
import datetime
import os
import time
from collections import deque

from alerts import AlertManager
from face_tools import build_face_monitor, draw_faces

# ----------------------------- CONFIG ---------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- Detection / box quality ---
BLUR_KERNEL_SIZE = (7, 7)
BG_HISTORY = 500
BG_VAR_THRESHOLD = 32
MIN_CONTOUR_AREA = 800
OPEN_KERNEL_SIZE = 3
CLOSE_KERNEL_SIZE = 15
MERGE_DISTANCE = 25
BOX_PADDING = 6
MAX_FOREGROUND_RATIO = 0.6

# --- Recording ---
RECORD_FPS = 20
PRE_ROLL_SECONDS = 2
POST_MOTION_SECONDS = 5
MAX_CLIP_SECONDS = 120
WARMUP_FRAMES = 60

SNAPSHOT_DIR = "motion_snapshots"
RECORDING_DIR = "recordings"
SHOW_PREVIEW = True


CAMERA_NAME = "Front Door Camera"
FACE_RECOGNITION_ENABLED = True
NOTIFY_KNOWN_PEOPLE = False
ALERT_CONSOLE = True
ALERT_SOUND = True
ALERT_TELEGRAM = True
ALERT_EMAIL = False
# -----------------------------------------------------------------------

_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE))
_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE))


def initialize_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera with index {index}. "
            "Please check if the camera is connected and accessible."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


def create_subtractor():
    # detectShadows=True marks shadows as gray (127) so we can throw them away later
    return cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY, varThreshold=BG_VAR_THRESHOLD, detectShadows=True
    )


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    # Light blur only; colour is kept because MOG2 uses it to tell shadows from objects
    return cv2.GaussianBlur(frame, BLUR_KERNEL_SIZE, 0)


def _boxes_close(a, b, margin):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw + margin < bx or bx + bw + margin < ax or
                ay + ah + margin < by or by + bh + margin < ay)


def merge_boxes(boxes, margin=MERGE_DISTANCE):
    """Merge boxes that overlap or sit within `margin` pixels, so one person = one box."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        result = []
        while boxes:
            a = boxes.pop()
            i = 0
            while i < len(boxes):
                if _boxes_close(a, boxes[i], margin):
                    b = boxes.pop(i)
                    x1, y1 = min(a[0], b[0]), min(a[1], b[1])
                    x2 = max(a[0] + a[2], b[0] + b[2])
                    y2 = max(a[1] + a[3], b[1] + b[3])
                    a = [x1, y1, x2 - x1, y2 - y1]
                    changed = True
                    i = 0  # box grew, so re-check against all the others
                else:
                    i += 1
            result.append(a)
        boxes = result
    return [tuple(b) for b in boxes]


def detect_motion(subtractor, blurred: np.ndarray):
    """
    Returns: (cleaned foreground mask, list of (x, y, w, h) boxes).
    """
    mask = subtractor.apply(blurred)

    # Shadows are 127, real foreground is 255 -> keep only the real foreground
    mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]

    # Remove speckles, then fill gaps so one object becomes one solid blob
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE_KERNEL)

    # A sudden lighting change lights up most of the frame - that's not an intruder
    if cv2.countNonZero(mask) / mask.size > MAX_FOREGROUND_RATIO:
        return mask, []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    return mask, merge_boxes(boxes)


def draw_detections(frame: np.ndarray, boxes) -> bool:
    # Draw one rectangle per detected object; return True if there is any motion
    h, w = frame.shape[:2]
    for (x, y, bw, bh) in boxes:
        x1 = max(x - BOX_PADDING, 0)
        y1 = max(y - BOX_PADDING, 0)
        x2 = min(x + bw + BOX_PADDING, w - 1)
        y2 = min(y + bh + BOX_PADDING, h - 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return len(boxes) > 0


def add_overlay_text(frame: np.ndarray, motion_detected: bool):
    status_text = "Motion Detected" if motion_detected else "No Motion"
    status_color = (0, 0, 255) if motion_detected else (0, 255, 0)
    cv2.putText(frame, f"Status: {status_text}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)


class ClipRecorder:
    """
    Writes clips that play back at REAL speed.

    A VideoWriter only stamps an fps number into the file header. If the loop really
    runs at 8 fps but the header says 20, the clip plays 2.5x too fast. Here every
    frame is placed by its real capture time: repeated if the loop is slow, skipped
    if the camera is faster than RECORD_FPS.
    """

    def __init__(self, frame_size, fps):
        self.frame_size = frame_size  # (width, height)
        self.fps = fps
        self.writer = None
        self.start_ts = 0.0
        self.frames_written = 0

    @property
    def active(self) -> bool:
        return self.writer is not None

    def start(self, path: str, start_ts: float):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, self.fps, self.frame_size)
        if not self.writer.isOpened():
            self.writer = None
            raise RuntimeError(f"Could not open video writer for {path}")
        self.start_ts = start_ts
        self.frames_written = 0

    def write(self, frame: np.ndarray, ts: float):
        if not self.active:
            return
        target = int((ts - self.start_ts) * self.fps) + 1
        while self.frames_written < target:
            self.writer.write(frame)
            self.frames_written += 1

    def duration(self, ts: float) -> float:
        return ts - self.start_ts

    def stop(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def main():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(RECORDING_DIR, exist_ok=True)

    cap = initialize_camera(CAMERA_INDEX)
    recorder = None
    alerts = None
    faces = None
    show_mask = False
    try:
        alerts = AlertManager(camera_name=CAMERA_NAME, console=ALERT_CONSOLE, sound=ALERT_SOUND,
                              telegram=ALERT_TELEGRAM, email=ALERT_EMAIL)
        if FACE_RECOGNITION_ENABLED:
            try:
                faces = build_face_monitor(notify=alerts.send, camera_name=CAMERA_NAME,
                                          notify_known=NOTIFY_KNOWN_PEOPLE)
            except Exception as exc:
                print(f"Face recognition is OFF (motion recording still works): {exc}")

        subtractor = create_subtractor()

        # Warm-up: let exposure settle and teach the model what "empty scene" looks like
        ok, frame = False, None
        for _ in range(WARMUP_FRAMES):
            ok, frame = cap.read()
            if ok:
                subtractor.apply(preprocess_frame(frame))
        if not ok:
            raise RuntimeError("Camera opened but returned no frames.")

        recorder = ClipRecorder((frame.shape[1], frame.shape[0]), RECORD_FPS)
        pre_roll = deque()          # (timestamp, frame) from just before motion
        last_motion_ts = 0.0

        print("Security camera running. Preview keys: 'q' = quit, 'm' = motion mask, 'r' = reload known faces.")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost camera feed, stopping.")
                break
            ts = time.time()

            mask, boxes = detect_motion(subtractor, preprocess_frame(frame))
            motion = len(boxes) > 0

            # Faces are checked on the CLEAN frame - drawn boxes would confuse the recogniser
            face_results = faces.update(frame, ts, motion) if faces else []
            unknown_present = any(r.unknown for r in face_results)
            active = motion or unknown_present   # a stranger standing still still counts

            draw_detections(frame, boxes)
            draw_faces(frame, face_results)
            add_overlay_text(frame, active)
            alerts.draw_banner(frame)

            if active:
                last_motion_ts = ts

            if recorder.active:
                recorder.write(frame, ts)
                quiet_for = ts - last_motion_ts
                if quiet_for > POST_MOTION_SECONDS or recorder.duration(ts) > MAX_CLIP_SECONDS:
                    recorder.stop()
                    print("Clip saved.")
            else:
                pre_roll.append((ts, frame.copy()))
                while pre_roll and ts - pre_roll[0][0] > PRE_ROLL_SECONDS:
                    pre_roll.popleft()

                if active:
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(os.path.join(SNAPSHOT_DIR, f"motion_{stamp}.jpg"), frame)
                    clip_path = os.path.join(RECORDING_DIR, f"motion_{stamp}.mp4")
                    recorder.start(clip_path, pre_roll[0][0])
                    for t, f in pre_roll:
                        recorder.write(f, t)
                    pre_roll.clear()
                    print(f"Motion! Recording to {clip_path}")

            if SHOW_PREVIEW:
                cv2.imshow("Security Camera", frame)
                if show_mask:
                    cv2.imshow("Motion Mask", mask)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("m"):
                    show_mask = not show_mask
                    if not show_mask:
                        cv2.destroyWindow("Motion Mask")
                if key == ord("r") and faces:
                    faces.reload_known()    # pick up newly enrolled people without restarting
    finally:
        if recorder is not None:
            recorder.stop()
        if faces:
            faces.close()
        if alerts:
            alerts.close()      # lets queued alerts finish sending
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()