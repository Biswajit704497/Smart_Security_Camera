import csv
import datetime
import json
import os
import re
import shutil
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

# ----------------------------- SETTINGS -------------------------------
KNOWN_DIR = "known_faces"
UNKNOWN_DIR = "unknown_faces"
FACE_LOG = "face_log.csv"
MODEL_DIR = "models"

MATCH_THRESHOLD = 0.363       # cosine similarity needed to call two faces "the same person"
                              # (0.363 is OpenCV's recommended value; raise it for stricter matching)
MIN_FACE_SCORE = 0.85         # ignore low-confidence detections
MIN_FACE_PIXELS = 60          # ignore faces narrower than this (too small to recognise reliably)

UNKNOWN_CONFIRM_COUNT = 3     # a stranger must be seen in 3 checks in a row before we alert
PENDING_TTL = 4.0             # ...within this many seconds (filters out one-frame mistakes)

CHECK_INTERVAL_MOTION = 0.2   # seconds between face checks while something is moving
CHECK_INTERVAL_IDLE = 1.0     # seconds between face checks when the scene is still
RESULT_HOLD_SECONDS = 1.0     # keep drawing the last face boxes this long

VISIT_GAP_SECONDS = 60        # absent this long = the next sighting counts as a new visit
ALERT_COOLDOWN_SECONDS = 300  # don't re-alert about the SAME stranger more often than this
MAX_IMAGES_PER_UNKNOWN = 10   # photo limit per stranger (stops the disk filling up)
IMAGE_SAVE_INTERVAL = 3.0     # seconds between saved photos of the same stranger
MAX_EMBEDDINGS_PER_UNKNOWN = 8
KNOWN_LOG_INTERVAL = 60       # log a known person at most once per minute
# -----------------------------------------------------------------------

_MODEL_FILES = {
    "face_detection_yunet_2023mar.onnx": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ],
    "face_recognition_sface_2021dec.onnx": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ],
}
_MIN_MODEL_BYTES = 50_000     # smaller than this = broken download / git-LFS pointer file
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ------------------------------------------------------------- helpers
def safe_name(name):
    """Make a person's name safe to use as a folder name."""
    cleaned = re.sub(r"[^\w\- ]", "", name, flags=re.UNICODE).strip()
    if not cleaned:
        raise ValueError("Please give a name using letters or numbers.")
    return cleaned


def crop_face(frame, box, margin=0.35):
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    mx, my = int(bw * margin), int(bh * margin)
    x1, y1 = max(x - mx, 0), max(y - my, 0)
    x2, y2 = min(x + bw + mx, w), min(y + bh + my, h)
    return frame[y1:y2, x1:x2].copy()


def ensure_model(filename):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path) and os.path.getsize(path) > _MIN_MODEL_BYTES:
        return path

    tmp = path + ".part"
    for url in _MODEL_FILES[filename]:
        try:
            print(f"[faces] Downloading {filename} ...")
            urllib.request.urlretrieve(url, tmp)
            if os.path.getsize(tmp) > _MIN_MODEL_BYTES:
                os.replace(tmp, path)
                return path
        except Exception as exc:
            print(f"[faces]   download failed: {exc}")
    if os.path.exists(tmp):
        os.remove(tmp)
    raise RuntimeError(
        f"Could not download {filename}.\n"
        f"Download it manually from https://github.com/opencv/opencv_zoo "
        f"(models/face_detection_yunet and models/face_recognition_sface) "
        f"and put it in the '{MODEL_DIR}' folder."
    )


@dataclass
class Face:
    box: tuple                    # (x, y, w, h)
    score: float
    row: object = field(default=None, repr=False)   # raw detector row, needed for alignment


@dataclass
class FaceResult:
    box: tuple
    label: str
    unknown: bool


# --------------------------------------------------------------- engine
class FaceEngine:
    def __init__(self, detect_threshold=0.7):
        detector_path = ensure_model("face_detection_yunet_2023mar.onnx")
        recognizer_path = ensure_model("face_recognition_sface_2021dec.onnx")
        self.detector = cv2.FaceDetectorYN.create(
            detector_path, "", (640, 480), detect_threshold, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")

    def detect(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, rows = self.detector.detect(frame)
        if rows is None:
            return []
        return [Face((int(r[0]), int(r[1]), int(r[2]), int(r[3])), float(r[14]), r)
                for r in rows]

    def embed(self, frame, face):
        """128-number fingerprint of a face, scaled so that dot product = cosine similarity."""
        aligned = self.recognizer.alignCrop(frame, face.row)
        feat = self.recognizer.feature(aligned).flatten().astype(np.float32)
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat


# ---------------------------------------------------------- known people
class KnownFaces:
    """
    known_faces/
        Alice/   photo1.jpg photo2.jpg ...       <- any photos with Alice's face
        Bob/     embeddings.npy  reference/...   <- created by --promote

    You can simply drop photos into a person's folder, or use enroll_face.py.
    """

    def __init__(self, root=KNOWN_DIR, threshold=MATCH_THRESHOLD):
        self.root = root
        self.threshold = threshold
        self.people = {}          # name -> array of fingerprints, shape (n, 128)

    def load(self, engine):
        self.people = {}
        os.makedirs(self.root, exist_ok=True)
        for name in sorted(os.listdir(self.root)):
            folder = os.path.join(self.root, name)
            if not os.path.isdir(folder):
                continue
            embs = []
            npy = os.path.join(folder, "embeddings.npy")
            if os.path.exists(npy):
                embs.extend(np.load(npy))
            for fn in sorted(os.listdir(folder)):
                if not fn.lower().endswith(IMAGE_EXTS):
                    continue
                img = cv2.imread(os.path.join(folder, fn))
                emb = self._embed_largest_face(engine, img) if img is not None else None
                if emb is None:
                    print(f"[faces]   no usable face in {name}/{fn} - skipped")
                else:
                    embs.append(emb)
            if embs:
                self.people[name] = np.array(embs, dtype=np.float32)
        summary = ", ".join(f"{n} ({len(e)})" for n, e in self.people.items()) or "nobody yet"
        print(f"[faces] Known people: {summary}")

    @staticmethod
    def _embed_largest_face(engine, img):
        longest = max(img.shape[:2])
        if longest > 1280:                       # phone photos are huge - shrink them
            scale = 1280 / longest
            img = cv2.resize(img, None, fx=scale, fy=scale)
        faces = engine.detect(img)
        if not faces:
            return None
        face = max(faces, key=lambda f: f.box[2] * f.box[3])
        return engine.embed(img, face)

    def identify(self, emb):
        """Returns (name, similarity) if emb matches a known person, else (None, best_similarity)."""
        best_name, best_sim = None, -1.0
        for name, arr in self.people.items():
            sim = float(np.max(arr @ emb))
            if sim > best_sim:
                best_name, best_sim = name, sim
        if best_sim >= self.threshold:
            return best_name, best_sim
        return None, best_sim


# ------------------------------------------------------- unknown visitors
class UnknownStore:
    """
    unknown_faces/
        unknown_0001/   info.json  embeddings.npy  face_*.jpg  scene_*.jpg
        unknown_0002/   ...
        _promoted/      strangers you later registered as known people
    """

    def __init__(self, root=UNKNOWN_DIR, threshold=MATCH_THRESHOLD):
        self.root = root
        self.threshold = threshold
        self.records = {}
        os.makedirs(root, exist_ok=True)
        self._load()

    def _load(self):
        for d in sorted(os.listdir(self.root)):
            folder = os.path.join(self.root, d)
            info_p = os.path.join(folder, "info.json")
            emb_p = os.path.join(folder, "embeddings.npy")
            if not (d.startswith("unknown_") and os.path.exists(info_p) and os.path.exists(emb_p)):
                continue
            with open(info_p) as f:
                info = json.load(f)
            self.records[d] = self._new_record(info, np.load(emb_p), folder)
        if self.records:
            print(f"[faces] Remembering {len(self.records)} earlier unknown visitor(s)")

    @staticmethod
    def _new_record(info, embs, folder):
        return {"info": info, "embs": np.atleast_2d(embs).astype(np.float32), "dir": folder,
                "last_seen_ts": 0.0, "last_alert_ts": 0.0, "last_image_ts": 0.0}

    def _next_index(self):
        highest = 0
        for base in (self.root, os.path.join(self.root, "_promoted")):
            if os.path.isdir(base):
                for d in os.listdir(base):
                    m = re.fullmatch(r"unknown_(\d+)", d)
                    if m:
                        highest = max(highest, int(m.group(1)))
        return highest + 1

    def match(self, emb):
        best_id, best_sim = None, -1.0
        for uid, rec in self.records.items():
            sim = float(np.max(rec["embs"] @ emb))
            if sim > best_sim:
                best_id, best_sim = uid, sim
        if best_sim >= self.threshold:
            return best_id, best_sim
        return None, best_sim

    def create(self, emb, face_img, scene_img, ts):
        uid = f"unknown_{self._next_index():04d}"
        folder = os.path.join(self.root, uid)
        os.makedirs(folder, exist_ok=True)
        now = datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds")
        info = {"id": uid, "first_seen": now, "last_seen": now, "visits": 1, "images": 0}
        rec = self._new_record(info, emb, folder)
        rec["last_seen_ts"] = ts
        self.records[uid] = rec
        self.save_image(uid, face_img, scene_img, ts)
        self.save_info(uid)
        return uid

    def sighting(self, uid, emb, ts, new_visit):
        rec = self.records[uid]
        rec["last_seen_ts"] = ts
        rec["info"]["last_seen"] = datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds")
        if new_visit:
            rec["info"]["visits"] += 1
        # Remember extra angles/lighting so the same stranger is recognised more reliably
        if len(rec["embs"]) < MAX_EMBEDDINGS_PER_UNKNOWN and float(np.max(rec["embs"] @ emb)) < 0.8:
            rec["embs"] = np.vstack([rec["embs"], emb])

    def wants_image(self, uid, ts):
        rec = self.records[uid]
        return (rec["info"]["images"] < MAX_IMAGES_PER_UNKNOWN
                and ts - rec["last_image_ts"] >= IMAGE_SAVE_INTERVAL)

    def save_image(self, uid, face_img, scene_img, ts):
        rec = self.records[uid]
        stamp = datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        cv2.imwrite(os.path.join(rec["dir"], f"face_{stamp}.jpg"), face_img)
        cv2.imwrite(os.path.join(rec["dir"], f"scene_{stamp}.jpg"), scene_img)
        rec["info"]["images"] += 1
        rec["last_image_ts"] = ts

    def save_info(self, uid):
        rec = self.records[uid]
        with open(os.path.join(rec["dir"], "info.json"), "w") as f:
            json.dump(rec["info"], f, indent=2)
        np.save(os.path.join(rec["dir"], "embeddings.npy"), rec["embs"])

    def flush(self):
        for uid in self.records:
            self.save_info(uid)


def promote_unknown(unknown_id, name, unknown_root=UNKNOWN_DIR, known_root=KNOWN_DIR):
    """Turn a stranger into a known person (e.g. a new neighbour or family friend)."""
    name = safe_name(name)
    src = os.path.join(unknown_root, unknown_id)
    if not os.path.isdir(src) or not os.path.exists(os.path.join(src, "embeddings.npy")):
        raise FileNotFoundError(f"No such unknown visitor: {unknown_id}")

    dst = os.path.join(known_root, name)
    ref = os.path.join(dst, "reference")
    os.makedirs(ref, exist_ok=True)

    embs = np.load(os.path.join(src, "embeddings.npy"))
    npy = os.path.join(dst, "embeddings.npy")
    if os.path.exists(npy):
        embs = np.vstack([np.load(npy), embs])
    np.save(npy, embs)

    for fn in os.listdir(src):
        if fn.startswith("face_") and fn.endswith(".jpg"):
            shutil.copy(os.path.join(src, fn), os.path.join(ref, f"{unknown_id}_{fn}"))

    archive = os.path.join(unknown_root, "_promoted")
    os.makedirs(archive, exist_ok=True)
    shutil.move(src, os.path.join(archive, unknown_id))
    return name


# -------------------------------------------------------------- monitor
class FaceMonitor:
    def __init__(self, engine, known, unknowns, notify=None, camera_name="Security Camera",
                 notify_known=False, log_path=FACE_LOG):
        self.engine = engine
        self.known = known
        self.unknowns = unknowns
        self.notify = notify              # callable(title, message, image)
        self.camera_name = camera_name
        self.notify_known = notify_known
        self.log_path = log_path

        self._pending = []                # strangers seen but not yet confirmed
        self._results = []
        self._results_ts = 0.0
        self._last_check = 0.0
        self._known_seen = {}             # name -> last logged timestamp

    # ---- called once per camera frame ----
    def update(self, frame, ts, motion):
        """Returns the list of FaceResult to draw. `frame` must be the clean, un-annotated frame."""
        interval = CHECK_INTERVAL_MOTION if motion else CHECK_INTERVAL_IDLE
        if ts - self._last_check >= interval:
            self._last_check = ts
            self._results = self._process(frame, ts)
            self._results_ts = ts
        elif ts - self._results_ts > RESULT_HOLD_SECONDS:
            self._results = []
        return self._results

    def reload_known(self):
        self.known.load(self.engine)

    def close(self):
        self.unknowns.flush()

    # ---- internals ----
    def _process(self, frame, ts):
        self._pending = [p for p in self._pending if ts - p["last"] <= PENDING_TTL]
        results = []
        for face in self.engine.detect(frame):
            if face.score < MIN_FACE_SCORE or face.box[2] < MIN_FACE_PIXELS:
                continue                               # too small / unclear to judge fairly
            emb = self.engine.embed(frame, face)

            name, _ = self.known.identify(emb)
            if name:
                results.append(FaceResult(face.box, name, False))
                self._note_known(name, frame, ts)
                continue

            uid, _ = self.unknowns.match(emb)
            if uid is None:
                uid = self._track_pending(emb, frame, face, ts)
            else:
                self._returning(uid, emb, frame, face, ts)

            label = self._label(uid) if uid else "Unknown"
            results.append(FaceResult(face.box, label, True))
        return results

    @staticmethod
    def _label(uid):
        return f"Unknown #{int(uid.split('_')[1])}"

    def _track_pending(self, emb, frame, face, ts):
        """A stranger must be seen several times in a row before we trust it."""
        for p in self._pending:
            if float(p["emb"] @ emb) >= MATCH_THRESHOLD:
                p["count"] += 1
                p["last"] = ts
                if face.score > p["face"].score:
                    p["face"], p["frame"] = face, frame.copy()
                if p["count"] >= UNKNOWN_CONFIRM_COUNT:
                    self._pending.remove(p)
                    return self._confirm_new(p, ts)
                return None
        self._pending.append({"emb": emb, "count": 1, "last": ts,
                              "face": face, "frame": frame.copy()})
        return None

    def _confirm_new(self, p, ts):
        scene = self._scene(p["frame"], p["face"].box)
        crop = crop_face(p["frame"], p["face"].box)
        uid = self.unknowns.create(p["emb"], crop, scene, ts)
        self._log("new_unknown", uid, ts)
        self._alert(uid, scene, ts, first_time=True)
        return uid

    def _returning(self, uid, emb, frame, face, ts):
        rec = self.unknowns.records[uid]
        new_visit = ts - rec["last_seen_ts"] > VISIT_GAP_SECONDS
        self.unknowns.sighting(uid, emb, ts, new_visit)

        need_alert = ts - rec["last_alert_ts"] >= ALERT_COOLDOWN_SECONDS
        need_image = self.unknowns.wants_image(uid, ts)
        if need_alert or need_image:
            scene = self._scene(frame, face.box)
            if need_image:
                self.unknowns.save_image(uid, crop_face(frame, face.box), scene, ts)
            if need_alert:
                self._log("unknown_returned", uid, ts)
                self._alert(uid, scene, ts, first_time=False)
        if new_visit or need_image:
            self.unknowns.save_info(uid)

    def _note_known(self, name, frame, ts):
        if ts - self._known_seen.get(name, 0.0) < KNOWN_LOG_INTERVAL:
            return
        self._known_seen[name] = ts
        self._log("known_person", name, ts)
        if self.notify_known and self.notify:
            self.notify(f"{name} arrived", f"{name} was recognised at {self.camera_name}.", None)

    def _alert(self, uid, scene, ts, first_time):
        rec = self.unknowns.records[uid]
        rec["last_alert_ts"] = ts
        when = datetime.datetime.fromtimestamp(ts).strftime("%d %b %Y, %I:%M:%S %p")
        if first_time:
            title = "Unknown person detected"
            message = f"A new face ({uid}) was seen at {self.camera_name} on {when}."
        else:
            visits = rec["info"]["visits"]
            title = "Unknown person is back"
            message = (f"{uid} was seen again at {self.camera_name} on {when} "
                       f"(visit #{visits}).")
        if self.notify:
            self.notify(title, message, scene)

    @staticmethod
    def _scene(frame, box):
        img = frame.copy()
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(img, "UNKNOWN", (x, max(y - 8, 18)), FONT, 0.7, (0, 0, 255), 2)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, stamp, (10, img.shape[0] - 10), FONT, 0.5, (255, 255, 255), 2)
        return img

    def _log(self, event, identity, ts):
        new_file = not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["time", "event", "identity"])
            writer.writerow([datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds"),
                             event, identity])


def build_face_monitor(notify=None, camera_name="Security Camera", notify_known=False):
    """Create everything needed for face recognition in one call."""
    engine = FaceEngine()
    known = KnownFaces()
    known.load(engine)
    return FaceMonitor(engine, known, UnknownStore(), notify=notify,
                       camera_name=camera_name, notify_known=notify_known)


def draw_faces(frame, results):
    """Known people get a blue box + name, strangers get a red box."""
    for r in results:
        x, y, w, h = r.box
        color = (0, 0, 255) if r.unknown else (255, 140, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(r.label, FONT, 0.55, 2)
        top = max(y - th - 10, 0)
        cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + 10), color, -1)
        cv2.putText(frame, r.label, (x + 4, top + th + 3), FONT, 0.55, (255, 255, 255), 2)