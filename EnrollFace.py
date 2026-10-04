import argparse
import json
import os
import time

import cv2

from face_tools import (IMAGE_EXTS, KNOWN_DIR, UNKNOWN_DIR, FaceEngine,
                        promote_unknown, safe_name)

PROMPTS = [
    "Look straight at the camera",
    "Turn your head slightly LEFT",
    "Turn your head slightly RIGHT",
    "Tilt your chin up a little",
    "Tilt your chin down a little",
    "Look straight again (smile!)",
]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def enroll(name, camera_index, samples):
    name = safe_name(name)
    folder = os.path.join(KNOWN_DIR, name)
    os.makedirs(folder, exist_ok=True)
    start_number = len([f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS)])

    engine = FaceEngine()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    saved, last_save = 0, 0.0
    print(f"Enrolling '{name}'. Follow the on-screen prompts. Press 'q' to stop early.")
    try:
        while saved < samples:
            ok, frame = cap.read()
            if not ok:
                break
            faces = [f for f in engine.detect(frame) if f.score >= 0.9 and f.box[2] >= 100]
            view = frame.copy()

            if len(faces) == 1:
                x, y, w, h = faces[0].box
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 200, 0), 2)
                status, color = PROMPTS[saved % len(PROMPTS)], (0, 255, 0)
                if time.time() - last_save > 1.0:        # one photo per second
                    path = os.path.join(folder, f"{name}_{start_number + saved + 1:02d}.jpg")
                    cv2.imwrite(path, frame)             # save the clean frame, not the annotated one
                    saved += 1
                    last_save = time.time()
            elif len(faces) > 1:
                status, color = "Only ONE person in view, please", (0, 165, 255)
            else:
                status, color = "No clear face - move closer / add light", (0, 0, 255)

            cv2.putText(view, status, (10, 30), FONT, 0.7, color, 2)
            cv2.putText(view, f"Saved {saved}/{samples}", (10, 60), FONT, 0.6, (255, 255, 255), 2)
            cv2.imshow("Enroll face", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Done: {saved} photo(s) saved in {folder}")


def list_people():
    print("\nKNOWN PEOPLE")
    if os.path.isdir(KNOWN_DIR):
        for name in sorted(os.listdir(KNOWN_DIR)):
            folder = os.path.join(KNOWN_DIR, name)
            if os.path.isdir(folder):
                photos = len([f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS)])
                print(f"  {name}: {photos} photo(s)")
    print("\nUNKNOWN VISITORS")
    found = False
    if os.path.isdir(UNKNOWN_DIR):
        for d in sorted(os.listdir(UNKNOWN_DIR)):
            info_path = os.path.join(UNKNOWN_DIR, d, "info.json")
            if d.startswith("unknown_") and os.path.exists(info_path):
                with open(info_path) as f:
                    info = json.load(f)
                print(f"  {d}: first seen {info['first_seen']}, last seen {info['last_seen']}, "
                      f"{info['visits']} visit(s), {info['images']} photo(s)")
                found = True
    if not found:
        print("  none")


def main():
    parser = argparse.ArgumentParser(description="Manage known faces for the security camera")
    parser.add_argument("--name", help="person's name (for enrolling or promoting)")
    parser.add_argument("--samples", type=int, default=10, help="photos to take (default 10)")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--list", action="store_true", help="list known people and unknown visitors")
    parser.add_argument("--promote", metavar="UNKNOWN_ID", help="e.g. unknown_0003")
    args = parser.parse_args()

    if args.list:
        list_people()
    elif args.promote:
        if not args.name:
            parser.error("--promote needs --name")
        name = promote_unknown(args.promote, args.name)
        print(f"{args.promote} is now registered as '{name}'. "
              "Press 'r' in the camera window (or restart) to apply.")
    elif args.name:
        enroll(args.name, args.camera, args.samples)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()