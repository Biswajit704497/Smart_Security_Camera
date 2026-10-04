# Smart Security Camera

A Python webcam security camera that detects motion, records video clips, recognises faces, and sends an alert when an **unknown person** appears.

## Features

- Motion detection with tight bounding boxes (OpenCV MOG2 background subtraction)
- Clips recorded at real-time speed, with 2 seconds of footage from before the motion
- Face recognition: known people are labelled, strangers are saved in their own folder
- Remembers returning strangers ("visit #3") and avoids duplicate alerts
- Alerts via on-screen banner, console + sound, Telegram (with photo) or email

## Setup

Requires Python 3.8+ and a webcam.

```bash
git clone https://github.com/Biswajit704497/Smart_Security_Camera.git
cd Smart_Security_Camera
python -m venv myenv
myenv\Scripts\activate          # Linux/macOS: source myenv/bin/activate
pip install -r requirements.txt
```

The face models (~40 MB) download automatically on the first run into `models/`.

## Usage

```bash
python EnrollFace.py --name Alice                        # register a known person via webcam
python EnrollFace.py --list                              # show known people and unknown visitors
python EnrollFace.py --promote unknown_0003 --name Bob   # make a stranger a known person
python model.py                                          # start the camera
```

Camera window keys: `q` quit, `m` show motion mask, `r` reload known faces.

You can also add a person by putting clear photos of them in `known_faces/<Name>/`.

## Alerts

On-screen, console and sound alerts work out of the box. For phone alerts, set your credentials as environment variables (never put them in the code):

```powershell
# Telegram (create a bot with @BotFather) - Windows PowerShell
$env:TELEGRAM_BOT_TOKEN = "your-bot-token"
$env:TELEGRAM_CHAT_ID   = "your-chat-id"
# Linux/macOS: export TELEGRAM_BOT_TOKEN="..."
```

Then set `ALERT_TELEGRAM = True` in `model.py`. For email, set `ALERT_EMAIL_FROM`, `ALERT_EMAIL_PASSWORD` (an app password) and `ALERT_EMAIL_TO`, and set `ALERT_EMAIL = True`.

## Project structure

| File | Purpose |
|---|---|
| `model.py` | Main program: motion detection, recording, ties everything together |
| `face_tools.py` | Face detection/recognition, known and unknown people storage |
| `alerts.py` | Alert channels (banner, sound, Telegram, email) |
| `EnrollFace.py` | Command-line tool to manage known people |

Folders created at runtime (not in the repo): `recordings/`, `motion_snapshots/`, `known_faces/`, `unknown_faces/`, `models/`.

## Configuration

Camera, sensitivity, recording and alert switches are at the top of `model.py`. Face-matching thresholds and cooldowns are at the top of `face_tools.py`.

## Privacy

Face images and recordings stay on your computer and are excluded from Git via `.gitignore`. Use this only where you have the right to record, and consider telling regular visitors that a camera is running.

## License

See [LICENSE](LICENSE).