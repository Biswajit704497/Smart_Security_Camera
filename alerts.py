import os
import queue
import smtplib
import threading
import time
from email.message import EmailMessage

import cv2


class AlertManager:
    def __init__(self, camera_name="Security Camera", console=True, sound=True,
                 telegram=False, email=False, banner_seconds=6):
        self.camera_name = camera_name
        self.banner_seconds = banner_seconds
        self.banner_text = ""
        self.banner_until = 0.0

        self._tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
        self._mail_from = os.environ.get("ALERT_EMAIL_FROM", "")
        self._mail_pass = os.environ.get("ALERT_EMAIL_PASSWORD", "")
        self._mail_to = os.environ.get("ALERT_EMAIL_TO", "")
        self._smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self._smtp_port = int(os.environ.get("SMTP_PORT", "587"))

        self._channels = []
        if console:
            self._channels.append(("console", self._console))
        if sound:
            self._channels.append(("sound", self._sound))
        if telegram:
            if self._tg_token and self._tg_chat:
                self._channels.append(("telegram", self._telegram))
            else:
                print("[alerts] Telegram is ON but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID "
                      "are not set - Telegram alerts disabled.")
        if email:
            if self._mail_from and self._mail_pass and self._mail_to:
                self._channels.append(("email", self._email))
            else:
                print("[alerts] Email is ON but ALERT_EMAIL_FROM / ALERT_EMAIL_PASSWORD / "
                      "ALERT_EMAIL_TO are not set - email alerts disabled.")

        print("[alerts] Active channels: on-screen banner, " +
              ", ".join(name for name, _ in self._channels))

        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ API
    def send(self, title, message, image=None):
        """Raise an alert. `image` is an optional BGR frame (numpy array)."""
        self.banner_text = title
        self.banner_until = time.time() + self.banner_seconds

        jpeg = None
        if image is not None:
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                jpeg = buf.tobytes()
        self._queue.put((title, message, jpeg))

    def draw_banner(self, frame):
        """Draw the red ALERT banner on the preview/recording while it is active."""
        if time.time() > self.banner_until:
            return
        w = frame.shape[1]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 38), (w, 78), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, f"ALERT: {self.banner_text}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def close(self):
        """Let queued alerts finish sending, then stop the worker."""
        self._queue.put(None)
        self._worker.join(timeout=15)

    # ------------------------------------------------------------- internals
    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            title, message, jpeg = item
            for name, send in self._channels:
                try:
                    send(title, message, jpeg)
                except Exception as exc:  # never let a failed alert crash the camera
                    print(f"[alerts] {name} failed: {self._scrub(exc)}")

    def _scrub(self, exc):
        # Network errors can contain the bot token inside the URL - hide it
        text = str(exc)
        for secret in (self._tg_token, self._mail_pass):
            if secret:
                text = text.replace(secret, "***")
        return text

    def _console(self, title, message, jpeg):
        print(f"\n*** ALERT [{self.camera_name}] {title} ***\n    {message}\n")

    def _sound(self, title, message, jpeg):
        try:
            import winsound  # Windows only
            for _ in range(3):
                winsound.Beep(1200, 350)
        except ImportError:
            print("\a", end="", flush=True)  # terminal bell on Linux / macOS

    def _telegram(self, title, message, jpeg):
        import requests
        base = f"https://api.telegram.org/bot{self._tg_token}"
        text = f"\U0001F6A8 {title}\n[{self.camera_name}]\n{message}"
        if jpeg:
            resp = requests.post(
                f"{base}/sendPhoto",
                data={"chat_id": self._tg_chat, "caption": text[:1024]},
                files={"photo": ("alert.jpg", jpeg, "image/jpeg")},
                timeout=20,
            )
        else:
            resp = requests.post(
                f"{base}/sendMessage",
                data={"chat_id": self._tg_chat, "text": text},
                timeout=20,
            )
        resp.raise_for_status()

    def _email(self, title, message, jpeg):
        msg = EmailMessage()
        msg["Subject"] = f"[{self.camera_name}] {title}"
        msg["From"] = self._mail_from
        msg["To"] = self._mail_to
        msg.set_content(f"{title}\n\n{message}\n")
        if jpeg:
            msg.add_attachment(jpeg, maintype="image", subtype="jpeg", filename="alert.jpg")
        with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=20) as server:
            server.starttls()
            server.login(self._mail_from, self._mail_pass)
            server.send_message(msg)