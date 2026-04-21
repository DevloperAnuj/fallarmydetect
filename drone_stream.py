"""KY UFO (com.cooingdv.kyufo) drone video receiver.

Protocol reverse-engineered from the official app:
  - Video:    rtsp://192.168.1.1:7070/webcam  (served by the drone's ijkplayer)
  - Heartbeat UDP 192.168.1.1:7099, payload {0x01, 0x01} every 1 second.
    The drone needs this heartbeat to keep the stream alive; without it, RTSP
    frames eventually stall.
  - Camera switch: UDP 7099, payload {0x06, 0x01} / {0x06, 0x02}
  - Screen flip:   UDP 7099, payload {0x09, 0x01} / {0x09, 0x02}

Keep your PC connected to the drone's WiFi AP before calling connect().
"""

import os
import socket
import threading
import time

import cv2
import numpy as np

DRONE_IP = "192.168.1.1"
RTSP_PORT = 7070
HEARTBEAT_PORT = 7099
HEARTBEAT_INTERVAL = 1.0  # seconds
HEARTBEAT_PAYLOAD = bytes([0x01, 0x01])

# Force TCP transport: the drone rejects RTP-over-UDP (SETUP 461).
# OpenCV uses FFmpeg under the hood and reads this env var at capture open.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def rtsp_url(drone_ip: str = DRONE_IP, port: int = RTSP_PORT) -> str:
    return f"rtsp://{drone_ip}:{port}/webcam"


class DroneStream:
    """RTSP video receiver for KY UFO drones with required UDP heartbeat."""

    def __init__(self, drone_ip: str = DRONE_IP):
        self.drone_ip = drone_ip
        self._hb_sock: socket.socket | None = None
        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._hb_thread: threading.Thread | None = None
        self._cap_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self.frames_received = 0

    def connect(self) -> bool:
        """Open heartbeat socket and RTSP capture."""
        try:
            self._hb_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Send an initial heartbeat so the drone wakes its video service
            # before we try to SETUP the RTSP session.
            self._hb_sock.sendto(HEARTBEAT_PAYLOAD,
                                 (self.drone_ip, HEARTBEAT_PORT))
        except OSError:
            return False

        self._cap = cv2.VideoCapture(rtsp_url(self.drone_ip), cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            self._hb_sock.close()
            self._hb_sock = None
            return False
        return True

    def _heartbeat_loop(self):
        while self._running and self._hb_sock is not None:
            try:
                self._hb_sock.sendto(HEARTBEAT_PAYLOAD,
                                     (self.drone_ip, HEARTBEAT_PORT))
            except OSError:
                break
            time.sleep(HEARTBEAT_INTERVAL)

    def _capture_loop(self):
        while self._running and self._cap is not None:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                # RTSP hiccup — brief pause then try again
                time.sleep(0.05)
                continue
            with self._lock:
                self._latest_frame = frame
            self.frames_received += 1

    def start(self):
        if self._running:
            return
        self._running = True
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True)
        self._cap_thread = threading.Thread(
            target=self._capture_loop, daemon=True)
        self._hb_thread.start()
        self._cap_thread.start()

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy()
        return False, None

    def stop(self):
        self._running = False
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        if self._hb_sock is not None:
            try:
                self._hb_sock.close()
            except OSError:
                pass
            self._hb_sock = None

    def is_opened(self) -> bool:
        return self._running and self._cap is not None
