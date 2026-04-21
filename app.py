import atexit
import json
import signal
import threading
import time
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from drone_stream import DroneStream


def _cleanup():
    """Clean up background resources on shutdown."""
    if "drone" in st.session_state and st.session_state.drone is not None:
        st.session_state.drone.stop()
        st.session_state.drone = None


atexit.register(_cleanup)


def _signal_handler(sig, frame):
    _cleanup()
    raise SystemExit(0)


# Catch Ctrl+C — ignore errors if signal isn't available (e.g. non-main thread)
try:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
except (OSError, ValueError):
    pass

DEFAULT_MODEL_PATH = Path("models/mobilenetv2_binary_run/best.keras")
DEFAULT_LABELS_PATH = Path("models/mobilenetv2_binary_run/labels.json")
DEFAULT_IMAGE_SIZE = 224


# Must match the loss class used during training for model deserialization
@tf.keras.utils.register_keras_serializable(package="FAW")
class SparseFocalCrossEntropy(tf.keras.losses.Loss):

    def __init__(self, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        true_probs = tf.gather(y_pred, y_true, batch_dims=1)
        focal_weight = tf.pow(1.0 - true_probs, self.gamma)
        loss = -focal_weight * tf.math.log(true_probs)
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config["gamma"] = self.gamma
        return config


@st.cache_resource
def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def load_labels(labels_path: Path) -> list[str]:
    if labels_path.exists():
        return json.loads(labels_path.read_text(encoding="utf-8"))
    return []


def preprocess_image(image: Image.Image, image_size: int) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    # Feed raw [0, 255] pixels — model's Rescaling layer handles
    # MobileNetV2 preprocessing internally (scale=1/127.5, offset=-1).
    return arr


def classify_frame(model, image: Image.Image, image_size: int,
                   labels: list[str], infected_idx, threshold: float):
    """Run inference on a PIL image and return (label, confidence, is_infected)."""
    batch = preprocess_image(image, image_size)
    probs = model.predict(batch, verbose=0)[0]

    active_labels = labels if labels else [
        f"Class_{i}" for i in range(len(probs))]

    if infected_idx is not None and float(probs[infected_idx]) >= threshold:
        best_idx = infected_idx
    else:
        best_idx = int(np.argmax(probs))

    return active_labels[best_idx], float(probs[best_idx]), active_labels[best_idx] == "infected"


def draw_overlay(img: np.ndarray, label: str, confidence: float,
                 is_infected: bool) -> np.ndarray:
    """Draw prediction overlay on a BGR numpy frame."""
    if not label:
        return img

    label_text = f"{label.upper()} ({confidence * 100:.1f}%)"

    if is_infected:
        color = (0, 0, 255)  # Red in BGR
        bg_color = (0, 0, 180)
    else:
        color = (0, 200, 0)  # Green in BGR
        bg_color = (0, 140, 0)

    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(
        label_text, font, font_scale, thickness
    )

    # Background rectangle at top
    cv2.rectangle(img, (0, 0), (text_w + 20, text_h +
                  baseline + 20), bg_color, -1)
    # Label text
    cv2.putText(
        img, label_text, (10, text_h + 10),
        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )
    # Colored border
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 3)

    return img


class FAWVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.model = None
        self.labels = []
        self.infected_idx = None
        self.image_size = DEFAULT_IMAGE_SIZE
        self.threshold = 0.35
        self.frame_count = 0
        self.lock = threading.Lock()
        self.last_label = ""
        self.last_confidence = 0.0
        self.last_is_infected = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        self.frame_count += 1
        # Process every 3rd frame to reduce CPU load
        if self.frame_count % 3 == 0 and self.model is not None:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            with self.lock:
                self.last_label, self.last_confidence, self.last_is_infected = (
                    classify_frame(
                        self.model, pil_img, self.image_size,
                        self.labels, self.infected_idx, self.threshold,
                    )
                )

        img = draw_overlay(img, self.last_label,
                           self.last_confidence, self.last_is_infected)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def run_ip_camera(model, labels, infected_idx, image_size, threshold, stream_url):
    """Read frames from an IP camera stream and display with detection overlay."""
    st.info(f"Connecting to: `{stream_url}`")

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        st.error(
            f"Cannot connect to stream: {stream_url}\n\n"
            "Make sure the URL is correct and the camera is on the same network.\n\n"
            "**Common URLs:**\n"
            "- ESP32-CAM: `http://192.168.1.x:81/stream`\n"
            "- IP Webcam (Android): `http://192.168.1.x:8080/video`\n"
            "- DroidCam: `http://192.168.1.x:4747/video`"
        )
        return

    st.success("Connected! Streaming...")
    frame_placeholder = st.empty()
    stop_btn = st.button("Stop Stream")

    frame_count = 0
    last_label, last_confidence, last_is_infected = "", 0.0, False

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            st.warning("Lost connection to stream. Retrying...")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(stream_url)
            continue

        frame_count += 1
        # Classify every 3rd frame
        if frame_count % 3 == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            last_label, last_confidence, last_is_infected = classify_frame(
                model, pil_img, image_size, labels, infected_idx, threshold,
            )

        frame = draw_overlay(
            frame, last_label, last_confidence, last_is_infected)

        # Convert BGR to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(
            frame_rgb, channels="RGB", width="stretch")

    cap.release()


def run_drone_camera(model, labels, infected_idx, image_size, threshold, drone_ip):
    """Read frames from a KY UFO / E58-style drone via UDP and display with detection."""

    # Use session state to persist the drone stream across Streamlit reruns
    if "drone" not in st.session_state:
        st.session_state.drone = None

    status_area = st.empty()
    frame_placeholder = st.empty()
    col1, col2 = st.columns(2)
    start_btn = col1.button("Start Stream")
    stop_btn = col2.button("Stop Stream")

    if stop_btn and st.session_state.drone is not None:
        st.session_state.drone.stop()
        st.session_state.drone = None
        status_area.info("Stream stopped.")
        return

    if start_btn or st.session_state.drone is not None:
        # Create new connection if needed
        if st.session_state.drone is None or not st.session_state.drone.is_opened():
            status_area.info(
                f"Connecting to drone at `{drone_ip}`...\n\n"
                "Make sure you are connected to the drone's WiFi."
            )
            drone = DroneStream(drone_ip=drone_ip)
            if not drone.connect():
                status_area.error(
                    f"Cannot connect to drone at {drone_ip} "
                    f"(RTSP :7070 / UDP heartbeat :7099)\n\n"
                    "**Troubleshooting:**\n"
                    "1. Connect your PC to the drone's WiFi first\n"
                    "2. Confirm the IP — default is `192.168.1.1`\n"
                    "3. Make sure the drone is powered on and the camera is active"
                )
                return
            drone.start()
            st.session_state.drone = drone

        drone = st.session_state.drone
        status_area.warning("Waiting for drone video stream...")

        frame_count = 0
        last_label, last_confidence, last_is_infected = "", 0.0, False
        no_frame_count = 0
        max_wait = 100  # ~5 seconds of no frames before showing help

        while drone.is_opened():
            ret, frame = drone.read()
            if not ret:
                no_frame_count += 1
                if no_frame_count == max_wait:
                    status_area.error(
                        "No video frames received from drone.\n\n"
                        "**Troubleshooting:**\n"
                        "1. Is your PC connected to the drone's WiFi?\n"
                        "2. Is the drone powered on with camera active?\n"
                        f"3. Try opening `rtsp://{drone_ip}:7070/webcam` "
                        "in VLC to confirm the stream is up."
                    )
                time.sleep(0.05)
                continue

            # Got frames — update status on first frame
            if no_frame_count > 0 or frame_count == 0:
                status_area.success(
                    f"Receiving drone video! (frames: {drone.frames_received})"
                )
                no_frame_count = 0

            frame_count += 1
            if frame_count % 3 == 0:
                pil_img = Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                last_label, last_confidence, last_is_infected = classify_frame(
                    model, pil_img, image_size, labels, infected_idx, threshold,
                )

            frame = draw_overlay(
                frame, last_label, last_confidence, last_is_infected)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(
                frame_rgb, channels="RGB", width="stretch")

        drone.stop()
        st.session_state.drone = None
    else:
        st.info(
            "**Instructions:**\n"
            "1. Power on the drone\n"
            "2. Connect your PC to the drone's WiFi "
            "(SSID like `WiFi-720P-XXXX` or `KY-XXXX`)\n"
            "3. Set the drone IP in the sidebar\n"
            "4. Click **Start Stream**"
        )


def main():
    st.set_page_config(page_title="FAW Detector",
                       page_icon="🌽", layout="wide",
                       initial_sidebar_state="collapsed")

    # Let the video feed fill the viewport width, especially on mobile.
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1rem; padding-left: 0.5rem;
                             padding-right: 0.5rem; max-width: 100%; }
          video { width: 100% !important; height: auto !important;
                  max-height: 80vh; border-radius: 8px; }
          div[data-testid="stImage"] img { width: 100% !important;
                                           height: auto !important; }
          iframe[title^="streamlit_webrtc"] { width: 100% !important;
                                              min-height: 70vh !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Fall Army Worm Detection")
    st.write("Real-time corn leaf infection detection.")

    # Sidebar: model config
    with st.sidebar:
        st.header("Model Settings")
        model_path_input = st.text_input("Model path", str(DEFAULT_MODEL_PATH))
        labels_path_input = st.text_input(
            "Labels path", str(DEFAULT_LABELS_PATH))
        image_size = st.number_input(
            "Image size", min_value=96, max_value=512, value=DEFAULT_IMAGE_SIZE, step=16)
        infected_threshold = st.slider(
            "Infected detection threshold",
            min_value=0.10, max_value=0.90, value=0.35, step=0.05,
            help="If the infected probability exceeds this, classify as infected. "
                 "Lower = catches more infections (higher recall).")

        st.header("Camera Source")
        camera_source = st.radio(
            "Select camera",
            [
                "Browser Webcam",
                "IP Camera (ESP32-CAM / Mobile)",
                "KY UFO Drone",
            ],
            help="**Browser Webcam**: uses your device's camera via WebRTC. "
                 "Works on desktop and mobile browsers.\n\n"
                 "**IP Camera**: connects to an MJPEG/RTSP stream URL from "
                 "ESP32-CAM, IP Webcam app, DroidCam, etc.\n\n"
                 "**KY UFO Drone**: connects directly to KY UFO / E58-style "
                 "WiFi drones via UDP. Connect to the drone's WiFi first.",
        )

        stream_url = ""
        drone_ip = ""
        if camera_source == "IP Camera (ESP32-CAM / Mobile)":
            stream_url = st.text_input(
                "Stream URL",
                "http://192.168.1.100:81/stream",
                help="ESP32-CAM: `http://<IP>:81/stream`\n\n"
                     "IP Webcam (Android): `http://<IP>:8080/video`\n\n"
                     "DroidCam: `http://<IP>:4747/video`",
            )
        elif camera_source == "KY UFO Drone":
            drone_ip = st.text_input(
                "Drone IP",
                "192.168.1.1",
                help="Connect to the drone's WiFi first, then enter its IP.\n\n"
                     "Default `192.168.1.1` matches the KY UFO app "
                     "(com.cooingdv.kyufo). Video is RTSP on :7070/webcam.",
            )

    model_path = Path(model_path_input)
    labels_path = Path(labels_path_input)

    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        st.stop()

    labels = load_labels(labels_path)
    model = load_model(str(model_path))

    if labels:
        with st.sidebar:
            st.write(f"**Classes:** {', '.join(labels)}")

    infected_idx = None
    if "infected" in labels:
        infected_idx = labels.index("infected")

    # --- Camera modes ---
    if camera_source == "Browser Webcam":
        ctx = webrtc_streamer(
            key="faw-detector",
            video_processor_factory=FAWVideoProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                    "facingMode": {"ideal": "environment"},
                },
                "audio": False,
            },
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.model = model
            ctx.video_processor.labels = labels
            ctx.video_processor.infected_idx = infected_idx
            ctx.video_processor.image_size = int(image_size)
            ctx.video_processor.threshold = infected_threshold

        st.info("Click **START** to begin detection. Allow camera access when prompted.\n\n"
                "**Mobile?** Open this app on your phone's browser "
                "(same WiFi) to use the phone camera.")

    elif camera_source == "IP Camera (ESP32-CAM / Mobile)":
        if stream_url:
            run_ip_camera(
                model, labels, infected_idx,
                int(image_size), infected_threshold, stream_url,
            )
        else:
            st.warning("Enter a stream URL in the sidebar to start.")

    else:  # KY UFO Drone
        if drone_ip:
            run_drone_camera(
                model, labels, infected_idx,
                int(image_size), infected_threshold, drone_ip,
            )
        else:
            st.warning("Enter the drone IP in the sidebar to start.")


if __name__ == "__main__":
    main()
