import json
import threading
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

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
            batch = preprocess_image(pil_img, self.image_size)

            with self.lock:
                probs = self.model.predict(batch, verbose=0)[0]

            labels = self.labels if self.labels else [
                f"Class_{i}" for i in range(len(probs))
            ]

            if (
                self.infected_idx is not None
                and float(probs[self.infected_idx]) >= self.threshold
            ):
                best_idx = self.infected_idx
            else:
                best_idx = int(np.argmax(probs))

            self.last_label = labels[best_idx]
            self.last_confidence = float(probs[best_idx])
            self.last_is_infected = self.last_label == "infected"

        # Draw overlay on every frame (using last prediction)
        if self.last_label:
            label_text = f"{self.last_label.upper()} ({self.last_confidence * 100:.1f}%)"

            if self.last_is_infected:
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

            # Draw background rectangle at top
            cv2.rectangle(img, (0, 0), (text_w + 20, text_h + baseline + 20), bg_color, -1)
            # Draw text
            cv2.putText(
                img, label_text, (10, text_h + 10),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
            )

            # Draw border around frame
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.set_page_config(page_title="FAW Detector", page_icon="🌽", layout="centered")
    st.title("Fall Army Worm Detection")
    st.write("Point your camera at a corn leaf for real-time infection detection.")

    # Sidebar: model config
    with st.sidebar:
        st.header("Model Settings")
        model_path_input = st.text_input("Model path", str(DEFAULT_MODEL_PATH))
        labels_path_input = st.text_input("Labels path", str(DEFAULT_LABELS_PATH))
        image_size = st.number_input(
            "Image size", min_value=96, max_value=512, value=DEFAULT_IMAGE_SIZE, step=16)
        infected_threshold = st.slider(
            "Infected detection threshold",
            min_value=0.10, max_value=0.90, value=0.35, step=0.05,
            help="If the infected probability exceeds this, classify as infected. "
                 "Lower = catches more infections (higher recall).")

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

    # Real-time camera detection
    ctx = webrtc_streamer(
        key="faw-detector",
        video_processor_factory=FAWVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Pass model and settings to the video processor
    if ctx.video_processor:
        ctx.video_processor.model = model
        ctx.video_processor.labels = labels
        ctx.video_processor.infected_idx = infected_idx
        ctx.video_processor.image_size = int(image_size)
        ctx.video_processor.threshold = infected_threshold

    st.info("Click **START** to begin real-time detection. Allow camera access when prompted.")


if __name__ == "__main__":
    main()
