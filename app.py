import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

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


def main():
    st.set_page_config(page_title="FAW Detector", page_icon="🌽", layout="centered")
    st.title("Fall Army Worm Detection")
    st.write("Upload a corn leaf image to check for worm infection.")

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

    # Find infected class index
    infected_idx = None
    if "infected" in labels:
        infected_idx = labels.index("infected")

    # Upload
    uploaded = st.file_uploader("Upload crop image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded image", use_container_width=True)

        batch = preprocess_image(image, int(image_size))
        probs = model.predict(batch, verbose=0)[0]

        if not labels:
            labels = [f"Class_{i}" for i in range(len(probs))]

        # Threshold-based decision for infected class
        if infected_idx is not None and float(probs[infected_idx]) >= infected_threshold:
            best_idx = infected_idx
        else:
            best_idx = int(np.argmax(probs))

        predicted_label = labels[best_idx]
        confidence = float(probs[best_idx])

        # Color-coded result
        st.subheader("Prediction")
        if predicted_label == "infected":
            st.error(f"**INFECTED** (confidence: {confidence * 100:.1f}%)")
        else:
            st.success(f"**NON-INFECTED** (confidence: {confidence * 100:.1f}%)")

        # Show all class probabilities
        if infected_idx is not None:
            st.write(f"Infected probability: **{float(probs[infected_idx]) * 100:.1f}%**")

        st.subheader("Class Probabilities")
        for i, label in enumerate(labels):
            st.progress(float(probs[i]), text=f"{label}: {float(probs[i]) * 100:.2f}%")


if __name__ == "__main__":
    main()
