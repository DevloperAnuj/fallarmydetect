# Fall Army Worm Detection

Real-time binary classifier for corn leaves — **infected** (fall army worm) vs **non_infected** — running on webcam, IP camera, or a KY UFO / E58-class WiFi drone feed. MobileNetV2 backbone, Streamlit UI.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501, pick a camera source in the sidebar, go.

For mobile access (phone camera, or viewing the feed on a phone), see [Remote access](#remote-access).

## Camera sources

The sidebar exposes three modes:

| Mode | Use when |
|---|---|
| **Browser Webcam** | Running on a laptop with a camera, or a phone browser (via [Remote access](#remote-access)) |
| **IP Camera (ESP32-CAM / Mobile)** | ESP32-CAM MJPEG, IP Webcam (Android), DroidCam, any MJPEG/RTSP URL |
| **KY UFO Drone** | `com.cooingdv.kyufo`-family WiFi drones. PC must be joined to the drone's AP. |

### KY UFO drone details

Connect your PC to the drone's WiFi first. Defaults in the sidebar (`192.168.1.1`) match the stock app. Under the hood, [drone_stream.py](drone_stream.py) opens `rtsp://192.168.1.1:7070/webcam` and keeps the stream alive with a 1 Hz UDP heartbeat `{0x01, 0x01}` → `192.168.1.1:7099` — the same handshake the app performs. See [Reverse-engineering notes](#reverse-engineering-notes) below.

## Dataset

Source images live in `dataset/` with subfolders mapped to the binary label:

```
dataset/
  Blight/            → non_infected
  Common_Rust/       → non_infected
  Gray_Leaf_Spot/    → non_infected
  Healthy/           → non_infected
  worm/              → infected
```

## Training (Google Colab)

Local training is optional — Colab's free T4 GPU is enough.

1. Upload the `dataset/` folder to Google Drive.
2. Open `notebooks/train_colab.ipynb` in Colab → **Runtime → Change runtime type → GPU (T4)**.
3. Edit `DRIVE_DATASET_PATH` in the config cell, **Run All**.
4. Download `best.keras` + `labels.json` from Colab into `models/mobilenetv2_binary_run/`.

The notebook splits with minority oversampling, runs two-phase training (frozen backbone → fine-tune top 30 layers), and evaluates with confusion matrix, ROC, and threshold sweep.

## Local scripts (optional)

```bash
# Split dataset locally with oversampling
python scripts/split_dataset.py \
    --dataset-dir dataset \
    --output-dir dataset_split_binary \
    --binary-mode --oversample-minority --force

# Evaluate a trained model
python scripts/evaluate_model.py \
    --data-dir dataset_split_binary/test \
    --model-path models/mobilenetv2_binary_run/best.keras \
    --threshold 0.35
```

## Remote access

Mobile browsers require HTTPS for `navigator.mediaDevices` — plain LAN `http://` won't expose the camera. [run.py](run.py) wraps this.

```bash
python run.py --lan                    # LAN-only HTTPS with self-signed cert
python run.py --token <ngrok_token>    # Public HTTPS via authenticated ngrok
python run.py                          # Public HTTPS via free ngrok (random URL)
```

`--lan` generates a self-signed cert in `.certs/` for `localhost` and your current LAN IP. Your phone will warn about the cert — tap **Advanced → Proceed**.

## Reverse-engineering notes

The drone integration was built by decompiling the official KY UFO app (`com.cooingdv.kyufo`) with [jadx](https://github.com/skylot/jadx). The relevant classes:

- `com.cooingdv.kyufo.socket.Config` — hardcodes `PREVIEW_ADDRESS = "rtsp://192.168.1.1:7070/webcam"`.
- `com.cooingdv.kyufo.socket.SocketClient` — hands that URL to ijkplayer; no custom video protocol.
- `com.cooingdv.kyufo.socket.UdpClient` — `HeartBeatTask` fires `{0x01, 0x01}` to UDP `:7099` every second. Without it, the RTSP stream stalls.
- Camera-switch commands: `{0x06, 0x01}` / `{0x06, 0x02}`. Screen-flip: `{0x09, 0x01}` / `{0x09, 0x02}`. Not wired into the app yet — [drone_stream.py](drone_stream.py) exposes the UDP socket path if you want to add them.

## File structure

```
app.py                               Streamlit app (webcam / IP camera / drone)
drone_stream.py                      KY UFO RTSP + heartbeat receiver
run.py                               LAN-HTTPS or ngrok launcher
requirements.txt
notebooks/
  train_colab.ipynb                  Training notebook (Colab)
scripts/
  split_dataset.py                   Split + oversample
  evaluate_model.py                  Evaluate with threshold sweep
models/
  mobilenetv2_binary_run/
    best.keras
    labels.json
artifacts/                           Evaluation outputs
dataset/                             Source images
```

## Bias fixes applied during training

| Issue | Fix |
|---|---|
| 5:1 class imbalance | Oversampling + class weights |
| Wrong preprocessing at inference | Model handles MobileNetV2 preprocessing internally via Rescaling layer |
| Accuracy-based checkpointing | Monitors val_loss |
| Frozen backbone | Fine-tunes top 30 layers |
| Standard cross-entropy | Focal loss (γ=2.0) |
| Weak augmentation | Flip, rotation, zoom, brightness, contrast, translation |
| Pure argmax | Threshold-based infected detection (sidebar slider) |
