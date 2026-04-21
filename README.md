# Fall Army Worm Detection + Auto-spray

Real-time corn-leaf infection detector with **auto-triggered pump** for targeted pesticide spray. Detection runs on webcam, an IP/Android camera, or the feed from a KY UFO WiFi drone. When sustained infection is seen, the app sends a Bluetooth command to an ESP32 that fires the pump.

| Detection input | → | Classifier | → | Bluetooth | → | ESP32 relay | → | Pump |

Stack: MobileNetV2 binary classifier · Streamlit UI · OpenCV / FFmpeg RTSP · `pyserial` over Bluetooth SPP.

---

## Table of contents

1. [Quick start](#quick-start)
2. [User manual (end-to-end)](#user-manual-end-to-end)
3. [Hardware setup](#hardware-setup)
4. [Camera sources](#camera-sources)
5. [Training the model](#training-the-model)
6. [Reverse-engineering notes (KY UFO drone)](#reverse-engineering-notes-ky-ufo-drone)
7. [File structure](#file-structure)
8. [Bias fixes applied during training](#bias-fixes-applied-during-training)

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate                  # Windows (cmd / PowerShell)
# source .venv/bin/activate             # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Pick a camera source. Done.

Every time you open a new terminal, re-activate the venv with `.venv\Scripts\activate` before running `streamlit`, `python run.py`, etc. For the full workflow including the ESP32/pump, drone, and remote-access steps, keep reading.

---

## User manual (end-to-end)

### 1. Prerequisites

- **Windows 10 or 11** (developed on LTSC 2019; `pyserial` over Bluetooth SPP expects Windows's virtual COM port).
- **Python 3.10+** with a venv (recommend using `.venv` at the project root).
- **Arduino IDE** (or `arduino-cli`) with the ESP32 board package installed.
- **Bluetooth** enabled on the PC.
- **Trained model** at `models/mobilenetv2_binary_run/best.keras` + `labels.json`. If you don't have one yet, see [Training the model](#training-the-model).

### 2. Install Python dependencies into a virtual environment

**First-time setup** from the project root:

```bash
python -m venv .venv
.venv\Scripts\activate                  # Windows (cmd / PowerShell)
# source .venv/bin/activate             # macOS / Linux
pip install -r requirements.txt
```

**Every subsequent terminal session**, re-activate before running any command below:

```bash
.venv\Scripts\activate                  # Windows
```

Your shell prompt should now start with `(.venv)`. If you forget activation, `streamlit` or `python` will resolve to your system Python and miss the dependencies.

Alternative without activation: prefix every command with the venv's interpreter, e.g. `.venv\Scripts\python -m streamlit run app.py`.

### 3. Flash the ESP32

Open [esp32_water_pump_sro4.ino](esp32_water_pump_sro4.ino) in Arduino IDE. Select your ESP32 board + port, then **Upload**.

Serial monitor at 115200 baud should print:

```
[BT] Started. Device name: FAW-Drone
[BT] Waiting for PC to connect...
```

The sketch advertises over Bluetooth Classic (SPP) as `FAW-Drone` and controls the relay on `GPIO 12`. See [Hardware setup](#hardware-setup) for wiring.

### 4. Pair the ESP32 with the PC (one time)

1. Windows **Settings → Bluetooth & devices → Add device → Bluetooth**.
2. Pick **FAW-Drone**. Pair (no PIN required, or `0000` / `1234` if prompted).
3. **Device Manager → Ports (COM & LPT)** — Windows creates two `Standard Serial over Bluetooth link` entries. The **outgoing** one (hwid contains the ESP32's MAC address) is the one you'll use. Note its `COMx` number.

Note: COM numbers differ from PC to PC. The app's sidebar dropdown auto-lists whatever ports are paired — you don't need to hardcode anything.

### 5. (If using the drone) Connect PC to the drone's WiFi

1. Power on the KY UFO drone — it creates its own WiFi AP.
2. On the PC, join the drone's WiFi (SSID usually starts with `WiFi-720P-` or `KY-`).
3. **The drone's AP allows only one client at a time.** Your phone must be disconnected. The PC replaces the phone as the video receiver.

### 6. Launch the app

With the venv active (`(.venv)` in your prompt):

```bash
streamlit run app.py
```

Browser opens at http://localhost:8501.

### 7. Configure in the sidebar

| Section | What to set |
|---|---|
| **Model Settings** | Leave defaults unless you trained a model elsewhere. |
| **Infected detection threshold** | Probability above which a frame is called *infected*. Lower = more sensitive. Start at `0.35`. |
| **Camera Source** | `Browser Webcam`, `IP Camera`, or `KY UFO Drone`. |
| **Drone IP** | `192.168.1.1` for KY UFO — do not change unless your drone uses a different IP. |
| **Pump (ESP32 — Bluetooth)** | Tick **Auto-fire pump on detection**. From the dropdown, pick the COM port tagged **(BT)** — usually the one whose hwid contains your ESP32's MAC. Hit 🔄 if the list is stale. |
| **Spray duration (ms)** | How long the pump runs per trigger. Default `5000` ms. |
| **Consecutive infected frames** | Debounce — avoids firing on a single flicker frame. Default `3`. |
| **Cooldown (s)** | Minimum gap between fires. Default `10` s. |

Sidebar status line shows: `🔵 connected on COM5` when the Bluetooth link is live. Click **Test fire** — relay should click and you should hear the pump for the configured duration.

### 8. Start detection

- **Browser Webcam** — click **START** in the WebRTC widget, allow camera access.
- **IP Camera** — enter the stream URL (e.g. `http://192.168.1.100:8080/video` for the Android IP Webcam app) and the feed starts automatically.
- **KY UFO Drone** — click **Start Stream**. First frame should appear within a few seconds.

Every 3rd frame is classified. When you hold an infected leaf (or point the camera at the field with worms) for `consec_frames` consecutive hits, the status flips to `💧 fired (5000 ms)` and the pump runs. Next fire is blocked for the cooldown period so one detection doesn't spam the pump.

### 9. Stop

- Click **Stop Stream** in the active camera view.
- For the drone, the UDP heartbeat thread and RTSP capture both close.
- Uncheck **Auto-fire pump on detection** to disconnect Bluetooth.
- `Ctrl+C` in the terminal stops Streamlit.

---

## Hardware setup

ESP32 pins (per [esp32_water_pump_sro4.ino](esp32_water_pump_sro4.ino)):

| ESP32 pin | Wire to |
|---|---|
| `GPIO 12` | Relay IN |
| `5 V` / `3.3 V` | Relay VCC (check your relay's logic level) |
| `GND` | Relay GND, common ground with pump supply |

Relay module is wired **NC** (normally closed) in the current sketch:
- `HIGH` on GPIO 12 → pump **ON**
- `LOW` on GPIO 12 → pump **OFF**
- Safety: Bluetooth disconnect forces the pin `LOW` (pump off) automatically.

Wire the pump's power line through the relay's switched contacts. Don't run the pump directly off the ESP32's 3.3 V rail — use a separate supply sized for the pump.

---

## Camera sources

| Mode | Use when |
|---|---|
| **Browser Webcam** | Running on a laptop with a camera, or a phone browser (HTTPS required — see [Remote access](#remote-access)) |
| **IP Camera (ESP32-CAM / Mobile)** | ESP32-CAM MJPEG, Android IP Webcam app, DroidCam — any MJPEG/RTSP URL |
| **KY UFO Drone** | `com.cooingdv.kyufo`-family WiFi drones. PC must be joined to the drone's AP. |

### Remote access

Mobile browsers require HTTPS for `navigator.mediaDevices` — plain LAN `http://` won't expose the camera. [run.py](run.py) wraps this. Run with the venv active:

```bash
python run.py --lan                    # LAN-only HTTPS with self-signed cert
python run.py --token <ngrok_token>    # Public HTTPS via authenticated ngrok
python run.py                          # Public HTTPS via free ngrok
```

`--lan` generates a self-signed cert in `.certs/` for `localhost` and your current LAN IP. Your phone will warn about the cert — tap **Advanced → Proceed**.

---

## Training the model

Local training is optional — Colab's free T4 GPU is enough.

Dataset layout:

```
dataset/
  Blight/            → non_infected
  Common_Rust/       → non_infected
  Gray_Leaf_Spot/    → non_infected
  Healthy/           → non_infected
  worm/              → infected
```

1. Upload the `dataset/` folder to Google Drive.
2. Open `notebooks/train_colab.ipynb` in Colab → **Runtime → Change runtime type → GPU (T4)**.
3. Edit `DRIVE_DATASET_PATH` in the config cell, **Run All**.
4. Download `best.keras` + `labels.json` from Colab into `models/mobilenetv2_binary_run/`.

The notebook splits with minority oversampling, runs two-phase training (frozen backbone → fine-tune top 30 layers), and evaluates with confusion matrix, ROC, and threshold sweep.

### Local scripts (optional)

Run with the venv active:

```bash
# Split dataset with oversampling
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

---

## Reverse-engineering notes (KY UFO drone)

The drone integration was built by decompiling the official KY UFO app (`com.cooingdv.kyufo`) with [jadx](https://github.com/skylot/jadx). The relevant classes:

- `com.cooingdv.kyufo.socket.Config` — hardcodes `PREVIEW_ADDRESS = "rtsp://192.168.1.1:7070/webcam"`.
- `com.cooingdv.kyufo.socket.SocketClient` — hands that URL to ijkplayer; no custom video protocol.
- `com.cooingdv.kyufo.socket.UdpClient` — `HeartBeatTask` fires `{0x01, 0x01}` to UDP `:7099` every second. Without it, the RTSP stream stalls.
- Camera-switch commands: `{0x06, 0x01}` / `{0x06, 0x02}`. Screen-flip: `{0x09, 0x01}` / `{0x09, 0x02}`. Not wired into the app yet — [drone_stream.py](drone_stream.py) exposes the UDP socket path if you want to add them.

Both pieces (RTSP capture + heartbeat thread) are in [drone_stream.py](drone_stream.py).

---

## File structure

```
app.py                               Streamlit app (webcam / IP camera / drone + pump)
drone_stream.py                      KY UFO RTSP + heartbeat receiver
esp32_water_pump_sro4.ino            ESP32 sketch: Bluetooth SPP + relay control
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

---

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

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Sidebar says `⚠️ SerialException: could not open port 'COMx'` | Wrong COM. Pick the other Bluetooth port from the dropdown, or 🔄 rescan after pairing. |
| Sidebar says `opening COMx...` forever | ESP32 in standby or out of range. Power-cycle the board. |
| Drone stream never opens | PC not on the drone's WiFi, or another client (e.g. phone) is using the AP. KY UFO drones allow only one client. |
| `461 Unsupported Transport` on RTSP | Benign — FFmpeg fell back from UDP to TCP. Already forced to TCP in [drone_stream.py](drone_stream.py). |
| Pump fires too often | Raise **Consecutive infected frames** or **Cooldown**. |
| Pump never fires despite detection | Confirm **Auto-fire** is ticked, status is `connected`, threshold isn't too high. Hit **Test fire** to isolate whether the issue is detection or plumbing. |
