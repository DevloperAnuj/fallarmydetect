# Fall Army Worm Detection (MobileNetV2 + Streamlit)

Binary classifier for corn leaf images: **infected** (worm) vs **non_infected** (healthy/other diseases).

## Dataset

Place source images in `dataset/` with subfolders:

```
dataset/
  Blight/
  Common_Rust/
  Gray_Leaf_Spot/
  Healthy/
  worm/
```

The `worm` folder maps to `infected`; all others map to `non_infected`.

## Training on Google Colab (Recommended)

Your local machine doesn't need a GPU — train on Colab's free T4 GPU.

### Step 1: Upload dataset to Google Drive

Upload your `dataset/` folder to Google Drive, e.g.:

```
My Drive/FAW_dataset/dataset/
  Blight/
  Common_Rust/
  ...
```

### Step 2: Open the training notebook

Upload `notebooks/train_colab.ipynb` to Google Colab, or open it directly from Drive.

Go to **Runtime > Change runtime type > GPU (T4)**.

### Step 3: Configure and run

Edit the `DRIVE_DATASET_PATH` in the config cell to match your Drive path, then **Run All**.

The notebook will:
- Split data into train/val/test with minority oversampling
- Train Phase 1: frozen MobileNetV2 backbone with focal loss + class weights
- Train Phase 2: fine-tune top 30 layers with lower learning rate
- Evaluate with confusion matrix, ROC curve, and threshold sweep
- Export the model to Drive and offer download

### Step 4: Use the model locally

Download `best.keras` and `labels.json` from Colab and place them in:

```
models/mobilenetv2_binary_run/
  best.keras
  labels.json
```

### Step 5: Run the Streamlit app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload an image and check the prediction. Adjust the **Infected detection threshold** slider in the sidebar to control sensitivity (lower = catches more infections).

## Local Scripts (Optional)

### Split dataset locally

```bash
python scripts/split_dataset.py --dataset-dir dataset --output-dir dataset_split_binary --binary-mode --oversample-minority --force
```

### Evaluate model locally

```bash
python scripts/evaluate_model.py --data-dir dataset_split_binary/test --model-path models/mobilenetv2_binary_run/best.keras --threshold 0.35
```

## Bias Fixes Applied

| Issue | Fix |
|-------|-----|
| 5:1 class imbalance | Oversampling + class weights |
| Wrong preprocessing at inference | Model handles preprocessing internally via Lambda layer |
| Accuracy-based checkpointing | Now monitors val_loss |
| Frozen backbone | Fine-tunes top 30 MobileNetV2 layers |
| Standard cross-entropy | Focal loss (gamma=2.0) |
| Weak augmentation | Stronger pipeline (flip, rotation, zoom, brightness, contrast, translation) |
| Pure argmax prediction | Threshold-based infected detection |

## File Structure

```
├── app.py                          # Streamlit inference app
├── requirements.txt                # Python dependencies
├── README.md
├── notebooks/
│   └── train_colab.ipynb           # Training notebook (run on Colab)
├── scripts/
│   ├── split_dataset.py            # Dataset splitting + oversampling
│   └── evaluate_model.py           # Evaluation with threshold sweep
├── models/
│   └── mobilenetv2_binary_run/
│       ├── best.keras              # Trained model (from Colab)
│       └── labels.json             # Class labels
├── artifacts/                      # Evaluation outputs
└── dataset/                        # Source images
```
