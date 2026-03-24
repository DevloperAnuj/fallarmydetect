import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


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


def build_dataset(data_dir: Path, image_size: int, batch_size: int):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred", label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size, shuffle=False,
    )
    return ds.prefetch(tf.data.AUTOTUNE), ds.class_names


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test split.")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("dataset_split_binary/test"))
    parser.add_argument("--model-path", type=Path,
                        default=Path("models/mobilenetv2_binary_run/best.keras"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Infected class threshold (if set, uses threshold instead of argmax)")
    parser.add_argument("--infected-label", type=str, default="infected",
                        help="Name of the infected class label")
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Test data not found: {args.data_dir}")

    test_ds, class_names = build_dataset(args.data_dir, args.image_size, args.batch_size)
    model = tf.keras.models.load_model(args.model_path)

    # Collect predictions
    y_true, y_probs = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_probs.extend(probs.tolist())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    # Find infected class index
    infected_idx = None
    if args.infected_label in class_names:
        infected_idx = class_names.index(args.infected_label)

    # Apply threshold or argmax
    if args.threshold is not None and infected_idx is not None:
        y_pred = np.where(y_probs[:, infected_idx] >= args.threshold, infected_idx,
                          1 - infected_idx)
        print(f"\nUsing threshold={args.threshold} for '{args.infected_label}' class")
    else:
        y_pred = np.argmax(y_probs, axis=1)
        print("\nUsing argmax for predictions")

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "classification_report_binary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("  Classification Report")
    print("=" * 60)
    for name in class_names:
        r = report[name]
        print(f"  {name:<20} precision={r['precision']:.4f}  recall={r['recall']:.4f}  f1={r['f1-score']:.4f}  n={r['support']}")
    print(f"  {'accuracy':<20} {report['accuracy']:.4f}")
    print("=" * 60)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = args.out_dir / "confusion_matrix_binary.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # Threshold sweep (infected class only)
    if infected_idx is not None:
        infected_probs = y_probs[:, infected_idx]
        actual_infected = y_true == infected_idx

        print("\nThreshold Sweep (infected class):")
        print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("  " + "-" * 45)

        sweep_results = {}
        best_f1, best_thresh = 0, 0.5

        for thresh in np.arange(0.10, 0.95, 0.05):
            pred_infected = infected_probs >= thresh
            tp = int((pred_infected & actual_infected).sum())
            fp = int((pred_infected & ~actual_infected).sum())
            fn = int((~pred_infected & actual_infected).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            print(f"  {thresh:>10.2f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")
            sweep_results[f"{thresh:.2f}"] = {"precision": prec, "recall": rec, "f1": f1}
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh

        print(f"\n  >>> Recommended threshold: {best_thresh:.2f} (F1={best_f1:.4f}) <<<")

        sweep_path = args.out_dir / "threshold_sweep.json"
        sweep_path.write_text(json.dumps(sweep_results, indent=2), encoding="utf-8")

        # ROC curve
        fpr, tpr, _ = roc_curve(actual_infected, infected_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Infected Detection")
        plt.legend()
        plt.tight_layout()
        roc_path = args.out_dir / "roc_curve.png"
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"  ROC AUC: {roc_auc:.4f}")

    print(f"\nArtifacts saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
