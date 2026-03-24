import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_list(items: list, train_ratio: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def copy_images(images: list[Path], dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing_names: set[str] = set()
    for img in images:
        name = img.name
        counter = 1
        while name in existing_names:
            name = f"{img.stem}_{counter}{img.suffix}"
            counter += 1
        existing_names.add(name)
        shutil.copy2(img, dest_dir / name)


def oversample_to_count(src_dir: Path, target_count: int, seed: int):
    images = sorted(p for p in src_dir.iterdir() if p.is_file())
    if not images or len(images) >= target_count:
        return 0
    rng = random.Random(seed)
    existing_names = {p.name for p in images}
    added = 0
    counter = 0
    while len(images) + added < target_count:
        src = rng.choice(images)
        counter += 1
        new_name = f"{src.stem}_dup{counter}{src.suffix}"
        while new_name in existing_names:
            counter += 1
            new_name = f"{src.stem}_dup{counter}{src.suffix}"
        shutil.copy2(src, src_dir / new_name)
        existing_names.add(new_name)
        added += 1
    return added


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"),
                        help="Source dataset directory with class subfolders")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_split_binary"),
                        help="Output directory for split dataset")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"),
                        help="Directory for summary JSON files")
    parser.add_argument("--binary-mode", action="store_true",
                        help="Binary classification: positive vs rest")
    parser.add_argument("--positive-class", type=str, default="worm",
                        help="Source folder name for the positive class")
    parser.add_argument("--positive-label", type=str, default="infected",
                        help="Label for the positive class")
    parser.add_argument("--negative-label", type=str, default="non_infected",
                        help="Label for all other classes")
    parser.add_argument("--oversample-minority", action="store_true",
                        help="Oversample minority class in train split to match majority count")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite output directory if it exists")
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Ratios must be >0 and train+val must be <1.")

    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")

    if args.output_dir.exists():
        if args.force:
            shutil.rmtree(args.output_dir)
        else:
            raise FileExistsError(
                f"Output directory exists: {args.output_dir}. Use --force to overwrite.")

    # Collect images per source class
    source_classes = sorted(
        d.name for d in args.dataset_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not source_classes:
        raise RuntimeError(f"No class folders found in {args.dataset_dir}")

    # Map source classes to target labels
    class_mapping = {}
    for cls in source_classes:
        if args.binary_mode:
            class_mapping[cls] = args.positive_label if cls.lower() == args.positive_class.lower() else args.negative_label
        else:
            class_mapping[cls] = cls

    # Collect and group images by target label
    images_by_target: dict[str, list[Path]] = defaultdict(list)
    for cls in source_classes:
        imgs = collect_images(args.dataset_dir / cls)
        if not imgs:
            print(f"Skipping empty class: {cls}")
            continue
        images_by_target[class_mapping[cls]].extend(imgs)

    if not images_by_target:
        raise RuntimeError(f"No images found in {args.dataset_dir}")

    # Split each target class into train/val/test
    summary = {}
    for target_label, images in sorted(images_by_target.items()):
        train, val, test = split_list(images, args.train_ratio, args.val_ratio, args.seed)
        copy_images(train, args.output_dir / "train" / target_label)
        copy_images(val, args.output_dir / "val" / target_label)
        copy_images(test, args.output_dir / "test" / target_label)
        summary[target_label] = {
            "total": len(images),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        }

    # Oversample minority class in train split only
    if args.oversample_minority and args.binary_mode:
        train_counts = {label: info["train"] for label, info in summary.items()}
        max_count = max(train_counts.values())
        min_label = min(train_counts, key=train_counts.get)

        if train_counts[min_label] < max_count:
            minority_dir = args.output_dir / "train" / min_label
            added = oversample_to_count(minority_dir, max_count, args.seed)
            summary[min_label]["train_oversampled"] = train_counts[min_label] + added
            print(f"\nOversampled '{min_label}': {train_counts[min_label]} -> {train_counts[min_label] + added}")

    # Save artifacts
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.artifacts_dir / "split_summary_binary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.binary_mode:
        mapping_path = args.artifacts_dir / "binary_class_mapping.json"
        mapping_path.write_text(json.dumps(class_mapping, indent=2), encoding="utf-8")

    # Print summary table
    print("\n" + "=" * 65)
    print("  Dataset Split Summary")
    print("=" * 65)
    print(f"  {'Class':<20} {'Train':>10} {'Val':>8} {'Test':>8}")
    print("-" * 65)
    for label, info in sorted(summary.items()):
        train_str = str(info["train"])
        if "train_oversampled" in info:
            train_str = f"{info['train']}->{info['train_oversampled']}"
        print(f"  {label:<20} {train_str:>10} {info['val']:>8} {info['test']:>8}")
    print("=" * 65)
    print(f"  Output: {args.output_dir}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
