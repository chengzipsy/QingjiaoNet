import argparse
import csv
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


# =========================
# 配置区：常用参数集中放这里，IDE 里直接改这里再运行即可
# =========================

# 输出数据集结构：
# dataset_num_synthetic_real_v1/
#   train/0/*.jpg ... train/9/*.jpg
#   val/0/*.jpg   ... val/9/*.jpg
#   negative/*.jpg
OUTPUT_DIR = "dataset_num_synthetic_real_v1"

# 合成图数据集。默认全部放入 train，不放入 val。
SYNTHETIC_DATASET_DIR = "synthetic_num_dataset"

# 实拍整图数据来源。每个来源会按 train_ratio 拆到 train/val。
REAL_SOURCES = [
    {
        "name": "num_tnt_original",
        "path": "num_tnt_original_classification_dataset",
        "train_ratio": 0.70,
        "enabled": True,
    },
    {
        "name": "runtime_digit_11",
        "path": "image_update_classified——11/digit",
        "train_ratio": 0.60,
        "enabled": True,
    },
    {
        "name": "num_photo_dataset_num2",
        "path": "num_photo/dataset_num2",
        "train_ratio": 0.75,
        "enabled": True,
    },
]

# 负样本来源。负样本不进入 train/val 类别目录，只复制到输出根目录的 negative。
NEGATIVE_SOURCE_DIRS = [
    "num_tnt_original_classification_dataset/negative",
    "num_photo/negative",
]

# 数字分类类别。
CLASS_NAMES = [str(index) for index in range(10)]

# 支持读取的图片格式。
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RANDOM_SEED = 42
CLEAN_OUTPUT = True
COPY_MODE = "copy"  # 可选：copy 或 hardlink；hardlink 省空间，但要求同一磁盘。
CONVERT_TO_JPG = True
JPEG_QUALITY = 95


class ChineseArgumentParser(argparse.ArgumentParser):
    def format_help(self):
        help_text = super().format_help()
        return (
            help_text
            .replace("usage:", "用法:")
            .replace("options:", "选项:")
            .replace("optional arguments:", "选项:")
        )


def parse_args():
    parser = ChineseArgumentParser(
        description="制作数字最终训练数据集：合成图进入 train，实拍原图按比例拆分到 train/val，负样本保留到 negative。",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出。")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="输出数据集目录。")
    parser.add_argument("--synthetic-dir", default=SYNTHETIC_DATASET_DIR, help="合成图数据集目录。")
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default=COPY_MODE,
                        help="文件写入方式：copy 为复制，hardlink 为硬链接。")
    parser.add_argument("--dry-run", action="store_true",
                        help="只统计和打印计划，不实际复制文件。")
    parser.add_argument("--clean-output", dest="clean_output", action="store_true", default=CLEAN_OUTPUT,
                        help="制作前清空输出目录，默认开启。")
    parser.add_argument("--no-clean-output", dest="clean_output", action="store_false",
                        help="制作前不清空输出目录。")
    return parser.parse_args()


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_from_class_dir(class_dir):
    return sorted(path for path in class_dir.rglob("*") if is_image(path))


def collect_class_images(root, class_names):
    """兼容 root/类别、root/train/类别、root/val/类别 等常见分类数据集结构。"""
    root = Path(root)
    result = {class_name: [] for class_name in class_names}
    used_dirs = set()

    if not root.exists():
        return result

    for class_name in class_names:
        candidate_dirs = [
            root / class_name,
            root / "train" / class_name,
            root / "val" / class_name,
            root / "digit" / class_name,
        ]
        for class_dir in candidate_dirs:
            if class_dir.exists() and class_dir.is_dir():
                resolved = class_dir.resolve()
                if resolved in used_dirs:
                    continue
                result[class_name].extend(collect_from_class_dir(class_dir))
                used_dirs.add(resolved)

    for class_dir in sorted(path for path in root.rglob("*") if path.is_dir() and path.name in class_names):
        resolved = class_dir.resolve()
        if resolved in used_dirs:
            continue
        result[class_dir.name].extend(collect_from_class_dir(class_dir))
        used_dirs.add(resolved)

    return result


def stable_shuffle(items, seed_text):
    rng = random.Random(f"{RANDOM_SEED}_{seed_text}")
    items = list(items)
    rng.shuffle(items)
    return items


def split_real_images(images, train_ratio):
    images = list(images)
    if not images:
        return [], []

    train_count = int(round(len(images) * train_ratio))
    if len(images) >= 2:
        train_count = max(1, min(len(images) - 1, train_count))
    else:
        train_count = len(images)

    return images[:train_count], images[train_count:]


def safe_reset_output(output_dir, clean_output):
    output_dir = output_dir.resolve()
    cwd = Path.cwd().resolve()

    if clean_output and output_dir.exists():
        try:
            output_dir.relative_to(cwd)
        except ValueError as exc:
            raise RuntimeError(f"为安全起见，不会清空工作区外的目录：{output_dir}") from exc
        if output_dir == cwd:
            raise RuntimeError("输出目录不能是当前工程根目录。")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def ensure_output_dirs(output_dir, class_names):
    for split in ["train", "val"]:
        for class_name in class_names:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    (output_dir / "negative").mkdir(parents=True, exist_ok=True)


def copy_or_link_file(source, target, copy_mode):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()

    if copy_mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            # 硬链接失败时退回普通复制，常见原因是跨磁盘或权限限制。
            pass

    shutil.copy2(source, target)


def save_as_jpg(source, target, copy_mode):
    target = target.with_suffix(".jpg")
    if source.suffix.lower() in {".jpg", ".jpeg"}:
        copy_or_link_file(source, target, copy_mode)
        return target

    if not CONVERT_TO_JPG:
        raise RuntimeError(f"遇到非 jpg 图片但 CONVERT_TO_JPG=False：{source}")

    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("需要 Pillow 才能把非 jpg 图片转换成 jpg，请先安装 pillow。") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"}:
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")
        image.save(target, "JPEG", quality=JPEG_QUALITY)
    return target


def copy_samples(samples, output_dir, split, class_name, source_name, copy_mode, manifest, counters, dry_run):
    copied = 0
    for source_path in samples:
        counters[(split, class_name, source_name)] += 1
        index = counters[(split, class_name, source_name)]
        target_name = f"{source_name}_{index:06d}.jpg"
        target_path = output_dir / split / class_name / target_name

        if not dry_run:
            target_path = save_as_jpg(source_path, target_path, copy_mode)

        manifest.append({
            "split": split,
            "class": class_name,
            "source_name": source_name,
            "source_path": str(source_path),
            "target_path": str(target_path),
        })
        copied += 1
    return copied


def copy_negative_samples(output_dir, copy_mode, manifest, counters, dry_run):
    copied = 0
    for negative_dir in NEGATIVE_SOURCE_DIRS:
        source_dir = Path(negative_dir)
        if not source_dir.exists():
            print(f"[跳过] 找不到 negative 目录：{source_dir}")
            continue

        source_name = f"negative_{source_dir.parent.name}"
        images = sorted(path for path in source_dir.rglob("*") if is_image(path))
        for source_path in images:
            counters[("negative", "_negative", source_name)] += 1
            index = counters[("negative", "_negative", source_name)]
            target_path = output_dir / "negative" / f"{source_name}_{index:06d}.jpg"

            if not dry_run:
                target_path = save_as_jpg(source_path, target_path, copy_mode)

            manifest.append({
                "split": "negative",
                "class": "_negative",
                "source_name": source_name,
                "source_path": str(source_path),
                "target_path": str(target_path),
            })
            copied += 1
    return copied


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary_rows):
    print("\n========== 数据集制作统计 ==========")
    for split in ["train", "val", "negative"]:
        rows = [row for row in summary_rows if row["split"] == split]
        if not rows:
            continue
        print(f"\n[{split}]")
        for row in rows:
            print(f"{row['class']:<10} {row['source_name']:<28} {row['count']}")
    print("====================================\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    manifest = []
    counters = defaultdict(int)
    summary_counts = defaultdict(int)

    if not args.dry_run:
        safe_reset_output(output_dir, args.clean_output)
        ensure_output_dirs(output_dir, CLASS_NAMES)

    synthetic_dir = Path(args.synthetic_dir)
    if synthetic_dir.exists():
        synthetic_images = collect_class_images(synthetic_dir, CLASS_NAMES)
        for class_name in CLASS_NAMES:
            copied = copy_samples(
                synthetic_images[class_name],
                output_dir,
                "train",
                class_name,
                "synthetic",
                args.copy_mode,
                manifest,
                counters,
                args.dry_run,
            )
            if copied:
                summary_counts[("train", class_name, "synthetic")] += copied
    else:
        print(f"[跳过] 找不到合成图数据集：{synthetic_dir}")

    for source in REAL_SOURCES:
        if not source.get("enabled", True):
            continue

        source_dir = Path(source["path"])
        if not source_dir.exists():
            print(f"[跳过] 找不到实拍图来源：{source['name']} -> {source_dir}")
            continue

        class_images = collect_class_images(source_dir, CLASS_NAMES)
        for class_name in CLASS_NAMES:
            images = stable_shuffle(class_images[class_name], f"{source['name']}_{class_name}")
            train_images, val_images = split_real_images(images, source["train_ratio"])

            for split, split_images in [("train", train_images), ("val", val_images)]:
                copied = copy_samples(
                    split_images,
                    output_dir,
                    split,
                    class_name,
                    source["name"],
                    args.copy_mode,
                    manifest,
                    counters,
                    args.dry_run,
                )
                if copied:
                    summary_counts[(split, class_name, source["name"])] += copied

    negative_count = copy_negative_samples(
        output_dir,
        args.copy_mode,
        manifest,
        counters,
        args.dry_run,
    )
    if negative_count:
        summary_counts[("negative", "_negative", "negative")] += negative_count

    summary_rows = [
        {
            "split": split,
            "class": class_name,
            "source_name": source_name,
            "count": count,
        }
        for (split, class_name, source_name), count in sorted(summary_counts.items())
    ]

    if not args.dry_run:
        write_csv(
            output_dir / "manifest.csv",
            manifest,
            ["split", "class", "source_name", "source_path", "target_path"],
        )
        write_csv(
            output_dir / "summary.csv",
            summary_rows,
            ["split", "class", "source_name", "count"],
        )
        with (output_dir / "config_snapshot.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "output_dir": str(output_dir),
                    "synthetic_dataset_dir": str(synthetic_dir),
                    "real_sources": REAL_SOURCES,
                    "negative_source_dirs": NEGATIVE_SOURCE_DIRS,
                    "class_names": CLASS_NAMES,
                    "random_seed": RANDOM_SEED,
                    "copy_mode": args.copy_mode,
                    "convert_to_jpg": CONVERT_TO_JPG,
                    "jpeg_quality": JPEG_QUALITY,
                },
                file,
                indent=4,
                ensure_ascii=False,
            )

    print_summary(summary_rows)
    if args.dry_run:
        print("[完成] dry-run 只统计，没有实际复制文件。")
    else:
        print(f"[完成] 已生成数据集：{output_dir.resolve()}")
        print(f"[记录] manifest：{(output_dir / 'manifest.csv').resolve()}")
        print(f"[记录] summary：{(output_dir / 'summary.csv').resolve()}")


if __name__ == "__main__":
    main()
