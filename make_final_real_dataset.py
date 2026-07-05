import argparse
import csv
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


# =========================
# 数据集制作配置：常改参数集中放这里
# =========================

# 输出数据集：
# dataset_final_real_v1/
#   train/类别名/*.jpg
#   val/类别名/*.jpg
#   negative/*.jpg
OUTPUT_DIR = "dataset_final_real_v1"

# 大规模合成数据。默认把原 train 和 val 都合并到最终 train。
SYNTHETIC_DATASET_DIR = "dataset_update_new_1"
SYNTHETIC_TRAIN_SUBDIR = "train"
SYNTHETIC_VAL_SUBDIR = "val"
SYNTHETIC_VAL_TO_TRAIN = True

# negative 保留给训练脚本使用；是否参与训练由训练脚本里的 NEGATIVE_LOSS_WEIGHT 控制。
NEGATIVE_SOURCE_DIR = "dataset_update_new_1/negative"

# 真实图来源。脚本会使用第一个存在的路径；不存在就跳过并打印提示。
# 如果实拍图已经混在 SYNTHETIC_DATASET_DIR/train 或 val 中，也可以通过 filename_prefixes 按文件名前缀提取。
REAL_SOURCES = [
    {
        "name": "real_red_occluded",
        "description": "偏红、遮挡、异常状态实拍图",
        "filename_prefixes": [
            "dataset_shijue_new",
            "dataset_shijue",
        ],
        "candidate_dirs": [
            "datasets/dataset_shijue_new",
            "dataset_shijue_new",
        ],
        "train_ratio": 0.60,
        "enabled": True,
        "review_only": False,
    },
    {
        "name": "real_normal",
        "description": "正常状态实拍图",
        "filename_prefixes": [
            "new_pridect_dataset",
            "new_pridect_datase",
            "new_pridect",
            "new_predict_dataset",
            "new_predict",
        ],
        "candidate_dirs": [
            "datasets/new_pridect_dataset",
            "datasets/new_pridect_datase",
            "new_pridect_dataset",
            "new_pridect_datase",
            "new_predict_dataset",
        ],
        "train_ratio": 0.70,
        "enabled": True,
        "review_only": False,
    },
    {
        "name": "real_recent_runtime",
        "description": "最近实际运行时的实拍图，已确认归类无误",
        "filename_prefixes": [],
        "candidate_dirs": [
            "image_update_classified/cartoon",
        ],
        "train_ratio": 0.60,
        "enabled": True,
        "review_only": False,
    },
]

# 固定 10 个卡通类别，避免把 digit 或其他文件夹误纳入。
CLASS_NAMES = [
    "00mickey_mouse",
    "01pikachu",
    "02spongebob_squarepants",
    "03pleasant_sheep",
    "04donald_duck",
    "05nezha",
    "06big_head_son",
    "07gg_bond",
    "08calabash_brothers",
    "09grey_wolf",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
RANDOM_SEED = 42
CLEAN_OUTPUT = True
COPY_MODE = "copy"  # 可选 copy 或 hardlink；hardlink 省空间，但要求同一磁盘。
CONVERT_TO_JPG = True
JPEG_QUALITY = 95


class ChineseArgumentParser(argparse.ArgumentParser):
    def format_help(self):
        help_text = super().format_help()
        return (help_text
                .replace("usage:", "用法:")
                .replace("options:", "选项:")
                .replace("optional arguments:", "选项:"))


def parse_args():
    parser = ChineseArgumentParser(
        description="制作最终训练数据集：合成图进入 train，可信实拍图按类别拆分到 train/val。",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出。")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="输出数据集目录。")
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default=COPY_MODE,
                        help="文件写入方式：copy 为复制，hardlink 为硬链接。")
    parser.add_argument("--dry-run", action="store_true",
                        help="只统计和打印计划，不实际复制文件。")
    parser.add_argument("--clean-output", dest="clean_output", action="store_true", default=CLEAN_OUTPUT,
                        help="制作前清空输出目录。")
    parser.add_argument("--no-clean-output", dest="clean_output", action="store_false",
                        help="制作前不清空输出目录。")
    return parser.parse_args()


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def resolve_first_existing(candidate_dirs):
    for item in candidate_dirs:
        path = Path(item)
        if path.exists() and path.is_dir():
            return path
    return None


def collect_from_class_dir(class_dir):
    return sorted(path for path in class_dir.rglob("*") if is_image(path))


def normalize_prefixes(prefixes):
    return tuple(prefix.lower() for prefix in prefixes if prefix)


def has_filename_prefix(path, prefixes):
    lowered_name = path.name.lower()
    return any(lowered_name.startswith(prefix) for prefix in normalize_prefixes(prefixes))


def get_all_real_filename_prefixes():
    prefixes = []
    for source in REAL_SOURCES:
        if source.get("enabled", True):
            prefixes.extend(source.get("filename_prefixes", []))
    return normalize_prefixes(prefixes)


def collect_prefixed_class_images(root, class_names, prefixes):
    """从混合数据集中按文件名前缀提取实拍图，类别取父目录名。"""
    root = Path(root)
    result = {class_name: [] for class_name in class_names}
    if not root.exists():
        return result

    normalized_prefixes = normalize_prefixes(prefixes)
    if not normalized_prefixes:
        return result

    for image_path in sorted(path for path in root.rglob("*") if is_image(path)):
        class_name = image_path.parent.name
        if class_name not in result:
            continue
        if has_filename_prefix(image_path, normalized_prefixes):
            result[class_name].append(image_path)
    return result


def collect_mixed_real_images_by_source(root, class_names, real_sources):
    """只扫描一次混合数据集，把匹配前缀的图片分给对应实拍来源。"""
    root = Path(root)
    empty = {class_name: [] for class_name in class_names}
    result = {
        source["name"]: {class_name: [] for class_name in class_names}
        for source in real_sources
        if source.get("enabled", True)
    }
    if not root.exists():
        return result

    prefix_entries = []
    for source in real_sources:
        if not source.get("enabled", True):
            continue
        prefixes = normalize_prefixes(source.get("filename_prefixes", []))
        if prefixes:
            prefix_entries.append((source["name"], prefixes))

    if not prefix_entries:
        return result

    for image_path in sorted(path for path in root.rglob("*") if is_image(path)):
        class_name = image_path.parent.name
        if class_name not in empty:
            continue
        lowered_name = image_path.name.lower()
        for source_name, prefixes in prefix_entries:
            if any(lowered_name.startswith(prefix) for prefix in prefixes):
                result[source_name][class_name].append(image_path)
                break

    return result


def merge_class_images(left, right, class_names):
    merged = {class_name: list(left.get(class_name, [])) for class_name in class_names}
    seen = {
        class_name: {path.resolve() for path in merged[class_name]}
        for class_name in class_names
    }
    for class_name in class_names:
        for image_path in right.get(class_name, []):
            resolved = image_path.resolve()
            if resolved in seen[class_name]:
                continue
            merged[class_name].append(image_path)
            seen[class_name].add(resolved)
    return merged


def collect_class_images(root, class_names):
    """兼容 root/类别、root/train/类别、root/val/类别 等常见结构。"""
    root = Path(root)
    result = {class_name: [] for class_name in class_names}
    used_dirs = set()

    for class_name in class_names:
        candidate_dirs = [
            root / class_name,
            root / "train" / class_name,
            root / "val" / class_name,
            root / "cartoon" / class_name,
        ]
        for class_dir in candidate_dirs:
            if class_dir.exists() and class_dir.is_dir():
                resolved = class_dir.resolve()
                if resolved not in used_dirs:
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


def ensure_output_dirs(output_dir, class_names):
    for split in ["train", "val", "real_review"]:
        for class_name in class_names:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    (output_dir / "negative").mkdir(parents=True, exist_ok=True)


def safe_reset_output(output_dir, clean_output):
    output_dir = output_dir.resolve()
    cwd = Path.cwd().resolve()
    if clean_output and output_dir.exists():
        try:
            output_dir.relative_to(cwd)
        except ValueError:
            raise RuntimeError(f"为安全起见，不会清空工作区外的目录：{output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


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
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("需要 Pillow 才能把非 jpg 图片转成 jpg，请先安装 pillow。") from e

    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
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


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_negative(output_dir, copy_mode, manifest, counters, dry_run):
    negative_dir = Path(NEGATIVE_SOURCE_DIR)
    if not negative_dir.exists():
        print(f"[跳过] 找不到 negative 目录：{negative_dir}")
        return 0

    images = sorted(path for path in negative_dir.rglob("*") if is_image(path))
    copied = 0
    for source_path in images:
        counters[("negative", "_negative", "negative")] += 1
        index = counters[("negative", "_negative", "negative")]
        target_path = output_dir / "negative" / f"negative_{index:06d}.jpg"
        if not dry_run:
            target_path = save_as_jpg(source_path, target_path, copy_mode)
        manifest.append({
            "split": "negative",
            "class": "_negative",
            "source_name": "negative",
            "source_path": str(source_path),
            "target_path": str(target_path),
        })
        copied += 1
    return copied


def print_summary(summary):
    print("\n========== 数据集制作统计 ==========")
    for split in ["train", "val", "real_review", "negative"]:
        rows = [row for row in summary if row["split"] == split]
        if not rows:
            continue
        print(f"\n[{split}]")
        for row in rows:
            print(f"{row['class']:<24} {row['source_name']:<22} {row['count']}")
    print("====================================\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    manifest = []
    counters = defaultdict(int)
    summary_counts = defaultdict(int)
    mixed_real_prefixes = get_all_real_filename_prefixes()
    mixed_real_images_by_source = collect_mixed_real_images_by_source(
        SYNTHETIC_DATASET_DIR,
        CLASS_NAMES,
        REAL_SOURCES,
    )

    if not args.dry_run:
        safe_reset_output(output_dir, args.clean_output)
        ensure_output_dirs(output_dir, CLASS_NAMES)

    synthetic_root = Path(SYNTHETIC_DATASET_DIR)
    if synthetic_root.exists():
        synthetic_parts = [
            ("synthetic_train", synthetic_root / SYNTHETIC_TRAIN_SUBDIR),
        ]
        if SYNTHETIC_VAL_TO_TRAIN:
            synthetic_parts.append(("synthetic_val_to_train", synthetic_root / SYNTHETIC_VAL_SUBDIR))

        for source_name, part_dir in synthetic_parts:
            if not part_dir.exists():
                print(f"[跳过] 找不到合成数据目录：{part_dir}")
                continue
            class_images = collect_class_images(part_dir, CLASS_NAMES)
            for class_name in CLASS_NAMES:
                images = [
                    path for path in class_images[class_name]
                    if not has_filename_prefix(path, mixed_real_prefixes)
                ]
                copied = copy_samples(
                    images, output_dir, "train", class_name, source_name,
                    args.copy_mode, manifest, counters, args.dry_run,
                )
                if copied:
                    summary_counts[("train", class_name, source_name)] += copied
    else:
        print(f"[跳过] 找不到合成数据集：{synthetic_root}")

    for source in REAL_SOURCES:
        if not source.get("enabled", True):
            continue
        mixed_class_images = mixed_real_images_by_source.get(
            source["name"],
            {class_name: [] for class_name in CLASS_NAMES},
        )
        source_dir = resolve_first_existing(source["candidate_dirs"])
        if source_dir is None:
            class_images = {class_name: [] for class_name in CLASS_NAMES}
            mixed_total = sum(len(items) for items in mixed_class_images.values())
            if mixed_total == 0:
                print(
                    f"[跳过] 找不到实拍来源：{source['name']} -> {source['candidate_dirs']}；"
                    f"混合数据前缀也没有匹配：{source.get('filename_prefixes', [])}"
                )
                continue
            print(
                f"[提取] {source['name']} 从混合数据文件名前缀提取 {mixed_total} 张："
                f"{source.get('filename_prefixes', [])}"
            )
        else:
            class_images = collect_class_images(source_dir, CLASS_NAMES)
            mixed_total = sum(len(items) for items in mixed_class_images.values())
            if mixed_total:
                print(
                    f"[合并] {source['name']} 独立目录 + 混合数据前缀 {mixed_total} 张："
                    f"{source.get('filename_prefixes', [])}"
                )

        class_images = merge_class_images(class_images, mixed_class_images, CLASS_NAMES)
        for class_name in CLASS_NAMES:
            images = stable_shuffle(class_images[class_name], f"{source['name']}_{class_name}")
            if source.get("review_only", False):
                train_images, val_images = [], []
                review_images = images
            else:
                train_images, val_images = split_real_images(images, source["train_ratio"])
                review_images = []

            for split, split_images in [
                ("train", train_images),
                ("val", val_images),
                ("real_review", review_images),
            ]:
                copied = copy_samples(
                    split_images, output_dir, split, class_name, source["name"],
                    args.copy_mode, manifest, counters, args.dry_run,
                )
                if copied:
                    summary_counts[(split, class_name, source["name"])] += copied

    negative_count = copy_negative(output_dir, args.copy_mode, manifest, counters, args.dry_run)
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
        write_csv(output_dir / "manifest.csv", manifest,
                  ["split", "class", "source_name", "source_path", "target_path"])
        write_csv(output_dir / "summary.csv", summary_rows,
                  ["split", "class", "source_name", "count"])
        with (output_dir / "config_snapshot.json").open("w", encoding="utf-8") as f:
            json.dump({
                "output_dir": str(output_dir),
                "synthetic_dataset_dir": SYNTHETIC_DATASET_DIR,
                "synthetic_val_to_train": SYNTHETIC_VAL_TO_TRAIN,
                "negative_source_dir": NEGATIVE_SOURCE_DIR,
                "real_sources": REAL_SOURCES,
                "mixed_real_filename_prefixes": list(mixed_real_prefixes),
                "class_names": CLASS_NAMES,
                "random_seed": RANDOM_SEED,
                "copy_mode": args.copy_mode,
                "convert_to_jpg": CONVERT_TO_JPG,
                "jpeg_quality": JPEG_QUALITY,
            }, f, indent=4, ensure_ascii=False)

    print_summary(summary_rows)
    if args.dry_run:
        print("[完成] dry-run 只统计，没有实际复制文件。")
    else:
        print(f"[完成] 已生成数据集：{output_dir.resolve()}")
        print(f"[记录] manifest：{(output_dir / 'manifest.csv').resolve()}")
        print(f"[记录] summary：{(output_dir / 'summary.csv').resolve()}")


if __name__ == "__main__":
    main()
