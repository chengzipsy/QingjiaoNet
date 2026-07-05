import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


# =========================
# 配置区：常用参数放这里，IDE 里直接改这里再运行即可
# =========================

# 原始图片目录。
IMAGE_DIR = "num_tnt/tn_images"

# JSON 标签目录。
JSON_DIR = "num_tnt/json"

# 输出的图像分类数据集目录。
OUTPUT_DIR = "num_tnt_classification_dataset"

# 没有标签的图片会放到这个类别文件夹中。
NEGATIVE_DIR_NAME = "negative"

# 裁剪框外扩比例。0.05 表示按框宽高各外扩 5%，默认不外扩。
BBOX_PADDING_RATIO = 0.0

# 太小的裁剪区域会被认为无效，避免生成异常小图。
MIN_CROP_SIZE = 2

# 输出图片扩展名。建议统一保存为 jpg，方便后续训练。
OUTPUT_EXT = ".jpg"

# jpg 保存质量。
JPEG_QUALITY = 95

# 重新生成数据集时是否清空旧输出目录。
CLEAN_OUTPUT = True

# 支持读取的图片格式。
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据 num_tnt 的 JSON 标注框裁剪目标，生成图像分类数据集。"
    )
    parser.add_argument(
        "--image-dir",
        default=IMAGE_DIR,
        help=f"原始图片目录，默认：{IMAGE_DIR}",
    )
    parser.add_argument(
        "--json-dir",
        default=JSON_DIR,
        help=f"JSON 标签目录，默认：{JSON_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"输出分类数据集目录，默认：{OUTPUT_DIR}",
    )
    parser.add_argument(
        "--negative-dir-name",
        default=NEGATIVE_DIR_NAME,
        help=f"无标签图片保存的类别文件夹名，默认：{NEGATIVE_DIR_NAME}",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=BBOX_PADDING_RATIO,
        help=f"裁剪框外扩比例，例如 0.05 表示外扩 5%，默认：{BBOX_PADDING_RATIO}",
    )
    parser.add_argument(
        "--min-crop-size",
        type=int,
        default=MIN_CROP_SIZE,
        help=f"最小有效裁剪尺寸，宽或高小于该值会跳过，默认：{MIN_CROP_SIZE}",
    )
    parser.add_argument(
        "--output-ext",
        default=OUTPUT_EXT,
        help=f"输出图片扩展名，默认：{OUTPUT_EXT}",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=JPEG_QUALITY,
        help=f"JPEG 保存质量，默认：{JPEG_QUALITY}",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        default=CLEAN_OUTPUT,
        help="生成前清空输出目录，默认开启。",
    )
    parser.add_argument(
        "--no-clean-output",
        action="store_false",
        dest="clean_output",
        help="生成前不清空输出目录。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计和打印将要生成的结果，不实际写入图片。",
    )
    return parser.parse_args()


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def safe_label_name(label):
    """把标签名转换为安全的文件夹名，同时尽量保留原始标签文本。"""
    label = str(label).strip()
    if not label:
        return ""
    invalid_chars = '<>:"/\\|?*'
    return "".join("_" if char in invalid_chars else char for char in label)


def assert_input_dir(path, name):
    path = path.resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"找不到{name}：{path}")
    return path


def assert_safe_output_dir(output_dir):
    output_dir = output_dir.resolve()
    cwd = Path.cwd().resolve()

    try:
        output_dir.relative_to(cwd)
    except ValueError as exc:
        raise RuntimeError(f"输出目录不在当前工程内，为避免误删，停止执行：{output_dir}") from exc

    blocked_names = {cwd.name, "num_tnt", "tn_images", "json", "tools"}
    if output_dir == cwd or output_dir.name in blocked_names:
        raise RuntimeError(f"输出目录过于危险，请换一个专门的数据集目录：{output_dir}")

    return output_dir


def prepare_output_dir(output_dir, clean_output, dry_run):
    if dry_run:
        return

    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def load_json(json_path):
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_json_for_image(image_path, json_dir):
    json_path = json_dir / f"{image_path.stem}.json"
    if json_path.exists():
        return json_path
    return None


def shape_to_bbox(shape, image_width, image_height, padding_ratio):
    points = shape.get("points") or []
    if not points:
        return None

    xs = []
    ys = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        xs.append(float(point[0]))
        ys.append(float(point[1]))

    if not xs or not ys:
        return None

    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)

    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None

    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(image_width, int(round(x2 + pad_x)))
    y2 = min(image_height, int(round(y2 + pad_y)))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def save_image(image, target_path, quality):
    target_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = target_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image = image.convert("RGB")
        image.save(target_path, quality=quality)
    else:
        image.save(target_path)


def make_target_path(output_dir, label, image_path, shape_index, output_ext):
    label_dir = output_dir / safe_label_name(label)
    base_name = f"{image_path.stem}_{shape_index:03d}{output_ext}"
    target_path = label_dir / base_name

    if not target_path.exists():
        return target_path

    index = 1
    while True:
        target_path = label_dir / f"{image_path.stem}_{shape_index:03d}_{index:03d}{output_ext}"
        if not target_path.exists():
            return target_path
        index += 1


def make_negative_path(output_dir, negative_dir_name, image_path, output_ext):
    negative_dir = output_dir / negative_dir_name
    target_path = negative_dir / f"{image_path.stem}{output_ext}"

    if not target_path.exists():
        return target_path

    index = 1
    while True:
        target_path = negative_dir / f"{image_path.stem}_{index:03d}{output_ext}"
        if not target_path.exists():
            return target_path
        index += 1


def add_manifest_row(
    rows,
    category,
    label,
    source_image,
    json_path,
    shape_index,
    target_path,
    bbox,
    reason,
):
    x1, y1, x2, y2 = bbox if bbox else ("", "", "", "")
    rows.append(
        {
            "category": category,
            "label": label,
            "source_image": str(source_image),
            "json_path": str(json_path) if json_path else "",
            "shape_index": shape_index,
            "target_path": str(target_path),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "reason": reason,
        }
    )


def write_csv_files(output_dir, manifest_rows, label_counter, dry_run):
    if dry_run:
        return

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "category",
            "label",
            "source_image",
            "json_path",
            "shape_index",
            "target_path",
            "x1",
            "y1",
            "x2",
            "y2",
            "reason",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "count"])
        for label, count in sorted(label_counter.items(), key=lambda item: item[0]):
            writer.writerow([label, count])


def process_images(args):
    image_dir = assert_input_dir(Path(args.image_dir), "原始图片目录")
    json_dir = assert_input_dir(Path(args.json_dir), "JSON 标签目录")
    output_dir = assert_safe_output_dir(Path(args.output_dir))
    output_ext = args.output_ext if args.output_ext.startswith(".") else f".{args.output_ext}"

    prepare_output_dir(output_dir, args.clean_output, args.dry_run)

    image_paths = sorted(path for path in image_dir.iterdir() if is_image(path))
    manifest_rows = []
    label_counter = Counter()
    stats = Counter()

    for image_path in image_paths:
        json_path = find_json_for_image(image_path, json_dir)

        if json_path is None:
            target_path = make_negative_path(output_dir, args.negative_dir_name, image_path, output_ext)
            if not args.dry_run:
                with Image.open(image_path) as image:
                    image = ImageOps.exif_transpose(image)
                    save_image(image, target_path, args.quality)
            label_counter[args.negative_dir_name] += 1
            stats["negative"] += 1
            add_manifest_row(
                manifest_rows,
                "negative",
                args.negative_dir_name,
                image_path,
                None,
                "",
                target_path,
                None,
                "没有找到同名 JSON",
            )
            continue

        try:
            data = load_json(json_path)
        except Exception as exc:
            target_path = make_negative_path(output_dir, args.negative_dir_name, image_path, output_ext)
            if not args.dry_run:
                with Image.open(image_path) as image:
                    image = ImageOps.exif_transpose(image)
                    save_image(image, target_path, args.quality)
            label_counter[args.negative_dir_name] += 1
            stats["negative"] += 1
            add_manifest_row(
                manifest_rows,
                "negative",
                args.negative_dir_name,
                image_path,
                json_path,
                "",
                target_path,
                None,
                f"JSON 读取失败：{exc}",
            )
            continue

        shapes = data.get("shapes") or []
        valid_crop_count = 0

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            image_width, image_height = image.size

            for shape_index, shape in enumerate(shapes):
                label = safe_label_name(shape.get("label", ""))
                if not label:
                    stats["invalid_shape"] += 1
                    continue

                bbox = shape_to_bbox(shape, image_width, image_height, args.padding_ratio)
                if bbox is None:
                    stats["invalid_shape"] += 1
                    continue

                x1, y1, x2, y2 = bbox
                if (x2 - x1) < args.min_crop_size or (y2 - y1) < args.min_crop_size:
                    stats["invalid_shape"] += 1
                    continue

                target_path = make_target_path(output_dir, label, image_path, shape_index, output_ext)
                if not args.dry_run:
                    crop = image.crop((x1, y1, x2, y2))
                    save_image(crop, target_path, args.quality)

                label_counter[label] += 1
                stats["crop"] += 1
                valid_crop_count += 1
                add_manifest_row(
                    manifest_rows,
                    "crop",
                    label,
                    image_path,
                    json_path,
                    shape_index,
                    target_path,
                    bbox,
                    "有效标注裁剪",
                )

        if valid_crop_count == 0:
            target_path = make_negative_path(output_dir, args.negative_dir_name, image_path, output_ext)
            if not args.dry_run:
                with Image.open(image_path) as image:
                    image = ImageOps.exif_transpose(image)
                    save_image(image, target_path, args.quality)
            label_counter[args.negative_dir_name] += 1
            stats["negative"] += 1
            add_manifest_row(
                manifest_rows,
                "negative",
                args.negative_dir_name,
                image_path,
                json_path,
                "",
                target_path,
                None,
                "JSON 中没有有效标注",
            )

    write_csv_files(output_dir, manifest_rows, label_counter, args.dry_run)
    return image_dir, json_dir, output_dir, stats, label_counter


def main():
    args = parse_args()
    image_dir, json_dir, output_dir, stats, label_counter = process_images(args)

    print("========== 处理完成 ==========")
    print(f"原始图片目录：{image_dir}")
    print(f"JSON 标签目录：{json_dir}")
    print(f"输出数据集目录：{output_dir}")
    print(f"裁剪样本数量：{stats['crop']}")
    print(f"negative 数量：{stats['negative']}")
    print(f"无效标注数量：{stats['invalid_shape']}")
    print("")
    print("各类别数量：")
    for label, count in sorted(label_counter.items(), key=lambda item: item[0]):
        print(f"  {label}: {count}")
    if args.dry_run:
        print("")
        print("当前是 dry-run 模式，没有实际写入图片。")
    print("==============================")


if __name__ == "__main__":
    main()
