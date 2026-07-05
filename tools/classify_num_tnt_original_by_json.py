import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


# =========================
# 配置区：常用参数放这里，IDE 里直接改这里再运行即可
# =========================

# 原始图片目录。
IMAGE_DIR = "num_tnt/tn_images"

# JSON 标签目录。
JSON_DIR = "num_tnt/json"

# 输出的整图分类数据集目录。
OUTPUT_DIR = "num_tnt_original_classification_dataset"

# 没有标签的图片会放到这个类别文件夹中。
NEGATIVE_DIR_NAME = "negative"

# 重新生成数据集时是否清空旧输出目录。
CLEAN_OUTPUT = True

# 如果一张图里出现多个不同标签，默认复制到所有对应类别。
# 可选：copy_all / first / negative
MULTI_LABEL_MODE = "copy_all"

# 支持读取的图片格式。
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据 num_tnt 的 JSON 标签，把原图整理成图像分类数据集。"
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
        help=f"输出整图分类数据集目录，默认：{OUTPUT_DIR}",
    )
    parser.add_argument(
        "--negative-dir-name",
        default=NEGATIVE_DIR_NAME,
        help=f"无标签图片保存的类别文件夹名，默认：{NEGATIVE_DIR_NAME}",
    )
    parser.add_argument(
        "--multi-label-mode",
        choices=["copy_all", "first", "negative"],
        default=MULTI_LABEL_MODE,
        help=(
            "一张图存在多个不同标签时的处理方式："
            "copy_all=复制到所有类别；first=只用第一个标签；negative=放入 negative。"
        ),
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
        help="只统计和打印将要生成的结果，不实际复制图片。",
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


def unique_keep_order(items):
    result = []
    seen = set()
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


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


def find_json_for_image(image_path, json_dir):
    json_path = json_dir / f"{image_path.stem}.json"
    if json_path.exists():
        return json_path
    return None


def load_labels_from_json(json_path):
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    labels = []
    for shape in data.get("shapes") or []:
        label = safe_label_name(shape.get("label", ""))
        if label:
            labels.append(label)
    return unique_keep_order(labels)


def make_unique_target(label_dir, image_path):
    target_path = label_dir / image_path.name
    if not target_path.exists():
        return target_path

    index = 1
    while True:
        target_path = label_dir / f"{image_path.stem}_{index:03d}{image_path.suffix}"
        if not target_path.exists():
            return target_path
        index += 1


def copy_original_image(image_path, output_dir, label, dry_run):
    label_dir = output_dir / label
    target_path = make_unique_target(label_dir, image_path)

    if not dry_run:
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_path)

    return target_path


def add_manifest_row(rows, category, label, source_image, json_path, target_path, reason):
    rows.append(
        {
            "category": category,
            "label": label,
            "source_image": str(source_image),
            "json_path": str(json_path) if json_path else "",
            "target_path": str(target_path),
            "reason": reason,
        }
    )


def write_csv_files(output_dir, manifest_rows, label_counter, dry_run):
    if dry_run:
        return

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = ["category", "label", "source_image", "json_path", "target_path", "reason"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "count"])
        for label, count in sorted(label_counter.items(), key=lambda item: item[0]):
            writer.writerow([label, count])


def choose_output_labels(labels, multi_label_mode, negative_dir_name):
    if not labels:
        return [negative_dir_name], "negative", "没有有效标签"

    if len(labels) == 1:
        return labels, "image", "按 JSON 标签复制原图"

    if multi_label_mode == "copy_all":
        return labels, "image", "多标签图片复制到所有对应类别"
    if multi_label_mode == "first":
        return [labels[0]], "image", "多标签图片只使用第一个标签"

    return [negative_dir_name], "negative", "多标签图片按配置放入 negative"


def process_images(args):
    image_dir = assert_input_dir(Path(args.image_dir), "原始图片目录")
    json_dir = assert_input_dir(Path(args.json_dir), "JSON 标签目录")
    output_dir = assert_safe_output_dir(Path(args.output_dir))

    prepare_output_dir(output_dir, args.clean_output, args.dry_run)

    image_paths = sorted(path for path in image_dir.iterdir() if is_image(path))
    manifest_rows = []
    label_counter = Counter()
    stats = Counter()

    for image_path in image_paths:
        json_path = find_json_for_image(image_path, json_dir)

        if json_path is None:
            labels = []
            reason = "没有找到同名 JSON"
        else:
            try:
                labels = load_labels_from_json(json_path)
                reason = ""
            except Exception as exc:
                labels = []
                reason = f"JSON 读取失败：{exc}"

        output_labels, category, default_reason = choose_output_labels(
            labels,
            args.multi_label_mode,
            args.negative_dir_name,
        )
        if not reason:
            reason = default_reason

        for label in output_labels:
            target_path = copy_original_image(image_path, output_dir, label, args.dry_run)
            label_counter[label] += 1
            stats[category] += 1
            add_manifest_row(
                manifest_rows,
                category,
                label,
                image_path,
                json_path,
                target_path,
                reason,
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
    print(f"整图分类样本数量：{stats['image']}")
    print(f"negative 数量：{stats['negative']}")
    print("")
    print("各类别数量：")
    for label, count in sorted(label_counter.items(), key=lambda item: item[0]):
        print(f"  {label}: {count}")
    if args.dry_run:
        print("")
        print("当前是 dry-run 模式，没有实际复制图片。")
    print("==============================")


if __name__ == "__main__":
    main()
