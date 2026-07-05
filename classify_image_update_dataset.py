import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CARTOON_LABELS = [
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
DEFAULT_DIGIT_LABELS = [str(i) for i in range(10)]


class ChineseArgumentParser(argparse.ArgumentParser):
    def format_help(self):
        help_text = super().format_help()
        return (help_text
                .replace("usage:", "用法:")
                .replace("options:", "选项:")
                .replace("optional arguments:", "选项:"))


@dataclass(frozen=True)
class ParsedName:
    prefix: str
    dataset_type: str
    label: str
    confidence: float


def parse_args():
    parser = ChineseArgumentParser(
        description="根据 image_update 图片文件名中的推理标签和置信度，整理成卡通人物与数字两份图像分类数据集。",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help",
                        help="显示帮助信息并退出。")
    parser.add_argument("--input-dir", default="image",
                        help="待归类图片目录。")
    parser.add_argument("--output-root", default="image_update_classified",
                        help="输出目录根路径。")
    parser.add_argument("--cartoon-label-json", default="class_indices.json",
                        help="卡通类别 json 文件；读取其中的类别名作为卡通标签。")
    parser.add_argument("--prefix-len", type=int, default=1,
                        help="文件名前面表示轮次或张数的前缀长度；你的当前命名规则默认为 1。")
    parser.add_argument("--low-confidence-threshold", type=float, default=0.8,
                        help="低于该置信度的图片会额外复制到复查目录；设为负数可关闭。")
    parser.add_argument("--copy-review", action="store_true", default=True,
                        help="把低置信度图片额外复制到 _review_low_confidence 目录。")
    parser.add_argument("--no-copy-review", action="store_false", dest="copy_review",
                        help="不额外复制低置信度复查图片，只在 manifest.csv 中记录。")
    parser.add_argument("--clean-output", action="store_true",
                        help="生成前先删除输出目录。")
    parser.add_argument("--move", action="store_true",
                        help="移动原图而不是复制原图；默认是复制，更安全。")
    parser.add_argument("--dry-run", action="store_true",
                        help="只解析和统计，不实际复制或移动文件。")
    return parser.parse_args()


def load_cartoon_labels(json_path):
    path = Path(json_path)
    if not path.exists():
        return DEFAULT_CARTOON_LABELS

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    labels = [data[key] for key in sorted(data, key=lambda item: int(item))]
    return labels or DEFAULT_CARTOON_LABELS


def list_images(input_dir):
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"输入目录不存在：{root}")
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_image_name(path, known_labels, prefix_len):
    stem = path.stem
    if len(stem) <= prefix_len:
        return None

    prefix = stem[:prefix_len]
    payload = stem[prefix_len:]
    for dataset_type, label in known_labels:
        if not payload.startswith(label):
            continue

        confidence_text = payload[len(label):]
        try:
            confidence = float(confidence_text)
        except ValueError:
            continue

        if 0 <= confidence <= 1:
            return ParsedName(prefix, dataset_type, label, confidence)

    return None


def unique_destination(path):
    if not path.exists():
        return path

    index = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def transfer_file(source, destination, move=False, dry_run=False):
    if dry_run:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = unique_destination(destination)
    if move:
        shutil.move(str(source), str(destination))
    else:
        shutil.copy2(source, destination)
    return destination


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "source_path",
            "output_path",
            "dataset_type",
            "label",
            "prefix",
            "confidence",
            "low_confidence",
            "parsed",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, count_by_type_label, unknown_count, low_conf_count):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset_type", "label", "count"])
        for (dataset_type, label), count in sorted(count_by_type_label.items()):
            writer.writerow([dataset_type, label, count])
        writer.writerow(["_unknown", "_unknown", unknown_count])
        writer.writerow(["_review_low_confidence", "_all", low_conf_count])


def main():
    args = parse_args()
    if args.prefix_len < 0:
        raise ValueError("--prefix-len 必须大于或等于 0")
    if args.low_confidence_threshold > 1:
        raise ValueError("--low-confidence-threshold 必须小于或等于 1")

    cartoon_labels = load_cartoon_labels(args.cartoon_label_json)
    known_labels = (
        [("cartoon", label) for label in cartoon_labels]
        + [("digit", label) for label in DEFAULT_DIGIT_LABELS]
    )
    known_labels.sort(key=lambda item: len(item[1]), reverse=True)

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    if args.clean_output and output_root.exists() and not args.dry_run:
        shutil.rmtree(output_root)
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    manifest_rows = []
    count_by_type_label = Counter()
    low_conf_count = 0
    unknown_count = 0

    for index, image_path in enumerate(images, start=1):
        parsed = parse_image_name(image_path, known_labels, args.prefix_len)
        source_text = str(image_path)

        if parsed is None:
            unknown_count += 1
            unknown_destination = output_root / "_unknown" / image_path.name
            output_path = transfer_file(image_path, unknown_destination, args.move, args.dry_run)
            manifest_rows.append({
                "source_path": source_text,
                "output_path": str(output_path),
                "dataset_type": "_unknown",
                "label": "_unknown",
                "prefix": "",
                "confidence": "",
                "low_confidence": "",
                "parsed": "no",
            })
        else:
            target = output_root / parsed.dataset_type / parsed.label / image_path.name
            output_path = transfer_file(image_path, target, args.move, args.dry_run)
            count_by_type_label[(parsed.dataset_type, parsed.label)] += 1

            is_low_conf = (
                args.low_confidence_threshold >= 0
                and parsed.confidence < args.low_confidence_threshold
            )
            if is_low_conf:
                low_conf_count += 1
                if args.copy_review:
                    review_source = output_path if args.move else image_path
                    review_target = (
                        output_root
                        / "_review_low_confidence"
                        / parsed.dataset_type
                        / parsed.label
                        / image_path.name
                    )
                    transfer_file(review_source, review_target, False, args.dry_run)

            manifest_rows.append({
                "source_path": source_text,
                "output_path": str(output_path),
                "dataset_type": parsed.dataset_type,
                "label": parsed.label,
                "prefix": parsed.prefix,
                "confidence": f"{parsed.confidence:.7f}",
                "low_confidence": "yes" if is_low_conf else "no",
                "parsed": "yes",
            })

        print(f"\r正在处理 {index}/{len(images)}", end="", flush=True)

    print()
    if not args.dry_run:
        write_manifest(output_root / "manifest.csv", manifest_rows)
        write_summary(output_root / "summary.csv", count_by_type_label, unknown_count, low_conf_count)

    print("处理完成。")
    print(f"图片总数：{len(images)}")
    print(f"无法解析：{unknown_count}")
    print(f"低置信度复查：{low_conf_count}")
    for dataset_type in ("cartoon", "digit"):
        total = sum(
            count for (kind, _), count in count_by_type_label.items()
            if kind == dataset_type
        )
        print(f"{dataset_type}: {total}")
        for (_, label), count in sorted(
            ((key, count) for key, count in count_by_type_label.items() if key[0] == dataset_type),
            key=lambda item: item[0][1],
        ):
            print(f"  {label}: {count}")

    if args.dry_run:
        print("当前是 dry-run，没有实际复制或移动文件。")
    else:
        print(f"输出目录：{output_root.resolve()}")


if __name__ == "__main__":
    main()
