import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ChineseArgumentParser(argparse.ArgumentParser):
    def format_help(self):
        help_text = super().format_help()
        return (help_text
                .replace("usage:", "用法:")
                .replace("options:", "选项:")
                .replace("optional arguments:", "选项:"))


def parse_args():
    parser = ChineseArgumentParser(
        description="加载 ckpt 权重评估验证集，并把识别错误的图片复制到输出目录。",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help",
                        help="显示帮助信息并退出。")
    parser.add_argument("--dataset-dir", default="dataset_final_real_v1",
                        help="数据集根目录；默认读取其中的 val 子目录作为验证集。")
    parser.add_argument("--val-subdir", default="val",
                        help="验证集子目录名称。")
    parser.add_argument("--class-json", default="class_indices.json",
                        help="训练时生成的类别索引 json 文件。")
    parser.add_argument("--weights-path", default=r"save_weights\try4_final_real_v1_20260702_225804_96x72\best.ckpt",
                        help="ckpt 权重路径，不带 .index 后缀；留空时自动选择最新权重。")
    parser.add_argument("--save-root", default="save_weights",
                        help="自动选择最新权重时搜索的根目录。")
    parser.add_argument("--save-prefix", default="num_synth_real_v1",
                        help="自动选择最新权重时使用的目录名前缀。")
    parser.add_argument("--output-dir", default="output_wrong_predictions",
                        help="保存错分图片和 csv 报告的输出目录。")
    parser.add_argument("--image-width", type=int, default=96,
                        help="模型输入宽度。")
    parser.add_argument("--image-height", type=int, default=72,
                        help="模型输入高度。")
    parser.add_argument("--no-openmv-center-crop", dest="openmv_center_crop", action="store_false",
                        help="关闭 OpenMV 风格的等比放大中心裁剪，改用普通 resize。")
    parser.set_defaults(openmv_center_crop=True)
    parser.add_argument("--alpha", type=float, default=0.35,
                        help="MobileNetV2 的 alpha 参数，需要与训练时一致。")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="验证推理时的 batch size。")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="可选 unknown 阈值；最高置信度低于该值时记为 _unknown，0 表示关闭。")
    parser.add_argument("--clean-output", action="store_true",default=True,
                        help="评估前先删除输出目录。")
    return parser.parse_args()


def load_class_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def find_latest_weights(save_root, save_prefix):
    candidates = [
        path for path in Path(save_root).glob(f"{save_prefix}*")
        if path.is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"在 {save_root} 下找不到前缀为 {save_prefix} 的权重目录")

    latest_dir = max(candidates, key=lambda path: path.stat().st_mtime)
    ckpt_files = sorted(latest_dir.glob("*.ckpt.index"))
    if not ckpt_files:
        raise FileNotFoundError(f"在 {latest_dir} 中找不到 .ckpt.index 文件")

    return str(ckpt_files[0]).replace(".ckpt.index", ".ckpt")


def list_validation_images(val_dir, name_to_index):
    samples = []
    for class_dir in sorted(path for path in Path(val_dir).iterdir() if path.is_dir()):
        if class_dir.name not in name_to_index:
            continue
        label = name_to_index[class_dir.name]
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((str(image_path), label, class_dir.name))
    return samples


def resize_like_openmv_classify(tf, image, image_height, image_width):
    image_shape = tf.shape(image)
    height = image_shape[0]
    width = image_shape[1]
    target_height = tf.constant(image_height, dtype=tf.int32)
    target_width = tf.constant(image_width, dtype=tf.int32)

    image_too_wide = width * target_height > height * target_width
    crop_width_from_height = tf.cast(
        tf.round(tf.cast(height * target_width, tf.float32) / tf.cast(target_height, tf.float32)),
        tf.int32,
    )
    crop_height_from_width = tf.cast(
        tf.round(tf.cast(width * target_height, tf.float32) / tf.cast(target_width, tf.float32)),
        tf.int32,
    )

    crop_width = tf.cond(
        image_too_wide,
        lambda: tf.minimum(width, tf.maximum(1, crop_width_from_height)),
        lambda: width,
    )
    crop_height = tf.cond(
        image_too_wide,
        lambda: height,
        lambda: tf.minimum(height, tf.maximum(1, crop_height_from_width)),
    )

    offset_width = tf.maximum(0, (width - crop_width) // 2)
    offset_height = tf.maximum(0, (height - crop_height) // 2)
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_height, crop_width)
    return tf.image.resize(image, [image_height, image_width])


def preprocess_image(tf, img_path, image_height, image_width, openmv_center_crop=True):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if openmv_center_crop:
        image = resize_like_openmv_classify(tf, image, image_height, image_width)
    else:
        image = tf.image.resize(image, [image_height, image_width])
    image = (image - 0.5) * 2.0
    return image


def safe_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
        return destination

    index = 1
    while True:
        candidate = destination.with_name(f"{destination.stem}_{index:03d}{destination.suffix}")
        if not candidate.exists():
            shutil.copy2(source, candidate)
            return candidate
        index += 1


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    import tensorflow as tf
    from model_v2 import MobileNetV2

    class_dict = load_class_dict(args.class_json)
    name_to_index = {name: index for index, name in class_dict.items()}
    class_num = len(class_dict)

    val_dir = Path(args.dataset_dir) / args.val_subdir
    assert val_dir.exists(), f"找不到验证集目录：{val_dir}"

    weights_path = args.weights_path or find_latest_weights(args.save_root, args.save_prefix)
    if weights_path.endswith(".index"):
        weights_path = weights_path[:-len(".index")]
    assert Path(weights_path + ".index").exists(), f"找不到权重文件：{weights_path}.index"

    output_dir = Path(args.output_dir)
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = list_validation_images(val_dir, name_to_index)
    if not samples:
        raise RuntimeError(f"在验证集目录中找不到图片：{val_dir}")

    print(f"权重路径：{weights_path}")
    print(f"验证图片数：{len(samples)}")
    print(f"输入尺寸：{args.image_width}x{args.image_height}")
    print(f"OpenMV中心裁剪：{args.openmv_center_crop}")

    model = MobileNetV2(
        im_height=args.image_height,
        im_width=args.image_width,
        num_classes=class_num,
        alpha=args.alpha,
    )
    model.load_weights(weights_path).expect_partial()

    paths = [item[0] for item in samples]
    labels = [item[1] for item in samples]
    true_names = [item[2] for item in samples]

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels, true_names))
    dataset = dataset.map(
        lambda path, label, true_name: (
            preprocess_image(tf, path, args.image_height, args.image_width, args.openmv_center_crop),
            label,
            true_name,
            path,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    total = 0
    correct = 0
    unknown_count = 0
    class_total = Counter()
    class_correct = Counter()
    confusion = defaultdict(Counter)
    wrong_rows = []

    for images, batch_labels, batch_true_names, batch_paths in tqdm(dataset, desc="验证", dynamic_ncols=True):
        logits = model(images, training=False)
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        batch_labels_np = batch_labels.numpy()
        batch_paths_np = [p.decode("utf-8") for p in batch_paths.numpy()]
        batch_true_names_np = [n.decode("utf-8") for n in batch_true_names.numpy()]

        pred_indices = np.argmax(probs, axis=1)
        pred_scores = np.max(probs, axis=1)

        for path, true_label, true_name, pred_index, pred_score, prob in zip(
            batch_paths_np,
            batch_labels_np,
            batch_true_names_np,
            pred_indices,
            pred_scores,
            probs,
        ):
            total += 1
            true_label = int(true_label)
            pred_index = int(pred_index)
            pred_score = float(pred_score)
            class_total[true_label] += 1

            if args.threshold > 0 and pred_score < args.threshold:
                pred_name = "_unknown"
                unknown_count += 1
                is_correct = False
            else:
                pred_name = class_dict[pred_index]
                is_correct = pred_index == true_label

            confusion[true_name][pred_name] += 1

            if is_correct:
                correct += 1
                class_correct[true_label] += 1
                continue

            source = Path(path)
            target = (
                output_dir
                / "wrong_images"
                / true_name
                / f"pred_{pred_name}"
                / source.name
            )
            copied_path = safe_copy(source, target)

            top3_indices = np.argsort(prob)[-3:][::-1]
            top3 = ";".join(
                f"{class_dict[int(idx)]}:{float(prob[int(idx)]):.4f}"
                for idx in top3_indices
            )
            wrong_rows.append({
                "source_path": path,
                "copied_path": str(copied_path),
                "true_label": true_name,
                "pred_label": pred_name,
                "pred_score": f"{pred_score:.6f}",
                "top3": top3,
            })

    summary_rows = []
    for index in sorted(class_dict):
        total_i = class_total[index]
        correct_i = class_correct[index]
        acc_i = correct_i / total_i if total_i else 0.0
        summary_rows.append({
            "class": class_dict[index],
            "total": total_i,
            "correct": correct_i,
            "accuracy": f"{acc_i:.6f}",
        })

    confusion_rows = []
    all_pred_names = [class_dict[index] for index in sorted(class_dict)]
    if args.threshold > 0:
        all_pred_names.append("_unknown")
    for true_name in [class_dict[index] for index in sorted(class_dict)]:
        row = {"true_label": true_name}
        for pred_name in all_pred_names:
            row[pred_name] = confusion[true_name][pred_name]
        confusion_rows.append(row)

    write_csv(
        output_dir / "wrong_predictions.csv",
        wrong_rows,
        ["source_path", "copied_path", "true_label", "pred_label", "pred_score", "top3"],
    )
    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        ["class", "total", "correct", "accuracy"],
    )
    write_csv(
        output_dir / "confusion_matrix.csv",
        confusion_rows,
        ["true_label"] + all_pred_names,
    )

    accuracy = correct / total if total else 0.0
    print("\n========== 验证结果 ==========")
    print(f"准确率：{accuracy * 100:.2f}% ({correct}/{total})")
    print(f"错误数量：{len(wrong_rows)}")
    if args.threshold > 0:
        print(f"被阈值判为 unknown 的数量：{unknown_count}")
    print(f"输出目录：{output_dir.resolve()}")
    print("==============================")


if __name__ == "__main__":
    main()
