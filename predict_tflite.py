import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm


# =========================
# TFLite 推理配置：常改参数集中放这里
# =========================

# 留空时自动选择 save_weights 下最新 try4_final_real_v1* 目录里的 best_int8.tflite。
TFLITE_PATH = ""
SAVE_ROOT = "save_weights"
SAVE_PREFIX = "num_synth_real_v1"
TFLITE_NAME = "best_int8.tflite"

# 默认评估最终数据集中的真实验证集。
DATASET_DIR = "dataset_num_synthetic_real_v1"
VAL_SUBDIR = "val"

# 类别 json 留空时优先使用 tflite 同目录下的 class_indices_num.json。
CLASS_JSON = ""

# 错分图片、csv 报告、柱状图输出目录。
OUTPUT_DIR = "output_wrong_predictions_tflite"
CLEAN_OUTPUT = True

# 模型输入参数会优先从 tflite 同目录的 config.json 读取；读不到时使用这里的默认值。
IMAGE_WIDTH = 96
IMAGE_HEIGHT = 72
OPENMV_CENTER_CROP = True

# TFLite 批量推理设置；若模型不支持该 batch，会自动退回 batch=1。
BATCH_SIZE = 64
NUM_THREADS = 4

# 最高置信度阈值。0 表示关闭 unknown 判断；例如 0.55 表示低于 0.55 记为 _unknown。
THRESHOLD = 0.55

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
        description="使用 INT8 TFLite 模型批量评估数据集，并把错分图片复制到输出目录。",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出。")
    parser.add_argument("--tflite-path", default=TFLITE_PATH,
                        help="TFLite 模型路径；留空时自动选择最新模型。")
    parser.add_argument("--save-root", default=SAVE_ROOT,
                        help="自动选择最新模型时搜索的根目录。")
    parser.add_argument("--save-prefix", default=SAVE_PREFIX,
                        help="自动选择最新模型时使用的目录名前缀。")
    parser.add_argument("--tflite-name", default=TFLITE_NAME,
                        help="自动选择最新模型时使用的 tflite 文件名。")
    parser.add_argument("--dataset-dir", default=DATASET_DIR,
                        help="数据集根目录。")
    parser.add_argument("--val-subdir", default=VAL_SUBDIR,
                        help="要评估的数据集子目录，默认 val。")
    parser.add_argument("--class-json", default=CLASS_JSON,
                        help="类别索引 json；留空时自动使用模型目录下的 class_indices_num.json。")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="保存错分图片和报告的输出目录。")
    parser.add_argument("--image-width", type=int, default=IMAGE_WIDTH,
                        help="模型输入宽度；通常会从 config.json 自动覆盖。")
    parser.add_argument("--image-height", type=int, default=IMAGE_HEIGHT,
                        help="模型输入高度；通常会从 config.json 自动覆盖。")
    parser.add_argument("--no-openmv-center-crop", dest="openmv_center_crop", action="store_false",
                        help="关闭 OpenMV 风格中心裁剪，改用普通 resize。")
    parser.set_defaults(openmv_center_crop=OPENMV_CENTER_CROP)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="TFLite 批量推理 batch size。")
    parser.add_argument("--num-threads", type=int, default=NUM_THREADS,
                        help="TFLite 推理线程数。")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="unknown 阈值；最高置信度低于该值时记为 _unknown，0 表示关闭。")
    parser.add_argument("--clean-output", dest="clean_output", action="store_true", default=CLEAN_OUTPUT,
                        help="评估前先删除输出目录。")
    parser.add_argument("--no-clean-output", dest="clean_output", action="store_false",
                        help="评估前不删除输出目录。")
    return parser.parse_args()


def find_latest_tflite(save_root, save_prefix, tflite_name):
    candidates = [
        path for path in Path(save_root).glob(f"{save_prefix}*")
        if path.is_dir() and (path / tflite_name).exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"在 {save_root} 下找不到 {save_prefix}*\\{tflite_name}")

    latest_dir = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest_dir / tflite_name


def resolve_tflite_path(args):
    if args.tflite_path:
        tflite_path = Path(args.tflite_path)
    else:
        tflite_path = find_latest_tflite(args.save_root, args.save_prefix, args.tflite_name)
    if not tflite_path.exists():
        raise FileNotFoundError(f"找不到 TFLite 模型：{tflite_path}")
    return tflite_path


def load_config(tflite_path):
    config_path = tflite_path.parent / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_settings(args, config):
    return {
        "image_width": int(config.get("image_width", args.image_width)),
        "image_height": int(config.get("image_height", args.image_height)),
        "openmv_center_crop": bool(config.get("enable_openmv_center_crop", args.openmv_center_crop)),
    }


def resolve_class_json(args, tflite_path):
    if args.class_json:
        class_json = Path(args.class_json)
    else:
        class_json = tflite_path.parent / "class_indices_num.json"
        if not class_json.exists():
            class_json = Path("class_indices_num.json")
    if not class_json.exists():
        raise FileNotFoundError(f"找不到类别 json：{class_json}")
    return class_json


def load_class_dict(path):
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


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


def resize_like_openmv_classify(image, image_height, image_width):
    """模拟 OpenMV tf.classify：等比缩放到覆盖模型输入，再中心裁剪。"""
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
    return tf.image.resize(image, [image_height, image_width], method=tf.image.ResizeMethod.BILINEAR)


def preprocess_image(img_path, settings):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    if settings["openmv_center_crop"]:
        image = resize_like_openmv_classify(
            image,
            settings["image_height"],
            settings["image_width"],
        )
    else:
        image = tf.image.resize(image, [settings["image_height"], settings["image_width"]])

    # 与训练和量化校准保持一致。TFLite INT8 输入会在后面再按模型量化参数转成 int8。
    image = (image - 0.5) * 2.0
    return image


def quantize_input(images_float, input_detail):
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        return images_float.astype(np.float32)

    scale, zero_point = input_detail["quantization"]
    if scale == 0:
        raise ValueError("TFLite 输入量化 scale 为 0，无法量化输入。")

    quantized = np.round(images_float / scale + zero_point)
    if dtype == np.int8:
        return np.clip(quantized, -128, 127).astype(np.int8)
    if dtype == np.uint8:
        return np.clip(quantized, 0, 255).astype(np.uint8)

    raise TypeError(f"暂不支持的 TFLite 输入 dtype：{dtype}")


def dequantize_output(outputs, output_detail):
    dtype = output_detail["dtype"]
    if dtype == np.float32:
        return outputs.astype(np.float32)

    scale, zero_point = output_detail["quantization"]
    return (outputs.astype(np.float32) - zero_point) * scale


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def make_interpreter(tflite_path, batch_size, settings, num_threads):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=num_threads)
    input_detail = interpreter.get_input_details()[0]
    wanted_shape = [batch_size, settings["image_height"], settings["image_width"], 3]
    try:
        interpreter.resize_tensor_input(input_detail["index"], wanted_shape, strict=False)
        interpreter.allocate_tensors()
        return interpreter, batch_size
    except Exception as e:
        print(f"[提示] 当前 TFLite 不支持 batch={batch_size}，自动退回 batch=1。原因：{e}")
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=num_threads)
        input_detail = interpreter.get_input_details()[0]
        interpreter.resize_tensor_input(
            input_detail["index"],
            [1, settings["image_height"], settings["image_width"], 3],
            strict=False,
        )
        interpreter.allocate_tensors()
        return interpreter, 1


def ensure_interpreter_batch(interpreter, actual_batch, settings):
    input_detail = interpreter.get_input_details()[0]
    current_shape = list(input_detail["shape"])
    wanted_shape = [actual_batch, settings["image_height"], settings["image_width"], 3]
    if current_shape != wanted_shape:
        interpreter.resize_tensor_input(input_detail["index"], wanted_shape, strict=False)
        interpreter.allocate_tensors()


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


def save_accuracy_plot(output_dir, summary_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        with (output_dir / "accuracy_plot_error.txt").open("w", encoding="utf-8") as f:
            f.write(repr(e))
        return

    class_names = [row["class"] for row in summary_rows]
    accuracies = [float(row["accuracy"]) * 100 for row in summary_rows]

    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(12, 7))
    bars = plt.bar(class_names, accuracies, color="#4C98D9", edgecolor="black")
    plt.xlabel("类别")
    plt.ylabel("准确率 (%)")
    plt.title("TFLite INT8 各类别准确率")
    plt.ylim(0, 110)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        value = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 1.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_bar_chart.png", dpi=200)
    plt.close()


def main():
    args = parse_args()
    tflite_path = resolve_tflite_path(args)
    config = load_config(tflite_path)
    settings = get_settings(args, config)
    class_json = resolve_class_json(args, tflite_path)
    class_dict = load_class_dict(class_json)
    name_to_index = {name: index for index, name in class_dict.items()}

    val_dir = Path(args.dataset_dir) / args.val_subdir
    if not val_dir.exists():
        raise FileNotFoundError(f"找不到验证集目录：{val_dir}")

    output_dir = Path(args.output_dir)
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = list_validation_images(val_dir, name_to_index)
    if not samples:
        raise RuntimeError(f"在验证集目录中找不到图片：{val_dir}")

    interpreter, runtime_batch_size = make_interpreter(
        tflite_path,
        args.batch_size,
        settings,
        args.num_threads,
    )
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    print("========== TFLite 批量评估 ==========")
    print(f"TFLite 模型：{tflite_path}")
    print(f"类别 json：{class_json}")
    print(f"验证目录：{val_dir}")
    print(f"输出目录：{output_dir.resolve()}")
    print(f"验证图片数：{len(samples)}")
    print(f"输入尺寸：{settings['image_width']}x{settings['image_height']}")
    print(f"OpenMV中心裁剪：{settings['openmv_center_crop']}")
    print(f"batch size：{runtime_batch_size}")
    print(f"输入 dtype/量化：{input_detail['dtype']} / {input_detail['quantization']}")
    print(f"输出 dtype/量化：{output_detail['dtype']} / {output_detail['quantization']}")
    print("====================================")

    paths = [item[0] for item in samples]
    labels = [item[1] for item in samples]
    true_names = [item[2] for item in samples]

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels, true_names))
    dataset = dataset.map(
        lambda path, label, true_name: (
            preprocess_image(path, settings),
            label,
            true_name,
            path,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).batch(runtime_batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    total = 0
    correct = 0
    unknown_count = 0
    class_total = Counter()
    class_correct = Counter()
    confusion = defaultdict(Counter)
    wrong_rows = []

    for images, batch_labels, batch_true_names, batch_paths in tqdm(dataset, desc="TFLite推理", dynamic_ncols=True):
        images_np = images.numpy().astype(np.float32)
        actual_batch = images_np.shape[0]
        ensure_interpreter_batch(interpreter, actual_batch, settings)
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]

        input_data = quantize_input(images_np, input_detail)
        interpreter.set_tensor(input_detail["index"], input_data)
        interpreter.invoke()

        outputs_raw = interpreter.get_tensor(output_detail["index"])
        logits = dequantize_output(outputs_raw, output_detail)
        probs = softmax(logits)
        pred_indices = np.argmax(probs, axis=1)
        pred_scores = np.max(probs, axis=1)

        batch_labels_np = batch_labels.numpy()
        batch_paths_np = [p.decode("utf-8") for p in batch_paths.numpy()]
        batch_true_names_np = [n.decode("utf-8") for n in batch_true_names.numpy()]

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
                f"{class_dict[int(index)]}:{float(prob[int(index)]):.4f}"
                for index in top3_indices
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

    all_pred_names = [class_dict[index] for index in sorted(class_dict)]
    if args.threshold > 0:
        all_pred_names.append("_unknown")

    confusion_rows = []
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
    save_accuracy_plot(output_dir, summary_rows)

    accuracy = correct / total if total else 0.0
    print("\n========== TFLite 验证结果 ==========")
    print(f"准确率：{accuracy * 100:.2f}% ({correct}/{total})")
    print(f"错误数量：{len(wrong_rows)}")
    if args.threshold > 0:
        print(f"被阈值判为 unknown 的数量：{unknown_count}")
    print(f"输出目录：{output_dir.resolve()}")
    print("====================================")


if __name__ == "__main__":
    main()
