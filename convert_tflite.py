import glob
import json
import os
import random
from pathlib import Path

import tensorflow as tf

from model_v2 import MobileNetV2


# =========================
# INT8 TFLite 量化配置：常改参数集中放这里
# =========================

# 留空时自动选择 save_weights 下最新的 try4_final_real_v1* 权重目录。
# 也可以手动填写：
#   1. 权重目录：ROOT_PATH = r"save_weights\try4_final_real_v1_xxx_96x72"
#   2. ckpt 前缀：ROOT_PATH = r"save_weights\try4_final_real_v1_xxx_96x72\best.ckpt"
ROOT_PATH = ""
SAVE_ROOT = "save_weights"
SAVE_PREFIX = "num_synth_real_v1_20260705_215017_96x72"

# 默认转换 best.ckpt；如果想转换最后一轮，改成 "last.ckpt"。
WEIGHTS_NAME = "best.ckpt"

# 输出 tflite 文件名；留空时自动保存为 best_int8.tflite 或 last_int8.tflite。
OUTPUT_TFLITE_NAME = ""

# 模型参数会优先从权重目录里的 config.json 读取；读不到时使用下面这些默认值。
IMAGE_WIDTH = 96
IMAGE_HEIGHT = 72
NUM_CLASSES = 10
MODEL_ALPHA = 0.35
DROPOUT_RATE = 0.2
ENABLE_OPENMV_CENTER_CROP = True

# 代表性图片用于 INT8 校准，应该尽量接近 OpenMV 实际输入分布。
# 这里使用最终数据集里的合成/实拍训练图、真实验证图和 negative 图。
REPRESENTATIVE_DIRS = [
    r"dataset_num_synthetic_real_v1\train",
    r"dataset_num_synthetic_real_v1\val",
    r"dataset_num_synthetic_real_v1\negative",
]
REPRESENTATIVE_SAMPLE_LIMIT = 500
REPRESENTATIVE_RANDOM_SEED = 0
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def find_latest_run_dir():
    save_root = Path(SAVE_ROOT)
    candidates = [
        path for path in save_root.glob(f"{SAVE_PREFIX}*")
        if path.is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"在 {SAVE_ROOT} 下找不到前缀为 {SAVE_PREFIX} 的权重目录")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_weights_path():
    """解析要转换的 ckpt 权重路径，返回不带 .index 后缀的 ckpt 前缀。"""
    if ROOT_PATH:
        root_path = Path(ROOT_PATH)
        if root_path.is_dir():
            weights_path = root_path / WEIGHTS_NAME
        else:
            weights_path = root_path
    else:
        weights_path = find_latest_run_dir() / WEIGHTS_NAME

    weights_text = str(weights_path)
    if weights_text.endswith(".index"):
        weights_text = weights_text[:-len(".index")]
    if not weights_text.endswith(".ckpt"):
        weights_text += ".ckpt"

    index_path = Path(weights_text + ".index")
    if not index_path.exists():
        raise FileNotFoundError(f"找不到权重文件：{index_path}")
    return Path(weights_text)


def load_train_config(weights_path):
    config_path = weights_path.parent / "config.json"
    if not config_path.exists():
        print(f"[提示] 权重目录中没有 config.json，使用脚本顶部默认模型参数：{config_path}")
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_model_settings(config):
    image_width = int(config.get("image_width", IMAGE_WIDTH))
    image_height = int(config.get("image_height", IMAGE_HEIGHT))
    model_alpha = float(config.get("model_alpha", MODEL_ALPHA))
    dropout_rate = float(config.get("dropout_rate", DROPOUT_RATE))
    classes = config.get("classes", None)
    num_classes = len(classes) if classes else NUM_CLASSES
    openmv_center_crop = bool(config.get("enable_openmv_center_crop", ENABLE_OPENMV_CENTER_CROP))

    return {
        "image_width": image_width,
        "image_height": image_height,
        "model_alpha": model_alpha,
        "dropout_rate": dropout_rate,
        "num_classes": num_classes,
        "openmv_center_crop": openmv_center_crop,
    }


def collect_representative_images():
    rng = random.Random(REPRESENTATIVE_RANDOM_SEED)
    image_paths = []
    existing_dirs = [Path(directory) for directory in REPRESENTATIVE_DIRS if Path(directory).exists()]
    if not existing_dirs:
        raise FileNotFoundError(f"找不到代表性图片目录：{REPRESENTATIVE_DIRS}")

    per_dir_limit = max(1, REPRESENTATIVE_SAMPLE_LIMIT // len(existing_dirs))
    for directory in existing_dirs:
        paths = []
        for extension in IMAGE_EXTENSIONS:
            paths.extend(glob.glob(str(directory / "**" / f"*{extension}"), recursive=True))
        paths = sorted(set(paths))
        rng.shuffle(paths)
        image_paths.extend(paths[:per_dir_limit])
        print(f"[代表性数据] {directory}: 抽取 {min(len(paths), per_dir_limit)} / {len(paths)} 张")

    rng.shuffle(image_paths)
    if not image_paths:
        raise FileNotFoundError("没有找到任何代表性图片")

    return image_paths[:REPRESENTATIVE_SAMPLE_LIMIT]


def resize_like_openmv_classify(image, image_height, image_width):
    """模拟 OpenMV tf.classify：等比放大到覆盖输入尺寸，再中心裁剪。"""
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

    # 训练端使用 [-1, 1]；全 INT8 转换后，输入量化通常会接近 pixel - 128 的部署形式。
    image = (image - 0.5) * 2.0
    return tf.expand_dims(image, 0)


def make_output_path(weights_path):
    if OUTPUT_TFLITE_NAME:
        output_name = OUTPUT_TFLITE_NAME
    else:
        output_name = f"{weights_path.stem}_int8.tflite"
    return weights_path.parent / output_name


def print_tflite_quant_info(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    print("[TFLite 输入量化参数]")
    print(f"  dtype: {input_detail['dtype']}")
    print(f"  quantization: {input_detail['quantization']}")
    print("[TFLite 输出量化参数]")
    print(f"  dtype: {output_detail['dtype']}")
    print(f"  quantization: {output_detail['quantization']}")


def main():
    weights_path = resolve_weights_path()
    config = load_train_config(weights_path)
    settings = get_model_settings(config)
    output_path = make_output_path(weights_path)

    print("========== INT8 TFLite 转换 ==========")
    print(f"权重路径：{weights_path}")
    print(f"输出路径：{output_path}")
    print(f"输入尺寸：{settings['image_width']}x{settings['image_height']}")
    print(f"类别数：{settings['num_classes']}")
    print(f"模型 alpha：{settings['model_alpha']}")
    print(f"OpenMV中心裁剪：{settings['openmv_center_crop']}")
    print("=====================================")

    model = MobileNetV2(
        im_height=settings["image_height"],
        im_width=settings["image_width"],
        num_classes=settings["num_classes"],
        alpha=settings["model_alpha"],
        dropout_rate=settings["dropout_rate"],
    )
    model.load_weights(str(weights_path)).expect_partial()

    image_paths = collect_representative_images()
    print(f"用于 INT8 校准的代表性图片数量：{len(image_paths)}")

    def representative_data_gen():
        for img_path in image_paths:
            yield [preprocess_image(img_path, settings)]

    print("开始转换 full INT8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(tflite_quant_model)

    print(f"转换完成：{output_path}")
    print(f"模型大小：{os.path.getsize(output_path) / 1024.0:.2f} KB")
    print_tflite_quant_info(output_path)


if __name__ == "__main__":
    main()
