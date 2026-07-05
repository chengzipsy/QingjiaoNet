import glob
import csv
import json
import math
import os
import random
from datetime import datetime

import tensorflow as tf
from tqdm import tqdm

from model_v2 import MobileNetV2


# =========================
# 训练配置：常改参数集中放这里
# =========================

# 数据集结构：
# dataset_update_new_1/
#   train/类别名/*.jpg
#   val/类别名/*.jpg
#   negative/*.jpg
DATASET_DIR = "dataset_num_synthetic_real_v1"
TRAIN_SUBDIR = "train"
VAL_SUBDIR = "val"
NEGATIVE_SUBDIR = "negative"
RANDOM_SEED = 42

# 输入尺寸：尽量保持 4:3，和 OpenMV 画面比例接近；改大尺寸会增加部署端计算量。
IMAGE_WIDTH = 96
IMAGE_HEIGHT = 72

# 训练超参数
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
MODEL_ALPHA = 0.35
DROPOUT_RATE = 0.2

# 负样本策略：模型仍然是 10 分类；negative 训练目标是均匀分布，压低未知画面的最高置信度。
NEGATIVE_BATCH_RATIO = 0.125
NEGATIVE_LOSS_WEIGHT = 0
NEGATIVE_VAL_RATE = 0.2
NEGATIVE_WARMUP_EPOCHS = 0
BN_UPDATE_WITH_NEGATIVE = False
EVAL_TRAIN_INFERENCE_EACH_EPOCH = True

# 轻量颜色增强：模拟引爆炸弹变红、光照和饱和度轻微变化，幅度不要开太大。
ENABLE_COLOR_AUGMENTATION = True
BRIGHTNESS_DELTA = 0.05
CONTRAST_LOWER = 0.90
CONTRAST_UPPER = 1.10
SATURATION_LOWER = 0.85
SATURATION_UPPER = 1.15
HUE_DELTA = 0.015
ENABLE_OPENMV_CENTER_CROP = True
# 合成数据已经包含位置、尺度、透视变化；这里默认关闭额外几何增强，避免把任务变得过难。
ENABLE_GEOMETRIC_AUGMENTATION = False
TRANSLATION_FACTOR = 0.06
ROTATION_FACTOR = 0.035
ZOOM_FACTOR = 0.10

# 保存和早停
SAVE_ROOT = "save_weights"
SAVE_PREFIX = "num_synth_real_v1_noneg"
CLASS_INDEX_JSON = "class_indices_num.json"
# 可选 "val_accuracy" 或 "val_loss"；当前更关注分类是否正确，默认按验证准确率选 best。
SAVE_BEST_BY = "val_accuracy"
PATIENCE = 30
MIN_DELTA = 1e-5

# GPU 选择
CUDA_VISIBLE_DEVICES = "0"

# 让 IDE/Windows 控制台尽量按 UTF-8 输出中文日志。
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES


def reset_metric(metric):
    if hasattr(metric, "reset_state"):
        metric.reset_state()
    else:
        metric.reset_states()


def list_images(pattern):
    return sorted(glob.glob(pattern, recursive=True))


HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "train_positive_loss",
    "train_negative_loss",
    "train_accuracy",
    "train_inference_loss",
    "train_inference_accuracy",
    "train_negative_max_confidence",
    "val_loss",
    "val_accuracy",
    "negative_val_loss",
    "negative_val_max_confidence",
    "negative_loss_weight_current",
    "learning_rate",
    "is_best",
    "best_epoch",
    "best_val_loss",
    "best_val_accuracy",
    "save_best_by",
    "best_weights_path",
]


def make_class_mapping(train_dir):
    data_class = sorted([
        cla for cla in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, cla))
    ])
    class_dict = {value: index for index, value in enumerate(data_class)}
    inverse_dict = {value: key for key, value in class_dict.items()}

    with open(CLASS_INDEX_JSON, "w", encoding="utf-8") as json_file:
        json.dump(inverse_dict, json_file, indent=4, ensure_ascii=False)

    return data_class, class_dict


def metric_value(metric):
    return float(metric.result().numpy())


def resize_like_openmv_classify(image):
    """模拟 OpenMV tf.classify 的 SCALE_ASPECT_EXPAND + CENTER：等比放大后中心裁剪。"""
    if not ENABLE_OPENMV_CENTER_CROP:
        return tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH], method=tf.image.ResizeMethod.BILINEAR)

    image_shape = tf.shape(image)
    height = image_shape[0]
    width = image_shape[1]
    target_height = tf.constant(IMAGE_HEIGHT, dtype=tf.int32)
    target_width = tf.constant(IMAGE_WIDTH, dtype=tf.int32)

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
    return tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH], method=tf.image.ResizeMethod.BILINEAR)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_history_csv(path, history_rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history_rows)


def plot_history(history_rows, save_path):
    if not history_rows:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        error_path = os.path.join(os.path.dirname(save_path), "result_plot_error.txt")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write("result.png 生成失败，请确认 matplotlib 是否可用。\n")
            f.write(repr(e))
        return

    epochs = [row["epoch"] for row in history_rows]
    train_loss_values = [row["train_loss"] for row in history_rows]
    val_loss_values = [row["val_loss"] for row in history_rows]
    train_acc_values = [row["train_accuracy"] * 100 for row in history_rows]
    train_infer_acc_values = [row.get("train_inference_accuracy", 0.0) * 100 for row in history_rows]
    val_acc_values = [row["val_accuracy"] * 100 for row in history_rows]
    train_neg_loss_values = [row["train_negative_loss"] for row in history_rows]
    val_neg_loss_values = [row["negative_val_loss"] for row in history_rows]
    train_neg_conf_values = [row["train_negative_max_confidence"] for row in history_rows]
    val_neg_conf_values = [row["negative_val_max_confidence"] for row in history_rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, train_loss_values, label="train")
    axes[0, 0].plot(epochs, val_loss_values, label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, train_acc_values, label="train")
    axes[0, 1].plot(epochs, train_infer_acc_values, label="train infer")
    axes[0, 1].plot(epochs, val_acc_values, label="val")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("%")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, train_neg_loss_values, label="train negative")
    axes[1, 0].plot(epochs, val_neg_loss_values, label="val negative")
    axes[1, 0].set_title("Negative Loss")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, train_neg_conf_values, label="train negative")
    axes[1, 1].plot(epochs, val_neg_conf_values, label="val negative")
    axes[1, 1].set_title("Negative Max Confidence")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def main():
    random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(gpus)
    if gpus:
        try:
            for gpu in gpus:
                print(gpu, gpus)
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            exit(-1)

    data_root = os.path.abspath(os.getcwd())
    image_path = os.path.join(data_root, DATASET_DIR)
    train_dir = os.path.join(image_path, TRAIN_SUBDIR)
    validation_dir = os.path.join(image_path, VAL_SUBDIR)
    negative_dir = os.path.join(image_path, NEGATIVE_SUBDIR)

    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)
    assert os.path.exists(negative_dir), "cannot find {}".format(negative_dir)

    os.makedirs(SAVE_ROOT, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{SAVE_PREFIX}_{run_id}_{IMAGE_WIDTH}x{IMAGE_HEIGHT}"
    run_dir = os.path.join(SAVE_ROOT, run_name)
    os.makedirs(run_dir, exist_ok=True)
    history_csv_path = os.path.join(run_dir, "history.csv")
    result_png_path = os.path.join(run_dir, "result.png")
    best_weights_path = os.path.join(run_dir, "best.ckpt")
    last_weights_path = os.path.join(run_dir, "last.ckpt")

    negative_batch_size = max(1, int(BATCH_SIZE * NEGATIVE_BATCH_RATIO))

    data_class, class_dict = make_class_mapping(train_dir)
    class_num = len(data_class)
    uniform_negative_label = tf.ones([class_num], dtype=tf.float32) / float(class_num)

    train_image_list = list_images(os.path.join(train_dir, "*", "*.jpg"))
    random.shuffle(train_image_list)
    train_num = len(train_image_list)
    assert train_num > 0, "cannot find any .jpg file in {}".format(train_dir)
    train_label_list = [class_dict[os.path.basename(os.path.dirname(path))] for path in train_image_list]

    val_image_list = list_images(os.path.join(validation_dir, "*", "*.jpg"))
    random.shuffle(val_image_list)
    val_num = len(val_image_list)
    assert val_num > 0, "cannot find any .jpg file in {}".format(validation_dir)
    val_label_list = [class_dict[os.path.basename(os.path.dirname(path))] for path in val_image_list]

    negative_image_list = list_images(os.path.join(negative_dir, "**", "*.jpg"))
    random.shuffle(negative_image_list)
    negative_num = len(negative_image_list)
    assert negative_num > 0, "cannot find any .jpg file in {}".format(negative_dir)

    if negative_num == 1:
        negative_val_image_list = negative_image_list
        negative_train_image_list = negative_image_list
    else:
        negative_val_num = max(1, int(negative_num * NEGATIVE_VAL_RATE))
        negative_val_num = min(negative_val_num, negative_num - 1)
        negative_val_image_list = negative_image_list[:negative_val_num]
        negative_train_image_list = negative_image_list[negative_val_num:]

    negative_train_num = len(negative_train_image_list)
    negative_val_num = len(negative_val_image_list)

    print("\n========== Training Config ==========")
    print(f"dataset: {image_path}")
    print(f"run_dir: {run_dir}")
    print(f"input size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"classes: {class_num} -> {data_class}")
    print(f"batch_size: {BATCH_SIZE}, negative_batch_size: {negative_batch_size}")
    print(f"negative_loss_weight: {NEGATIVE_LOSS_WEIGHT}, warmup_epochs: {NEGATIVE_WARMUP_EPOCHS}")
    print(f"color_augmentation: {ENABLE_COLOR_AUGMENTATION}")
    print(f"geometric_augmentation: {ENABLE_GEOMETRIC_AUGMENTATION}")
    print(f"openmv_center_crop: {ENABLE_OPENMV_CENTER_CROP}")
    print(f"save_best_by: {SAVE_BEST_BY}")
    print("=====================================\n")
    print("using {} positive images for training, {} positive images for validation.".format(train_num, val_num))
    print("using {} negative images for training, {} negative images for validation.".format(
        negative_train_num, negative_val_num
    ))

    write_json(os.path.join(run_dir, "config.json"), {
        "dataset_dir": DATASET_DIR,
        "train_subdir": TRAIN_SUBDIR,
        "val_subdir": VAL_SUBDIR,
        "negative_subdir": NEGATIVE_SUBDIR,
        "random_seed": RANDOM_SEED,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "model_alpha": MODEL_ALPHA,
        "dropout_rate": DROPOUT_RATE,
        "negative_batch_ratio": NEGATIVE_BATCH_RATIO,
        "negative_loss_weight": NEGATIVE_LOSS_WEIGHT,
        "negative_val_rate": NEGATIVE_VAL_RATE,
        "negative_warmup_epochs": NEGATIVE_WARMUP_EPOCHS,
        "bn_update_with_negative": BN_UPDATE_WITH_NEGATIVE,
        "eval_train_inference_each_epoch": EVAL_TRAIN_INFERENCE_EACH_EPOCH,
        "enable_color_augmentation": ENABLE_COLOR_AUGMENTATION,
        "brightness_delta": BRIGHTNESS_DELTA,
        "contrast_lower": CONTRAST_LOWER,
        "contrast_upper": CONTRAST_UPPER,
        "saturation_lower": SATURATION_LOWER,
        "saturation_upper": SATURATION_UPPER,
        "hue_delta": HUE_DELTA,
        "enable_openmv_center_crop": ENABLE_OPENMV_CENTER_CROP,
        "enable_geometric_augmentation": ENABLE_GEOMETRIC_AUGMENTATION,
        "translation_factor": TRANSLATION_FACTOR,
        "rotation_factor": ROTATION_FACTOR,
        "zoom_factor": ZOOM_FACTOR,
        "save_best_by": SAVE_BEST_BY,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "train_num": train_num,
        "val_num": val_num,
        "negative_train_num": negative_train_num,
        "negative_val_num": negative_val_num,
        "classes": data_class,
    })
    write_json(os.path.join(run_dir, "class_indices_num.json"), {
        index: name for name, index in class_dict.items()
    })

    geometric_augmentation = None
    if ENABLE_GEOMETRIC_AUGMENTATION:
        geometric_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomTranslation(
                height_factor=TRANSLATION_FACTOR,
                width_factor=TRANSLATION_FACTOR,
                fill_mode="nearest",
            ),
            tf.keras.layers.RandomRotation(
                factor=ROTATION_FACTOR,
                fill_mode="nearest",
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-ZOOM_FACTOR, ZOOM_FACTOR),
                width_factor=(-ZOOM_FACTOR, ZOOM_FACTOR),
                fill_mode="nearest",
            ),
        ], name="light_geometric_augmentation")

    def load_image(img_path, training=False):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = resize_like_openmv_classify(image)

        if training and ENABLE_COLOR_AUGMENTATION:
            image = tf.image.random_brightness(image, max_delta=BRIGHTNESS_DELTA)
            image = tf.image.random_contrast(image, lower=CONTRAST_LOWER, upper=CONTRAST_UPPER)
            image = tf.image.random_saturation(image, lower=SATURATION_LOWER, upper=SATURATION_UPPER)
            image = tf.image.random_hue(image, max_delta=HUE_DELTA)
            image = tf.clip_by_value(image, 0.0, 1.0)

        if training and geometric_augmentation is not None:
            image = geometric_augmentation(tf.expand_dims(image, axis=0), training=True)[0]
            image = tf.clip_by_value(image, 0.0, 1.0)

        # 与部署量化输入保持一致：训练使用 [-1, 1] 范围。
        return (image - 0.5) * 2.0

    def process_train_img(img_path, label):
        label = tf.one_hot(label, depth=class_num)
        image = load_image(img_path, training=True)
        return image, label

    def process_val_img(img_path, label):
        label = tf.one_hot(label, depth=class_num)
        image = load_image(img_path, training=False)
        return image, label

    def process_negative_train_img(img_path):
        image = load_image(img_path, training=True)
        return image, uniform_negative_label

    def process_negative_val_img(img_path):
        image = load_image(img_path, training=False)
        return image, uniform_negative_label

    autotune = tf.data.experimental.AUTOTUNE

    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
    train_dataset = train_dataset.shuffle(buffer_size=train_num) \
        .map(process_train_img, num_parallel_calls=autotune) \
        .batch(BATCH_SIZE).repeat().prefetch(autotune)

    train_inference_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
    train_inference_dataset = train_inference_dataset.map(process_val_img, num_parallel_calls=autotune) \
        .batch(BATCH_SIZE)

    negative_train_dataset = tf.data.Dataset.from_tensor_slices(negative_train_image_list)
    negative_train_dataset = negative_train_dataset.shuffle(buffer_size=negative_train_num) \
        .map(process_negative_train_img, num_parallel_calls=autotune) \
        .batch(negative_batch_size).repeat().prefetch(autotune)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list, val_label_list))
    val_dataset = val_dataset.map(process_val_img, num_parallel_calls=autotune) \
        .batch(BATCH_SIZE)

    negative_val_dataset = tf.data.Dataset.from_tensor_slices(negative_val_image_list)
    negative_val_dataset = negative_val_dataset.map(process_negative_val_img, num_parallel_calls=autotune) \
        .batch(negative_batch_size)

    model = MobileNetV2(
        im_height=IMAGE_HEIGHT,
        im_width=IMAGE_WIDTH,
        num_classes=class_num,
        alpha=MODEL_ALPHA,
        dropout_rate=DROPOUT_RATE,
    )
    model.summary()

    loss_object = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=LABEL_SMOOTHING,
    )
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        jit_compile=False,
    )

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_positive_loss = tf.keras.metrics.Mean(name="train_positive_loss")
    train_negative_loss = tf.keras.metrics.Mean(name="train_negative_loss")
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
    train_negative_max_confidence = tf.keras.metrics.Mean(name="train_negative_max_confidence")
    train_inference_loss = tf.keras.metrics.Mean(name="train_inference_loss")
    train_inference_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_inference_accuracy")

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_negative_loss = tf.keras.metrics.Mean(name="test_negative_loss")
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")
    test_negative_max_confidence = tf.keras.metrics.Mean(name="test_negative_max_confidence")

    @tf.function
    def train_step(images, labels, negative_images, negative_labels, negative_loss_weight_current):
        with tf.GradientTape() as tape:
            output = model(images, training=True)
            positive_loss = loss_object(labels, output)

            negative_output = model(negative_images, training=BN_UPDATE_WITH_NEGATIVE)
            negative_loss = loss_object(negative_labels, negative_output)

            # 负样本 loss 单独按 negative batch 求平均；再乘 NEGATIVE_BATCH_RATIO，
            # 才能让它的实际影响接近“负样本占 batch 的比例”。
            loss = positive_loss + negative_loss_weight_current * NEGATIVE_BATCH_RATIO * negative_loss

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_positive_loss(positive_loss)
        train_negative_loss(negative_loss)
        train_accuracy(labels, output)

        negative_prob = tf.nn.softmax(negative_output, axis=-1)
        train_negative_max_confidence(tf.reduce_mean(tf.reduce_max(negative_prob, axis=-1)))

    @tf.function
    def test_step(images, labels):
        output = model(images, training=False)
        t_loss = loss_object(labels, output)

        test_loss(t_loss)
        test_accuracy(labels, output)

    @tf.function
    def test_negative_step(images, labels):
        output = model(images, training=False)
        t_loss = loss_object(labels, output)
        prob = tf.nn.softmax(output, axis=-1)

        test_negative_loss(t_loss)
        test_negative_max_confidence(tf.reduce_mean(tf.reduce_max(prob, axis=-1)))

    @tf.function
    def test_train_inference_step(images, labels):
        output = model(images, training=False)
        t_loss = loss_object(labels, output)

        train_inference_loss(t_loss)
        train_inference_accuracy(labels, output)

    best_test_loss = float("inf")
    best_val_accuracy = 0.0
    train_step_num = max(1, math.ceil(train_num / BATCH_SIZE))
    val_step_num = max(1, math.ceil(val_num / BATCH_SIZE))
    negative_val_step_num = max(1, math.ceil(negative_val_num / negative_batch_size))
    patience_counter = 0
    best_epoch = 0
    history_rows = []

    for epoch in range(1, EPOCHS + 1):
        for metric in [
            train_loss,
            train_positive_loss,
            train_negative_loss,
            train_accuracy,
            train_negative_max_confidence,
            train_inference_loss,
            train_inference_accuracy,
            test_loss,
            test_negative_loss,
            test_accuracy,
            test_negative_max_confidence,
        ]:
            reset_metric(metric)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        if epoch <= NEGATIVE_WARMUP_EPOCHS:
            negative_loss_weight_current = 0.0
        else:
            negative_loss_weight_current = float(NEGATIVE_LOSS_WEIGHT)
        negative_loss_weight_tensor = tf.constant(negative_loss_weight_current, dtype=tf.float32)

        train_bar = tqdm(total=train_step_num, desc="Train", unit="step", dynamic_ncols=True)
        train_batches = zip(train_dataset, negative_train_dataset)
        for index, ((images, labels), (negative_images, negative_labels)) in enumerate(train_batches):
            train_step(images, labels, negative_images, negative_labels, negative_loss_weight_tensor)
            train_bar.update(1)
            train_bar.set_postfix({
                "loss": f"{train_loss.result():.4f}",
                "pos": f"{train_positive_loss.result():.4f}",
                "neg": f"{train_negative_loss.result():.4f}",
                "acc": f"{train_accuracy.result() * 100:.2f}%",
                "neg_w": f"{negative_loss_weight_current:.2f}",
                "neg_conf": f"{train_negative_max_confidence.result():.3f}",
            })
            if index + 1 == train_step_num:
                break
        train_bar.close()

        if EVAL_TRAIN_INFERENCE_EACH_EPOCH:
            train_eval_bar = tqdm(total=train_step_num, desc="TrainEval", unit="step", dynamic_ncols=True)
            for index, (images, labels) in enumerate(train_inference_dataset):
                test_train_inference_step(images, labels)
                train_eval_bar.update(1)
                train_eval_bar.set_postfix({
                    "loss": f"{train_inference_loss.result():.4f}",
                    "acc": f"{train_inference_accuracy.result() * 100:.2f}%",
                })
            train_eval_bar.close()

        val_bar = tqdm(total=val_step_num, desc="Val  ", unit="step", dynamic_ncols=True)
        for index, (images, labels) in enumerate(val_dataset):
            test_step(images, labels)
            val_bar.update(1)
            val_bar.set_postfix({
                "loss": f"{test_loss.result():.4f}",
                "acc": f"{test_accuracy.result() * 100:.2f}%",
            })
            if index + 1 == val_step_num:
                break
        val_bar.close()

        neg_val_bar = tqdm(total=negative_val_step_num, desc="NegVal", unit="step", dynamic_ncols=True)
        for index, (negative_images, negative_labels) in enumerate(negative_val_dataset):
            test_negative_step(negative_images, negative_labels)
            neg_val_bar.update(1)
            neg_val_bar.set_postfix({
                "loss": f"{test_negative_loss.result():.4f}",
                "max_conf": f"{test_negative_max_confidence.result():.3f}",
            })
            if index + 1 == negative_val_step_num:
                break
        neg_val_bar.close()

        current_test_loss = metric_value(test_loss)
        current_val_accuracy = metric_value(test_accuracy)
        if SAVE_BEST_BY == "val_accuracy":
            accuracy_improved = current_val_accuracy > (best_val_accuracy + MIN_DELTA)
            accuracy_tied = abs(current_val_accuracy - best_val_accuracy) <= MIN_DELTA
            loss_improved_on_tie = accuracy_tied and current_test_loss < (best_test_loss - MIN_DELTA)
            is_best = accuracy_improved or loss_improved_on_tie
        elif SAVE_BEST_BY == "val_loss":
            is_best = current_test_loss < (best_test_loss - MIN_DELTA)
        else:
            raise ValueError('SAVE_BEST_BY 只能设置为 "val_accuracy" 或 "val_loss"')

        if is_best:
            best_test_loss = current_test_loss
            best_val_accuracy = current_val_accuracy
            best_epoch = epoch
            patience_counter = 0
            model.save_weights(best_weights_path, save_format="tf")
        else:
            patience_counter += 1

        model.save_weights(last_weights_path, save_format="tf")

        history_rows.append({
            "epoch": epoch,
            "train_loss": metric_value(train_loss),
            "train_positive_loss": metric_value(train_positive_loss),
            "train_negative_loss": metric_value(train_negative_loss),
            "train_accuracy": metric_value(train_accuracy),
            "train_inference_loss": metric_value(train_inference_loss),
            "train_inference_accuracy": metric_value(train_inference_accuracy),
            "train_negative_max_confidence": metric_value(train_negative_max_confidence),
            "val_loss": current_test_loss,
            "val_accuracy": current_val_accuracy,
            "negative_val_loss": metric_value(test_negative_loss),
            "negative_val_max_confidence": metric_value(test_negative_max_confidence),
            "negative_loss_weight_current": negative_loss_weight_current,
            "learning_rate": LEARNING_RATE,
            "is_best": int(is_best),
            "best_epoch": best_epoch,
            "best_val_loss": best_test_loss,
            "best_val_accuracy": best_val_accuracy,
            "save_best_by": SAVE_BEST_BY,
            "best_weights_path": best_weights_path,
        })
        write_history_csv(history_csv_path, history_rows)
        plot_history(history_rows, result_png_path)

        print(f"Run files: {run_dir}")
        if is_best:
            print(f"Best weights updated: {best_weights_path}")

        if patience_counter >= PATIENCE:
            print("")
            print(f"[Early Stop] {PATIENCE} epochs without improvement. Stop training.")
            print(f"Best weights saved at: {best_weights_path}")
            break



if __name__ == "__main__":
    main()
