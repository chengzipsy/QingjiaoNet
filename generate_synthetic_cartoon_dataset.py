import argparse
import math
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


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
        description="将按类别裁剪好的前景图随机贴到背景图上，生成新的图像分类数据集。",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help",
                        help="显示帮助信息并退出。")
    parser.add_argument("--foreground-root", default=r"num_photo\\photo_crop",
                        help="前景图数据集根目录，子目录名会作为类别名。")
    parser.add_argument("--background-dir", default="background_photo",
                        help="背景图目录。")
    parser.add_argument("--output-root", default="synthetic_num_dataset",
                        help="输出的新分类数据集根目录。")
    parser.add_argument("--num-per-class", type=int, default=4000,
                        help="每个类别生成多少张合成图片。")
    parser.add_argument("--scale-min", type=float, default=0.6,
                        help="前景图长边占背景短边比例的最小值。")
    parser.add_argument("--scale-max", type=float, default=1.8,
                        help="前景图长边占背景短边比例的最大值。")
    parser.add_argument("--scale-distribution", choices=("normal", "uniform"), default="normal",
                        help="前景图大小的采样方式：normal 偏向较大的目标，uniform 为均匀随机。")
    parser.add_argument("--scale-base", choices=("long_edge", "height", "width"), default="height",
                        help="缩放比例参考方式：long_edge=前景长边占背景短边比例；height=前景高度占背景高度比例；width=前景宽度占背景宽度比例。")
    parser.add_argument("--scale-mean", type=float, default=0.98,
                        help="normal 大小采样的中心值；不填时默认取范围内偏大的位置。")
    parser.add_argument("--scale-std", type=float, default=0.2,
                        help="normal 大小采样的标准差；不填时默认约为取值范围的三分之一。")
    parser.add_argument("--scale-uniform-mix", type=float, default=0.2,
                        help="normal 模式下额外使用均匀采样的概率，用于保留少量小目标样本。")
    parser.add_argument("--overflow", type=float, default=0.25,
                        help="允许前景图超出背景边界的程度，按前景图自身尺寸比例计算。")
    parser.add_argument("--min-visible-ratio", type=float, default=0.35,
                        help="前景图随机越界后，要求至少可见的外接框面积比例。")
    parser.add_argument("--perspective", type=float, default=0.5,
                        help="左右转动的最大透视强度；不会模拟上下俯仰视角。")
    parser.add_argument("--perspective-distribution", choices=("normal", "uniform"), default="normal",
                        help="透视强度的采样方式：normal 偏向中心值附近，uniform 为均匀随机。")
    parser.add_argument("--perspective-mean", type=float, default=0,
                        help="normal 透视强度采样的中心值；不填时尽量默认使用 0.2。")
    parser.add_argument("--perspective-std", type=float, default=0.07,
                        help="normal 透视强度采样的标准差；不填时默认约为最大透视强度的七分之一。")
    parser.add_argument("--perspective-uniform-mix", type=float, default=0.10,
                        help="normal 模式下额外使用均匀透视采样的概率，用于保留少量弱透视和强透视样本。")
    parser.add_argument("--rotate", type=float, default=1.0,
                        help="轻微旋转的最大绝对角度，单位为度。")
    parser.add_argument("--canvas-size", type=int, nargs=2, metavar=("宽度", "高度"),
                        help="可选的固定输出尺寸；设置后背景图会先缩放到该尺寸再合成。")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子；设置后可以复现同样的生成结果。")
    parser.add_argument("--quality", type=int, default=99,
                        help="保存 JPG 图片时使用的 JPEG 质量。")
    parser.add_argument("--clean-output", action="store_true",default=True,
                        help="生成前先删除输出目录。")
    return parser.parse_args()


def list_images(path):
    root = Path(path)
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_class_images(foreground_root):
    root = Path(foreground_root)
    if not root.exists():
        raise FileNotFoundError(f"前景图数据集根目录不存在：{root}")

    class_to_images = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = list_images(class_dir)
        if images:
            class_to_images[class_dir.name] = images

    if not class_to_images:
        raise RuntimeError(f"没有在前景图数据集目录下找到类别图片：{root}")
    return class_to_images


def load_rgba(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGBA")


def sample_scale_ratio(args, rng):
    if args.scale_distribution == "uniform":
        return rng.uniform(args.scale_min, args.scale_max)

    scale_range = args.scale_max - args.scale_min
    mean = args.scale_mean
    if mean is None:
        mean = args.scale_min + scale_range * 0.83
    std = args.scale_std
    if std is None:
        std = scale_range / 3.2

    if rng.random() < args.scale_uniform_mix:
        return rng.uniform(args.scale_min, args.scale_max)

    for _ in range(100):
        scale = rng.gauss(mean, std)
        if args.scale_min <= scale <= args.scale_max:
            return scale

    return min(args.scale_max, max(args.scale_min, rng.gauss(mean, std)))


def resize_foreground(foreground, background_size, args, rng):
    bg_w, bg_h = background_size
    scale_ratio = sample_scale_ratio(args, rng)
    if args.scale_base == "height":
        target_size = scale_ratio * bg_h
        source_size = foreground.height
    elif args.scale_base == "width":
        target_size = scale_ratio * bg_w
        source_size = foreground.width
    else:
        target_size = scale_ratio * min(bg_w, bg_h)
        source_size = max(foreground.size)

    scale = target_size / max(1, source_size)
    new_w = max(1, int(round(foreground.width * scale)))
    new_h = max(1, int(round(foreground.height * scale)))
    return foreground.resize((new_w, new_h), Image.Resampling.LANCZOS)


def perspective_coefficients(src_points, dst_points):
    matrix = []
    vector = []
    for (x, y), (u, v) in zip(dst_points, src_points):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        vector.append(u)
        vector.append(v)
    return np.linalg.solve(np.asarray(matrix, dtype=np.float64), np.asarray(vector, dtype=np.float64))


def sample_perspective_depth(args, rng):
    if args.perspective <= 0:
        return 0.0
    if args.perspective_distribution == "uniform":
        return rng.uniform(0.0, args.perspective)

    mean = args.perspective_mean
    if mean is None:
        mean = 0.2 if args.perspective >= 0.2 else args.perspective * 0.65
    std = args.perspective_std
    if std is None:
        std = args.perspective / 7.0

    if rng.random() < args.perspective_uniform_mix:
        return rng.uniform(0.0, args.perspective)

    for _ in range(100):
        depth = rng.gauss(mean, std)
        if 0.0 <= depth <= args.perspective:
            return depth

    return min(args.perspective, max(0.0, rng.gauss(mean, std)))


def apply_perspective(foreground, args, rng):
    if args.perspective <= 0:
        return foreground

    width, height = foreground.size
    direction = rng.choice((-1, 1))
    depth = sample_perspective_depth(args, rng)
    side_scale = max(0.55, 1.0 - depth)
    side_inset = width * depth * 0.45

    if direction < 0:
        left_scale, right_scale = side_scale, 1.0
        left_x, right_x = side_inset, width
    else:
        left_scale, right_scale = 1.0, side_scale
        left_x, right_x = 0, width - side_inset

    left_top = (height * (1.0 - left_scale)) / 2.0
    left_bottom = height - left_top
    right_top = (height * (1.0 - right_scale)) / 2.0
    right_bottom = height - right_top

    raw_dst = [
        (left_x, left_top),
        (right_x, right_top),
        (right_x, right_bottom),
        (left_x, left_bottom),
    ]

    min_x = min(x for x, _ in raw_dst)
    min_y = min(y for _, y in raw_dst)
    max_x = max(x for x, _ in raw_dst)
    max_y = max(y for _, y in raw_dst)

    dst = [(x - min_x, y - min_y) for x, y in raw_dst]
    out_w = max(1, int(math.ceil(max_x - min_x)))
    out_h = max(1, int(math.ceil(max_y - min_y)))

    src = [(0, 0), (width, 0), (width, height), (0, height)]
    coeffs = perspective_coefficients(src, dst)
    return foreground.transform(
        (out_w, out_h),
        Image.Transform.PERSPECTIVE,
        coeffs,
        Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )


def apply_rotation(foreground, max_degrees, rng):
    if max_degrees <= 0:
        return foreground

    angle = rng.uniform(-max_degrees, max_degrees)
    return foreground.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0, 0),
    )


def visible_ratio(background_size, foreground_size, x, y):
    bg_w, bg_h = background_size
    fg_w, fg_h = foreground_size
    visible_w = max(0, min(bg_w, x + fg_w) - max(0, x))
    visible_h = max(0, min(bg_h, y + fg_h) - max(0, y))
    return (visible_w * visible_h) / max(1, fg_w * fg_h)


def random_position(background_size, foreground_size, overflow, min_visible_ratio, rng):
    bg_w, bg_h = background_size
    fg_w, fg_h = foreground_size

    low_x = int(round(-fg_w * overflow))
    high_x = int(round(bg_w - fg_w + fg_w * overflow))
    low_y = int(round(-fg_h * overflow))
    high_y = int(round(bg_h - fg_h + fg_h * overflow))
    if low_x > high_x:
        low_x, high_x = high_x, low_x
    if low_y > high_y:
        low_y, high_y = high_y, low_y

    for _ in range(100):
        x = rng.randint(low_x, high_x)
        y = rng.randint(low_y, high_y)
        if visible_ratio(background_size, foreground_size, x, y) >= min_visible_ratio:
            return x, y

    return (bg_w - fg_w) // 2, (bg_h - fg_h) // 2


def paste_with_clipping(background, foreground, x, y):
    bg_w, bg_h = background.size
    fg_w, fg_h = foreground.size

    left = max(0, x)
    top = max(0, y)
    right = min(bg_w, x + fg_w)
    bottom = min(bg_h, y + fg_h)
    if right <= left or bottom <= top:
        return background

    crop_box = (left - x, top - y, right - x, bottom - y)
    foreground_crop = foreground.crop(crop_box)
    background.alpha_composite(foreground_crop, (left, top))
    return background


def build_one_sample(foreground_path, background_path, args, rng):
    background = load_rgba(background_path)
    if args.canvas_size:
        background = background.resize(tuple(args.canvas_size), Image.Resampling.LANCZOS)

    foreground = load_rgba(foreground_path)
    foreground = resize_foreground(foreground, background.size, args, rng)
    foreground = apply_perspective(foreground, args, rng)
    foreground = apply_rotation(foreground, args.rotate, rng)

    x, y = random_position(
        background.size,
        foreground.size,
        args.overflow,
        args.min_visible_ratio,
        rng,
    )
    return paste_with_clipping(background, foreground, x, y).convert("RGB")


def validate_args(args):
    if args.num_per_class <= 0:
        raise ValueError("--num-per-class 必须大于 0")
    if args.scale_min <= 0 or args.scale_max <= 0:
        raise ValueError("--scale-min 和 --scale-max 必须大于 0")
    if args.scale_min > args.scale_max:
        raise ValueError("--scale-min 必须小于或等于 --scale-max")
    if args.scale_mean is not None and not args.scale_min <= args.scale_mean <= args.scale_max:
        raise ValueError("--scale-mean 必须位于 [--scale-min, --scale-max] 范围内")
    if args.scale_std is not None and args.scale_std <= 0:
        raise ValueError("--scale-std 必须大于 0")
    if not 0 <= args.scale_uniform_mix <= 1:
        raise ValueError("--scale-uniform-mix 必须位于 0 到 1 之间")
    if args.overflow < 0:
        raise ValueError("--overflow 必须大于或等于 0")
    if not 0 <= args.min_visible_ratio <= 1:
        raise ValueError("--min-visible-ratio 必须位于 0 到 1 之间")
    if args.perspective < 0:
        raise ValueError("--perspective 必须大于或等于 0")
    if args.perspective_mean is not None and not 0 <= args.perspective_mean <= args.perspective:
        raise ValueError("--perspective-mean 必须位于 [0, --perspective] 范围内")
    if args.perspective_std is not None and args.perspective_std <= 0:
        raise ValueError("--perspective-std 必须大于 0")
    if not 0 <= args.perspective_uniform_mix <= 1:
        raise ValueError("--perspective-uniform-mix 必须位于 0 到 1 之间")
    if args.rotate < 0:
        raise ValueError("--rotate 必须大于或等于 0")
    if args.canvas_size and any(v <= 0 for v in args.canvas_size):
        raise ValueError("--canvas-size 的宽高值必须大于 0")


def main():
    args = parse_args()
    validate_args(args)

    rng = random.Random(args.seed)
    class_to_images = collect_class_images(args.foreground_root)
    backgrounds = list_images(args.background_dir)
    if not backgrounds:
        raise RuntimeError(f"没有在背景图目录下找到图片：{args.background_dir}")

    output_root = Path(args.output_root)
    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    total = len(class_to_images) * args.num_per_class
    done = 0
    for class_name, foregrounds in class_to_images.items():
        class_output = output_root / class_name
        class_output.mkdir(parents=True, exist_ok=True)

        for index in range(args.num_per_class):
            foreground_path = rng.choice(foregrounds)
            background_path = rng.choice(backgrounds)
            image = build_one_sample(foreground_path, background_path, args, rng)
            image.save(class_output / f"{index:06d}.jpg", quality=args.quality, optimize=True)

            done += 1
            print(f"\r正在生成 {done}/{total}: {class_name}", end="", flush=True)
        print()

    print(f"完成。合成数据集已保存到：{output_root.resolve()}")


if __name__ == "__main__":
    main()
