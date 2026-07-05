import argparse
import shutil
from pathlib import Path


# =========================
# 配置区：常改参数放这里
# =========================

# 要整理的目录。脚本会把它所有子文件夹中的图片移动到这个目录根部。
TARGET_DIR = "num_tnt"

# 支持识别为图片的扩展名。
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 如果移动后子文件夹里还剩非图片文件，默认不强删该文件夹，避免误删。
# 确认子文件夹里只有无用文件时，可以运行时加 --force-delete-dirs。
FORCE_DELETE_DIRS = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="把 num_tnt 子文件夹中的图片全部移动到 num_tnt 根目录，并删除原子文件夹。"
    )
    parser.add_argument(
        "--target-dir",
        default=TARGET_DIR,
        help="要整理的目标目录，默认 num_tnt。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的操作，不实际移动或删除文件。",
    )
    parser.add_argument(
        "--force-delete-dirs",
        action="store_true",
        default=FORCE_DELETE_DIRS,
        help="移动图片后强制删除子文件夹，即使里面还有非图片文件。",
    )
    return parser.parse_args()


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def make_unique_target(root_dir, source_path):
    """生成不会覆盖已有文件的目标路径。"""
    target_path = root_dir / source_path.name
    if not target_path.exists():
        return target_path

    # 同名时加上来源文件夹名，仍冲突则继续加编号。
    parent_name = source_path.parent.name
    stem = source_path.stem
    suffix = source_path.suffix
    candidate = root_dir / f"{parent_name}_{stem}{suffix}"
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = root_dir / f"{parent_name}_{stem}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def assert_safe_target(target_dir):
    target_dir = target_dir.resolve()
    cwd = Path.cwd().resolve()
    if not target_dir.exists() or not target_dir.is_dir():
        raise FileNotFoundError(f"找不到目标目录：{target_dir}")
    try:
        target_dir.relative_to(cwd)
    except ValueError as exc:
        raise RuntimeError(f"目标目录不在当前工作区内，为安全起见停止执行：{target_dir}") from exc
    return target_dir


def move_images_to_root(target_dir, dry_run):
    image_paths = [
        path for path in sorted(target_dir.rglob("*"))
        if is_image(path) and path.parent.resolve() != target_dir
    ]

    moved_count = 0
    for source_path in image_paths:
        destination_path = make_unique_target(target_dir, source_path)
        print(f"[移动] {source_path} -> {destination_path}")
        if not dry_run:
            shutil.move(str(source_path), str(destination_path))
        moved_count += 1

    return moved_count


def delete_child_dirs(target_dir, dry_run, force_delete_dirs):
    child_dirs = [
        path for path in sorted(target_dir.iterdir(), reverse=True)
        if path.is_dir()
    ]

    deleted_count = 0
    skipped_count = 0
    for child_dir in child_dirs:
        remaining_items = list(child_dir.rglob("*"))
        if remaining_items and not force_delete_dirs:
            print(f"[跳过删除] {child_dir} 仍有非图片文件或未处理内容，确认后可加 --force-delete-dirs")
            skipped_count += 1
            continue

        print(f"[删除文件夹] {child_dir}")
        if not dry_run:
            shutil.rmtree(child_dir)
        deleted_count += 1

    return deleted_count, skipped_count


def main():
    args = parse_args()
    target_dir = assert_safe_target(Path(args.target_dir))

    print(f"目标目录：{target_dir}")
    if args.dry_run:
        print("当前为 dry-run 模式，不会实际移动或删除文件。")

    moved_count = move_images_to_root(target_dir, args.dry_run)
    deleted_count, skipped_count = delete_child_dirs(
        target_dir,
        args.dry_run,
        args.force_delete_dirs,
    )

    print("\n========== 完成 ==========")
    print(f"移动图片数量：{moved_count}")
    print(f"删除文件夹数量：{deleted_count}")
    print(f"跳过文件夹数量：{skipped_count}")
    print("==========================")


if __name__ == "__main__":
    main()
