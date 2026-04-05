import os
from PIL import Image

def batch_crop_images(input_root, output_root, roi):
    """
    按指定 ROI 批量裁剪图片，并保持原有的文件夹结构。
    
    :param input_root: 原图片所在的总文件夹路径
    :param output_root: 裁剪后图片保存的总文件夹路径
    :param roi: 裁剪区域 (左上角 x, 左上角 y, 宽度 w, 高度 h)
    """
    # 解析 ROI: PIL 的 crop 函数需要 (left, upper, right, lower) 格式
    x, y, w, h = roi
    crop_box = (x, y, x + w, y + h)

    # 支持的图片格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    # 遍历输入文件夹
    for dirpath, dirnames, filenames in os.walk(input_root):
        
        # 获取当前文件夹相对于根目录的相对路径
        # 例如 input_root 是 "data/in"，dirpath 是 "data/in/folder1"
        # 那么 rel_path 就是 "folder1"
        rel_path = os.path.relpath(dirpath, input_root)
        
        # 拼接出对应的输出文件夹路径
        out_dir = os.path.join(output_root, rel_path)
        
        # 检查当前文件夹内是否有图片，如果有，才创建输出文件夹
        image_files = [f for f in filenames if f.lower().endswith(valid_extensions)]
        if image_files:
            os.makedirs(out_dir, exist_ok=True)
            print(f"📁 正在处理文件夹: {rel_path} ({len(image_files)} 张图片)")

            # 遍历并处理每张图片
            for filename in image_files:
                in_file_path = os.path.join(dirpath, filename)
                out_file_path = os.path.join(out_dir, filename)

                try:
                    # 打开图片 -> 裁剪 -> 保存
                    with Image.open(in_file_path) as img:
                        cropped_img = img.crop(crop_box)
                        # quality=100 保证 JPG 保存时不压缩画质
                        cropped_img.save(out_file_path, quality=100)
                except Exception as e:
                    print(f"❌ 处理图片 {filename} 时出错: {e}")

if __name__ == "__main__":
    # ================= 配置区域 =================
    
    # 1. 设置输入和输出文件夹路径
    # 注意：请使用完整的路径，或者确保代码在这个路径的上一级运行
    INPUT_FOLDER = r"num_photo\photo2\photos"   # 替换为你的原文件夹路径
    OUTPUT_FOLDER = r"num_photo\photo3" # 替换为你想要保存的文件夹路径
    
    # 2. 设置你的 ROI (Region of Interest)
    # 格式为: (x, y, width, height)
    # 因为你的原图是 320x240，请确保 x+width <= 320 且 y+height <= 240
    # 例如：从坐标(50, 50)开始，裁剪一个 200x150 的区域
    MY_ROI = (45, 35, 222, 165) 
    
    # ===========================================

    # 检查输入文件夹是否存在
    if not os.path.exists(INPUT_FOLDER):
        print("⚠️ 错误：找不到输入文件夹，请检查 INPUT_FOLDER 路径是否正确。")
    else:
        print("🚀 开始批量裁剪...")
        batch_crop_images(INPUT_FOLDER, OUTPUT_FOLDER, MY_ROI)
        print("✅ 所有图片裁剪并保存完毕！")