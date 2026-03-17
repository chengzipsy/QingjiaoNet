import os
import random
import glob
from PIL import Image, ImageEnhance
from tqdm import tqdm

# ================= 配置区域 =================
# 文件夹路径设置
FG_DIR = "dataset_orign/cartooon_character"  # 前景图目录
BG_DIR = "dataset_background"                # 背景图目录（保持原始大小，如 208x150）
OUT_DIR = "dataset_dealed/added_cartoon"                   # 生成的合成数据集输出目录

# 参数设置
NUM_PER_CLASS = 500           # 每个类别想要生成的图片数量

BRIGHTNESS_RANGE = (0.5, 1.0) 

FG_SCALE_RANGE = (90, 130)    
# ============================================

def setup_directories(classes):
    """在输出目录中创建对应的类别文件夹"""
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    for cls_name in classes:
        cls_path = os.path.join(OUT_DIR, cls_name)
        if not os.path.exists(cls_path):
            os.makedirs(cls_path)

def get_image_paths(directory):
    """获取目录下所有的图片路径（自动去重）"""
    valid_exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    paths = []
    for ext in valid_exts:
        paths.extend(glob.glob(os.path.join(directory, ext)))
        paths.extend(glob.glob(os.path.join(directory, ext.upper())))
    return list(set(paths))

def synthesize_images():
    bg_paths = get_image_paths(BG_DIR)
    if not bg_paths:
        raise ValueError(f"在 {BG_DIR} 中没有找到背景图片！")

    classes = [d for d in os.listdir(FG_DIR) if os.path.isdir(os.path.join(FG_DIR, d))]
    if not classes:
        raise ValueError(f"在 {FG_DIR} 中没有找到类别文件夹！")
    
    setup_directories(classes)
    
    print(f"检测到 {len(classes)} 个类别: {classes}")
    print(f"检测到 {len(bg_paths)} 张背景图")
    print(f"开始生成数据集，每类生成 {NUM_PER_CLASS} 张...\n")

    for cls_name in classes:
        cls_dir = os.path.join(FG_DIR, cls_name)
        fg_paths = get_image_paths(cls_dir)
        
        if not fg_paths:
            print(f"警告：类别 {cls_name} 中没有图片，跳过。")
            continue

        print(f"正在生成类别 [{cls_name}] 的数据...")
        for i in tqdm(range(NUM_PER_CLASS)):
            bg_path = random.choice(bg_paths)
            fg_path = random.choice(fg_paths)

            # 1. 打开图像并统一转为 RGB 格式
            bg_img = Image.open(bg_path).convert("RGB")
            fg_img = Image.open(fg_path).convert("RGB")
            
            # 获取当前背景图的实际大小 (比如 208x150)
            bg_w, bg_h = bg_img.size

            # 2. 【修改点】：亮度数据增强 (仅针对前景图)
            enhancer = ImageEnhance.Brightness(fg_img)
            # 在设定范围内随机取一个亮度系数
            brightness_factor = random.uniform(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
            fg_img = enhancer.enhance(brightness_factor)

            # 3. 处理前景：等比缩放，并防止前景图大于背景图
            # 动态计算安全的最大边长（留 5 个像素的边距）
            safe_max_size = min(bg_w, bg_h) - 5 
            
            # 如果配置的最小值都比背景大，就强行限制住
            safe_min_scale = min(FG_SCALE_RANGE[0], safe_max_size)
            safe_max_scale = min(FG_SCALE_RANGE[1], safe_max_size)
            
            target_scale = random.randint(safe_min_scale, safe_max_scale)
            
            w, h = fg_img.size
            ratio = target_scale / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            fg_img = fg_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

            # 4. 计算随机粘贴的位置
            max_x = max(0, bg_w - new_w)
            max_y = max(0, bg_h - new_h)
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)

            # 5. 叠加图片
            bg_img.paste(fg_img, (paste_x, paste_y))
            
            # 6. 保存图片
            save_name = f"{cls_name}_{i:04d}.jpg"
            save_path = os.path.join(OUT_DIR, cls_name, save_name)
            bg_img.save(save_path, quality=95)

    print("\n✅ 数据集生成完毕！已保存在:", OUT_DIR)

if __name__ == "__main__":
    synthesize_images()