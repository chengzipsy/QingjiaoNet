import os
import random
import glob
from PIL import Image
from tqdm import tqdm

# ================= 配置区域 =================
# 文件夹路径设置
FG_DIR = "dataset_orign\cartooon_character"       # 前景图目录（存放 1417x1417 的 JPG 图片）
BG_DIR = "dataset_background"           # 背景图目录（存放 208x150 的纯背景图）
OUT_DIR = "dataset_dealed"   # 生成的合成数据集输出目录

# 参数设置
NUM_PER_CLASS = 500           # 每个类别想要生成的图片数量
TARGET_BG_SIZE = (224, 224)   # 最终输出的背景图大小 (你提到的训练大小)
FG_SCALE_RANGE = (110, 200)    # 前景图叠加时的随机大小范围（像素边长）
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
    
    # 使用 set() 去除重复的路径，然后再转回 list
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

            # 1. 打开图像并统一转为 RGB 格式（去除任何可能干扰的通道）
            bg_img = Image.open(bg_path).convert("RGB")
            fg_img = Image.open(fg_path).convert("RGB")

            # 2. 处理背景：拉伸到训练所需的 224x224
            bg_img = bg_img.resize(TARGET_BG_SIZE, Image.Resampling.BILINEAR)

            # 3. 处理前景：等比缩放
            # 获取一个随机目标大小
            target_scale = random.randint(FG_SCALE_RANGE[0], FG_SCALE_RANGE[1])
            
            # 严格等比缩放逻辑 (虽然你的图是 1417x1417，但这样写可以兼容以后非正方形的图)
            w, h = fg_img.size
            ratio = target_scale / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            
            fg_img = fg_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

            # 4. 计算随机粘贴的位置 (确保箱子图案不会超出背景边界)
            max_x = max(0, TARGET_BG_SIZE[0] - new_w)
            max_y = max(0, TARGET_BG_SIZE[1] - new_h)
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)

            # 5. 叠加图片！
            # 注意：因为是 JPG，没有透明度，所以这里只有 2 个参数，去掉了 mask 遮罩
            bg_img.paste(fg_img, (paste_x, paste_y))
            
            # 6. 保存图片
            save_name = f"{cls_name}_{i:04d}.jpg"
            save_path = os.path.join(OUT_DIR, cls_name, save_name)
            bg_img.save(save_path, quality=95)

    print("\n✅ 数据集生成完毕！已保存在:", OUT_DIR)

if __name__ == "__main__":
    synthesize_images()