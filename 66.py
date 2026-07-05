import os
import cv2
import random
import numpy as np

def overlay_image(background, foreground, x, y):
    """
    智能叠加函数：
    如果 foreground 带有 Alpha 通道（经过斜拍处理），则进行透明度混合，去除空白边缘。
    如果只是普通 JPG，则直接覆盖。
    """
    h_bg, w_bg = background.shape[:2]
    h_fg, w_fg = foreground.shape[:2]

    # 计算交集区域 (防越界)
    fg_x1 = max(0, -x)
    fg_y1 = max(0, -y)
    eff_w = min(w_fg - fg_x1, w_bg - max(0, x))
    eff_h = min(h_fg - fg_y1, h_bg - max(0, y))

    if eff_w <= 0 or eff_h <= 0:
        return background

    bg_x1 = max(0, x)
    bg_y1 = max(0, y)

    fg_roi = foreground[fg_y1 : fg_y1 + eff_h, fg_x1 : fg_x1 + eff_w]
    roi_bg = background[bg_y1 : bg_y1 + eff_h, bg_x1 : bg_x1 + eff_w]

    # 判断是否包含透明通道 (4通道)
    if fg_roi.shape[2] == 4:
        # 分离出 Alpha 通道并归一化
        alpha_fg = fg_roi[:, :, 3] / 255.0
        alpha_bg = 1.0 - alpha_fg

        # 对 BGR 三个通道分别进行透明度混合
        for c in range(3):
            roi_bg[:, :, c] = (alpha_fg * fg_roi[:, :, c] + alpha_bg * roi_bg[:, :, c])
            
        background[bg_y1 : bg_y1 + eff_h, bg_x1 : bg_x1 + eff_w] = roi_bg
    else:
        # 普通 3 通道直接覆盖
        background[bg_y1 : bg_y1 + eff_h, bg_x1 : bg_x1 + eff_w] = fg_roi

    return background

def apply_perspective_warp(image, max_extent):
    """
    对图像进行随机透视变换（仅模拟左右斜拍）。
    自动添加 Alpha 通道，并将变换后空出的区域设为完全透明。
    """
    h, w = image.shape[:2]

    # 1. 将 JPG (BGR) 转换为带有透明通道的 BGRA
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # 随机生成形变程度 (0.05 到 max_extent 之间)
    extent = random.uniform(0.05, max_extent)
    
    # 计算 Y 轴和 X 轴的缩进像素量
    dy = int(h * extent)
    dx = int(w * extent * 0.3) # X 轴稍微缩进，模拟更真实的纵深感

    # 原图的四个角坐标: [左上, 右上, 右下, 左下]
    pts_src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # 修改点：仅在左右斜拍中随机选择 ⭐
    direction = random.choice(['left', 'right'])

    if direction == 'right': 
        # 往右拍：右边离镜头远，右边变短
        pts_dst = np.float32([[0, 0], [w - dx, dy], [w - dx, h - dy], [0, h]])
    else: 
        # 往左拍：左边离镜头远，左边变短
        pts_dst = np.float32([[dx, dy], [w, 0], [w, h], [dx, h - dy]])

    # 2. 计算透视变换矩阵并执行形变
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    
    # 将变换后空出的背景填充为纯黑+全透明
    warped = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    return warped

def generate_composite_dataset(src_fg_dir, src_bg_dir, dest_dir, config):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    background_pool = []
    for root, _, files in os.walk(src_bg_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                background_pool.append(os.path.join(root, file))
                
    if not background_pool:
        print("❌ 背景图文件夹为空！")
        return

    for class_name in os.listdir(src_fg_dir):
        fg_class_dir = os.path.join(src_fg_dir, class_name)
        if not os.path.isdir(fg_class_dir):
            continue

        fg_files = [f for f in os.listdir(fg_class_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
        if not fg_files:
            continue

        target_class_dir = os.path.join(dest_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)

        print(f"📂 正在处理类别: {class_name}，计划生成 {config['count_per_class']} 张图片...")

        for i in range(config['count_per_class']):
            fg_name = random.choice(fg_files)
            fg_path = os.path.join(fg_class_dir, fg_name)
            bg_path = random.choice(background_pool)
            
            foreground = cv2.imread(fg_path)
            background = cv2.imread(bg_path)
            
            if foreground is None or background is None:
                continue

            h_bg, w_bg = background.shape[:2]
            h_fg_orig, w_fg_orig = foreground.shape[:2]

            # --- 1. 随机缩放逻辑 ---
            scale_factor = random.uniform(config['scale_min'], config['scale_max'])
            target_h_fg = int(h_bg * scale_factor)
            ratio = w_fg_orig / h_fg_orig
            target_w_fg = int(target_h_fg * ratio)
            
            if target_h_fg > 0 and target_w_fg > 0:
                foreground_processed = cv2.resize(foreground, (target_w_fg, target_h_fg), interpolation=cv2.INTER_AREA)
            else:
                continue

            # --- 2. 随机斜拍透视变换逻辑 (仅左右) ---
            if random.random() < config['skew_probability']:
                foreground_processed = apply_perspective_warp(foreground_processed, config['skew_max_extent'])

            h_fg_s, w_fg_s = foreground_processed.shape[:2]

            # --- 3. 随机位置与出界逻辑 ---
            y_pos = random.randint(0, max(0, h_bg - h_fg_s))

            default_x_min = 0
            default_x_max = w_bg - w_fg_s
            x_pos = random.randint(default_x_min, max(0, default_x_max))

            if random.random() < config['oob_probability']:
                actual_oob_range = random.uniform(0.0, config['oob_max_extent'])
                max_oob_px = int(w_fg_s * actual_oob_range)
                
                if max_oob_px > 0:
                    if random.choice(['left', 'right']) == 'left':
                        x_pos = random.randint(-max_oob_px, 0)
                    else:
                        x_pos = random.randint(default_x_max, default_x_max + max_oob_px)

            # --- 4. 执行叠加 ---
            result = overlay_image(background.copy(), foreground_processed, x_pos, y_pos)

            save_name = f"{class_name}_{i:04d}_2.jpg"
            cv2.imwrite(os.path.join(target_class_dir, save_name), result)

    print("\n✅ 所有类别图片生成完毕！")

# ==========================================
# 🛑 主配置项
# ==========================================
if __name__ == "__main__":
    CONFIG = {
        'count_per_class': 500,     
        'scale_min': 0.7,          
        'scale_max': 0.98,          
        'oob_probability': 0.3,    
        'oob_max_extent': 0.3, 
        
        # 斜拍配置
        'skew_probability': 0.1,   # 触发斜拍的概率 (0.5 = 50% 概率会倾斜)
        'skew_max_extent': 0.25,   # 倾斜最大形变程度 
    }

    FG_DIR = r"dataset_orign\cartooon_character"
    BG_DIR = r"background_photo"
    OUTPUT_DIR = r"dataset_new_dealed_6\cartoon"

    generate_composite_dataset(FG_DIR, BG_DIR, OUTPUT_DIR, CONFIG)