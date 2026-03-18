import os
import random
import glob
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

# ================= 配置區域 =================
# 1. 數據夾路徑設置
FG_DIR = "dataset_orign/cartooon_character"  # 前景圖目錄
BG_DIR = "dataset_background"                # 背景圖目錄（保持原始大小，如 208x150）
OUT_DIR = "dataset_new_dealed/added_cartoon"     # 生成的合成數據集輸出目錄

# 2. 基礎生成參數
NUM_PER_CLASS = 1000           # 每個類別想要生成的圖片數量
FG_SCALE_RANGE = (90, 140)    # 前景圖隨機縮放範圍

# --------------------------------------------
# 3. ⭐️ 實拍數據增強參數 (Sim2Real 調整區) ⭐️
# 你可以根據實拍圖的劣化程度，隨時在這裡微調這些參數
# --------------------------------------------
SIM_CONTRAST_RANGE = (0.8, 1)     # 全局對比度範圍 (降低對比度以模擬螢幕發白)
SIM_BRIGHTNESS_RANGE = (0.7, 1.2)   # 全局亮度範圍
SIM_SATURATION_RANGE = (0.8, 1)   # 全局色彩飽和度 (實拍圖通常色彩會變淡)
SIM_ROTATION_RANGE = (-3, 3)        # 隨機旋轉角度 (模擬單片機/攝像頭輕微歪斜)
SIM_BLUR_RADIUS = (0.2,0.5)        # 高斯模糊半徑 (模擬鏡頭失焦/邊緣模糊)
SIM_NOISE_INTENSITY = 6           # CMOS 噪點強度 (數值越大雪花點越多，0-255)
SIM_SCANLINE_INTENSITY = (8, 16)    # 螢幕掃描線/摩爾紋的強度 (透明度範圍)
# ============================================

def setup_directories(classes):
    """在輸出目錄中創建對應的類別數據夾"""
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    for cls_name in classes:
        cls_path = os.path.join(OUT_DIR, cls_name)
        if not os.path.exists(cls_path):
            os.makedirs(cls_path)

def get_image_paths(directory):
    """獲取目錄下所有的圖片路徑（自動去重）"""
    valid_exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    paths = []
    for ext in valid_exts:
        paths.extend(glob.glob(os.path.join(directory, ext)))
        paths.extend(glob.glob(os.path.join(directory, ext.upper())))
    return list(set(paths))

def generate_moire_pattern(size, intensity=15):
    """
    使用二維正弦波干涉來模擬真實螢幕拍攝時的彎曲摩爾紋
    """
    w, h = size
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)

    # 1. 隨機生成干涉中心點 (模擬鏡頭與螢幕的不完全平行)
    # 中心點可以偏出畫面外，這樣能產生大弧度的波浪紋
    cx = random.randint(-w, w * 2)
    cy = random.randint(-h, h * 2)
    
    # 計算每個像素點到干涉中心的距離
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    # 2. 模擬螢幕底層的物理網格頻率 (高頻)
    # freq_base 決定了底層網格的細密程度
    freq_x = random.uniform(0.5, 1.5)
    freq_y = random.uniform(0.5, 1.5)
    base_grid = np.sin(X * freq_x) * np.sin(Y * freq_y)
    
    # 3. 模擬鏡頭與螢幕疊加產生的空間干涉波紋 (低頻彎曲波)
    # freq_r 決定了摩爾紋同心圓/波浪的寬窄
    freq_r = random.uniform(0.02, 0.1)
    interference = np.sin(R * freq_r)
    
    # 4. 將網格與干涉波相乘，得出最終的摩爾紋形狀
    pattern = base_grid * interference
    
    # 5. 歸一化並轉換為 Alpha 透明通道
    # pattern 的範圍原本是 -1 到 1，將其平移映射到 0 到 1
    pattern = (pattern + 1) / 2.0 
    
    # 根據 intensity 參數控制摩爾紋的深淺
    alpha = (pattern * intensity).astype(np.uint8)
    
    # 6. 【可選增強】：真實的摩爾紋往往帶有輕微的 RGB 色偏
    # 這裡我們創建一個稍微偏灰黑色的圖層，你也可以加入微弱的彩色噪點
    color_layer = np.zeros((h, w, 3), dtype=np.uint8) 
    
    # 為了模擬螢幕發光中的暗條紋，將顏色設為深灰黑
    # color_layer[:, :] = [30, 30, 30] 
    
    rgba = np.dstack((color_layer, alpha))
    return Image.fromarray(rgba, mode='RGBA')

def apply_sim2real_augmentation(image):
    """將合成好的完美圖片，劣化為單片機實拍效果"""
    
    # 1. 降低對比度 (模擬螢幕反光泛白)
    if random.random() < 0.8:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(*SIM_CONTRAST_RANGE))
    
    # 2. 調整亮度
    if random.random() < 0.8:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(*SIM_BRIGHTNESS_RANGE))
        
    # 3. 降低飽和度
    if random.random() < 0.7:
        image = ImageEnhance.Color(image).enhance(random.uniform(*SIM_SATURATION_RANGE))

    # 4. 輕微旋轉 (模擬攝像頭沒對正，expand=False 確保圖片大小不變)
    angle = random.uniform(*SIM_ROTATION_RANGE)
    image = image.rotate(angle, expand=False, resample=Image.BICUBIC)

    # 5. 高斯模糊 (模擬鏡頭低畫質與失焦)
    if random.random() < 0.7:
        blur = random.uniform(*SIM_BLUR_RADIUS)
        image = image.filter(ImageFilter.GaussianBlur(radius=blur))

    # 6. CMOS 噪點 (ISO 噪點)
    if random.random() < 0.8:
        img_array = np.array(image, dtype=np.int16)
        noise = np.random.normal(0, SIM_NOISE_INTENSITY, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)

    # 7. 螢幕掃描線/摩爾紋疊加 (最關鍵的特徵)
    intensity = random.randint(*SIM_SCANLINE_INTENSITY)
    scanlines = generate_moire_pattern(image.size, intensity=intensity)
    image = Image.alpha_composite(image.convert('RGBA'), scanlines).convert('RGB')
    
    return image


def synthesize_images():
    bg_paths = get_image_paths(BG_DIR)
    if not bg_paths:
        raise ValueError(f"在 {BG_DIR} 中沒有找到背景圖片！")

    classes = [d for d in os.listdir(FG_DIR) if os.path.isdir(os.path.join(FG_DIR, d))]
    if not classes:
        raise ValueError(f"在 {FG_DIR} 中沒有找到類別數據夾！")
    
    setup_directories(classes)
    
    print(f"檢測到 {len(classes)} 個類別: {classes}")
    print(f"檢測到 {len(bg_paths)} 張背景圖")
    print(f"開始生成抗干擾數據集，每類生成 {NUM_PER_CLASS} 張...\n")

    for cls_name in classes:
        cls_dir = os.path.join(FG_DIR, cls_name)
        fg_paths = get_image_paths(cls_dir)
        
        if not fg_paths:
            print(f"警告：類別 {cls_name} 中沒有圖片，跳過。")
            continue

        print(f"正在生成類別 [{cls_name}] 的數據...")
        for i in tqdm(range(NUM_PER_CLASS)):
            bg_path = random.choice(bg_paths)
            fg_path = random.choice(fg_paths)

            # 1. 打開圖像並轉為 RGB
            bg_img = Image.open(bg_path).convert("RGB")
            fg_img = Image.open(fg_path).convert("RGB")
            bg_w, bg_h = bg_img.size

            # 2. 處理前景：等比縮放
            safe_max_size = min(bg_w, bg_h) - 5 
            safe_min_scale = min(FG_SCALE_RANGE[0], safe_max_size)
            safe_max_scale = min(FG_SCALE_RANGE[1], safe_max_size)
            
            target_scale = random.randint(safe_min_scale, safe_max_scale)
            
            w, h = fg_img.size
            ratio = target_scale / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            fg_img = fg_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

            # 3. 計算隨機黏貼的位置並組合 (此時還是完美的合成圖)
            max_x = max(0, bg_w - new_w)
            max_y = max(0, bg_h - new_h)
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)

            bg_img.paste(fg_img, (paste_x, paste_y))
            
            # 4. ⭐️ 核心：對整張組合後的圖片進行實拍劣化模擬 ⭐️
            final_img = apply_sim2real_augmentation(bg_img)
            
            # 5. 保存圖片 (適當降低 JPEG 質量以增加真實感)
            save_name = f"{cls_name}_{i:04d}.jpg"
            save_path = os.path.join(OUT_DIR, cls_name, save_name)
            final_img.save(save_path, quality=85)

    print("\n✅ 實拍風格數據集生成完畢！已保存在:", OUT_DIR)

if __name__ == "__main__":
    synthesize_images()