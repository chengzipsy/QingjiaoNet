import os
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

def simulate_screen_capture(
    image, 
    color_range,
    contrast_range,
    brightness_range,
    blur_radius_range,
    noise_range,
    rotation_range
):
    """
    模擬「用廉價相機拍攝螢幕」的全局劣化效果 (參數化版本)
    """
    # 1. 全局色彩與對比度劣化 
    image = ImageEnhance.Color(image).enhance(random.uniform(*color_range))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(*contrast_range))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(*brightness_range))

    # 2. 全局高斯模糊
    blur_radius = random.uniform(*blur_radius_range)
    if blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 3. 模擬高 ISO 的 RGB 彩色噪點
    img_array = np.array(image, dtype=np.float32)
    noise_intensity = random.uniform(*noise_range)
    if noise_intensity > 0:
        noise = np.random.normal(0, noise_intensity, img_array.shape)
        noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(noisy_img_array)

    # 4. 全局輕微旋轉 (模擬拍到螢幕黑邊)
    angle = random.uniform(*rotation_range)
    if angle != 0:
        image = image.rotate(angle, expand=False, resample=Image.BICUBIC, fillcolor=(0, 0, 0))

    return image

def synthesize_advanced_sim2real_dataset(
    bg_dir, 
    fg_dir, 
    output_dir, 
    samples_per_class, 
    # --- 增強參數接收 ---
    scale_range,
    color_range,
    contrast_range,
    brightness_range,
    blur_radius_range,
    noise_range,
    rotation_range,
    jpeg_quality_range
):
    os.makedirs(output_dir, exist_ok=True)
    
    bg_paths = list(Path(bg_dir).rglob("*.jpg")) + list(Path(bg_dir).rglob("*.jpeg")) + list(Path(bg_dir).rglob("*.png"))
    if not bg_paths:
        print(f"❌ 錯誤：在 {bg_dir} 中沒有找到背景圖片！")
        return

    fg_root = Path(fg_dir)
    class_dirs = [d for d in fg_root.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"❌ 錯誤：在 {fg_dir} 中未找到類別子資料夾！")
        return

    for class_dir in class_dirs:
        class_name = class_dir.name
        fg_paths = list(class_dir.rglob("*.jpg")) + list(class_dir.rglob("*.jpeg")) + list(class_dir.rglob("*.png"))
        
        if not fg_paths:
            continue
            
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        print(f"\n⏳ 正在準備生成類別 '{class_name}' 的實拍模擬數據...")
        
        for i in tqdm(range(samples_per_class), desc=f"生成進度 ({class_name})", unit="張"):
            bg = Image.open(random.choice(bg_paths)).convert("RGB")
            fg = Image.open(random.choice(fg_paths)).convert("RGB") 
            
            # --- 1. 完美的物理疊加 ---
            min_bg_side = min(bg.width, bg.height)
            target_size = max(1, int(min_bg_side * random.uniform(*scale_range)))
            
            fg_resized = fg.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
            max_x = max(0, bg.width - fg_resized.width)
            max_y = max(0, bg.height - fg_resized.height)
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)
            
            final_image = bg.copy()
            final_image.paste(fg_resized, (paste_x, paste_y))
            
            # --- 2. 模擬螢幕拍攝的全局劣化 (傳入所有參數) ---
            final_sim2real_image = simulate_screen_capture(
                final_image, 
                color_range=color_range,
                contrast_range=contrast_range,
                brightness_range=brightness_range,
                blur_radius_range=blur_radius_range,
                noise_range=noise_range,
                rotation_range=rotation_range
            )
            
            # --- 3. 儲存圖片 ---
            save_name = f"{class_name}_{i:04d}.jpg"
            jpeg_quality = random.randint(*jpeg_quality_range)
            final_sim2real_image.save(os.path.join(class_output_dir, save_name), quality=jpeg_quality)
            
    print(f"\n✅ 所有實拍模擬數據生成完畢！資料集保存在: {output_dir}")

if __name__ == "__main__":
    synthesize_advanced_sim2real_dataset(
        # 📂 1. 路徑與基礎設定
        bg_dir="dataset_background",               
        fg_dir=r"dataset_orign\cartooon_character",                   
        output_dir=r"dataset_new_dealed_2\added_cartoon",     
        samples_per_class=1000,                    
        
        scale_range=(0.7, 0.95),         # 前景方塊佔畫面「最短邊」的比例 (0.4 到 0.8)
        rotation_range=(-3.0, 3.0),     # 全局畫面的歪斜角度 (負為逆時針，正為順時針)，模擬手持相機沒拿正
        
        # 🎨 3. 色彩與曝光模擬 (1.0為原圖無變化)
        color_range=(0.8, 1),         # 飽和度：實拍螢幕通常顏色會變淡 (越小越黑白)
        contrast_range=(0.8, 1),      # 對比度：實拍因為螢幕發光，暗部會變灰 (越小越灰濛濛)
        brightness_range=(0.9, 1.2),    # 亮度波動：模擬環境光源的明暗變化
        
        blur_radius_range=(0.0, 0.4),   # 高斯模糊半徑：模擬相機失焦、廉價鏡頭邊緣不清晰 (0.0=不模糊)
        noise_range=(0, 10),       # RGB晶格噪點強度：實拍圖中最明顯的高 ISO 彩色雪花 (0.0=無噪點)
        jpeg_quality_range=(60, 90)     # JPEG壓縮失真度：100最高，降低可以產生不規則的方塊狀失真，增強真實感
    )