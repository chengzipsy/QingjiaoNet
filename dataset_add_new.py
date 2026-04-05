import os
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

def recolor_foreground_advanced(image, dark_color, light_color):
    """
    保留了不均匀的物理质感和正确的蓝黑色，但去除了任何条纹干扰。
    """
    image = image.filter(ImageFilter.GaussianBlur(radius=0.6))
    
    gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape
    
    # 液晶分子的微弱不均匀斑块（保留这个能让纯黑色看起来像实拍）
    small_noise = np.random.rand(h // 4, w // 4).astype(np.float32)
    noise_img = Image.fromarray((small_noise * 255).astype(np.uint8))
    noise_img = noise_img.resize((w, h), Image.Resampling.BICUBIC)
    macro_noise = (np.array(noise_img, dtype=np.float32) / 255.0 - 0.5) * 0.3 
    
    dc = np.array(dark_color, dtype=np.float32)
    lc = np.array(light_color, dtype=np.float32)
    
    gray_3d = np.expand_dims(gray, axis=-1)
    macro_noise_3d = np.expand_dims(macro_noise, axis=-1)
    
    dark_variant = dc * (1.0 + macro_noise_3d * (1.0 - gray_3d))
    dark_variant = np.clip(dark_variant, 0, 255)
    
    mapped = gray_3d * lc + (1.0 - gray_3d) * dark_variant
    
    return Image.fromarray(np.clip(mapped, 0, 255).astype(np.uint8))

def apply_global_lighting_and_lens(image, params):
    """
    【核心修改】：去除了所有条纹，极大地增强了中心高光和四周暗角。
    """
    img_array = np.array(image, dtype=np.float32)
    h, w = img_array.shape[:2]
    
    # 1. 强烈的径向渐变光影遮罩 (中间亮，四周暗)
    Y, X = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    norm_distance = distance / max_dist
    
    center_b = random.uniform(*params['glare_range'])
    edge_b = random.uniform(*params['vignette_range'])
    
    # 使用 1.5 次方让明暗过渡更自然，光斑更集中在中心
    lighting_mask = center_b - (center_b - edge_b) * (norm_distance ** 1.5)
    lighting_mask = np.clip(lighting_mask, 0, 2.0)
    lighting_mask = np.expand_dims(lighting_mask, axis=-1)
    
    img_array = img_array * lighting_mask
    
    # 【注意：这里已经彻底删除了 LCD 网格条纹的代码，不会再有格子了】

    # 2. 相机亮度噪点 (Luma Noise)
    noise_intensity = random.uniform(*params['noise_range'])
    if noise_intensity > 0:
        noise = np.random.normal(0, noise_intensity, (h, w, 1))
        img_array = img_array + noise
        
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    image = Image.fromarray(img_array)

    # 3. 对比度与模糊劣化
    image = ImageEnhance.Contrast(image).enhance(random.uniform(*params['contrast_range']))
    
    blur_radius = random.uniform(*params['blur_radius_range'])
    if blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 4. 倾斜
    angle = random.uniform(*params['rotation_range'])
    if angle != 0:
        image = image.rotate(angle, expand=False, resample=Image.BICUBIC, fillcolor=(0, 0, 0))

    return image

def synthesize_advanced_sim2real_dataset(CONFIG):
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    bg_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        bg_paths.extend(list(Path(CONFIG['bg_dir']).rglob(ext)))
        
    fg_root = Path(CONFIG['fg_dir'])
    class_dirs = [d for d in fg_root.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        class_name = class_dir.name
        fg_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            fg_paths.extend(list(class_dir.rglob(ext)))
            
        if not fg_paths: continue
        class_output_dir = os.path.join(CONFIG['output_dir'], class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        for i in tqdm(range(CONFIG['samples_per_class']), desc=f"生成 ({class_name})"):
            bg = Image.open(random.choice(bg_paths)).convert("RGB")
            fg = Image.open(random.choice(fg_paths)).convert("RGB") 
            
            target_size = max(1, int(min(bg.width, bg.height) * random.uniform(*CONFIG['scale_range'])))
            fg_resized = fg.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
            fg_recolored = recolor_foreground_advanced(
                fg_resized, 
                dark_color=CONFIG['fg_dark_color'], 
                light_color=CONFIG['fg_light_color']
            )
            
            final_image = bg.copy()
            paste_x = random.randint(0, max(0, bg.width - fg_resized.width))
            paste_y = random.randint(0, max(0, bg.height - fg_resized.height))
            final_image.paste(fg_recolored, (paste_x, paste_y))
            
            final_sim2real_image = apply_global_lighting_and_lens(final_image, CONFIG)
            
            save_name = f"{class_name}_{i:04d}_final_1.jpg"
            final_sim2real_image.save(os.path.join(class_output_dir, save_name), quality=random.randint(*CONFIG['jpeg_quality_range']))


# ==========================================
# ⚙️ 强光影无条纹版本配置
# ==========================================
if __name__ == "__main__":
    CONFIG = {
        'bg_dir': "dataset_background",               
        'fg_dir': r"dataset_orign\cartooon_character",                   
        'output_dir': r"dataset_new_dealed_5\added_cartoon",     
        'samples_per_class': 20,
        
        'scale_range': (0.7, 1),         
        'rotation_range': (-1.0, 1.0),     
        
        # 🎨 【色彩区：保留你认可的深蓝/灰蓝】
        'fg_dark_color': (12, 16, 25),     
        'fg_light_color': (160, 180, 225), 
        
        # 💡 【核心修改区：极端强化“中间亮、四周暗”】
        'glare_range': (1.3, 1.5),       # ★ 大幅提亮中心（甚至会有轻微过曝感，模拟屏幕中心发光）
        'vignette_range': (0.1, 0.3),    # ★ 大幅压暗边缘（边缘几乎会变成深灰色/黑色）
        
        # 📷 【相机劣化特质】
        'contrast_range': (1.1, 1.3),    # 保持高对比度，让黑的更黑
        'blur_radius_range': (0.1, 0.4), 
        'noise_range': (2, 7),           
        
        'jpeg_quality_range': (65, 75)
    }
    
    synthesize_advanced_sim2real_dataset(CONFIG)