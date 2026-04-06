import os
import random
from PIL import Image

def synthesize_images(bg_dir, fg_dir, output_dir, 
                      images_per_class=100, 
                      random_size=True,   # 【開關】是否開啟隨機大小
                      min_scale=0.1,      # 當開啟隨機大小時，最小縮放比例
                      max_scale=0.9):     # 當開啟隨機大小時，最大縮放比例
    """
    將前景圖(0-9)疊加到背景圖上。
    開啟隨機大小時會等比縮放；關閉時則保持原圖大小，直接隨機位置黏貼。
    """
    # 1. 準備輸出資料夾
    for i in range(10):
        os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

    # 2. 載入所有背景圖路徑
    bg_paths = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not bg_paths:
        print("❌ 錯誤：找不到任何背景圖！請檢查 bg_dir 路徑。")
        return

    print("🚀 開始合成圖片...")
    if random_size:
        print(f"⚙️ 當前模式: 隨機大小 (比例範圍: {min_scale} ~ {max_scale})")
    else:
        print(f"⚙️ 當前模式: 保持原圖大小直接黏貼")
        
    total_generated = 0

    # 3. 遍歷 0-9 每個類別
    for class_id in range(10):
        class_dir = os.path.join(fg_dir, str(class_id))
        if not os.path.exists(class_dir):
            continue

        fg_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not fg_paths:
            continue

        print(f"正在合成類別 {class_id} ...")

        for i in range(images_per_class):
            bg_path = random.choice(bg_paths)
            fg_path = random.choice(fg_paths)

            with Image.open(bg_path).convert('RGB') as bg_img, \
                 Image.open(fg_path).convert('RGB') as fg_img:

                bg_w, bg_h = bg_img.size
                fg_w, fg_h = fg_img.size

                # ==========================================
                # 大小處理邏輯
                # ==========================================
                if random_size:
                    # 【開啟隨機大小】：根據背景尺寸進行等比縮放
                    base_dim = min(bg_w, bg_h)
                    current_scale = random.uniform(min_scale, max_scale)
                    
                    target_fg_w = int(base_dim * current_scale)
                    aspect_ratio = fg_h / fg_w
                    target_fg_h = int(target_fg_w * aspect_ratio)

                    # 防出界安全鎖
                    target_fg_w = min(max(10, target_fg_w), bg_w - 1)
                    target_fg_h = min(max(10, target_fg_h), bg_h - 1)

                    paste_img = fg_img.resize((target_fg_w, target_fg_h), Image.Resampling.LANCZOS)
                    paste_w, paste_h = target_fg_w, target_fg_h
                else:
                    # 【關閉隨機大小】：不縮放，直接使用原圖
                    paste_img = fg_img
                    paste_w, paste_h = fg_w, fg_h

                    # 簡單防呆：如果你的原圖不小心比背景還大，PIL 也能處理，這裡只確保下面計算座標不報錯
                    paste_w = min(paste_w, bg_w)
                    paste_h = min(paste_h, bg_h)

                # ==========================================
                # 計算隨機位置 (保證不出界)
                # ==========================================
                max_x = bg_w - paste_w
                max_y = bg_h - paste_h

                paste_x = random.randint(0, max(0, max_x))
                paste_y = random.randint(0, max(0, max_y))

                # 複製背景並黏貼
                composite = bg_img.copy()
                composite.paste(paste_img, (paste_x, paste_y))

                save_name = f"synth_2_{class_id}_{i:04d}.jpg"
                save_path = os.path.join(output_dir, str(class_id), save_name)
                
                composite.save(save_path, quality=95)
                total_generated += 1

    print(f"\n✅ 任務完成！共成功合成了 {total_generated} 張圖片。")

# ================= 參數設定區 =================
if __name__ == '__main__':
    BG_DIR = r"dataset_background" 
    FG_DIR = r"num_photo\photo_crop" 
    OUTPUT_DIR = r"dataset_num_2" 
    
    synthesize_images(
        bg_dir=BG_DIR, 
        fg_dir=FG_DIR, 
        output_dir=OUTPUT_DIR,
        
        images_per_class=1000,      
        
        # --- 核心控制區 ---
        random_size=True,  # 👈 設為 False 時，完全不縮放，直接原圖黏貼
        
        # 當 random_size=True 時，才會使用下面這兩個比例進行隨機縮放：
        min_scale=0.8,     
        max_scale=0.98      
    )