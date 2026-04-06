import os
from PIL import Image

def crop_yolo_to_classification(img_dir, label_dir, output_dir):
    """
    根据 YOLO 标签精准裁剪图片，并按类别放入对应的子文件夹中
    """
    # 1. 预先创建 0 到 9 的文件夹
    for i in range(10):
        os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)
    
    print(f"📁 已在 {output_dir} 创建分类文件夹")
    print("🚀 开始精准裁剪...")
    
    # 记录裁剪总数
    total_cropped = 0
    
    # 2. 遍历 label 文件夹中的所有 txt 文件
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
            
        base_name = os.path.splitext(label_file)[0]
        
        # 3. 寻找对应的图片文件 (支持多种常见格式)
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
            temp_path = os.path.join(img_dir, base_name + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
                
        if not img_path:
            print(f"⚠️ 找不到 {label_file} 对应的图片，已跳过。")
            continue
            
        try:
            # 4. 打开图片并获取真实宽高
            with Image.open(img_path) as img:
                img_width, img_height = img.size
                
                # 5. 读取对应的 YOLO label 文件
                with open(os.path.join(label_dir, label_file), 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                # 遍历这张图片里的每一个目标 (一张图可能含有多个数字)
                for idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                        
                    # YOLO 格式：class_id x_center y_center width height (后面都是 0~1 的比例值)
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_width = float(parts[3])
                    box_height = float(parts[4])
                    
                    # 6. 将比例值还原为真实的像素坐标 [x_min, y_min, x_max, y_max]
                    x_min = int((x_center - box_width / 2) * img_width)
                    y_min = int((y_center - box_height / 2) * img_height)
                    x_max = int((x_center + box_width / 2) * img_width)
                    y_max = int((y_center + box_height / 2) * img_height)
                    
                    # 防呆设计：确保坐标不会超出图片边界
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(img_width, x_max)
                    y_max = min(img_height, y_max)
                    
                    # 检查框是否有效 (防止出现面积为 0 的异常框)
                    if x_max <= x_min or y_max <= y_min:
                        continue
                        
                    # 7. 严丝合缝地裁剪！
                    cropped_img = img.crop((x_min, y_min, x_max, y_max))
                    
                    # 8. 保存到对应的类别文件夹
                    # 命名规则：原图名_目标序号.jpg (防止一张图有多个数字导致覆盖)
                    save_name = f"{base_name}_{idx}.jpg"
                    
                    # 如果推理出了 0-9 以外的意外类别，也为它动态创建文件夹
                    class_folder = os.path.join(output_dir, str(class_id))
                    if not os.path.exists(class_folder):
                        os.makedirs(class_folder)
                        
                    save_path = os.path.join(class_folder, save_name)
                    
                    # 强转为 RGB 并保存 (防止原图是 RGBA 带有透明通道导致保存 JPG 报错)
                    cropped_img.convert('RGB').save(save_path, quality=100)
                    total_cropped += 1
                    
        except Exception as e:
            print(f"❌ 处理 {label_file} 时发生错误: {e}")

    print(f"\n✅ 任務完成！共成功裁剪並分類了 {total_cropped} 個目標數字。")

# ================= 使用方法 =================
if __name__ == '__main__':
    # 1. 存放原圖的資料夾路徑
    IMG_DIR = r"num_photo\images" 
    
    # 2. 存放 YOLO txt 標籤的資料夾路徑
    LABEL_DIR = r"num_photo\labels" 
    
    # 3. 裁剪後想要存放的總資料夾路徑 (程式會自動在裡面建 0-9 資料夾)
    OUTPUT_DIR = r"num_photo\photo_crop" 
    
    crop_yolo_to_classification(IMG_DIR, LABEL_DIR, OUTPUT_DIR)