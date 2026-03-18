import os
import glob
import tensorflow as tf
from model_v2 import MobileNetV2

# ================= 1. 参数配置 (必须与训练时完全一致) =================
im_height = 160
im_width = 160
num_classes = 10
alpha = 0.35  # 确保这里和你 trainGPU_mobilenet_v2.py 里的保持一致
weights_path = r'save_weights\new_epoch150_batch64_adamW_lr0.001_wh160\new_epoch150_batch64_adamW_lr0.001_wh160.ckpt'
dataset_dir = './dataset_new_dealed/val' # 使用验证集的一小部分作为代表性数据

def main():
    # ================= 2. 实例化模型并加载权重 =================
    print("正在加载模型和权重...")
    model = MobileNetV2(im_height=im_height, im_width=im_width, num_classes=num_classes, alpha=alpha)
    # 使用 expect_partial() 消除一些非关键变量未加载的警告
    model.load_weights(weights_path).expect_partial() 

    # ================= 3. 构建代表性数据生成器 =================
    # 获取 100 张验证集图片路径用于校准
    image_paths = glob.glob(os.path.join(dataset_dir, "*/*.jpg"))[:100]

    def representative_data_gen():
        for img_path in image_paths:
            # 这里的预处理必须和 trainGPU_mobilenet_v2.py 的 process_val_img 完全一致
            image = tf.io.read_file(img_path)
            image = tf.image.decode_jpeg(image)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize(image, [im_height, im_width])
            image = (image - 0.5) * 2.0  # 归一化到 [-1, 1]
            
            # TFLite 要求输入带有 batch 维度: [1, height, width, channels]
            image = tf.expand_dims(image, 0)
            yield [image]

    # ================= 4. 配置转换器 =================
    print("开始 TFLite 转换与 INT8 量化...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 开启默认优化
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # 提供代表性数据集
    converter.representative_dataset = representative_data_gen
    
    # 强制要求所有算子都被量化为 INT8，如果有不支持的算子会报错 (保证纯净性)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # 强制设定模型的输入和输出数据类型为 INT8 (这对微控制器极其重要)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()

    # ================= 5. 保存量化模型 =================
    save_path = r'save_weights\new_epoch150_batch64_adamW_lr0.001_wh160\new_epoch150_batch64_adamW_lr0.001_wh160'+'.tflite'
    with open(save_path, 'wb') as f:
        f.write(tflite_quant_model)
    print(f"转换成功！全整型量化模型已保存至: {save_path}")
    print(f"模型大小: {os.path.getsize(save_path) / 1024.0:.2f} KB")

if __name__ == '__main__':
    main()