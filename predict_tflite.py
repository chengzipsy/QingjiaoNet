import os
import glob
import numpy as np
import tensorflow as tf
import time

def main():
    tflite_path = "mobilenet_v2_int8.tflite"
    test_images_dir = "./dataset_dealed/val" # 测试用的文件夹
    im_height, im_width = 160, 160
    
    # 获取测试图片 (这里我们取前 32 张作为一组 Batch)
    image_paths = glob.glob(os.path.join(test_images_dir, "*/*.jpg"))[:32]
    batch_size = len(image_paths)

    # ================= 1. 加载 TFLite 解释器 =================
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    
    # 获取输入和输出的详细信息
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # 【关键】：获取量化参数 (用于 Float 和 INT8 之间的相互转换)
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    # ================= 2. 调整解释器以支持 Batch 输入 =================
    # 默认情况下 TFLite 输入 Batch 是 1，我们需要 Resize Tensor 来支持批量
    interpreter.resize_tensor_input(input_details['index'], [batch_size, im_height, im_width, 3])
    interpreter.allocate_tensors() # Resize 后必须重新分配内存

    # ================= 3. 准备批量输入数据 =================
    batch_data = []
    for img_path in image_paths:
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [im_height, im_width])
        image = (image - 0.5) * 2.0  # Float32 预处理
        
        # 【量化输入】：Float32 -> INT8
        # 公式: quantized_val = (float_val / scale) + zero_point
        image_np = image.numpy() / input_scale + input_zero_point
        image_int8 = np.clip(np.round(image_np), -128, 127).astype(np.int8)
        
        batch_data.append(image_int8)

    # 堆叠成 NumPy 数组，形状为 (batch_size, 160, 160, 3)
    batch_data = np.stack(batch_data)

    # ================= 4. 执行推理 =================
    t1 = time.perf_counter()
    
    interpreter.set_tensor(input_details['index'], batch_data)
    interpreter.invoke()
    outputs_int8 = interpreter.get_tensor(output_details['index'])
    
    print(f"Batch 推理耗时: {time.perf_counter() - t1:.4f} 秒")

    # ================= 5. 解析输出 =================
    # 【反量化输出】：INT8 -> Float32
    # 公式: float_val = (quantized_val - zero_point) * scale
    outputs_float = (outputs_int8.astype(np.float32) - output_zero_point) * output_scale

    # 打印预测结果
    predictions = np.argmax(outputs_float, axis=1)
    print("批量预测类别索引:", predictions)
    
    # 如果你有 class_indices.json，可以在这里加载并打印实际名称

if __name__ == '__main__':
    main()