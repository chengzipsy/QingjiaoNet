import os
import glob
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    # ================= 1. 配置路径 =================
    # 【注意】请将下面的路径替换为你实际生成的 tflite 模型名！
    tflite_path = "save_weights\epoch30_batch32_learningrate0.001_test_loss0.2049214243888855_wh160.tflite" 
    test_images_dir = "./dataset_dealed/val"
    im_height, im_width = 160, 160
    
    # ================= 2. 加载类别索引映射 =================
    try:
        with open('class_indices.json', 'r') as f:
            class_dict = json.load(f)
        # class_dict 格式形如: {"0": "class_A", "1": "class_B"}
    except FileNotFoundError:
        print("错误: 找不到 class_indices.json！请确保在训练生成的目录下。")
        return

    # 制作一个通过名字反查索引的字典 (形如: {"class_A": 0})
    name_to_index = {v: int(k) for k, v in class_dict.items()}
    
    # 初始化统计数据字典
    class_correct = {int(k): 0 for k in class_dict.keys()}
    class_total = {int(k): 0 for k in class_dict.keys()}

    # 获取所有的验证图片路径
    image_paths = glob.glob(os.path.join(test_images_dir, "*/*.jpg"))
    if not image_paths:
        print(f"错误: 在 {test_images_dir} 未找到测试图片！")
        return

    print(f"共找到 {len(image_paths)} 张验证图片，开始进行 TFLite 评估...")

    # ================= 3. 加载 TFLite 解释器 =================
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # 获取量化参数
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    # 为了稳定处理整个验证集，我们将 batch 设置为 1 逐张推理
    interpreter.resize_tensor_input(input_details['index'], [1, im_height, im_width, 3])
    interpreter.allocate_tensors() 

    # ================= 4. 遍历整个验证集进行推理 =================
    for img_path in tqdm(image_paths, desc="推理进度"):
        # 1. 获取真实标签 (根据文件夹名称)
        class_name = os.path.basename(os.path.dirname(img_path))
        if class_name not in name_to_index:
            continue
        true_label = name_to_index[class_name]
        class_total[true_label] += 1

        # 2. 图像预处理
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [im_height, im_width])
        image = (image - 0.5) * 2.0  # Float32 预处理 [-1, 1]
        
        # 3. 量化输入: Float32 -> INT8
        image_np = image.numpy() / input_scale + input_zero_point
        image_int8 = np.clip(np.round(image_np), -128, 127).astype(np.int8)
        
        # 扩展维度为 (1, 160, 160, 3) 以匹配模型输入
        input_data = np.expand_dims(image_int8, axis=0)

        # 4. 执行推理
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        outputs_int8 = interpreter.get_tensor(output_details['index'])

        # 5. 反量化输出: INT8 -> Float32
        outputs_float = (outputs_int8.astype(np.float32) - output_zero_point) * output_scale

        # 6. 获取预测结果并统计
        prediction = np.argmax(outputs_float, axis=1)[0]
        if prediction == true_label:
            class_correct[true_label] += 1

    # ================= 5. 计算并打印结果 =================
    print("\n" + "="*40)
    print("模型各类准确率统计报告:")
    print("="*40)
    
    class_names_list = []
    accuracies_list = []
    
    for idx in sorted(class_total.keys()):
        name = class_dict[str(idx)]
        total = class_total[idx]
        correct = class_correct[idx]
        acc = correct / total if total > 0 else 0
        
        class_names_list.append(name)
        accuracies_list.append(acc * 100)
        
        print(f"类别: {name:<15} | 总数: {total:<4} | 正确: {correct:<4} | 准确率: {acc*100:5.1f}%")

    # ================= 6. 画柱状图 =================
    # 解决 matplotlib 在部分系统中不显示中文字符的问题
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] 
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 7))
    bars = plt.bar(class_names_list, accuracies_list, color='#4C98D9', edgecolor='black')
    
    plt.xlabel('类别 (Classes)', fontsize=12)
    plt.ylabel('准确率 / Accuracy (%)', fontsize=12)
    plt.title('TFLite INT8 量化模型各类别识别准确率', fontsize=14, fontweight='bold')
    plt.ylim(0, 110) # 给顶部留点空间写字
    plt.xticks(rotation=45, ha='right') # 倾斜标签防止重叠
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 在柱子上方标注具体百分比数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.1f}%', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path = 'accuracy_bar_chart.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ 柱状图已成功保存为: {os.path.abspath(save_path)}")
    plt.show() # 弹窗显示图片

if __name__ == '__main__':
    main()