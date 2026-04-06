import os
import glob
import tensorflow as tf
from model_v2 import MobileNetV2


im_height = 64
im_width = 64
num_classes = 10
alpha = 0.35
root_path=r'save_weights\num_newdata_2_modelv2_epoch18_batch128_adamW_lr0.001_loss0.5023_wh64\num_newdata_2_modelv2_epoch18_batch128_adamW_lr0.001_loss0.5023_wh64'
weights_path = root_path+'.ckpt'
dataset_dir = r'dataset_num_2\val'

def main():
    print("正在加载模型和权重...")
    model = MobileNetV2(im_height=im_height, im_width=im_width, num_classes=num_classes, alpha=alpha)
    model.load_weights(weights_path).expect_partial() 

    image_paths = glob.glob(os.path.join(dataset_dir, "*/*.jpg"))[:100]

    random_translate = tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='nearest')    
    random_zoom = tf.keras.layers.RandomZoom(height_factor=(-0.15, 0.15), width_factor=(-0.15, 0.15), fill_mode='nearest')
    def representative_data_gen():
        for img_path in image_paths:
            image = tf.io.read_file(img_path)
            image = tf.image.decode_jpeg(image)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize(image, [im_height, im_width])

            image = random_translate(image, training=True)
            image = random_zoom(image, training=True)
            image = (image - 0.5) * 2.0
            
            image = tf.expand_dims(image, 0)
            yield [image]

    print("开始 TFLite 转换与 INT8 量化...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()

    save_path = root_path+'.tflite'
    with open(save_path, 'wb') as f:
        f.write(tflite_quant_model)
    print(f"转换成功！全整型量化模型已保存至: {save_path}")
    print(f"模型大小: {os.path.getsize(save_path) / 1024.0:.2f} KB")

if __name__ == '__main__':
    main()