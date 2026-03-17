import os
from model_v2 import MobileNetV2
import tensorflow as tf
import json
import time
import glob
import random
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(gpus)
    if gpus:
        try:
            for gpu in gpus:
                print(gpu,gpus)
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            exit(-1)

    data_root = os.path.abspath(os.path.join(os.getcwd(), ""))  # get data root path
    image_path = os.path.join(data_root, "", "dataset_dealed")  
    train_dir = os.path.join(image_path, "train")
    validation_dir = os.path.join(image_path, "val")
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    # create direction for saving weights
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    im_height = 160
    im_width = 160
    batch_size = 48
    epochs = 30

    # class dict
    data_class = [cla for cla in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cla))]
    class_num = len(data_class)
    class_dict = dict((value, index) for index, value in enumerate(data_class))

    # reverse value and key of dict
    inverse_dict = dict((val, key) for key, val in class_dict.items())
    # write dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # load train images list
    train_image_list = glob.glob(train_dir+"/*/*.jpg")
    random.shuffle(train_image_list)
    train_num = len(train_image_list)
    assert train_num > 0, "cannot find any .jpg file in {}".format(train_dir)
    train_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in train_image_list]

    # load validation images list
    val_image_list = glob.glob(validation_dir+"/*/*.jpg")
    random.shuffle(val_image_list)
    val_num = len(val_image_list)
    assert val_num > 0, "cannot find any .jpg file in {}".format(validation_dir)
    val_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in val_image_list]

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    def process_train_img(img_path, label):
        label = tf.one_hot(label, depth=class_num)
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [im_height, im_width])
        image = tf.image.random_flip_left_right(image)
        # image = (image - 0.5) / 0.5
        image = (image - 0.5) * 2.0
        return image, label

    def process_val_img(img_path, label):
        label = tf.one_hot(label, depth=class_num)
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [im_height, im_width])
        # image = (image - 0.5) / 0.5
        image = (image - 0.5) * 2.0
        return image, label

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # load train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
    train_dataset = train_dataset.shuffle(buffer_size=train_num)\
                                 .map(process_train_img, num_parallel_calls=AUTOTUNE)\
                                 .repeat().batch(batch_size).prefetch(AUTOTUNE)

    # load train dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list, val_label_list))
    val_dataset = val_dataset.map(process_val_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                             .repeat().batch(batch_size)

    # 实例化模型
    model = MobileNetV2(num_classes=10,alpha=0.35)

    #pre_weights_path = './pretrain0weights.ckpt'
    #assert len(glob.glob(pre_weights_path + "*")), "cannot find {}".format(pre_weights_path)
    #model.load_weights(pre_weights_path)
    # for layer_t in model.layers[:-1]:
    #     layer_t.trainable = False

    model.summary()

    # using keras low level api for training
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            output = model(images, training=True)
            loss = loss_object(labels, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, output)

    @tf.function
    def test_step(images, labels):
        output = model(images, training=False)
        t_loss = loss_object(labels, output)

        test_loss(t_loss)
        test_accuracy(labels, output)

    best_test_loss = float('inf')
    train_step_num = train_num // batch_size
    val_step_num = val_num // batch_size
    for epoch in range(1, epochs+1):
        train_loss.reset_states()        # clear history info
        train_accuracy.reset_states()    # clear history info
        test_loss.reset_states()         # clear history info
        test_accuracy.reset_states()     # clear history info

        print(f"\nEpoch {epoch}/{epochs}")

        train_bar = tqdm(total=train_step_num, desc="Train", unit="step", dynamic_ncols=True)
        for index, (images, labels) in enumerate(train_dataset):
            train_step(images, labels)
            train_bar.update(1)
            train_bar.set_postfix({
                'loss': f"{train_loss.result():.4f}", 
                'acc': f"{train_accuracy.result() * 100:.2f}%"
            })
            if index+1 == train_step_num:
                break
        train_bar.close()

        val_bar = tqdm(total=val_step_num, desc="Val  ", unit="step", dynamic_ncols=True)
        for index, (images, labels) in enumerate(val_dataset):
            test_step(images, labels)
            val_bar.update(1)
            val_bar.set_postfix({
                'loss': f"{test_loss.result():.4f}", 
                'acc': f"{test_accuracy.result() * 100:.2f}%"
            })
            if index+1 == val_step_num:
                break
        val_bar.close()

        if test_loss.result() < best_test_loss:
            best_test_loss = test_loss.result()
            save_path = f"./save_weights/epoch{epochs}_batch{batch_size}_learningrate0.001_wh{im_height}.ckpt"
            model.save_weights(save_path, save_format='tf')


if __name__ == '__main__':
    main()
