from keras import layers, Model, Sequential
import tensorflow as tf

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(layers.Layer):
    def __init__(self, out_channel, kernel_size=3, stride=1, **kwargs):
        super(ConvBNReLU, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=out_channel, kernel_size=kernel_size,
                                  strides=stride, padding='SAME', use_bias=False, name='Conv2d')
        self.bn = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='BatchNorm')
        self.activation = layers.ReLU(max_value=6.0)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x


class SEBlock(layers.Layer):
    def __init__(self, in_channels, reduction_ratio=4, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        reduced_ch = max(8, in_channels // reduction_ratio)
        
        self.pool = layers.GlobalAveragePooling2D()
        self.reshape_layer = layers.Reshape((1, 1, in_channels))
        
        self.conv1 = layers.Conv2D(filters=reduced_ch, kernel_size=1, strides=1, 
                                   padding='SAME', use_bias=False, activation='relu')
        self.conv2 = layers.Conv2D(filters=in_channels, kernel_size=1, strides=1, 
                                   padding='SAME', use_bias=False, activation='sigmoid')
        
        self.multiply = layers.Multiply()

    def call(self, inputs, training=False, **kwargs):
        x = self.pool(inputs)
        x = self.reshape_layer(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.multiply([inputs, x])

class InvertedResidual(layers.Layer):
    def __init__(self, in_channel, out_channel, stride, expand_ratio, **kwargs):
        super(InvertedResidual, self).__init__(**kwargs)
        self.hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layer_list = []
        if expand_ratio != 1:
            # 1x1 pointwise conv (擴展維度)
            layer_list.append(ConvBNReLU(out_channel=self.hidden_channel, kernel_size=1, name='expand'))

        layer_list.extend([
            # 3x3 depthwise conv (提取空間特徵)
            layers.DepthwiseConv2D(kernel_size=3, padding='SAME', strides=stride,
                                   use_bias=False, name='depthwise'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='depthwise/BatchNorm'),
            layers.ReLU(max_value=6.0),
        ])
        
        if expand_ratio != 1 and out_channel >= 32:
            layer_list.append(SEBlock(in_channels=self.hidden_channel, reduction_ratio=4))

        layer_list.extend([
            # 1x1 pointwise conv (線性降維投影)
            layers.Conv2D(filters=out_channel, kernel_size=1, strides=1,
                          padding='SAME', use_bias=False, name='project'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='project/BatchNorm')
        ])
        self.main_branch = Sequential(layer_list, name='expanded_conv')

    def call(self, inputs, training=False, **kwargs):
        if self.use_shortcut:
            return inputs + self.main_branch(inputs, training=training)
        else:
            return self.main_branch(inputs, training=training)


def MobileNetV2(im_height=128,
                im_width=128,
                num_classes=10,
                alpha=0.35,
                round_nearest=8,
                include_top=True):
    block = InvertedResidual
    input_channel = _make_divisible(32 * alpha, round_nearest)
    last_channel = _make_divisible(1280 * alpha, round_nearest)
    
    inverted_residual_setting = [
        # t (expand_ratio), c (channels), n (blocks), s (stride)
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        #[6, 320, 1, 1],
    ]

    input_image = layers.Input(shape=(im_height, im_width, 3), dtype='float32')
    # conv1
    x = ConvBNReLU(input_channel, stride=2, name='Conv')(input_image)
    
    # building inverted residual blocks
    for idx, (t, c, n, s) in enumerate(inverted_residual_setting):
        output_channel = _make_divisible(c * alpha, round_nearest)
        for i in range(n):
            stride = s if i == 0 else 1
            x = block(x.shape[-1],
                      output_channel,
                      stride,
                      expand_ratio=t)(x)
                      
    # building last several layers
    x = ConvBNReLU(last_channel, kernel_size=1, name='Conv_1')(x)

    if include_top is True:
        # building classifier
        x = layers.GlobalAveragePooling2D()(x)  # pool + flatten
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(num_classes, name='Logits')(x)
    else:
        output = x

    model = Model(inputs=input_image, outputs=output)
    return model