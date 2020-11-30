# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Layers.py
import tensorflow as tf
from tensorflow import keras


class AddSingletonDepth(keras.layers.Layer):

    def call(self, x, mask=None):
        x = keras.backend.expand_dims(x, -1)  # add a dimension of the right

        if keras.backend.ndim(x) == 4:
            return keras.backend.permute_dimensions(x, (0, 3, 1, 2))
        else:
            return x

    def get_output_shape_for(self, input_shape):
        if len(input_shape) == 3:
            return input_shape[0], 1, input_shape[1], input_shape[2]
        else:
            return input_shape[0], input_shape[1], 1


class Subtract(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Subtract, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[0] - x[1]

    def get_output_shape_for(self, input_shape):
        return input_shape[0]


class CorrectFourier(keras.layers.Layer):
    def __init__(self, units=32,filter_size =500,stride = 1,padding ="SAME",**kwargs):
        self.units = units
        self.stride = stride
        self.padding = padding
        self.filter_size = filter_size
        super(CorrectFourier, self).__init__(**kwargs)
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            # 'selector': self.selector,
            'units': self.units,
            'stride': self.stride,
            'padding': self.padding,
            'filter_size': self.filter_size,
        })
        return config
    def build(self, input_shape):
        assert input_shape[-1] == 1
        w_init = tf.random_uniform_initializer(0,3.14)
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, self.units), dtype=tf.float32),
            trainable=True,
        )
        self.coeff = tf.constant(tf.range(self.filter_size,dtype=tf.float32),shape=(self.filter_size,1))

    def call(self, inputs):
        wn = tf.matmul(self.coeff, self.w)
        sin_kernels = tf.expand_dims(tf.sin(wn),1)
        cos_kernels = tf.expand_dims(tf.cos(wn),1)
        conv_sin = tf.nn.conv1d(inputs,sin_kernels,stride=self.stride , padding = self.padding)
        conv_cos = tf.nn.conv1d(inputs,cos_kernels,stride=self.stride , padding = self.padding)
        return tf.math.log(conv_sin**2 + conv_cos**2)

class CosineExtractor(keras.layers.Layer):
    def __init__(self, units=32,filter_size =500,stride = 1,padding ="SAME",**kwargs):
        self.units = units
        self.stride = stride
        self.padding = padding
        self.filter_size = filter_size
        super(CosineExtractor, self).__init__(**kwargs)
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            # 'selector': self.selector,
            'units': self.units,
            'stride': self.stride,
            'padding': self.padding,
            'filter_size': self.filter_size,
        })
        return config
    def build(self, input_shape):
        assert input_shape[-1] == 1
        w_init = tf.random_uniform_initializer(0,3.14)
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, self.units), dtype=tf.float32),
            trainable=True,
        )
        self.coeff = tf.constant(tf.range(self.filter_size,dtype=tf.float32),shape=(self.filter_size,1))

    def call(self, inputs):
        wn = tf.matmul(self.coeff, self.w)
        cos_kernels = tf.expand_dims(tf.cos(wn),1)
        conv_cos = tf.nn.conv1d(inputs,cos_kernels,stride=self.stride , padding = self.padding)
        return  tf.nn.conv1d(inputs,cos_kernels,stride=self.stride , padding = self.padding) / float(self.filter_size/2)

class Slice(keras.layers.Layer):

    def __init__(self, selector, output_shape, **kwargs):
        self.selector = selector
        self.desired_output_shape = output_shape
        super(Slice, self).__init__(**kwargs)
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            # 'selector': self.selector,
            'desired_output_shape': self.desired_output_shape,
        })
        return config
    def call(self, x, mask=None):

        selector = self.selector
        if len(self.selector) == 2 and not type(self.selector[1]) is slice and not type(self.selector[1]) is int:
            x = keras.backend.permute_dimensions(x, [0, 2, 1])
            selector = (self.selector[1], self.selector[0])

        y = x[selector]

        if len(self.selector) == 2 and not type(self.selector[1]) is slice and not type(self.selector[1]) is int:
            y = keras.backend.permute_dimensions(y, [0, 2, 1])

        return y


    def get_output_shape_for(self, input_shape):

        output_shape = (None,)
        for i, dim_length in enumerate(self.desired_output_shape):
            if dim_length == Ellipsis:
                output_shape = output_shape + (input_shape[i+1],)
            else:
                output_shape = output_shape + (dim_length,)
        return output_shape
