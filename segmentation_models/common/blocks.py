from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import BatchNormalization
import keras


def Conv2DBlock(n_filters, kernel_size,
                activation='relu',
                use_batchnorm=True,
                name='conv_block',
                use_regularization=False,
                **kwargs):
    """Extension of Conv2D layer with batchnorm"""
    def layer(input_tensor):

        if use_regularization:
            x = Conv2D(n_filters, kernel_size, use_bias=not(use_batchnorm), name=name+'_conv', **kwargs)(input_tensor)
        else:
            x = Conv2D(n_filters, kernel_size, use_bias=not(use_batchnorm), kernel_regularizer=keras.regularizers.l2(0.0001), name=name + '_conv', **kwargs)(input_tensor)
        if use_batchnorm:
            x = BatchNormalization(name=name+'_bn',)(x)
        x = Activation(activation, name=name+'_'+activation)(x)

        return x
    return layer