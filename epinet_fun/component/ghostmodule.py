from math import ceil
import tensorflow.keras.layers
from tensorflow.keras.layers import SeparableConv2D, Conv2D, Concatenate, DepthwiseConv2D, Lambda, Layer, Activation
from tensorflow.keras import Model

class GhostModule_Org(Model):
    """
    The main Ghost module
    """
    def __init__(self, out, ratio, convkernel, dwkernel):
        super(GhostModule, self).__init__()
        self.ratio = ratio
        self.out = out
        self.conv_out_channel = ceil(self.out * 1.0 / ratio)
        self.conv = Conv2D(int(self.conv_out_channel), (convkernel, convkernel), use_bias=False,
                           strides=(1, 1), padding='valid', activation=None)
        self.depthconv = DepthwiseConv2D(dwkernel, 1, padding='same', use_bias=False,
                                         depth_multiplier=ratio-1, activation=None)
        self.slice = Lambda(self._return_slices, arguments={'channel': int(self.out - self.conv_out_channel)})
        self.concat = Concatenate()

    @staticmethod
    def _return_slices(x, channel):
        return x[:, :, :, :channel]

    def call(self, inputs):
        x = self.conv(inputs)
        if self.ratio == 1:
            return x
        dw = self.depthconv(x)
        dw = self.slice(dw)
        output = self.concat([x, dw])
        return output


def _return_slices(x, channel):
    return x[:, :, :, :channel]

def GhostModule(inputs, out,  ratio, convkernel, dwkernel):
    conv_out_channel = ceil(out * 1.0 / ratio)
    i = tensorflow.keras.layers.Input(shape=inputs)
    x = Conv2D(int(conv_out_channel), (convkernel, convkernel), use_bias=False,
                           strides=(1, 1), padding='valid', activation=None)(i)
    if ratio == 1:
        return x
    dw = DepthwiseConv2D(dwkernel, 1, padding='same', use_bias=False,
                                         depth_multiplier=ratio-1, activation=None)(x)
    dw = Lambda(_return_slices, arguments={'channel': int(out - conv_out_channel)})(dw)
    y = Concatenate()([x, dw])
    model = Model(inputs=i, outputs=y)
    return model

def GhostModuleDWSC(inputs, out,  ratio, convkernel, dwkernel):
    conv_out_channel = ceil(out * 1.0 / ratio)
    i = tensorflow.keras.layers.Input(shape=inputs)
    x = SeparableConv2D(int(conv_out_channel), (convkernel, convkernel),
                           strides=(1, 1), padding='valid')(i)
    if ratio == 1:
        return x
    dw = DepthwiseConv2D(dwkernel, 1, padding='same', use_bias=False,
                                         depth_multiplier=ratio-1, activation=None)(x)
    dwl = Lambda(_return_slices, arguments={'channel': int(out - conv_out_channel)})(dw)
    y = Concatenate()([x, dwl])
    model = Model(inputs=i, outputs=y)
    return model

def GhostModuleSeq(inputs, out,  ratio, convkernel, dwkernel):
    conv_out_channel = ceil(out * 1.0 / ratio)
    i = tensorflow.keras.layers.Input(shape=inputs)
    x = Conv2D(int(conv_out_channel), (convkernel, convkernel), use_bias=False,
                           strides=(1, 1), padding='valid', activation=None)(i)
    if ratio == 1:
        return x
    dw = DepthwiseConv2D(dwkernel, 1, padding='same', use_bias=False,
                                         depth_multiplier=ratio-1, activation=None)(x)
    dw = Lambda(_return_slices, arguments={'channel': int(out - conv_out_channel)})(dw)
    y = Concatenate()([x, dw])
    model = Model(inputs=i, outputs=y)
    return model