"""Code Snippet for counting FLOPs of a model.

Not final version, it will be updated to improve the usability.
https://github.com/ckyrkou/Keras_FLOP_Estimator
"""

import os.path

import tensorflow as tf
from tensorflow.python.keras import Model, Sequential
from keras_flops import get_flops
from tensorflow.keras import backend as K
from model_profiler import model_profiler #https://pypi.org/project/model-profiler/
import keras_flops
import math

def get_flops(model):
    flops2 = keras_flops.get_flops(model, batch_size=1)
    print('FLOPS2: ', flops2 / 10 ** 9)
    use_units = ['GPU IDs', 'GFLOPs', 'MB', 'Million', 'MB']
    profile, values = model_profiler(model, Batch_size=1, use_units=use_units)
    #model.summary(expand_nested=True)
    print(profile)

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    #return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred -
y_true))))


def cFlops(vList):
    # values = [gpus, flops, mem, param, mem_req]
    tflops = 0
    tmem = 0
    tparam = 0
    tmem_req = 0
    for arr in vList:
        tflops += arr[1]
        tmem += arr[2]
        tparam += arr[3]
        tmem_req += arr[4]
    print('Total FlOPS (GFLOPs): ',tflops)
    print('Total Memory (MB): ', tmem)
    print('Total Parameters (Million): ', tparam)
    print('Total Memory Requirement (MB): ', tmem_req)


def net_flops(model, table=False):
    if (table == True):
        print('%25s \t| %25s \t| %16s \t| %16s \t| %16s \t| %16s \t| %6s \t| %6s' % (
            'Layer Name', 'Layer Type', 'Input Shape', 'Output Shape', 'Kernel Size', 'Filters', 'Strides', 'FLOPS'))
        print('-' * 170)

    t_flops = 0
    t_macc = 0

    for l in model.layers:

        o_shape, i_shape, strides, ks, filters = ['', '', ''], ['', '', ''], [1, 1], [0, 0], [0, 0]
        flops = 0
        macc = 0
        name = l.name
        ltype = str(l)

        factor = 1000000

        if ('InputLayer' in str(l)):
            i_shape = l.input.get_shape()[1:4].as_list()
            o_shape = i_shape
            ltype = 'InputLayer'

        if ('Reshape' in str(l)):
            i_shape = l.input.get_shape()[1:4].as_list()
            o_shape = l.output.get_shape()[1:4].as_list()
            ltype = 'Reshape'

        if ('Add' in str(l) or 'Maximum' in str(l) or 'Concatenate' in str(l)):
            i_shape = l.input[0].get_shape()[1:4].as_list() + [len(l.input)]
            o_shape = l.output.get_shape()[1:4].as_list()
            flops = (len(l.input) - 1) * i_shape[0] * i_shape[1] * i_shape[2]
            ltype = 'Add'

        if ('Average' in str(l) and 'pool' not in str(l)):
            i_shape = l.input[0].get_shape()[1:4].as_list() + [len(l.input)]
            o_shape = l.output.get_shape()[1:4].as_list()
            flops = len(l.input) * i_shape[0] * i_shape[1] * i_shape[2]
            ltype = 'Average'

        if ('BatchNormalization' in str(l)):
            i_shape = l.input.get_shape()[1:4].as_list()
            o_shape = l.output.get_shape()[1:4].as_list()
            ltype = 'BatchNormalization'
            bflops = 1
            for i in range(len(i_shape)):
                bflops *= i_shape[i]
            flops /= factor


        if ('Activation' in str(l) or 'activation' in str(l)):
            i_shape = l.input.get_shape()[1:4].as_list()
            o_shape = l.output.get_shape()[1:4].as_list()
            bflops = 1
            ltype = 'Activation'
            for i in range(len(i_shape)):
                bflops *= i_shape[i]
            flops /= factor

        if ('pool' in str(l) and ('Global' not in str(l))):
            i_shape = l.input.get_shape()[1:4].as_list()
            strides = l.strides
            ks = l.pool_size
            ltype = 'pool'
            flops = ((i_shape[0] / strides[0]) * (i_shape[1] / strides[1]) * (ks[0] * ks[1] * i_shape[2]))

        if ('Flatten' in str(l)):
            i_shape = l.input.shape[1:4].as_list()
            flops = 1
            out_vec = 1
            ltype = 'Flatten'
            for i in range(len(i_shape)):
                flops *= i_shape[i]
                out_vec *= i_shape[i]
            o_shape = flops
            flops = 0

        if ('Dense' in str(l)):
            #print(l.input)
            ltype = 'Dense'
            i_shape = l.input.shape[1:4].as_list()[0]
            if (i_shape == None):
                i_shape = out_vec

            o_shape = l.output.shape[1:4].as_list()
            flops = 2 * (o_shape[0] * i_shape)
            macc = flops / 2

        if ('Padding' in str(l)):
            flops = 0
            ltype = 'Padding'

        if (('Global' in str(l))):
            i_shape = l.input.get_shape()[1:4].as_list()
            flops = ((i_shape[0]) * (i_shape[1]) * (i_shape[2]))
            o_shape = [l.output.get_shape()[1:4].as_list(), 1, 1]
            out_vec = o_shape
            ltype = 'Global'

        if ('Conv2D ' in str(l) and 'DepthwiseConv2D' not in str(l) and 'SeparableConv2D' not in str(l)):
            strides = l.strides
            ks = l.kernel_size
            filters = l.filters
            i_shape = l.input.get_shape()[1:4].as_list()
            o_shape = l.output.get_shape()[1:4].as_list()
            pad = l.padding
            ltype = 'Conv2D'

            if (filters == None):
                filters = i_shape[2]

            #flops = 2 * ((filters * ks[0] * ks[1] * i_shape[2]) * (
            #        (i_shape[0] / strides[0]) * (i_shape[1] / strides[1])))
            flops = (2 * ((filters * ks[0] * ks[1] * i_shape[2]) * (
                    (o_shape[0] * (o_shape[1])))))-(o_shape[0] * (o_shape[1]) * o_shape[2])
            macc = flops / 2

        if ('Conv2D ' in str(l) and 'DepthwiseConv2D' in str(l) and 'SeparableConv2D' not in str(l)):
            strides = l.strides
            ks = l.kernel_size
            filters = l.filters
            i_shape = l.input.get_shape()[1:4].as_list()
            o_shape = l.output.get_shape()[1:4].as_list()
            ltype = 'Conv2D/DepthwiseConv2D'
            if (filters == None):
                filters = i_shape[2]

            flops = 2 * (
                    (ks[0] * ks[1] * i_shape[2]) * ((i_shape[0] / strides[0]) * (i_shape[1] / strides[1]))) / factor
            macc = flops / 2

        t_macc += macc

        t_flops += flops

        if (table == True):
            print('%25s \t| %25s \t| %16s \t| %16s \t| %16s \t| %16s \t| %6s \t| %5.4f' % (
                name, ltype, str(i_shape), str(o_shape), str(ks), str(filters), str(strides), flops))
    t_flops = t_flops / factor

    print('\nTotal FLOPS (x 10^-6): %10.8f' % (t_flops))
    print('Total MACCs: %10.8f\n' % (t_macc))
    return 0

# nflops = keras_flops.get_flops(model, batch_size=1)
# print(nflops)
# flops = evaluation_metric.net_flops(model, table=True)
# nflops = keras_flops.get_flops(model, batch_size=1)
# print(nflops)