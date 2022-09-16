# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:54:06 2018

@author: shinyonsei2
"""
from tensorflow.python.ops.image_ops_impl import psnr
from tensorflow.keras import metrics
from tensorflow.python.keras import backend
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Reshape, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.backend import concatenate
import numpy as np
from tensorflow.keras import applications
from model_profiler import model_profiler  # https://pypi.org/project/model-profiler/
from tensorflow.keras import backend as K
from epinet_fun import evaluation_metric
import keras_flops

printStat = False
paddingType = 'valid'  # valid
strides = (1, 1)
depth_multiplier = 1
vList = []


def layer1_multistream_cs(input_dim1, input_dim2, input_dim3, filt_num):
    seq = Sequential()
    ''' Multi-Stream layer : Conv - Relu - Conv - BN - Relu  '''

    #    seq.add(Reshape((input_dim1,input_dim12,input_dim3),input_shape=(input_dim1, input_dim2, input_dim3,1)))
    for i in range(3):
        seq.add(Conv2D(int(filt_num), (2, 2), input_shape=(input_dim1, input_dim2, input_dim3),
                                padding=paddingType, name='conv_dw_%d' % (i)))
        seq.add(Activation('relu', name='S1_relu1%d' % (i)))
        seq.add(SeparableConv2D(int(filt_num), (2, 2), padding=paddingType, name='S1_c2%d' % (i)))
        seq.add(BatchNormalization(axis=-1, name='S1_BN%d' % (i)))
        seq.add(Activation('relu', name='S1_relu2%d' % (i)))

        # seq.add(Reshape((input_dim1-6,input_dim2-6,int(filt_num))))
    if (printStat):
        # flops = evaluation_metric.net_flops(seq, table=True)
        # flops2 = keras_flops.get_flops(seq,batch_size=1)
        # print('FLOPS2: ',flops2)
        use_units = ['GPU IDs', 'GFLOPs', 'MB', 'Million', 'MB']
        profile, values = model_profiler(seq, Batch_size=1, use_units=use_units)
        vList.append(values)
        print(profile)
    return seq


def layer2_merged_cs(input_dim1, input_dim2, input_dim3, filt_num, conv_depth):
    ''' Merged layer : Conv - Relu - Conv - BN - Relu '''

    seq = Sequential()

    for i in range(conv_depth):
        seq.add(Conv2D(filt_num, (2, 2), padding=paddingType, input_shape=(input_dim1, input_dim2, input_dim3),
                                name='S2_c1%d' % (i)))
        seq.add(Activation('relu', name='S2_relu1%d' % (i)))
        seq.add(SeparableConv2D(filt_num, (2, 2), padding=paddingType, name='S2_c2%d' % (i)))
        seq.add(BatchNormalization(axis=-1, name='S2_BN%d' % (i)))
        seq.add(Activation('relu', name='S2_relu2%d' % (i)))

    if (printStat):
        # flops = evaluation_metric.net_flops(seq, table=True)
        # flops2 = keras_flops.get_flops(seq,batch_size=1)
        # print('FLOPS2: ',flops2)
        use_units = ['GPU IDs', 'GFLOPs', 'MB', 'Million', 'MB']
        profile, values = model_profiler(seq, Batch_size=1, use_units=use_units)
        vList.append(values)
        print(profile)
    return seq


def layer3_last_cs(input_dim1, input_dim2, input_dim3, filt_num):
    ''' last layer : Conv - Relu - Conv '''

    seq = Sequential()

    for i in range(1):
        seq.add(SeparableConv2D(filt_num, (2, 2), padding=paddingType, input_shape=(input_dim1, input_dim2, input_dim3),
                                name='S3_c1%d' % (i)))  # pow(25/23,2)*12*(maybe7?) 43 3
        seq.add(Activation('relu', name='S3_relu1%d' % (i)))

    seq.add(Conv2D(1, (2, 2), padding='valid', use_bias=False, name='S3_last'))
    if (printStat):
        # flops = evaluation_metric.net_flops(seq, table=True)
        # flops2 = keras_flops.get_flops(seq,batch_size=1)
        # print('FLOPS2: ',flops2)
        use_units = ['GPU IDs', 'GFLOPs', 'MB', 'Million', 'MB']
        profile, values = model_profiler(seq, Batch_size=1, use_units=use_units)
        vList.append(values)
        print(profile)
    return seq


def define_epinet_cs(sz_input, sz_input2, view_n, conv_depth, filt_num, learning_rate):
    tFlops = 0
    tMem = 0
    tParam = 0
    tMem_req = 0

    ''' 4-Input : Conv - Relu - Conv - BN - Relu '''
    input_stack_90d = Input(shape=(sz_input, sz_input2, len(view_n)), name='input_stack_90d')
    input_stack_0d = Input(shape=(sz_input, sz_input2, len(view_n)), name='input_stack_0d')
    input_stack_45d = Input(shape=(sz_input, sz_input2, len(view_n)), name='input_stack_45d')
    input_stack_M45d = Input(shape=(sz_input, sz_input2, len(view_n)), name='input_stack_M45d')

    ''' 4-Stream layer : Conv - Relu - Conv - BN - Relu '''
    mid_90d = layer1_multistream_cs(sz_input, sz_input2, len(view_n), int(filt_num))(input_stack_90d)
    mid_0d = layer1_multistream_cs(sz_input, sz_input2, len(view_n), int(filt_num))(input_stack_0d)
    mid_45d = layer1_multistream_cs(sz_input, sz_input2, len(view_n), int(filt_num))(input_stack_45d)
    mid_M45d = layer1_multistream_cs(sz_input, sz_input2, len(view_n), int(filt_num))(input_stack_M45d)

    ''' Merge layers '''
    mid_merged = concatenate([mid_90d, mid_0d, mid_45d, mid_M45d])

    ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
    mid_merged_ = layer2_merged_cs(sz_input - 6, sz_input2 - 6, int(4 * filt_num), int(4 * filt_num), conv_depth)(
        mid_merged)

    ''' Last Conv layer : Conv - Relu - Conv '''
    output = layer3_last_cs(sz_input - 20, sz_input2 - 20, int(4 * filt_num), int(4 * filt_num))(mid_merged_)

    model_512 = Model(inputs=[input_stack_90d, input_stack_0d,
                              input_stack_45d, input_stack_M45d], outputs=[output])
    opt = RMSprop(learning_rate=learning_rate)
    model_512.compile(optimizer=opt, loss='mae', metrics=[metrics.MeanSquaredError(), evaluation_metric.PSNR])
    return model_512


def evaluateEPINet_cs():
    image_w = 512
    image_h = 512
    model_conv_depth = 7  # 7 convolutional blocks for second layer
    model_filt_num = 70
    model_learning_rate = 0.1 ** 4
    Setting02_AngualrViews = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])  # number of views ( 0~8 for 9x9 )
    model = define_epinet_cs(image_w, image_h,
                               Setting02_AngualrViews,
                               model_conv_depth,
                               model_filt_num,
                               model_learning_rate)
    # flops1 = evaluation_metric.net_flops(model, table=True)
    # print('FLOPS1: ', flops1)
    flops2 = keras_flops.get_flops(model, batch_size=1)
    print('FLOPS2: ', flops2 / 10 ** 9)
    use_units = ['GPU IDs', 'GFLOPs', 'MB', 'Million', 'MB']
    profile, values = model_profiler(model, Batch_size=1, use_units=use_units)
    model.summary(expand_nested=True)
    print(profile)


if __name__ == '__main__':
    evaluateEPINet_cs()