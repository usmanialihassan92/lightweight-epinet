# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 14:41:04 2018

@author: shinyonsei2
"""

''' 
The order of LF image files may be different with this file.
(Top to Bottom, Left to Right, and so on..)

If you use different LF images, 

you should change our 'func_makeinput.py' file.

# Light field images: input_Cam000-080.png
# All viewpoints = 9x9(81)

# -- LF viewpoint ordering --
# 00 01 02 03 04 05 06 07 08
# 09 10 11 12 13 14 15 16 17
# 18 19 20 21 22 23 24 25 26
# 27 28 29 30 31 32 33 34 35
# 36 37 38 39 40 41 42 43 44
# 45 46 47 48 49 50 51 52 53
# 54 55 56 57 58 59 60 61 62
# 63 64 65 66 67 68 69 70 71
# 72 73 74 75 76 77 78 79 80


# We use star-shape 9x9 viewpoints 
# for depth estimation
#
# 00          04          08
#    10       13       16 
#       20    22    24 
#          30 31 32 
# 36 37 38 39 40 41 42 43 44
#          48 49 50 
#       56    58    60 
#    64       67       70 
# 72          76          80    

'''

import numpy as np
import os
import time
from epinet_fun.func_pfm import write_pfm
from epinet_fun.func_makeinput import make_epiinput
from epinet_fun.func_makeinput import make_multiinput
from epinet_fun.func_epinetmodel import layer1_multistream
from epinet_fun.func_epinetmodel import layer2_merged
from epinet_fun.func_epinetmodel import layer3_last
from epinet_fun.func_epinetmodel import define_epinet
from epinet_fun.func_epinetmodel_dwsc import define_epinet_dwsc
from epinet_fun.func_epinetmodel_ghost import define_epinet_ghost
from epinet_fun.func_epinetmodel_ghostdwsc import define_epinet_ghostdwsc
from epinet_fun.func_epinetmodel_conv_sep import define_epinet_cs
from epinet_fun.func_epinetmodel_conv_ghost import define_epinet_cg
from epinet_fun.func_epinetmodel_sep_conv import define_epinet_sc
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.python.profiler.model_analyzer import profile
# from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from model_profiler import model_profiler  # https://pypi.org/project/model-profiler/
from epinet_fun import evaluation_metric


def get_flops(model):
    graph = tf.compat.v1.get_default_graph()
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    # We use the Keras session graph in the call to the profiler.
    flops = tf.compat.v1.profiler.profile(graph=graph,
                                          run_meta=run_meta, cmd='op', options=opts)
    tf.compat.v1.reset_default_graph()
    return flops.total_float_ops  # Prints the "flops" of the model.


def evaluateArchitecture(architectureType):
    if architectureType == 0:
        networkname = 'EPINET'
    elif architectureType == 1:
        networkname = 'CS_EPINET'
    elif architectureType == 2:
        networkname = 'CG_EPINET'
    elif architectureType == 3:
        networkname = 'Ghost_EPINET'
    elif architectureType == 4:
        networkname = 'GhostDWSC_EPINET'
    elif architectureType == 5:
        networkname = 'DWSC_EPINET'
    elif architectureType == 6:
        networkname = 'SC_EPINET'
    else:
        print('Invalid Argument!')
        return 0

    # Input : input_Cam000-080.png
    # Depth output : image_name.pfm
    dir_output = 'epinet_output'
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

        # GPU setting ( gtx 1080ti - gpu0 )
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"

    '''
    /// Setting 1. LF Images Directory

    Setting01_LFdir = 'synthetic': Test synthetic LF images (from 4D Light Field Benchmark)
                                 "A Dataset and Evaluation Methodology for 
                                 Depth Estimation on 4D Light Fields".
                                http://hci-lightfield.iwr.uni-heidelberg.de/

    Setting01_LFdir = 'Lytro': Test real LF images(Lytro)

    '''
    Setting01_LFdir = 'synthetic'
    #   Setting01_LFdir='Lytro'

    if (Setting01_LFdir == 'synthetic'):
        dir_LFimages = ['hci_dataset/training/dino', 'hci_dataset/training/cotton']
        image_w = 512
        image_h = 512

    elif (Setting01_LFdir == 'Lytro'):
        dir_LFimages = ['lytro/2067']
        image_w = 552
        image_h = 383

    '''
    /// Setting 2. Angular Views 

    Setting02_AngualrViews = [2,3,4,5,6] : 5x5 viewpoints

    Setting02_AngualrViews = [0,1,2,3,4,5,6,7,8] : 9x9 viewpoints

    # ------ 5x5 viewpoints -----
    #                                  
    #       20    22    24 
    #          30 31 32 
    #       38 39 40 41 42      
    #          48 49 50 
    #       56    58    60 
    #                         
    # ---------------------------                      

    # ------ 9x9 viewpoints -----
    # 
    # 00          04          08
    #    10       13       16 
    #       20    22    24 
    #          30 31 32 
    # 36 37 38 39 40 41 42 43 44
    #          48 49 50 
    #       56    58    60 
    #    64       67       70 
    # 72          76          80       
    #
    # ---------------------------
    '''

    #   Setting02_AngualrViews = [2,3,4,5,6]  # number of views ( 2~6 for 5x5 )
    Setting02_AngualrViews = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # number of views ( 0~8 for 9x9 )

    if (len(Setting02_AngualrViews) == 5):
        path_weight = 'epinet_checkpoints/pretrained_5x5.hdf5'  # sample weight.
    if (len(Setting02_AngualrViews) == 9):
        path_weight = 'epinet_checkpoints/pretrained_9x9.hdf5'  # sample weight.
    #        path_weight='epinet_checkpoints/EPINET_train_ckp/iter0097_trainmse2.706_bp12.06.hdf5'

    img_scale = 1  # 1 for small_baseline(default) <3.5px,
    # 0.5 for large_baseline images   <  7px

    img_scale_inv = int(1 / img_scale)

    ''' Define Model ( set parameters )'''

    model_conv_depth = 7
    model_filt_num = 70
    model_learning_rate = 0.1 ** 5

    input_sizeh = round(img_scale * image_h)
    input_sizew = round(img_scale * image_w)
    if architectureType == 0:
        model = define_epinet(input_sizeh, input_sizew, Setting02_AngualrViews, model_conv_depth, model_filt_num,
                              model_learning_rate)
    elif architectureType == 1:
        model = define_epinet_cs(input_sizeh, input_sizew, Setting02_AngualrViews, model_conv_depth, model_filt_num,
                                 model_learning_rate)
    elif architectureType == 2:
        model = define_epinet_cg(input_sizeh, input_sizew, Setting02_AngualrViews, model_conv_depth, model_filt_num,
                                 model_learning_rate)
    elif architectureType == 3:
        model = define_epinet_ghost(input_sizeh, input_sizew, Setting02_AngualrViews, model_conv_depth, model_filt_num,
                                    model_learning_rate)
    elif architectureType == 4:
        model = define_epinet_ghostdwsc(input_sizeh, input_sizew, Setting02_AngualrViews, model_conv_depth,
                                        model_filt_num, model_learning_rate)
    elif architectureType == 5:
        model = define_epinet_dwsc(input_sizeh, input_sizew, Setting02_AngualrViews, model_conv_depth, model_filt_num,
                                   model_learning_rate)
    elif architectureType == 6:
        model = define_epinet_sc(input_sizeh, input_sizew, Setting02_AngualrViews, model_conv_depth, model_filt_num,
                                 model_learning_rate)

    ''' Model Initialization '''

    model.load_weights(path_weight)
    use_units = ['GPU IDs', 'GFLOPs', 'MB', 'Million', 'MB']
    profile, values = model_profiler(model, Batch_size=1, use_units=use_units)
    # model_512.summary(expand_nested=True)
    print(profile)

    """  Depth Estimation  """
    for image_path in dir_LFimages:
        (val_90d, val_0d, val_45d, val_M45d) = make_multiinput(image_path,
                                                               image_h,
                                                               image_w,
                                                               Setting02_AngualrViews)
        start = time.clock()
        # predict
        val_output_tmp = model.predict([val_90d[:, ::img_scale_inv, ::img_scale_inv],
                                        val_0d[:, ::img_scale_inv, ::img_scale_inv],
                                        val_45d[:, ::img_scale_inv, ::img_scale_inv],
                                        val_M45d[:, ::img_scale_inv, ::img_scale_inv]],
                                       batch_size=1)

        runtime = time.clock() - start
        plt.imshow(val_output_tmp[0, :, :, 0])
        print("runtime: %.5f(s)" % runtime)

        # save .pfm file
        write_pfm(val_output_tmp[0, :, :, 0], dir_output + '/%s.pfm' % (image_path.split('/')[-1]))
        print('pfm file saved in %s/%s.pfm' % (dir_output, image_path.split('/')[-1]))


if __name__ == '__main__':
    evaluateArchitecture(0)