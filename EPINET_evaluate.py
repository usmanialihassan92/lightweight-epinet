# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:31:58 2017
@author: shinyonsei2
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from epinet_fun.func_generate_traindata import generate_traindata512
from epinet_fun.func_epinetmodel import define_epinet
from epinet_fun.func_epinetmodel_dwsc import define_epinet_dwsc
from epinet_fun.func_epinetmodel_ghost import define_epinet_ghost
from epinet_fun.func_epinetmodel_ghostdwsc import define_epinet_ghostdwsc
from epinet_fun.func_epinetmodel_conv_sep import define_epinet_cs
from epinet_fun.func_epinetmodel_conv_ghost import define_epinet_cg
from epinet_fun.func_epinetmodel_sep_conv import define_epinet_sc
from epinet_fun.func_savedata import measurePerformance
from epinet_fun.util import load_LFdata
import numpy as np
import os
import time
import pandas as pd


def evaluateArchitecture(architectureType, directory_ckp):
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
    print('Network: ', networkname)
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    # config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    ''' 
    Define Model parameters    
        first layer:  3 convolutional blocks, 
        second layer: 7 convolutional blocks, 
        last layer:   1 convolutional block
    '''
    model_conv_depth = 7  # 7 convolutional blocks for second layer
    model_filt_num = 70
    model_learning_rate = 0.1 ** 4

    ''' 
    Define Patch-wise training parameters
    '''
    Setting02_AngualrViews = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])  # number of views ( 0~8 for 9x9 )

    ''' 
    Model for predicting full-size LF images  
    '''
    image_w = 512
    image_h = 512

    if architectureType == 0:
        model = define_epinet(image_w, image_h, Setting02_AngualrViews, model_conv_depth, model_filt_num,model_learning_rate)
    elif architectureType == 1:
        model = define_epinet_cs(image_w, image_h, Setting02_AngualrViews, model_conv_depth, model_filt_num,model_learning_rate)
    elif architectureType == 2:
        model = define_epinet_cg(image_w, image_h, Setting02_AngualrViews, model_conv_depth, model_filt_num,model_learning_rate)
    elif architectureType == 3:
        model = define_epinet_ghost(image_w, image_h, Setting02_AngualrViews, model_conv_depth, model_filt_num,model_learning_rate)
    elif architectureType == 4:
        model = define_epinet_ghostdwsc(image_w, image_h, Setting02_AngualrViews, model_conv_depth, model_filt_num,model_learning_rate)
    elif architectureType == 5:
        model = define_epinet_dwsc(image_w, image_h, Setting02_AngualrViews, model_conv_depth, model_filt_num,model_learning_rate)
    elif architectureType == 6:
        model = define_epinet_sc(image_w, image_h, Setting02_AngualrViews, model_conv_depth, model_filt_num,model_learning_rate)

    """ 
    load latest_checkpoint
    """
    model.load_weights(directory_ckp)
    LFname = []
    mseList = []
    bp7List = []
    bp3List = []
    bp1List = []
    psnrList = []
    timeList = []
    ''' 
    Load Test data from LF .png files
    '''
    #print('Load test data...')
    dir_LFimages = ['stratified/backgammon', 'stratified/dots', 'stratified/pyramids', 'stratified/stripes',
        'training/boxes', 'training/cotton', 'training/dino', 'training/sideboard']
    for currentLFimages in dir_LFimages:
        LFname.append(currentLFimages)
        LFimages_list = [currentLFimages]
        valdata_all, valdata_label = load_LFdata(LFimages_list)
        valdata_90d, valdata_0d, valdata_45d, valdata_m45d, valdata_label = generate_traindata512(valdata_all,valdata_label,Setting02_AngualrViews)
        # (valdata_90d, 0d, 45d, m45d) to validation or test
        # print('Result of :',currentLFimages)
        start = time.time()
        model_output = model.predict([valdata_90d, valdata_0d,
                                          valdata_45d, valdata_m45d], batch_size=1)
        end = time.time()
        timeList.append(end - start)
        valid_error, valid_bp7, valid_bp3, valid_bp1, valid_psnr = measurePerformance(model_output, valdata_label, networkname,currentLFimages)
        valid_mean_squared_error = np.average(np.square(valid_error))  # Multiplied by 100
        valid_bad_pixel_ratio7 = np.average(valid_bp7)  # Multiplied by 100
        valid_bad_pixel_ratio3 = np.average(valid_bp3)  # Multiplied by 100
        valid_bad_pixel_ratio1 = np.average(valid_bp1)  # Multiplied by 100
        print('Name: ',currentLFimages,' Time: ',end - start, ' MSE: ',valid_mean_squared_error, ' BPR7: ',valid_bad_pixel_ratio7, ' BPR3: ',valid_bad_pixel_ratio3,' BPR1: ',valid_bad_pixel_ratio1, ' PSNR: ', valid_psnr)
        mseList.append(valid_mean_squared_error)
        bp7List.append(valid_bad_pixel_ratio7)
        bp3List.append(valid_bad_pixel_ratio3)
        bp1List.append(valid_bad_pixel_ratio1)
        psnrList.append(valid_psnr)
    r = np.array([np.array(LFname).astype('str').T, np.array(mseList).astype('float32').T, np.array(bp7List).astype('float32').T, np.array(bp3List).astype('float32').T, np.array(bp1List).astype('float32').T, np.array(psnrList).astype('float32').T, np.array(timeList).astype('float32').T])
    pd.DataFrame(r.T).to_csv('./epinet_output/'+networkname+'.csv', header=['Name', 'MSE', 'BP7', 'BP3', 'BP1', 'PSNR', 'Time'], index=False)
    print('Average MSE: ', np.average(valid_mean_squared_error))
    print('Average BPR7: ', np.average(valid_bad_pixel_ratio7))
    print('Average BPR3: ', np.average(valid_bad_pixel_ratio3))
    print('Average BPR1: ', np.average(valid_bad_pixel_ratio1))
    print('Average PSNR: ', np.average(valid_psnr))
    print('Average Inference Time: ', np.average(timeList))
    print('Done!')

if __name__ == '__main__':
    # Architecture Type = 0: EPINet Original, 1: Ghost-EPINet, 2: DWSC-EPINet 3: GhostDWSC-EPINet
    # EPINET CS_EPINET CG_EPINET Ghost_EPINET GhostDWSC_EPINET DWSC_EPINET SC_EPINET
    architectureType = 0
    if architectureType == 0:
        directory_ckp = './epinet_checkpoints/EPINET_ckp/EPINET_005_0.0523_27.97.hdf5'
    elif architectureType == 1:
        directory_ckp = './epinet_checkpoints/CS_EPINET_ckp/.hdf5'
    elif architectureType == 2:
        directory_ckp = './epinet_checkpoints/CG_EPINET_ckp/.hdf5'
    elif architectureType == 3:
        directory_ckp = './epinet_checkpoints/Ghost_EPINET_ckp/.hdf5'
    elif architectureType == 4:
        directory_ckp = './epinet_checkpoints/GhostDWSC_EPINET_ckp/.hdf5'
    elif architectureType == 5:
        directory_ckp = './epinet_checkpoints/DWSC_EPINET_ckp/.hdf5'
    elif architectureType == 6:
        directory_ckp = './epinet_checkpoints/SC_EPINET_ckp/.hdf5'
    if os.path.exists(directory_ckp):
        evaluateArchitecture(architectureType, directory_ckp)
    else:
        print('Unable to find model weights, correct path!')