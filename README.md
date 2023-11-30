# Light-Weight EPINET Architecture for Fast Light Field Disparity Estimation
Ali Hassan, Marten Sjostrom, Tingting Zhang and Karen Egiazarian
IEEE Conference on Multimedia Signal Processing (MMSP), Sept 2022 

:page_facing_up:Summary: It is forked version of EPINET architecture paper, and we tried to compress it.

:e-mail:Contact: ali.hassan@miun.se

# Environments
- Python 3.7.1 and Tensorflow 2.10
- Use `pip install imageio numpy tensorflow-gpu==2.4.0 keras matplotlib keras-flops model-profiler pandas` to install all required libraries.

# Hardware
- 2x NVIDIA GeForce RTX 2070 graphics cards
- 64GB of Random Access Memory (RAM)
- Ubuntu 18.04 operating system

# Preparing HCI 4D LF Dataset
 First, you need to download HCI Light field dataset from http://hci-lightfield.iwr.uni-heidelberg.de/.
 Unzip the LF dataset and move 'additional/, training/, test/, stratified/ ' into the 'hci_dataset/'.

 # Configuring EPINET variant
 Choose the desired variant of EPINET by importing desired model function at `Line # 16` of the `EPINET_train.py` code. 
 For example, if you want to train Depthwise Separable Convolution based EPINET variant, change the `Line # 16` to:
 `from epinet_fun.func_epinetmodel_dwsc import define_epinet_dwsc as define_epinet`

 The available variants of EPINET are:
  `func_epinetmodel_conv_ghost`,
  `func_epinetmodel_conv_sep`,
  `func_epinetmodel_dwsc`,
  `func_epinetmodel_ghost`,
  `func_epinetmodel_ghostdwsc`,
  `func_epinetmodel_sep_conv`
 
 And then run
  `python EPINET_train.py`
 
 - Checkpoint files will be saved in 'epinet_checkpoints/EPINET_train_ckp/iterXXX_XX.hdf5', it could be used for test EPINET model.
 - Training process will be saved 'epinet_output/EPINET_train/train_XX.jpg'. (XX is iteration number). 
 - You might be change the setting 'learning rate','patch_size' and so on to get better result.

# Test the EPINET

Run `python EPINET_plusX_9conv22_save.py`

 - To test your own trained model from `python EPINET_train.py`, you need to modify the line 141-142 like below
`path_weight='epinet_checkpoints/EPINET_train_ckp/iter0097_trainmse2.706_bp12.06.hdf5'`

Last modified date: 09/29/2022
