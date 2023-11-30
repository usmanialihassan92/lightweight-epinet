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
 
 `from epinet_fun.func_epinetmodel import define_epinet`
 
 For example, if you want to train Depthwise Separable Convolution based EPINET variant, change the line to:
 
 `from epinet_fun.func_epinetmodel_dwsc import define_epinet_dwsc as define_epinet`

 The available variants of EPINET are:
 - File: `func_epinetmodel_conv_sep`, Function: `define_epinet_cs` -> EPINET-CD
 - File: `func_epinetmodel_dwsc`, Function: `define_epinet_dwsc` -> EPINET-D
 - File: `func_epinetmodel_ghost`, Function: `define_epinet_ghost` -> EPINET-G
 - File: `func_epinetmodel_sep_conv`, Function: `define_epinet_sc` -> EPINET-DC
 
 And then run
  `python EPINET_train.py`
 
 - Checkpoint files will be saved in 'epinet_checkpoints/EPINET_train_ckp/iterXXX_XX.hdf5', it could be used for test EPINET model.
 - Training process will be saved 'epinet_output/EPINET_train/train_XX.jpg'. (XX is iteration number). 
 - You might be change the setting 'learning rate','patch_size' and so on to get better result.

# Test the EPINET
 Edit the EPINET_test file by choosing the architecture type:
 - 0: EPINET
 - 1: EPINET-CD
 - 3: EPINET-G
 - 5: EPINET-D
 - 6: EPINET-DC

 - and then run
`python EPINET_test`

 - To test your own trained model from `python EPINET_train.py`, you need to modify the line 141-142 like below
`path_weight='epinet_checkpoints/EPINET_train_ckp/iter0097_trainmse2.706_bp12.06.hdf5'`

# Citation
```
@INPROCEEDINGS{hassan2022,
  author={Hassan, Ali and Sjöström, Mårten and Zhang, Tingting and Egiazarian, Karen},
  booktitle={2022 IEEE 24th International Workshop on Multimedia Signal Processing (MMSP)}, 
  title={Light-Weight EPINET Architecture for Fast Light Field Disparity Estimation}, 
  year={2022},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/MMSP55362.2022.9949378}}
```


# Acknowledgement

Last modified data: 2023/11/30.

The code is modified and heavily borrowed from EPINET: [https://github.com/LIAGM/LFattNet](https://github.com/chshin10/epinet)

The code they provided is greatly appreciated.
