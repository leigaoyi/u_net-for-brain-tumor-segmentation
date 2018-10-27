# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 10:17:57 2018

@author: kasy
"""

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import model
import glob
import os
import nibabel as nib

task = 'all'
test_type = 'normal'
save_dir = 'checkpoint'
save_seg_path = './result/HGG/{0}/'.format(str(task))
data_path = './data/MICCAI_BraTS17_Data_Training/HGG/' # ubuntu data path
#data_path = './data/MICCAI_BraTS17_Data_Training_IPP/MICCAI_BraTS17_Data_Training/HGG/' #windows
batch_size = 10
nw = 240
nh = 240
nz = 4

#init the network
sess = tf.Session()
t_image = tf.placeholder('float32', [batch_size, nw, nh, nz], name='input_image')
## labels are either 0 or 1
t_seg = tf.placeholder('float32', [batch_size, nw, nh, 1], name='target_segment')
## train inference
net = model.u_net(t_image, is_train=False, reuse=False, n_out=1)
net_seg = net.outputs
sess = tf.Session()
tl.layers.initialize_global_variables(sess)
## load existing model if possible
tl.files.load_and_assign_npz(sess=sess, name=save_dir+'/u_net_{}.npz'.format(task), network=net)

# read the files
if not os.path.exists(save_seg_path):
    os.makedirs(save_seg_path)

with open('HGG_list.txt', 'r') as f:
    files = f.readlines()

if test_type == 'small':# file paths
    file_list = files[:20]
if test_type == 'normal':
    file_list = files[:50]
if test_type == 'all':
    file_list = files    



def input_slice(path):
    data_types = ['flair', 't1', 't1ce', 't2']
    test_data = []
    test_input = []
    for i in data_types:
        img = nib.load(path+'_'+i+'.nii.gz')
        test_data.append(img.get_data())
    
    label_test = nib.load(path+'_seg'+'.nii.gz')
    label_affine = label_test.affine
    for i in range(test_data[0].shape[2]):
        slice_combine = np.stack((test_data[0][..., i], test_data[1][..., i],
                                        test_data[2][..., i], test_data[3][..., i]),axis=2)
        slice_combine = np.transpose(slice_combine, [1, 0, 2])
        #transpose
        test_input.append(slice_combine)
    if task=='all':
        label_seg = (label_test.get_data()>0).astype(np.float32)
    return test_input, label_seg, label_affine # (155, 240, 240, 4), nifty image

def test_single_seg_out(input_arr):
    test_out = []
    test_input = input_arr
    for i in range(15):
        test_slice = np.array(test_input[10*i:10*(i+1)])
        seg = sess.run(net_seg, feed_dict={t_image:test_slice})
        test_out.append(seg)
        
    test_slice = np.array(test_input[-10:])
    seg = sess.run(net_seg, feed_dict={t_image:test_slice})
    test_out.append(seg[-5:]) 
    
    test_out = np.concatenate(test_out, axis=0)
    seg_out_data = np.concatenate([test_out[i,...] for i in range(155)], axis=2)
    seg_out_data = np.transpose(seg_out_data, [1, 0, 2])
    return seg_out_data
    

# test the dice scores
input_list = []
gt_list = []
gt_affine_list = []
seg_out_list = []
base_name_list = []
for i in file_list:
    base_name = i[:-1]
    base_name_list.append(base_name)
    base_path = data_path+'/'+base_name+'/'+base_name
    input_case, label_slice, affine = input_slice(base_path)
    input_list.append(input_case)
    gt_list.append(label_slice)
    gt_affine_list.append(affine)
print('load data successfule')
mean = np.mean(input_list)
std = np.std(input_list)
input_list = (input_list-mean)/std #image pre-process
print('number of input: ',len(input_list))    
# run the seg of model and save the image
for i in range(len(input_list)):
    single_seg = test_single_seg_out(input_list[i])
    seg_out_list.append(single_seg)
    save_path = save_seg_path+file_list[i][:-1]+'_seg.nii.gz'
    nii_img = nib.Nifti1Image(single_seg, affine=affine)
    nib.save(nii_img, save_path)
    if i%5 == 0:
        print('running ', i)
    
# compute the dice score
dice_score_list = []
for i in range(len(seg_out_list)):
    dice_score = tl.cost.dice_coe(seg_out_list[i], gt_list[i], axis=[0, 1, 2])
    dice_score = sess.run(dice_score)
    dice_score_list.append(dice_score)
    if i%5 == 0:
        print('Computing ', int(i+1/len(seg_out_list)*100),'%')
#    print('type ', dice_score.dtype)
dice_mean = np.mean(dice_score_list)
dice_std = np.std(dice_score_list)
dice_median = np.median(dice_score_list)   
print('Dice mean ', dice_mean)
print('Dice std ', dice_std)
print('Dice median ', dice_median) 
print(dice_score_list)
np.savetxt('./result/dice_score_{0}.txt'.format(str(task)), dice_score_list)