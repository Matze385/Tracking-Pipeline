import numpy as np
from skimage.transform import rotate
import h5py 

"""
take every 4th image of rawdata and perform all 90 degrees rotations
"""

filename_raw ='../80_raw_cropped.h5'
dataset_raw = 'data'

every_i_img = 4 #must be >=4
filename_augmented = 'rawdata_aug.h5'
dataset_aug = 'data'

rawdata_f = h5py.File(filename_raw,'r')
shape = rawdata_f[dataset_raw].shape
augdata_f = h5py.File(filename_augmented,'w')
augdata_f.create_dataset(dataset_aug, shape , dtype = rawdata_f[dataset_raw].dtype)

n_img = shape[0]
for i in np.arange(n_img/every_i_img):
    idx = i*every_i_img 
    unrotated = rawdata_f[dataset_raw][idx,:,:,0]
    augdata_f[dataset_aug][i,:,:,0] = unrotated
    for i_rot in np.arange(3):
        i_rot += 1
        augdata_f[dataset_aug][i+i_rot*n_img/every_i_img,:,:,0] = np.rot90(unrotated, i_rot) 
        # augdata_f[dataset_aug][i*4+i_rot,:,:,0] = rotate(unrotated, 90.*i_rot, resize=False, center=(shape[2], shape[1]), preserve_range=True )



augdata_f.close()
rawdata_f.close()
