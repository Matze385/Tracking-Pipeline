"""
counts the number of pixels belonging to each class in one image, to estimate maximal allowed number of trainingspixel
"""

import numpy as np
import h5py

n_folds = 5
n_classes = 7
i_img = 1

with h5py.File('labels_1700_29_1200_800.h5', 'r') as f:
    img = f['exported_data'][i_img,:,:,0]
    xDim = img.shape[0]
    yDim = img.shape[1]
    img = img.reshape((xDim*yDim))
    class_count = np.zeros((n_classes+1,))
    for cl in np.arange(n_classes+1):
        length = len(filter(lambda x: x==cl, img))
        class_count[cl] = length
        print('class: ', cl,'length: ',length )
    print('maximal possible number of trainingspixels: ', class_count.min()*n_classes*(n_folds-1))
