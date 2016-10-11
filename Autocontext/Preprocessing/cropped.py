import numpy as np
import h5py

# img_idx that should be in one labelmap hdf5 file
files = [1750, 400, 50, 1400, 300, 1300, 200]

#name of hdf5 file that containes cropped rawimages
trainimages_name = 'cropped.h5'
dataset_train = 'data'
#locateion rawdata
raw_path = '../80_raw_cropped.h5'
dataset_raw = 'data' #'/volume/data'


def create_filename(idx):
    return 'Labels_' + str(idx) + '.h5'

#open rawdata
rawdata_f = h5py.File(raw_path, 'r')
shape = rawdata_f[dataset_raw][files[0]].shape
x_dim = shape[0]
y_dim = shape[1]

trainimages = np.zeros((len(files), x_dim, y_dim, 1) )


for idx in np.arange(len(files)):
    trainimages[idx,:,:,0] = rawdata_f[dataset_raw][files[idx],:,:,0]
    
rawdata_f.close()

#produce cropped trainingsimages
cropped_f = h5py.File(trainimages_name, 'w')
cropped_f.create_dataset(dataset_train, data=trainimages)
cropped_f.close()
