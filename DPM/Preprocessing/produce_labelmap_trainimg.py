import numpy as np
import h5py

# img_idx that should be in one labelmap hdf5 file
files = [1700, 29, 1200, 800, 0] #last one is not true
dataset_name_in = 'exported_data'
#name of hdf5 file labelmap that should be produced
labelmap_name = 'labels_1700_29_1200_800.h5'
dataset_name_out = 'exported_data'
#name of hdf5 file that containes cropped rawimages
trainimages = 'trainimages.h5'
dataset_train = 'data'
#locateion rawdata
raw_path = '../80_raw_cropped.h5'
dataset_raw = 'data' #'/volume/data'

def create_filename(idx):
    return 'Labels_' + str(idx) + '.h5'

#get dimension of images
first_in_file = h5py.File(create_filename(files[0]), 'r')
shape = first_in_file[dataset_name_in].shape
x_dim = shape[1]
y_dim = shape[2]
first_in_file.close()

#create empty labelmap
labelmap_f = h5py.File(labelmap_name, 'w')
labelmap = np.zeros( (len(files), x_dim, y_dim, 1), dtype = np.int16)

#read in label data
for idx in np.arange(len(files)):
    i_file = h5py.File(create_filename(files[idx]), 'r')
    labelmap[idx,:,:,0] = i_file[dataset_name_in][0,:,:,0]
    i_file.close()

labelmap_f.create_dataset(dataset_name_in, data=labelmap)
labelmap_f.close()

#produce cropped trainingsimages
trainimages_f = h5py.File(trainimages, 'w')
trainimages = np.zeros((len(files), x_dim, y_dim, 1) )
#open rawdata
rawdata_f = h5py.File(raw_path, 'r')

for idx in np.arange(len(files)):
    trainimages[idx,:,:,0] = rawdata_f[dataset_raw][files[idx],:,:,0]
    
rawdata_f.close()
trainimages_f.create_dataset(dataset_train, data=trainimages)
trainimages_f.close()
