"""
important: must be executed out of folder Train_Pred when used without train.py to have correct file pathes
select for each of the 4 labelmaps a balanced trainingset consisting of coordinates and labels 
input: 
number of trainingspixel in total 
labelmaps (labelmap_1, labelmap_2, ...) hdf5 file in format [1, x,y,ch=1]
trainimages (trainimg_1, ...) hdf5 file in format [1, x, y, ch=1] rawdata of labeled images
output:
preprocessing: produce one file with labelmaps out of single labelmaps, produce one file with trainingsimages out of single trainimgs
coordinates and labels for training saved in CrossValFolds
"""

import sys, getopt
import numpy as np
import h5py
import vigra as vg
import skimage as skim

from skimage.transform import rotate 
from sampling import *
import configargparse

if __name__ == "__main__":
    #fixed parameters
    N_IMG = 5                              #number of labeled images correspond to number of folds in crossvalidation
    filename_labels = 'labels.h5'          #name of hdf5 file with all 4 label maps in format [img, x, y, ch=1]
    labels_dataset = 'data'                #name of dataset in labels.h5
    filename_trainimg = 'trainimages.h5'   #name of file with rawdata of trainimgs in format [img, x, y, ch=1]
    trainimg_dataset = 'data'
    relative_path_labelmaps = 'Autocontext/'    #relative path from train.py to labelmaps 
    relative_path_autocontext = '../Autocontext/' #relative path from train.py to Autocontext working dir

    #default parameters when no command line parameter is used
    use_config = True                     #must be False if executed by hand
    N_TRAIN = 500                         #approximate number of trainingspixel in total, reset by command line parameter
    N_CLASS = 7                             #number of classes
    filename_labelmap_1 = 'Labels_1.h5'
    filename_labelmap_2 = 'Labels_2.h5'
    filename_labelmap_3 = 'Labels_3.h5'
    filename_labelmap_4 = 'Labels_4.h5'
    labelmap_dataset = 'data'
    filename_rawdata = 'RawData.h5'         #filename of rawdata for cropping out trainimg, in format [i_img, x, y, ch=0]
    datasetname_rawdata = 'data'            
    cropp_trainimg = True                   #should rawadata of trainimg cropped out of rawdata
    cropped_trainimg_1 = 0                  #idx of trainimg in rawdata if cropp_trainimg=True
    cropped_trainimg_2 = 1 
    cropped_trainimg_3 = 2 
    cropped_trainimg_4 = 3 
    filename_trainimg_1 = 'trainimg_1.h5'            #if cropp_trainimg = False filenames of trainimg
    filename_trainimg_2 = 'trainimg_2.h5'             
    filename_trainimg_3 = 'trainimg_3.h5'
    filename_trainimg_4 = 'trainimg_4.h5'
    

    #derived parameters
    n_pix_per_class = 0     #number of pixel per class per image
    size_of_fold = 0        #size of each fold, is also size of testset
    n_train_exact = 0       #exact number of trainingspixel for unrotated case
    n_train_exact_rot=0

    #config arg parse
    p = configargparse.ArgParser()#(default_config_files=['config.ini'])
    p.add('-c', '--config', is_config_file=True, help='config file path')
    p.add('--autocontext-n_classes', default=N_CLASS, type=int, help='number of classes to train the autocontext')
    p.add('--autocontext-n_trainpxl', default=N_TRAIN, type=int, help='total number of trainingspxl to train the autocontext')
    p.add('--autocontext-labelmap_1', default=filename_labelmap_1, type=str, help='filename of partially labeled labelmaps for training autocontext as hdf5 files in format [x, y, ch=1]')
    p.add('--autocontext-labelmap_2', default=filename_labelmap_2, type=str, help='filename of partially labeled labelmaps for training autocontext as hdf5 files in format [x, y, ch=1]')
    p.add('--autocontext-labelmap_3', default=filename_labelmap_3, type=str, help='filename of partially labeled labelmaps for training autocontext as hdf5 files in format [x, y, ch=1]')
    p.add('--autocontext-labelmap_4', default=filename_labelmap_4, type=str, help='filename of partially labeled labelmaps for training autocontext as hdf5 files in format [x, y, ch=1]')
    p.add('--autocontext-labelmap_dataset', default=labelmap_dataset, type=str, help='name of dataset in labelmaps')
    p.add('--global-filename_rawdata', default=filename_rawdata, type=str, help='filename of rawdata for cropping out trainimg, format [i_img, x, y, ch=0]')
    p.add('--global-datasetname_rawdata', default=datasetname_rawdata, type=str, help='datasetname of rawdata')
    p.add('--autocontext-cropp_trainimg', default=False, action='store_true', help='if true: trainimg are cropped out of rawdata')
    p.add('--autocontext-cropped_trainimg_1', default=cropped_trainimg_1, type=int, help='if cropp_trainimg true: take idx to choose trainimg out of rawdata')
    p.add('--autocontext-cropped_trainimg_2', default=cropped_trainimg_2, type=int, help='if cropp_trainimg true: take idx to choose trainimg out of rawdata')
    p.add('--autocontext-cropped_trainimg_3', default=cropped_trainimg_3, type=int, help='if cropp_trainimg true: take idx to choose trainimg out of rawdata')
    p.add('--autocontext-cropped_trainimg_4', default=cropped_trainimg_4, type=int, help='if cropp_trainimg true: take idx to choose trainimg out of rawdata')
    p.add('--autocontext-trainimg_1', default=filename_trainimg_1, type=str, help='if cropp_trainimg false: filename of rawdata for training autocontext as hdf5 files in format [x, y, ch=1]')
    p.add('--autocontext-trainimg_2', default=filename_trainimg_2, type=str, help='if cropp_trainimg false: filename of rawdata for training autocontext as hdf5 files in format [x, y, ch=1]')
    p.add('--autocontext-trainimg_3', default=filename_trainimg_3, type=str, help='if cropp_trainimg false: filename of rawdata for training autocontext as hdf5 files in format [x, y, ch=1]')
    p.add('--autocontext-trainimg_4', default=filename_trainimg_4, type=str, help='if cropp_trainimg false: filename of rawdata for training autocontext as hdf5 files in format [x, y, ch=1]')
    options, unknown = p.parse_known_args()
    
    #parse parameter
    if use_config==True:
        N_TRAIN = options.autocontext_n_trainpxl                         
        N_CLASS = options.autocontext_n_classes
        filename_labelmap_1 = options.autocontext_labelmap_1
        filename_labelmap_2 = options.autocontext_labelmap_2
        filename_labelmap_3 = options.autocontext_labelmap_3
        filename_labelmap_4 = options.autocontext_labelmap_4
        labelmap_dataset = options.autocontext_labelmap_dataset
        filename_rawdata = options.global_filename_rawdata
        datasetname_rawdata = options.global_datasetname_rawdata
        cropp_trainimg = options.autocontext_cropp_trainimg 
        cropped_trainimg_1 = options.autocontext_cropped_trainimg_1
        cropped_trainimg_2 = options.autocontext_cropped_trainimg_2
        cropped_trainimg_3 = options.autocontext_cropped_trainimg_3
        cropped_trainimg_4 = options.autocontext_cropped_trainimg_4
        filename_trainimg_1 = options.autocontext_trainimg_1   
        filename_trainimg_2 = options.autocontext_trainimg_2                
        filename_trainimg_3 = options.autocontext_trainimg_3
        filename_trainimg_4 = options.autocontext_trainimg_4

        
    #set derived parameters
    n_pix_per_class = int(round(N_TRAIN/(N_IMG-1)/N_CLASS))     #number of pixel per class per image
    filenames_labelmaps = [filename_labelmap_1, filename_labelmap_2, filename_labelmap_3, filename_labelmap_4]
    filenames_trainimgs = [filename_trainimg_1, filename_trainimg_2, filename_trainimg_3, filename_trainimg_4]
    cropped_trainimg = [cropped_trainimg_1, cropped_trainimg_2, cropped_trainimg_3, cropped_trainimg_4]         #list of idx of trainimgs
    size_of_fold = n_pix_per_class*N_CLASS  #size of each fold, is also size of testset
    n_train_exact = size_of_fold*(N_IMG-1)          #exact number of trainingspixel for unrotated 
    print 'number of classes: ', N_CLASS
    print 'exact number trainingspxl: ', n_train_exact 
    x_dim = 0   #x dimension of labelmap/trainimg, deduced later on
    y_dim = 0   #y dimension of labelmap/trainimg, deduced later on
    #main program
    
    #put labelmaps to one h5 file
    labelmap_1_f = h5py.File(relative_path_labelmaps + filename_labelmap_1, 'r')
    x_dim = labelmap_1_f[labelmap_dataset].shape[1]
    y_dim = labelmap_1_f[labelmap_dataset].shape[2]
    labelmap_1_f.close()    

    labelmap_f = h5py.File(relative_path_autocontext + filename_labels, 'w')
    labelmap = np.zeros( (N_IMG, x_dim, y_dim, 1), dtype = np.int16)

    for idx, filename_labelmap in enumerate(filenames_labelmaps):
        i_file = h5py.File(relative_path_labelmaps + filename_labelmap, 'r')
        assert(i_file[labelmap_dataset].shape[1]==x_dim),'dimensions of labelmaps must be equal'
        assert(i_file[labelmap_dataset].shape[2]==y_dim),'dimensions of labelmaps must be equal'
        labelmap[idx,:,:,0] = i_file[labelmap_dataset][0,:,:,0]
        i_file.close()

    labelmap_f.create_dataset(labels_dataset, data=labelmap)
    labelmap_f.close()
        
    
    
    #put trainimgs in one file
    if cropp_trainimg==True:
        trainimages_f = h5py.File(relative_path_autocontext + filename_trainimg, 'w')
        trainimages = np.zeros((len(cropped_trainimg), x_dim, y_dim, 1) )
        #open rawdata
        rawdata_f = h5py.File(filename_rawdata, 'r')

        for i,idx in enumerate(cropped_trainimg):
            assert( 0 <= idx < rawdata_f[datasetname_rawdata].shape[0]), 'idx of trainimg out of rawdata must be equal or greater 0 and smaller img_dim of rawdata'
            assert( rawdata_f[datasetname_rawdata].shape[1]==x_dim), 'x_dim of trainimg out of rawdata must be equal to x_dim of labelmap'
            assert( rawdata_f[datasetname_rawdata].shape[2]==y_dim), 'y_dim of trainimg out of rawdata must be equal to y_dim of labelmap'
            trainimages[i,:,:,0] = rawdata_f[datasetname_rawdata][idx,:,:,0]
            
        rawdata_f.close()
        trainimages_f.create_dataset(trainimg_dataset, data=trainimages)
        trainimages_f.close()
    else:
        trainimages_f = h5py.File(relative_path_autocontext + filename_trainimg, 'w')
        trainimages = np.zeros((len(filenames_trainimgs), x_dim, y_dim, 1) )

        for i,filename_trainimg in enumerate(filenames_trainimgs):
            trainimg_f = h5py.File(relative_path_labelmaps + filename_trainimg, 'r')
            assert( trainimg_f[trainimg_dataset].shape[1]==x_dim), 'x_dim of trainimg out of single file must be equal to x_dim of labelmap'
            assert( trainimg_f[trainimg_dataset].shape[2]==y_dim), 'y_dim of trainimg out of single file must be equal to y_dim of labelmap'
            trainimages[i,:,:,0] = trainimg_f[trainimg_dataset][0,:,:,0]
            trainimg_f.close()
            
        trainimages_f.create_dataset(trainimg_dataset, data=trainimages)
        trainimages_f.close()
    

    
    with h5py.File(relative_path_autocontext + filename_labels,'r') as train_lab_f: #trainingsLabels.
        #data structure to store selected coordinates of pixels for test/training-set for each image, number of rotated labels is due to nearest neighbor interpolation not fix->size_of_fold*1.1
        coord = np.zeros((N_IMG, 2, size_of_fold), dtype = np.int32)
        labels = np.zeros((N_IMG, size_of_fold), dtype=np.uint8)
        for i in np.arange(N_IMG-1):
            lab_img = train_lab_f[labels_dataset][i,:,:,0]
            selected_map = selectPixelsBalanced(lab_img, n_pix_per_class, N_CLASS)
            #save coordinates
            x, y = np.nonzero(selected_map)
            coord[i,0,:] = x
            coord[i,1,:] = y
            labels[i,:] = selected_map[x,y]

        with h5py.File(relative_path_autocontext + 'CrossValFolds.h5','w') as foldsF:
            #format of coord [n_img_rot, x/y, sample]: coord
            foldsF.create_dataset('folds', data=coord)       
            #format of labels [n_img_rot,sample]: label, labels with 0 are not used 
            foldsF.create_dataset('labels', data=labels) 
            

     


