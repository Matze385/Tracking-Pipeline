#script to produce negfile.txt for DPM indices_neg.h5 for Autocontext and posfile.txt for DPM and indices_pos.h5 for Autocontext

#negative examples
#produce .txt file of negative examples in the following format:
#probmap_12.h5
#and produce .h5 file with indices of negative images that should be predicted by Autocontext
#need as input indices of images written by hand 
#name of hdf5 files
#RawData_00124.xml -> probmap_124.h5
#RawData_00124_2.xml -> probmap_124.h5

#positive examples
#produce .txt file of positive examples in the following format:
#name_of_file x y width height
#and produce .h5 file with indices of images that should be predicted by Autocontext
#need as input to be in folder with .xml files with name in following format: beginning + '_' + frame_idx + ending
#RawData_00124.xml -> probmap_124.h5
#RawData_00124_2.xml -> probmap_124.h5


import os
import xml.etree.ElementTree as ET
import re
import h5py
import numpy as np
import configargparse

if __name__ == '__main__':
    
    """
    parameter
    """
    print 'read trainingsdata for DPM'
    #parsed parameters
    use_config = True                     #must be False if executed by hand
    filename_neg_ex_idx = 'neg_ex_idx.txt'
    cropp_neg_ex = True         
    
    #config arg parse
    p = configargparse.ArgParser()#(default_config_files=['config.ini'])
    p.add('-c', '--config', is_config_file=True, help='config file path')
    p.add('--dpm-neg_ex_idx', default=filename_neg_ex_idx, type=str, help='filename of .txt file with indices of negative examples for DPM')    
    p.add('--dpm-cropp_neg_ex', default=False, action='store_true', help='if true: cropp negative examples out of rawdata else use files in directory Train_Pred/DPM/neg_ex')  
    options, unknown = p.parse_known_args()

    #parse parameter
    if use_config==True:
        filename_neg_ex_idx = options.dpm_neg_ex_idx
        cropp_neg_ex = options.dpm_cropp_neg_ex

    #fixed parameters
    #relative path to Autocontext
    relative_path_autocontext = '../Autocontext/' 
    relative_path_DPM_trainings_data = '../DPM/samples/train/' 
    relative_path_neg_ex_idx = 'DPM/'
    relative_path_neg_ex = 'DPM/neg_ex/'
    #fixed parameters
    max_imgs = 1000 #different imgs with bboxes
    #filename of output txt file
    neg_filename_txt = 'negfile.txt'
    #filename of output h5 file with selected img indices
    neg_filename_h5 = 'indices_neg.h5'
    #path and filename for files within output txt file
    path = '../neg/'
    neg_beginning = 'probmap' #beginning of filename of 
    neg_ending = '.h5'

    posfile = open(relative_path_DPM_trainings_data + neg_filename_txt, "w")

    if cropp_neg_ex == True:
        negcount = 0
        
        #read in idx_neg out of neg_ex_idx.txt file
        
        idx_neg_str = [line.rstrip('\n') for line in open(relative_path_neg_ex_idx + filename_neg_ex_idx, 'r')]
        idx_neg = [int(idx_str) for idx_str in idx_neg_str]

        #save indices of negative images
        img_indices = np.zeros((max_imgs,), dtype=np.int16)
        n_img = 0 
        #go in directory and loop over all xml files
        for idx in idx_neg:
            negcount += 1
            filename = path + neg_beginning + '_' + str(idx) + neg_ending
            posfile.write(filename + '\n')
            #read idx of img
            #print idx
            if idx not in img_indices:
                img_indices[n_img] = idx
                n_img += 1
            
        #save indices of images with bbox for prediction of Autocontext
        img_idx = h5py.File(relative_path_autocontext + neg_filename_h5, "w")
        img_idx.create_dataset("data", data=img_indices)
        img_idx.create_dataset("n_img", (1,), dtype=np.int16)
        img_idx["n_img"][0] = n_img
        img_idx.close()
        print 'number of neg images for DPM: {0}'.format(negcount)           
    else:
        negcount = 0
        for file in os.listdir(relative_path_neg_ex):
            if file.endswith('.h5'):
                filename = path + neg_beginning + '_' + file + neg_ending
                posfile.write(filename + '\n')
                negcount += 1
        print 'number of neg images for DPM: {0}'.format(negcount)           

    posfile.close()

    #positive examples

    #path to directory with bboxes in xml format
    rect_dir = 'DPM/Bounding_Boxes/'
    #filename of output txt file
    pos_filename_txt = 'posfile.txt'
    #filename of output h5 file with selected img indices
    pos_filename_h5 = 'indices_pos.h5'
    #filenames for files within output txt file
    pos_beginning = 'probmap' #beginning of filename of 
    pos_ending = '.h5'

    posfile = open(relative_path_DPM_trainings_data + pos_filename_txt, "w")
    poscount = 0

    #save indices of images with bbox
    img_indices = np.zeros((max_imgs,), dtype=np.int16)
    n_img = 0 
    #go in directory and loop over all xml files
    for file in os.listdir(rect_dir):
        if file.endswith('.xml'):
            #print(file)
            poscount += 1
            tree = ET.parse(rect_dir+file)
            root = tree.getroot()

            #bounding box information, parameters that must be read for each xml file
            x = 0
            y = 0
            width = 0
            height = 0

            #root[6][4]: annotation-> object -> bndbox
            x_min = int(root[6][4][0].text)
            y_min = int(root[6][4][1].text) 
            x_max = int(root[6][4][2].text) 
            y_max = int(root[6][4][3].text) 

            x = x_min
            y = y_min
            width = x_max - x_min
            height = y_max - y_min
            file_idx_re = re.search(r'[0-9]{1,}', file)
            idx = int(file_idx_re.group())
            filename = pos_beginning + '_' + str(idx) + pos_ending
            posfile.write(filename + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + '\n')
            #posfile.write(filename + ' ' + str(y) + ' ' + str(x) + ' ' + str(height) + ' ' + str(width) + '\n')

            #read idx of img
            #print idx
            if idx not in img_indices[:n_img]:
                img_indices[n_img] = idx
                n_img += 1
            
    #save indices of images with bbox for prediction of Autocontext
    img_idx = h5py.File(relative_path_autocontext + pos_filename_h5, "w")
    img_idx.create_dataset("data", data=img_indices)
    img_idx.create_dataset("n_img", (1,), dtype=np.int16)
    img_idx["n_img"][0] = n_img
    img_idx.close()

    posfile.close()

    print 'number of pos examples for DPM: {0}'.format(poscount)           
