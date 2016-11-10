import numpy as np
import h5py
import vigra as vg
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from skimage.transform import rotate
import configargparse

from featureFct import *
from Functions import * 
from mergeClasses import *


if __name__ == '__main__':
    print 'begin prediction of trainimg for DPM with Autocontext'

    """
    parameters
    """
    
    #fixed parameters
    
    #for prediction
    relative_path_preprocessing = '../Preprocessing/'       #path to cropped rawdata
    filename_new_rawdata = 'rawdata.h5'
    datasetname_new_rawdata = 'data'
    relative_path_autocontext = '../Autocontext/'       #path to cropped Autocontext
    relative_path_pred = '../DPM/samples/pred/'                     #path to DPM where prediction.txt file is saved
    relative_path_pred_soft = '../DPM/samples/pred/probmap/'                 
    relative_path_pred_hard = '../DPM/samples/pred/hardseg/'                   
    relative_path_train_pos_soft = '../DPM/samples/train/pos/'                 
    relative_path_train_neg_soft = '../DPM/samples/train/neg/'                 
    relative_path_train_pos_hard = '../DPM/samples/train/hardseg/pos/'                 
    relative_path_train_neg_hard = '../DPM/samples/train/hardseg/neg/'      
    relative_path_neg_ex = 'DPM/neg_ex/'           
    filename_clf = 'clf'
    filename_extension_clf = '.pkl'
    output_dir_pred_soft = 'output/pred/soft/'
    output_dir_pred_hard = 'output/pred/hard/'
    output_dir_pos_soft = 'output/pos/soft/'
    output_dir_pos_hard = 'output/pos/hard/'
    output_dir_neg_soft = 'output/neg/soft/'
    output_dir_neg_hard = 'output/neg/hard/'
    filename_txt = 'predfile.txt' # txt file containing images that should be predicted by DPM later on
    #input: files with indices of images out of filenname_rawdata that should be predicted
    filename_indices_pos ='indices_pos.h5'
    filename_indices_neg ='indices_neg.h5'
    #output directory:
    output_dir_pos = 'output/pos'
    output_dir_neg = 'output/neg'
    #output: beginning  of filename for saving probmaps
    filename_seg_soft = 'probmap'
    #output: beginning  of filename for saving segmentation results 
    filename_seg_hard = 'hard_segmentation'
    
    
    """
    #for training
    #random forest parameters
    N_TREES = 600
    N_STAGES = 2
    """

    #possible values for sigmas: 0.3, 1.0, 1.6, 3.5, 5.0, 10.0
    #sigmas for features on raw data
    sigmasGaussian = [0.3,1.0,1.6,3.5, 5.0, 10.0]
    sigmasLoG = [0.3,1.0,1.6,3.5, 5.0, 10.0]
    sigmasGGM = [0.3,1.0,1.6,3.5, 5.0, 10.0]
    sigmasSTE = [0.3,1.0,1.6,3.5, 5.0, 10.0]# [1.0,1.6,3.5]
    sigmasHoGE = [0.3,1.0,1.6,3.5, 5.0, 10.0]#[1.0,1.6,3.5]
    #sigmas for features on prob maps
    sigmasGaussian_prob = [0.3,1.0,1.6,3.5, 5.0, 10.0]
    sigmasLoG_prob = [0.3,1.0,1.6,3.5, 5.0, 10.0]
    sigmasGGM_prob = [0.3,1.0,1.6,3.5, 5.0, 10.0]
    sigmasSTE_prob = [0.3,1.0,1.6,3.5, 5.0, 10.0]# [1.0,1.6,3.5]
    sigmasHoGE_prob = [0.3,1.0,1.6,3.5, 5.0, 10.0]#[1.0,1.6,3.5]


    #parameters set by configfile
    use_config = True                     #must be False if executed by hand
    N_STAGES = 2
    N_CLASSES = 7
    filename_rawdata = 'RawData.h5'
    dataset_rawdata = 'data'
    #auxiliary classes that should be merged by testing
    merged_classes_1 = [1]              #classes background(4) background (2) boundary(5) bright edge (6) are merged to background for testing
    merged_classes_2 = [2]              #classes background(4) background (2) boundary(5) bright edge (6) are merged to background for testing
    merged_classes_3 = [3]              #classes background(4) background (2) boundary(5) bright edge (6) are merged to background for testing
    merged_classes_4 = []               #classes background(4) background (2) boundary(5) bright edge (6) are merged to background for testing
    save_many_class_probmaps = True     #if True probmaps before merging are saved    
    save_hard_seg = True                #if True also hard prediction is saved 
    cropp_neg_ex = True         
    neg_ex_dataset = 'data'     

    #config arg parse
    p = configargparse.ArgParser()#(default_config_files=['config.ini'])
    p.add('-c', '--config', is_config_file=True, help='config file path')
    p.add('--autocontext-n_classes', default=N_CLASSES, type=int, help='number of classes in autocontext')    
    p.add('--autocontext-n_stages', default=N_STAGES, type=int, help='number of stages in autocontext')    
    p.add('--global-filename_rawdata', default=filename_rawdata, type=str, help='filename of rawdata')    
    p.add('--global-datasetname_rawdata', default=dataset_rawdata, type=str, help='datasetname of rawdata')    
    p.add('--autocontext-1_merged_class_1_old_class', default=-1, type=int, help='old class that belong to 1th merged class')    
    p.add('--autocontext-1_merged_class_2_old_class', default=-1, type=int, help='old class that belong to 1th merged class')    
    p.add('--autocontext-1_merged_class_3_old_class', default=-1, type=int, help='old class that belong to 1th merged class')    
    p.add('--autocontext-1_merged_class_4_old_class', default=-1, type=int, help='old class that belong to 1th merged class')    
    p.add('--autocontext-2_merged_class_1_old_class', default=-1, type=int, help='old class that belong to 2th merged class')    
    p.add('--autocontext-2_merged_class_2_old_class', default=-1, type=int, help='old class that belong to 2th merged class')    
    p.add('--autocontext-2_merged_class_3_old_class', default=-1, type=int, help='old class that belong to 2th merged class')    
    p.add('--autocontext-2_merged_class_4_old_class', default=-1, type=int, help='old class that belong to 2th merged class')    
    p.add('--autocontext-3_merged_class_1_old_class', default=-1, type=int, help='old class that belong to 3th merged class')    
    p.add('--autocontext-3_merged_class_2_old_class', default=-1, type=int, help='old class that belong to 3th merged class')    
    p.add('--autocontext-3_merged_class_3_old_class', default=-1, type=int, help='old class that belong to 3th merged class')    
    p.add('--autocontext-3_merged_class_4_old_class', default=-1, type=int, help='old class that belong to 3th merged class')    
    p.add('--autocontext-4_merged_class_1_old_class', default=-1, type=int, help='old class that belong to 4th merged class')    
    p.add('--autocontext-4_merged_class_2_old_class', default=-1, type=int, help='old class that belong to 4th merged class')    
    p.add('--autocontext-4_merged_class_3_old_class', default=-1, type=int, help='old class that belong to 4th merged class')    
    p.add('--autocontext-4_merged_class_4_old_class', default=-1, type=int, help='old class that belong to 4th merged class')    
    p.add('--autocontext-save_many_class_probmaps', default=False, action='store_true', help='if true: probmaps before merging are saved')    
    p.add('--autocontext-save_hard_seg', default=False, action='store_true', help='if true: hard segmentation are saved')  
    p.add('--dpm-cropp_neg_ex', default=False, action='store_true', help='if true: cropp negative examples out of rawdata else use files in directory Train_Pred/DPM/neg_ex')  
    p.add('--dpm_neg_ex_dataset', default=neg_ex_dataset, type=str, help='name of dataset in neg_ex files if they are given as hdf5 files in Train_Pred/DPM/neg_ex')    
    
    options, unknown = p.parse_known_args()
    
    #parse parameter
    if use_config==True:
        N_CLASSES = options.autocontext_n_classes
        N_STAGES = options.autocontext_n_stages                         
        filename_rawdata = options.global_filename_rawdata
        dataset_rawdata = options.global_datasetname_rawdata
        merged_classes_1 = []
        merged_classes_2 = []
        merged_classes_3 = []
        merged_classes_4 = []
        if N_CLASSES>=options.autocontext_1_merged_class_1_old_class>0:
            merged_classes_1.append(options.autocontext_1_merged_class_1_old_class)
        if N_CLASSES>=options.autocontext_1_merged_class_2_old_class>0:
            merged_classes_1.append(options.autocontext_1_merged_class_2_old_class)
        if N_CLASSES>=options.autocontext_1_merged_class_3_old_class>0:
            merged_classes_1.append(options.autocontext_1_merged_class_3_old_class)
        if N_CLASSES>=options.autocontext_1_merged_class_4_old_class>0:
            merged_classes_1.append(options.autocontext_1_merged_class_4_old_class)
        if N_CLASSES>=options.autocontext_2_merged_class_1_old_class>0:
            merged_classes_2.append(options.autocontext_2_merged_class_1_old_class)
        if N_CLASSES>=options.autocontext_2_merged_class_2_old_class>0:
            merged_classes_2.append(options.autocontext_2_merged_class_2_old_class)
        if N_CLASSES>=options.autocontext_2_merged_class_3_old_class>0:
            merged_classes_2.append(options.autocontext_2_merged_class_3_old_class)
        if N_CLASSES>=options.autocontext_2_merged_class_4_old_class>0:
            merged_classes_2.append(options.autocontext_2_merged_class_4_old_class)
        if N_CLASSES>=options.autocontext_3_merged_class_1_old_class>0:
            merged_classes_3.append(options.autocontext_3_merged_class_1_old_class)
        if N_CLASSES>=options.autocontext_3_merged_class_2_old_class>0:
            merged_classes_3.append(options.autocontext_3_merged_class_2_old_class)
        if N_CLASSES>=options.autocontext_3_merged_class_3_old_class>0:
            merged_classes_3.append(options.autocontext_3_merged_class_3_old_class)
        if N_CLASSES>=options.autocontext_3_merged_class_4_old_class>0:
            merged_classes_3.append(options.autocontext_3_merged_class_4_old_class)
        if N_CLASSES>=options.autocontext_4_merged_class_1_old_class>0:
            merged_classes_4.append(options.autocontext_4_merged_class_1_old_class)
        if N_CLASSES>=options.autocontext_4_merged_class_2_old_class>0:
            merged_classes_4.append(options.autocontext_4_merged_class_2_old_class)
        if N_CLASSES>=options.autocontext_4_merged_class_3_old_class>0:
            merged_classes_4.append(options.autocontext_4_merged_class_3_old_class)
        if N_CLASSES>=options.autocontext_4_merged_class_4_old_class>0:
            merged_classes_4.append(options.autocontext_4_merged_class_4_old_class)
        save_many_class_probmaps = options.autocontext_save_many_class_probmaps 
        save_hard_seg = options.autocontext_save_hard_seg
        cropp_neg_ex = options.dpm_cropp_neg_ex
        neg_ex_dataset = options.dpm_neg_ex_dataset
        

    #derived parameters
    mergers = [merged_classes_1, merged_classes_2, merged_classes_3, merged_classes_4]    
    #number of features
    n_feat_prob = len(sigmasGaussian_prob)+len(sigmasLoG_prob)+len(sigmasGGM_prob)+2*len(sigmasSTE_prob)+2*len(sigmasHoGE_prob)
    n_feat_raw = len(sigmasGaussian)+len(sigmasLoG)+len(sigmasGGM)+2*len(sigmasSTE)+2*len(sigmasHoGE)

    
    
    #read in clf
    print 'load Random Forest classifier'
    clf = []
    for i in np.arange(N_STAGES):
        i_clf = joblib.load(relative_path_autocontext + filename_clf + '_' + str(i) + filename_extension_clf)        
        clf.append(i_clf)
       

    """
    prediction for rawdata
    """
    
    print 'prediction of prob maps with autocontext for positive trainingsdata of DPM'    
       
    #do prediction for positive examples 
    img_idx_file_pos = h5py.File(relative_path_autocontext + filename_indices_pos,'r')
    n_img_pos = img_idx_file_pos['n_img'][0]
    for i_img in np.arange(n_img_pos): 
        progress = float(i_img)/float(n_img_pos)
        print 'progress: {0}%\r'.format(int(progress*100)),

        #predict img with selected idx (only images with bbox)
        idx = img_idx_file_pos['data'][i_img]
        y2D_pred_soft, y2D_pred_hard = predict_one_image_new_data(idx, filename_rawdata ,clf, N_CLASSES, sigmasGaussian, dataset_rawdata, sigmasLoG, sigmasGGM, sigmasSTE, sigmasHoGE, sigmasGaussian_prob, sigmasLoG_prob, sigmasGGM_prob, sigmasSTE_prob, sigmasHoGE_prob)
        y2D_pred_soft_merged = merge(y2D_pred_soft, mergers)
        #write prob map
        filename_soft = filename_seg_soft+ '_' + str(idx) + '.h5'
        filename_hard = filename_seg_hard + '_' + str(idx) + '.h5'
        prob_file = h5py.File(relative_path_train_pos_soft + filename_soft, 'w' )
        prob_file.create_dataset('data', data=np.swapaxes(y2D_pred_soft_merged,0,1)) #take only probmaps of foreground classes 
        prob_file.close()
        if save_hard_seg == True:
            prob_file = h5py.File(relative_path_train_pos_hard + filename_hard, 'w' )
            y2D_pred_hard_merged = hard_seg(y2D_pred_soft_merged, normalized=False)
            prob_file.create_dataset('data', data=y2D_pred_hard_merged[:,:]) 
            prob_file.close()
        if save_many_class_probmaps == True:
            prob_file = h5py.File(relative_path_autocontext + output_dir_pos_soft + filename_soft, 'w' )
            prob_file.create_dataset('data', data=y2D_pred_soft[:,:,:]) 
            prob_file.close()
            if save_hard_seg == True:
                prob_file = h5py.File(relative_path_autocontext + output_dir_pos_hard + filename_hard, 'w' )
                y2D_pred_hard = hard_seg(y2D_pred_soft)
                prob_file.create_dataset('data', data=y2D_pred_hard[:,:])  
                prob_file.close()   
    img_idx_file_pos.close()
    
    print 'prediction of prob maps with autocontext for negative trainingsdata of DPM'    
    
    
    #do prediction for negative examples
    if cropp_neg_ex == True:
        img_idx_file_neg = h5py.File(relative_path_autocontext + filename_indices_neg,'r')
        n_img_neg = img_idx_file_neg['n_img'][0] 
        for i_img in np.arange(n_img_neg): 
            progress = float(i_img)/float(n_img_neg)
            print 'progress: {0}%\r'.format(int(progress*100)),
            #predict img with selected idx (only images with bbox)
            idx = img_idx_file_neg['data'][i_img]
            y2D_pred_soft, y2D_pred_hard = predict_one_image_new_data(idx, filename_rawdata ,clf, N_CLASSES, sigmasGaussian, dataset_rawdata, sigmasLoG, sigmasGGM, sigmasSTE, sigmasHoGE, sigmasGaussian_prob, sigmasLoG_prob, sigmasGGM_prob, sigmasSTE_prob, sigmasHoGE_prob)
            y2D_pred_soft_merged = merge(y2D_pred_soft, mergers)
            #write prob map
            filename_soft = filename_seg_soft+ '_' + str(idx) + '.h5'
            filename_hard = filename_seg_hard + '_' + str(idx) + '.h5'
            prob_file = h5py.File(relative_path_train_neg_soft + filename_soft, 'w' )
            prob_file.create_dataset('data', data=np.swapaxes(y2D_pred_soft_merged,0,1)) #take only probmaps of foreground classes 
            prob_file.close()
            if save_hard_seg == True:
                prob_file = h5py.File(relative_path_train_neg_hard + filename_hard, 'w' )
                y2D_pred_hard_merged = hard_seg(y2D_pred_soft_merged, normalized=False)
                prob_file.create_dataset('data', data=y2D_pred_hard_merged[:,:]) 
                prob_file.close()
            if save_many_class_probmaps == True:
                prob_file = h5py.File(relative_path_autocontext + output_dir_neg_soft + filename_soft, 'w' )
                prob_file.create_dataset('data', data=y2D_pred_soft[:,:,:]) 
                prob_file.close()
                if save_hard_seg == True:
                    prob_file = h5py.File(relative_path_autocontext + output_dir_neg_hard + filename_hard, 'w' )
                    y2D_pred_hard = hard_seg(y2D_pred_soft)
                    prob_file.create_dataset('data', data=y2D_pred_hard[:,:])  
                    prob_file.close()   
        img_idx_file_neg.close()
    else:
        for file in os.listdir(relative_path_neg_ex):
            if file.endswith('.h5'):
                y2D_pred_soft, y2D_pred_hard = predict_one_image_new_data(0, relative_path_neg_ex + file ,clf, N_CLASSES, sigmasGaussian, neg_ex_dataset, sigmasLoG, sigmasGGM, sigmasSTE, sigmasHoGE, sigmasGaussian_prob, sigmasLoG_prob, sigmasGGM_prob, sigmasSTE_prob, sigmasHoGE_prob)
                y2D_pred_soft_merged = merge(y2D_pred_soft, mergers)
                #write prob map
                filename_soft = filename_seg_soft+ '_' + file 
                filename_hard = filename_seg_hard + '_' + file 
                prob_file = h5py.File(relative_path_train_neg_soft + filename_soft, 'w' )
                prob_file.create_dataset('data', data=np.swapaxes(y2D_pred_soft_merged,0,1)) #take only probmaps of foreground classes 
                prob_file.close()
                if save_hard_seg == True:
                    prob_file = h5py.File(relative_path_train_neg_hard + filename_hard, 'w' )
                    y2D_pred_hard_merged = hard_seg(y2D_pred_soft_merged, normalized=False)
                    prob_file.create_dataset('data', data=y2D_pred_hard_merged[:,:]) 
                    prob_file.close()
                if save_many_class_probmaps == True:
                    prob_file = h5py.File(relative_path_autocontext + output_dir_neg_soft + filename_soft, 'w' )
                    prob_file.create_dataset('data', data=y2D_pred_soft[:,:,:]) 
                    prob_file.close()
                    if save_hard_seg == True:
                        prob_file = h5py.File(relative_path_autocontext + output_dir_neg_hard + filename_hard, 'w' )
                        y2D_pred_hard = hard_seg(y2D_pred_soft)
                        prob_file.create_dataset('data', data=y2D_pred_hard[:,:])  
                        prob_file.close()   
        
                
            
