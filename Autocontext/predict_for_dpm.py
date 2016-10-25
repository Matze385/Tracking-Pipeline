import numpy as np
import h5py
import vigra as vg

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from skimage.transform import rotate
import configargparse

from featureFct import *
from Functions import * 
from mergeClasses import *


if __name__ == '__main__':

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
    filename_clf = 'clf'
    filename_extension_clf = '.pkl'
    output_dir_pred_soft = 'output/pred/soft/'
    output_dir_pred_hard = 'output/pred/hard/'
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
    arrow_orientation = True     
    n_rot = 8
    x_center = 510
    y_center = 514
    #auxiliary classes that should be merged by testing
    merged_classes_1 = [1]              #classes background(4) background (2) boundary(5) bright edge (6) are merged to background for testing
    merged_classes_2 = [2]              #classes background(4) background (2) boundary(5) bright edge (6) are merged to background for testing
    merged_classes_3 = [3]              #classes background(4) background (2) boundary(5) bright edge (6) are merged to background for testing
    merged_classes_4 = []               #classes background(4) background (2) boundary(5) bright edge (6) are merged to background for testing
    save_many_class_probmaps = True     #if True probmaps before merging are saved    
    save_hard_seg = True                #if True also hard prediction is saved 

    #config arg parse
    p = configargparse.ArgParser()#(default_config_files=['config.ini'])
    p.add('-c', '--config', is_config_file=True, help='config file path')
    p.add('--autocontext-n_classes', default=N_CLASSES, type=int, help='number of classes in autocontext')    
    p.add('--autocontext-n_stages', default=N_STAGES, type=int, help='number of stages in autocontext')    
    p.add('--global-arrow_orientation', default=False, action='store_true', help='if true: arrow orientation exist, if wrong: only axis orientation exist')    
    p.add('--global-n_rot', default=n_rot, type=int, help='number of rotations of probability maps')    
    p.add('--global-x_center', default=x_center, type=int, help='coordinates of center for rotations')    
    p.add('--global-y_center', default=y_center, type=int, help='coordinates of center for rotations')    
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
    
    options, unknown = p.parse_known_args()
    
    #parse parameter
    if use_config==True:
        N_CLASSES = options.autocontext_n_classes
        N_STAGES = options.autocontext_n_stages                         
        arrow_orientation = options.global_arrow_orientation
        n_rot = options.global_n_rot
        x_center = options.global_x_center
        y_center = options.global_y_center
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


    #derived parameters
    mergers = [merged_classes_1, merged_classes_2, merged_classes_3, merged_classes_4]    
    #number of features
    n_feat_prob = len(sigmasGaussian_prob)+len(sigmasLoG_prob)+len(sigmasGGM_prob)+2*len(sigmasSTE_prob)+2*len(sigmasHoGE_prob)
    n_feat_raw = len(sigmasGaussian)+len(sigmasLoG)+len(sigmasGGM)+2*len(sigmasSTE)+2*len(sigmasHoGE)

    
    #image used for qualitative analysis (idx 4 corresponds to image 461)
    idx_test_img = 4 

    #derived parameters (set automatically later by programm)
    xDim = 0				
    yDim = 0
    #number of used features
    n_features = 0
    #number of folds
    n_folds = 0
    #derived automatically out of file filename_rawdata
    dFrames = h5py.File(relative_path_preprocessing + filename_new_rawdata, 'r')
    n_frames_out = dFrames[datasetname_new_rawdata].shape[0]        #format of rawdata [t,x,y,c] 
    xDim = dFrames[datasetname_new_rawdata].shape[1]
    yDim = dFrames[datasetname_new_rawdata].shape[2]
    dFrames.close()

    #read in clf
    print 'load Random Forest classifier'
    clf = []
    for i in np.arange(N_STAGES):
        i_clf = joblib.load(relative_path_autocontext + filename_clf + '_' + str(i) + filename_extension_clf)        
        clf.append(i_clf)
       
    print 'begin prediction of prob maps with autocontext'    

    """
    prediction for rawdata
    """
       
    """
    #do prediction for positive examples 
    img_idx_file_pos = h5py.File(filename_indices_pos,'r')
    n_img_pos = img_idx_file_pos['n_img'][0]
    for i_img in np.arange(n_img_pos): 
        #predict img with selected idx (only images with bbox)
        idx = img_idx_file_pos['data'][i_img]
        y2D_pred_soft, y2D_pred_hard = predict_one_image_new_data(idx, filename_rawdata ,clf, N_CLASSES, sigmasGaussian, sigmasLoG, sigmasGGM, sigmasSTE, sigmasHoGE, sigmasGaussian_prob, sigmasLoG_prob, sigmasGGM_prob, sigmasSTE_prob, sigmasHoGE_prob)
        #write prob map
        path_soft = output_dir_pos + '/soft/' + filename_seg_soft+ '_'+ str(idx) + '.h5'
        prob_file = h5py.File(path_soft, 'w' )
        prob_file.create_dataset('data', data=y2D_pred_soft[:,:,:]) #take only probmaps of foreground classes 
        prob_file.close()
        if seg_hard_prediction == True:
            #merge auxiliary background classes
            y2D_pred_hard = merge_classes(y2D_pred_hard.reshape((xDim*yDim,)), merged_classes).reshape((xDim, yDim))
            y2D_pred_hard = merge_classes(y2D_pred_hard.reshape((xDim*yDim,)), merged_classes_2).reshape((xDim, yDim))
            #write segmenatation map
            path_hard = output_dir_pos + '/hard/' + filename_seg_hard+ '_'+ str(idx) + '.h5'
            seg_file = h5py.File(path_hard, 'w' )
            seg_file.create_dataset('data', (xDim, yDim, 1), dtype=np.uint32)  
            seg_file['data'][:,:,0] = y2D_pred_hard[:,:]
            seg_file.close()
        print('progress: {i}/{total} '.format(i=i_img, total=n_img_pos))
    img_idx_file_pos.close()
     
    
    #do prediction for negative examples
    img_idx_file_neg = h5py.File(filename_indices_neg,'r')
    n_img_neg = img_idx_file_neg['n_img'][0] 
    for i_img in np.arange(n_img_neg): 
        #predict img with selected idx (only images with bbox)
        idx = img_idx_file_neg['data'][i_img]
        y2D_pred_soft, y2D_pred_hard = predict_one_image_new_data(idx, filename_rawdata ,clf, N_CLASSES, sigmasGaussian, sigmasLoG, sigmasGGM, sigmasSTE, sigmasHoGE, sigmasGaussian_prob, sigmasLoG_prob, sigmasGGM_prob, sigmasSTE_prob, sigmasHoGE_prob)
        #write prob map
        path_soft = output_dir_neg + '/soft/' + filename_seg_soft+ '_'+ str(idx) + '.h5'
        prob_file = h5py.File(path_soft, 'w' )
        prob_file.create_dataset('data', data=y2D_pred_soft[:,:,:]) #take only probmaps of foreground classes 
        prob_file.close()
        if seg_hard_prediction == True:
            #merge auxiliary background classes
            y2D_pred_hard = merge_classes(y2D_pred_hard.reshape((xDim*yDim,)), merged_classes).reshape((xDim, yDim))
            y2D_pred_hard = merge_classes(y2D_pred_hard.reshape((xDim*yDim,)), merged_classes_2).reshape((xDim, yDim))
            #write segmenatation map
            path_hard = output_dir_neg + '/hard/' + filename_seg_hard+ '_'+ str(idx) + '.h5'
            seg_file = h5py.File(path_hard, 'w' )
            seg_file.create_dataset('data', (xDim, yDim, 1), dtype=np.uint32)  
            seg_file['data'][:,:,0] = y2D_pred_hard[:,:]
            seg_file.close()
        print('progress: {i}/{total} '.format(i=i_img, total=n_img_neg))
    img_idx_file_neg.close()
    """   
    
    
    predfile = open(relative_path_pred + filename_txt, 'w')
    for i_img in np.arange(n_frames_out):
        progress = float(i_img)/float(n_frames_out)
        print 'progress: {0}%\r'.format(int(progress*100)),
        y2D_pred_soft, y2D_pred_hard = predict_one_image_new_data(i_img, relative_path_preprocessing + filename_new_rawdata ,clf, N_CLASSES, sigmasGaussian, datasetname_new_rawdata ,sigmasLoG, sigmasGGM, sigmasSTE, sigmasHoGE, sigmasGaussian_prob, sigmasLoG_prob, sigmasGGM_prob, sigmasSTE_prob, sigmasHoGE_prob)
        x_dim = y2D_pred_soft.shape[0]
        y_dim = y2D_pred_soft.shape[1]
        angles = np.linspace(0., 360., num=n_rot, endpoint=False)
        if arrow_orientation == False:
            angles = np.linspace(0., 180., num=n_rot, endpoint=False)
        for i_rot, angle in enumerate(angles):
            filename_soft = filename_seg_soft + '_' + str(i_img) + '_' + str(i_rot) + '.h5'
            filename_hard = filename_seg_hard + '_' + str(i_img) + '_' + str(i_rot) + '.h5'
            predfile.write(filename_soft + '\n')
            if i_rot==0:
                if save_many_class_probmaps == True:
                    prob_file = h5py.File(relative_path_autocontext + output_dir_pred_soft + filename_soft, 'w' )
                    prob_file.create_dataset('data', data=np.swapaxes(y2D_pred_soft,0,1)) 
                    prob_file.close()
                    if save_hard_seg == True:
                        prob_file = h5py.File(relative_path_autocontext + output_dir_pred_hard + filename_hard, 'w' )
                        y2D_pred_hard = hard_seg(y2D_pred_soft)
                        prob_file.create_dataset('data', data=y2D_pred_hard[:,:])  
                        prob_file.close()   
                y2D_pred_soft_merged = merge(y2D_pred_soft, mergers)
                prob_file = h5py.File(relative_path_pred_soft + filename_soft, 'w' )
                prob_file.create_dataset('data', data=y2D_pred_soft_merged[:,:,:]) 
                prob_file.close()
                if save_hard_seg == True:
                    prob_file = h5py.File(relative_path_pred_hard + filename_hard, 'w' )
                    y2D_pred_hard_merged = hard_seg(y2D_pred_soft_merged, normalized=False)
                    prob_file.create_dataset('data', data=y2D_pred_hard_merged[:,:]) 
                    prob_file.close()   
            else: 
                y2D_pred_soft_rot = np.zeros((x_dim, y_dim, y2D_pred_soft.shape[2]), dtype=np.float32)
                for channel in np.arange(y2D_pred_soft_rot.shape[2]):
                    y2D_pred_soft_rot[:,:,channel] = rotate(y2D_pred_soft[:,:,channel], angle, resize=False, center=(y_center, x_center) )
                if save_many_class_probmaps == True:
                    prob_file = h5py.File(relative_path_autocontext + output_dir_pred_soft + filename_soft, 'w' )
                    prob_file.create_dataset('data', data=np.swapaxes(y2D_pred_soft_rot,0,1)) 
                    prob_file.close()
                    if save_hard_seg == True:
                        prob_file = h5py.File(relative_path_autocontext + output_dir_pred_hard + filename_hard, 'w' )
                        y2D_pred_hard_rot = hard_seg(y2D_pred_soft_rot)
                        prob_file.create_dataset('data', data=y2D_pred_hard_rot[:,:]) 
                        prob_file.close()   
                y2D_pred_soft_rot_merged = merge(y2D_pred_soft_rot, mergers)
                prob_file = h5py.File(relative_path_pred_soft + filename_soft, 'w' )
                prob_file.create_dataset('data', data=y2D_pred_soft_rot_merged[:,:,:]) 
                prob_file.close()
                if save_hard_seg == True:
                    prob_file = h5py.File(relative_path_pred_hard + filename_hard, 'w' )
                    y2D_pred_hard_rot_merged = hard_seg(y2D_pred_soft_rot_merged, normalized=False)
                    prob_file.create_dataset('data', data=y2D_pred_hard_rot_merged[:,:]) 
                    prob_file.close()

    predfile.close()


    
