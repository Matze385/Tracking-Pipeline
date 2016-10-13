import numpy as np
import h5py
import vigra as vg

from sklearn.ensemble import RandomForestClassifier
from skimage.transform import rotate

from featureFct import *
from Functions import * 
from mergeClasses import *

"""
parameters
"""

#random forest parameters
N_TREES = 600
N_STAGES = 2
#input: file with rawdata for prediction
filename_rawdata = 'rawdata_aug.h5'
#input: inidices of images that should be predicted
idx_start = 200
number_predicted = 200
output_dir_pred = 'output/pred/soft/'
n_rot = 8
x_center = 510
y_center = 514
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
seg_hard_prediction = True #if True also hard prediction is done 
filename_seg_hard = 'hard_segmentation'

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

#number of features
n_feat_prob = len(sigmasGaussian_prob)+len(sigmasLoG_prob)+len(sigmasGGM_prob)+2*len(sigmasSTE_prob)+2*len(sigmasHoGE_prob)
n_feat_raw = len(sigmasGaussian)+len(sigmasLoG)+len(sigmasGGM)+2*len(sigmasSTE)+2*len(sigmasHoGE)

#dataaugmentation, rotational settings
X_CENTER = 510		                #center coordinates for rotation
Y_CENTER = 514
RADIUS = 500
N_ROT = 16                              #number of rotations per image, determines rotational angle
N_CLASSES = 7                           #number of different labels, here edge, background, boundary, head, upper body, lower body, wings
#testparameters
#auxiliary classes that should be merged by testing
merged_classes = np.array([4,5,6])              #classes background(4) background (2) boundary(5) bright edge (6) are merged to background for testing
merged_classes_2 = np.array([1,2])              #classes tail
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
dFrames = h5py.File(filename_rawdata, 'r')
n_frames_out = dFrames['data'].shape[0]        #format of rawdata [t,x,y,c] 
dFrames.close()


"""
read in data
"""

labelsF = h5py.File('CrossValFolds.h5','r')
#folds: [img, x/y, selected_pixels] coordinates of labels
folds = vg.readHDF5(labelsF,'folds')
#labels: labels [img, selected_pixels] c: class labels c=0 no label
labels = vg.readHDF5(labelsF,'labels')
labelsF.close()

#labels_all: [img, x, y, 0] needed for test sets with pixels out of complete interesting region
labelsAllF = h5py.File('labels_1700_29_1200_800.h5','r')
labels_all = labelsAllF['data'][:,:,:,0]
labelsAllF.close()

print("read in finished")
n_folds  = labels.shape[0]
#n_folds = 5
print ('n_folds:', n_folds)

#n_features = features.shape[3]
xDim = labels_all.shape[1]
yDim = labels_all.shape[2]


#feature selection on unrotated data
#dont forget to integrate following comments into code when using subset of std feat on rawdata
#select_std_features(0,'rawdata','features.h5', sigmasGaussian, sigmasLoG, sigmasGGM, sigmasSTE, sigmasHoGE) #features [x,y,img,f]
#select_std_features(0,'rawdata_rotated', 'featuresRot.h5', sigmasGaussian, sigmasLoG, sigmasGGM, sigmasSTE, sigmasHoGE)
print("feature calc finished")

#array for number of trainingssamples in different trainingssplits of crossval
n_train = np.zeros((n_folds))

"""
read in list of images that should be predicted
"""



"""
main programm
"""

#n_fold crossvalidation1

for testindex in np.arange(1):#n_folds):
    #do not forget to comment out when using testindex in arange(n_folds)
    testindex += 4
    """
    training
    """
    #list for RF of different stages
    clf = []
    #cummulate trainingssamples in different stages
    n_train_1_stage = 0
    #train different stages
    for i_stage in np.arange(N_STAGES):
        #train stages on different trainingssets
        if i_stage==0:
            X_train, y_train= train_sets_stages_several_files(0, N_STAGES, labels, folds,['selected_std_feat_rawdata_stage_0.h5'], testindex, rot=False, n_rot=N_ROT, x_center=X_CENTER, y_center=Y_CENTER)
            #X_train, y_train= train_sets_stages(0, N_STAGES, labels, folds,'selected_std_feat_rawdata_stage_0.h5', testindex, rot=False, n_rot=N_ROT, x_center=X_CENTER, y_center=Y_CENTER)
            n_train_1_stage += len(y_train)
            #add RF object to list clf
            clf.append(RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1))
            #train random forest of this stage
            clf[i_stage].fit(X_train, y_train)
        else:
            #indices of images prob maps are computed on
            images = img_split(i_stage, N_STAGES, testindex, n_folds, labels, rot=False, n_rot=N_ROT)
            #images = img_split(i_stage-1, n_stages, testindex, n_folds, labels, rot=True, n_rot=N_ROT)
            print images
            #create prob map of RF of stage before, saved in 'probability_map_stage_'+str(i_stage).h5 in dataset 'data' array shape [img,x,y,c]
            create_probability_maps(i_stage-1, clf[i_stage-1], n_folds, images, N_CLASSES)
            #create_probability_maps(i_stage-1, clf[i_stage-1], n_folds*N_ROT, images, N_CLASSES, rot=True)
            print('prob maps finished of stage: ',i_stage)  
            #delete features out of prob maps of stage before and prob_map of two stages before if i_stage>=2 
            if i_stage>1:
                filename_std_feat_out_of_prob_before = 'selected_std_feat_prob_stage_'+str(i_stage-1)+'.h5' 
                f = h5py.File(filename_std_feat_out_of_prob_before,'w')
                f.create_dataset('data',(1,))
                f.close()
                filename_prob_before = 'probability_map_stage_'+str(i_stage-2)+'.h5'
                f = h5py.File(filename_prob_before,'w')
                f.create_dataset('data',(1,))
                f.close()
            #calculation of features out of prob maps of stage before, saved in 'selected_std_feat_prob_stage_'+str(i_stage)+'.h5'
            filename_prob = 'probability_map_stage_'+str(i_stage-1)+'.h5'
            filename_std_feat_out_of_prob = 'selected_std_feat_prob_stage_'+str(i_stage)+'.h5' 
            #change line down
            calc_selected_std_features(filename_prob, filename_std_feat_out_of_prob, n_folds, images, sigmasGaussian_prob, sigmasLoG_prob, sigmasGGM_prob, sigmasSTE_prob, sigmasHoGE_prob)
            #for data augmentation
            #calc_selected_std_features(filename_prob, filename_std_feat_out_of_prob, n_folds*N_ROT, images, sigmasGaussian_prob, sigmasLoG_prob, sigmasGGM_prob, sigmasSTE_prob, sigmasHoGE_prob)
            print('feature calc out of prob maps finished')
            feature_filenames = ['selected_std_feat_rawdata_stage_0.h5',filename_std_feat_out_of_prob]
            X_train, y_train= train_sets_stages_several_files(i_stage, N_STAGES, labels, folds, feature_filenames, testindex, rot=False, n_rot=N_ROT, x_center=X_CENTER, y_center=Y_CENTER)
            #feature_filenames = ['selected_std_feat_rawdata_rotated_stage_0.h5',filename_std_feat_out_of_prob]
            #X_train, y_train= train_sets_stages_several_files(i_stage, N_STAGES, labels, folds, feature_filenames, testindex, rot=True, n_rot=N_ROT, x_center=X_CENTER, y_center=Y_CENTER)
            print('trainingsdata sampling finished')
            #add RF object to list clf
            clf.append(RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1))
            #train random forest of this stage
            clf[i_stage].fit(X_train, y_train)
    

    n_train =  n_train_1_stage*N_STAGES
    print('train number', n_train)
    
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
    
    
    predfile = open(output_dir_pred + filename_txt, 'w')
    for i_img in np.arange(idx_start, idx_start+number_predicted, 1):
        y2D_pred_soft, y2D_pred_hard = predict_one_image_new_data(i_img, filename_rawdata ,clf, N_CLASSES, sigmasGaussian, sigmasLoG, sigmasGGM, sigmasSTE, sigmasHoGE, sigmasGaussian_prob, sigmasLoG_prob, sigmasGGM_prob, sigmasSTE_prob, sigmasHoGE_prob)
        #write prob map
        if n_rot==1:
            filename = filename_seg_soft+ '_' + str(i_img) + '_' + str(i_rot) + '.h5'
            predfile.write(filename + '\n')
            prob_file = h5py.File(output_dir_pred + filename, 'w' )
            prob_file.create_dataset('data', data=y2D_pred_soft[:,:,:]) #take only probmaps of foreground classes 
            prob_file.close() 
        else:
            x_dim = y2D_pred_soft.shape[0]
            y_dim = y2D_pred_soft.shape[1]
            angles = np.linspace(0., 360., num=n_rot, endpoint=False)
            for i_rot, angle in enumerate(angles):
                filename = filename_seg_soft+ '_' + str(i_img) + '_' + str(i_rot) + '.h5'
                predfile.write(filename + '\n')
                prob_file = h5py.File(output_dir_pred + filename, 'w' )
                if i_rot==0:
                    prob_file.create_dataset('data', data=y2D_pred_soft[:,:,:]) #take only probmaps of foreground classes 
                else: 
                    y2D_pred_soft_rot = np.zeros((x_dim, y_dim, y2D_pred_soft.shape[2]), dtype=np.float32)
                    for channel in np.arange(y2D_pred_soft_rot.shape[2]):
                        y2D_pred_soft_rot[:,:,channel] = rotate(y2D_pred_soft[:,:,channel], angle, resize=False, center=(y_center, x_center) )
                    prob_file.create_dataset('data', data=y2D_pred_soft_rot) #take only probmaps of foreground classes 
                prob_file.close()

    predfile.close()


    
