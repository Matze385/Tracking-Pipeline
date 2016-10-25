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
    #for training
    relative_path_autocontext = '../Autocontext/'
    filename_labels = 'labels.h5'
    labels_dataset = 'data'
    filename_clf = 'clf'
    filename_extension_clf = '.pkl'
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
    #random forest parameters
    use_config = True                     #must be False if executed by hand
    N_CLASSES = 7                          #number of different labels, here edge, background, boundary, head, upper body, lower body, wings
    N_TREES = 600
    N_STAGES = 2

    #config arg parse
    p = configargparse.ArgParser()#(default_config_files=['config.ini'])
    p.add('-c', '--config', is_config_file=True, help='config file path')
    p.add('--autocontext-n_classes', default=N_CLASSES, type=int, help='number of classes in autocontext')    
    p.add('--autocontext-n_stages', default=N_STAGES, type=int, help='number of stages in autocontext')    
    p.add('--autocontext-n_trees', default=N_TREES, type=int, help='number of trees in Random Forest of autocontext')    
    
    options, unknown = p.parse_known_args()
    
    #parse parameter
    if use_config==True:
        N_CLASSES = options.autocontext_n_classes
        N_STAGES = options.autocontext_n_stages                         
        N_TREES =  options.autocontext_n_trees                         

    #derived parameters
    #number of features
    n_feat_prob = len(sigmasGaussian_prob)+len(sigmasLoG_prob)+len(sigmasGGM_prob)+2*len(sigmasSTE_prob)+2*len(sigmasHoGE_prob)
    n_feat_raw = len(sigmasGaussian)+len(sigmasLoG)+len(sigmasGGM)+2*len(sigmasSTE)+2*len(sigmasHoGE)
    #dataaugmentation, rotational settings
    X_CENTER = 510		                #center coordinates for rotation
    Y_CENTER = 514
    RADIUS = 500
    N_ROT = 16                              #number of rotations per image, determines rotational angle
    #testparameters
    #auxiliary classes that should be merged by testing
    merged_classes = np.array([4,5,6])              #classes background(4) background (2) boundary(5) bright edge (6) are merged to background for testing
    merged_classes_2 = np.array([1,2])              #classes tail
    #image used for qualitative analysis (idx 4 corresponds to image 461)
    idx_test_img = 4 
    xDim = 0				
    yDim = 0
    #number of used features
    n_features = 0
    #number of folds
    n_folds = 0


    """
    read in data
    """

    labelsF = h5py.File(relative_path_autocontext + 'CrossValFolds.h5','r')
    #folds: [img, x/y, selected_pixels] coordinates of labels
    folds = vg.readHDF5(labelsF,'folds')
    #labels: labels [img, selected_pixels] c: class labels c=0 no label
    labels = vg.readHDF5(labelsF,'labels')
    labelsF.close()

    #labels_all: [img, x, y, 0] needed for test sets with pixels out of complete interesting region
    labelsAllF = h5py.File(relative_path_autocontext + filename_labels,'r')
    labels_all = labelsAllF[labels_dataset][:,:,:,0]
    labelsAllF.close()

    n_folds  = labels.shape[0]
    #n_folds = 5
    #print ('n_folds:', n_folds)

    #n_features = features.shape[3]
    xDim = labels_all.shape[1]
    yDim = labels_all.shape[2]


    #feature selection on unrotated data
    #dont forget to integrate following comments into code when using subset of std feat on rawdata
    select_std_features(0,'rawdata',relative_path_autocontext, relative_path_autocontext + 'features.h5', sigmasGaussian, sigmasLoG, sigmasGGM, sigmasSTE, sigmasHoGE) #features [x,y,img,f]
    #select_std_features(0,'rawdata_rotated', 'featuresRot.h5', sigmasGaussian, sigmasLoG, sigmasGGM, sigmasSTE, sigmasHoGE)

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
                print '{0} Stage: select features/labels for X_train, y_train'.format(i_stage+1)
                X_train, y_train= train_sets_stages_several_files(0, N_STAGES, labels, folds,[relative_path_autocontext + 'selected_std_feat_rawdata_stage_0.h5'], testindex, rot=False, n_rot=N_ROT, x_center=X_CENTER, y_center=Y_CENTER)
                #X_train, y_train= train_sets_stages(0, N_STAGES, labels, folds,'selected_std_feat_rawdata_stage_0.h5', testindex, rot=False, n_rot=N_ROT, x_center=X_CENTER, y_center=Y_CENTER)
                n_train_1_stage += len(y_train)
                #add RF object to list clf
                clf.append(RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1))
                print '{0} Stage: train RF'.format(i_stage+1)
                #train random forest of this stage
                clf[i_stage].fit(X_train, y_train)
            else:
                print '{0} Stage: create prob maps with RF of stage before'.format(i_stage+1)
                #indices of images prob maps are computed on
                images = img_split(i_stage, N_STAGES, testindex, n_folds, labels, rot=False, n_rot=N_ROT)
                #images = img_split(i_stage-1, n_stages, testindex, n_folds, labels, rot=True, n_rot=N_ROT)
                #print images
                #create prob map of RF of stage before, saved in 'probability_map_stage_'+str(i_stage).h5 in dataset 'data' array shape [img,x,y,c]
                create_probability_maps(i_stage-1, clf[i_stage-1], n_folds, images, N_CLASSES, relative_path_autocontext)
                #create_probability_maps(i_stage-1, clf[i_stage-1], n_folds*N_ROT, images, N_CLASSES, rot=True)
                #delete features out of prob maps of stage before and prob_map of two stages before if i_stage>=2 
                if i_stage>1:
                    filename_std_feat_out_of_prob_before = relative_path_autocontext + 'selected_std_feat_prob_stage_'+str(i_stage-1)+'.h5' 
                    f = h5py.File(filename_std_feat_out_of_prob_before,'w')
                    f.create_dataset('data',(1,))
                    f.close()
                    filename_prob_before = relative_path_autocontext + 'probability_map_stage_'+str(i_stage-2)+'.h5'
                    f = h5py.File(filename_prob_before,'w')
                    f.create_dataset('data',(1,))
                    f.close()
                #calculation of features out of prob maps of stage before, saved in 'selected_std_feat_prob_stage_'+str(i_stage)+'.h5'
                filename_prob = relative_path_autocontext + 'probability_map_stage_'+str(i_stage-1)+'.h5'
                filename_std_feat_out_of_prob = relative_path_autocontext + 'selected_std_feat_prob_stage_'+str(i_stage)+'.h5' 
                #change line down
                print '{0} Stage: calc features on prob maps of RF of stage before'.format(i_stage+1)
                calc_selected_std_features(filename_prob, filename_std_feat_out_of_prob, n_folds, images, sigmasGaussian_prob, sigmasLoG_prob, sigmasGGM_prob, sigmasSTE_prob, sigmasHoGE_prob)
                #for data augmentation
                #calc_selected_std_features(filename_prob, filename_std_feat_out_of_prob, n_folds*N_ROT, images, sigmasGaussian_prob, sigmasLoG_prob, sigmasGGM_prob, sigmasSTE_prob, sigmasHoGE_prob)
                print '{0} Stage: select features/labels for X_train, y_train'.format(i_stage+1)
                feature_filenames = [relative_path_autocontext + 'selected_std_feat_rawdata_stage_0.h5', filename_std_feat_out_of_prob]
                X_train, y_train= train_sets_stages_several_files(i_stage, N_STAGES, labels, folds, feature_filenames, testindex, rot=False, n_rot=N_ROT, x_center=X_CENTER, y_center=Y_CENTER)
                #feature_filenames = ['selected_std_feat_rawdata_rotated_stage_0.h5',filename_std_feat_out_of_prob]
                #X_train, y_train= train_sets_stages_several_files(i_stage, N_STAGES, labels, folds, feature_filenames, testindex, rot=True, n_rot=N_ROT, x_center=X_CENTER, y_center=Y_CENTER)
                #add RF object to list clf
                clf.append(RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1))
                #train random forest of this stage
                print '{0} Stage: train RF'.format(i_stage+1)
                clf[i_stage].fit(X_train, y_train)
        
        
        n_train =  n_train_1_stage*N_STAGES
        #print 'train number', n_train
        print 'Training of Autocontext finished'
        print 'save Random Forest classifier'
        for i,i_clf in enumerate(clf):
            joblib.dump(i_clf, relative_path_autocontext + filename_clf + '_' + str(i) + filename_extension_clf)        
        
        
