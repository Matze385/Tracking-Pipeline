import h5py
import numpy as np
import matplotlib.pyplot as plt
from grid_search import*
from sklearn.model_selection import KFold

"""
structure of hdf5 file with cluster ground truth and evaluation
-different datasets for clusters with different number of objects (maximal number of objects=8)
    -if number of objects in cluster is i the following datasets exist
        -nClusteri of shape (1,)
            contains number of clusters with i objects
        -clusteri of shape (n_of_grid_points, 1000, i, 3)
            zeros dimension (n_of_grid_points): index of grid point used for solutions
            first dimension (1000): ith cluster
            second dimension (i): rows for each object
            third dimension (3): columns for start_id (0), end_id_gt (1), end_id_pred (2)
"""

#generates list with one element for clusters with different number of objects. One element contains trainings and testdata splits for clusters with fixed number of elements, format [clusters_with_2_objects, clusters_with_3_objects, ... , clusters_with_upto_max_n_objects] cluster_with_i_objects: [(trainingset1, testset1), ... , (trainingset_n_folds, testset_n_folds)]
def generate_cross_val_idx(n_folds, upto_max_n_objects=3):
    assert(upto_max_n_objects<=8)
    assert(upto_max_n_objects>=2)
    result = []
    kf = KFold(n_splits=n_folds)
    for n_objects in np.arange(2, upto_max_n_objects + 1):
        filename_h5 = 'clusterEval.h5'
        dataset_name_n = 'nCluster' + str(n_objects)
        f_eval = h5py.File(filename_h5, 'r')
        n_cluster = f_eval[dataset_name_n][0]
        f_eval.close()
        X = np.arange(n_cluster)
        cluster_with_i_objects = []
        for train_set, test_set in kf.split(X):
            cluster_with_i_objects.append((train_set, test_set))
        result.append(cluster_with_i_objects)
        #result.append(kf.split(X))
    return result

#extract the train sets of one fold for clusters with different number of objects
#result: output of generate_cross_val_idx
#i_fold: 
def extract_train_sets_of_one_fold(result, i_fold):
    train_sets = []
    n_folds = 0
    for train_set, test_set in result[0]:
        n_folds += 1
    assert(i_fold>=0)
    assert(i_fold<n_folds)
    for i in np.arange(len(result)):
        train_set = result[i][i_fold][0]
        train_sets.append(train_set)
    return train_sets

#extract the test sets of one fold for clusters with different number of objects
def extract_test_sets_of_one_fold(result, i_fold):       
    test_sets = []
    n_folds = 0
    for train_set, test_set in result[0]:
        n_folds += 1
    assert(i_fold>=0)
    assert(i_fold<n_folds)
    for i in np.arange(len(result)):
        test_sets.append(result[i][i_fold][1])
    return test_sets


#id_matrix: id matrix for one cluster: (n_objects, 3), f[clusteri][j, :,:]
#first dimension rows for each object
#second dimension columns for start_id (0), end_id_gt (1), end_id_pred (2)

def accuracy_one_cluster(id_matrix):
    return np.mean(id_matrix[:,1] == id_matrix[:,2])
    
#n_objects: clusters with n_objects are evaluated
#eval_metric_single_cluster: function with one argument of type id_matrix (see accuracy_one_cluster)
#grid_idx: index of grid point that should be evaluated
#selected_cluster_idx: selected idx for evaluation, used for cross validation
#return: accuracy measure, number of clusters used for returned measure
def eval_clusters_i(n_objects, grid_idx, selected_cluster_idx, eval_metric_single_cluster):
    filename_h5 = 'clusterEval.h5'
    #dataset_name_n = 'nCluster' + str(n_objects)
    dataset_name = 'cluster' + str(n_objects)
    f_eval = h5py.File(filename_h5, 'r')
    #n_cluster = f_eval[dataset_name_n][0]
    n_cluster = len(selected_cluster_idx)
    accuracies = np.zeros((n_cluster,), dtype=float)
    all_cluster = f_eval[dataset_name][grid_idx,:,:,:]
    f_eval.close()
    #if n_cluster==0:
       # return 0., 0.
    for i, i_cluster in enumerate(selected_cluster_idx):
        accuracies[i] = eval_metric_single_cluster(all_cluster[i_cluster,:,:])
    return np.mean(accuracies), n_cluster

#n_objects_list: evaluates clusters with object numbers given in n_objects_list 
#grid_idx: tracking solution with weights of grid point with grid_idx is evaluated
#selected_cluster_idx: list with elements corresponding to clusters with fixed number of objects, specifies which clusters with different number of objects are used, length must correspond with length of n_objects_list
def eval_all_clusters(n_objects_list, grid_idx, selected_cluster_idx_list, eval_metric_single_cluster):
    assert(len(n_objects_list)==len(selected_cluster_idx_list))
    accum_accuracies = 0.
    n_tracked_objects = 0
    n_cluster = 0
    for i_objects, selected_cluster_idx in zip(n_objects_list, selected_cluster_idx_list):
        accuracy, i_cluster  = eval_clusters_i(i_objects, grid_idx, selected_cluster_idx, eval_metric_single_cluster)
        n_cluster += i_cluster
        i_tracked_objects = i_cluster*i_objects
        n_tracked_objects += i_tracked_objects
        accum_accuracies += i_tracked_objects*accuracy
    if n_cluster == 0:
        print 'n_cluster=0'
        return 0., 0
    total_accuracy = accum_accuracies/n_tracked_objects
    return total_accuracy, n_cluster


#train_sets_i_fold: list of train_set in i_fold for different number of objects, result of extract_train_sets_of_one_fold
#return grid_idx with maximal score and corresponding score, 
#average_grid_idx: when there are several subsequent grid indices with same maximal score take grid_idx in the middle when average_grid_idx=True and take first grid_idx when average_grid_idx=False
def grid_search(train_sets_i_fold, grid_idx_range, eval_metric_single_cluster, average_grid_idx=True):
    n_objects_list = []
    for n_objects in np.arange(2, 2+len(train_sets_i_fold)):
        n_objects_list.append(n_objects)
    max_score = 0.
    max_grid_idx_start = 0
    max_grid_idx_last = -1
    max_grid_idx = 0
    n_cluster = 0
    n_same_score = 0
    for grid_idx in grid_idx_range:
        total_accuracy, i_cluster = eval_all_clusters(n_objects_list, grid_idx, train_sets_i_fold, eval_metric_single_cluster )
        if total_accuracy >= max_score:
            n_same_score += 1
            if max_grid_idx_last  + 1 == grid_idx:
                max_grid_idx_last = grid_idx 
            if total_accuracy > max_score:
                max_score = total_accuracy
                max_grid_idx_start = grid_idx
                max_grid_idx_last = grid_idx
                max_grid_idx = grid_idx
                n_cluster = i_cluster
                n_same_score = 0
    print 'number of grid_idx with maximal score: {0}'.format(n_same_score)
    if average_grid_idx:
        max_grid_idx = int((max_grid_idx_start + max_grid_idx_last)/2)
        print 'max_grid_idx_start: {0}'.format(max_grid_idx_start)
        print 'max_grid_idx_last: {0}'.format(max_grid_idx_last)
    return max_grid_idx, max_score, n_cluster

#perform crossvalidation with n_folds
def perform_cross_val(n_folds, upto_max_n_objects, grid_idx_range, eval_metric_single_cluster, average_grid_idx=True):
    result = generate_cross_val_idx(n_folds, upto_max_n_objects)

    #data that should be collected    
    test_accuracies = []
    n_cluster_test_average = 0.

    test_accuracies_separate_cluster_size = np.zeros((upto_max_n_objects-2+1, n_folds), dtype=float)
    n_cluster_test_sepa_average = np.zeros((upto_max_n_objects-2+1), dtype=float)

    train_accuracies = []
    list_max_grid_idx = []
    n_cluster_train_average = 0.

    train_accuracies_separate_cluster_size = np.zeros((upto_max_n_objects-2+1, n_folds), dtype=float)
    n_cluster_train_sepa_average = np.zeros((upto_max_n_objects-2+1), dtype=float)
    

    #collect data
    for i_fold in np.arange(n_folds):
        train_sets_i_fold = extract_train_sets_of_one_fold(result, i_fold)
        max_grid_idx, max_score, n_cluster_train = grid_search(train_sets_i_fold, grid_idx_range, eval_metric_single_cluster, average_grid_idx)
        n_cluster_train_average += float(n_cluster_train)
        list_max_grid_idx.append(max_grid_idx)
        train_accuracies.append(max_score)
        test_sets_i_fold = extract_test_sets_of_one_fold(result, i_fold)
        n_objects_list = []
        for i, n_objects in enumerate(np.arange(2, 2+len(test_sets_i_fold))):
            n_objects_list.append(n_objects)
            n_objects_list_sepa = [n_objects,]
            train_sets_i_fold_n_objects = [train_sets_i_fold[i],]
            total_accuracy_separate, n_cluster_sepa = eval_all_clusters(n_objects_list_sepa, max_grid_idx, train_sets_i_fold_n_objects, eval_metric_single_cluster)
            train_accuracies_separate_cluster_size[i, i_fold] = total_accuracy_separate
            n_cluster_train_sepa_average[i] += float(n_cluster_sepa)
            
        total_accuracy, n_cluster_test = eval_all_clusters(n_objects_list, max_grid_idx, test_sets_i_fold, eval_metric_single_cluster)
        n_cluster_test_average += float(n_cluster_test)
        test_accuracies.append(total_accuracy)
        for i, n_objects in enumerate(np.arange(2, 2+len(test_sets_i_fold))):
            n_objects_list_sepa = [n_objects,]
            test_sets_i_fold_n_objects = [test_sets_i_fold[i],]
            total_accuracy_separate, n_cluster_sepa = eval_all_clusters(n_objects_list_sepa, max_grid_idx, test_sets_i_fold_n_objects, eval_metric_single_cluster)
            n_cluster_test_sepa_average[i] += float(n_cluster_sepa)
            test_accuracies_separate_cluster_size[i, i_fold] = total_accuracy_separate

    #extract information
    test_accuracy_mean = np.mean(test_accuracies)
    test_error_on_mean = np.std(test_accuracies)/np.sqrt(n_folds)
    n_cluster_test_average /= float(n_folds)

    test_accuracy_mean_sepa = np.mean(test_accuracies_separate_cluster_size, axis=1)
    test_accuracy_sepa_error_on_mean = np.std(test_accuracies_separate_cluster_size, axis=1)/np.sqrt(n_folds)
    n_cluster_test_sepa_average /= float(n_folds)

    train_accuracy_mean = np.mean(train_accuracies)
    train_error_on_mean = np.std(train_accuracies)/np.sqrt(n_folds)
    n_cluster_train_average /= float(n_folds)

    train_accuracy_mean_sepa = np.mean(train_accuracies_separate_cluster_size, axis=1)
    train_accuracy_sepa_error_on_mean = np.std(train_accuracies_separate_cluster_size, axis=1)/np.sqrt(n_folds)
    n_cluster_train_sepa_average /= float(n_folds)

    return (test_accuracy_mean,         #0
    test_error_on_mean,                 #1
    train_accuracy_mean,                #2
    train_error_on_mean,                #3
    test_accuracy_mean_sepa,            #4
    test_accuracy_sepa_error_on_mean,   #5
    n_cluster_train_average,            #6
    n_cluster_test_average,             #7
    n_cluster_test_sepa_average,        #8
    train_accuracy_mean_sepa,           #9
    train_accuracy_sepa_error_on_mean,  #10
    n_cluster_train_sepa_average,       #11
    list_max_grid_idx)                  #12
        

#show evaluation metric as function of grid point index, 
#score_list: list of scores
def plot_grid_search(score_list):
    fig, ax = plt.subplots()
    ax.plot(score_list)
    plt.show()

#prints weights of grid point with index grid_idx, needs grid_search_helper as input
def print_weights(grid_idx, grid_search_helper, score_list):
    print 'weights at grid_idx:{0}, (weight_trans_move, weight_trans_angle, weight_det): {1},{2},{3}, score: {4}  '.format(grid_idx, grid_search_helper.get_weight_trans_move(grid_idx), grid_search_helper.get_weight_trans_angle(grid_idx), grid_search_helper.get_weight_det(grid_idx), score_list[grid_idx])

#def find_idx_with_maximal_score(score_list):
       
def print_cross_val_result(result_list, grid_search_helper):
    upto_max_n_objects = result_list[4].shape[0] + 1

    #test results
    print 'test_accuracy: {0:.2f} +/- {1:.2f}, n_cluster_average: {2:.1f}'.format(result_list[0], result_list[1], result_list[7])
    for n_objects in np.arange(2, upto_max_n_objects +1):
        print 'test_accuracy cluster with {2} objects: {0:.2f} +/- {1:.2f}, n_cluster_average {3:.1f} '.format(result_list[4][n_objects-2], result_list[5][n_objects-2], n_objects, result_list[8][n_objects-2])
    
    #train results
    print 'train_accuracy: {0:.2f} +/- {1:.2f}, n_cluster_average: {2:.1f}'.format(result_list[2], result_list[3], result_list[6])
    for n_objects in np.arange(2, upto_max_n_objects +1):
        print 'train_accuracy cluster with {2} objects: {0:.2f} +/- {1:.2f}, n_cluster_average: {3:.1f}'.format(result_list[9][n_objects-2], result_list[10][n_objects-2], n_objects, result_list[11][n_objects-2])
    
    for i_fold, grid_idx in enumerate(result_list[12]):
        print 'optimal weights in fold {0}: (weight_trans_move, weight_trans_angle, weight_det): {1},{2},{3}'.format(i_fold+1, grid_search_helper.get_weight_trans_move(grid_idx), grid_search_helper.get_weight_trans_angle(grid_idx), grid_search_helper.get_weight_det(grid_idx))

if __name__ == '__main__':
    n_folds = 5
    upto_max_n_objects = 4

    grid_search_helper = grid_search_helper()
    grid_idx_range = grid_search_helper.get_idx_range()
    result_cross_val = perform_cross_val(n_folds, upto_max_n_objects, grid_idx_range, accuracy_one_cluster, average_grid_idx=True)
    print_cross_val_result(result_cross_val, grid_search_helper)
    
