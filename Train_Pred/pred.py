"""
script for prediction of tracks
"""

import os
import sys
#sys.path.insert(0, os.path.abspath('..'))

from subprocess import check_call
import configargparse
import json
import h5py
import numpy as np

import sys
sys.path.append('.')

from grid_search import*

    

if __name__=='__main__':

    #read in json file generated with ImageJ plugin 
    filename_json = 'clusterGroundTruth.json'
    f = open(filename_json)
    #s = f.readlines()
    clusters = json.load(f)
    f.close()
    
    grid_search_helper = grid_search_helper()
    #number of grid points
    n_grid_points = grid_search_helper.n_idx


    #create hdf5 file for evaluation
    filename_h5 = 'clusterEval.h5'
    f_eval = h5py.File(filename_h5, 'w')
    f_eval.create_dataset('cluster2', (n_grid_points, 1000, 2, 3), dtype=int)
    f_eval.create_dataset('nCluster2', (1,), dtype=int)
    f_eval['nCluster2'][0] = 0
    f_eval.create_dataset('cluster3', (n_grid_points, 1000, 3, 3), dtype=int)
    f_eval.create_dataset('nCluster3', (1,), dtype=int)
    f_eval['nCluster3'][0] = 0    
    f_eval.create_dataset('cluster4', (n_grid_points, 1000, 4, 3), dtype=int)
    f_eval.create_dataset('nCluster4', (1,), dtype=int)
    f_eval['nCluster4'][0] = 0    
    f_eval.create_dataset('cluster5', (n_grid_points, 1000, 5, 3), dtype=int)
    f_eval.create_dataset('nCluster5', (1,), dtype=int)
    f_eval['nCluster5'][0] = 0    
    f_eval.create_dataset('cluster6', (n_grid_points, 1000, 6, 3), dtype=int)
    f_eval.create_dataset('nCluster6', (1,), dtype=int)
    f_eval['nCluster6'][0] = 0    
    f_eval.create_dataset('cluster7', (n_grid_points, 1000, 7, 3), dtype=int)
    f_eval.create_dataset('nCluster7', (1,), dtype=int)
    f_eval['nCluster7'][0] = 0    
    f_eval.create_dataset('cluster8', (n_grid_points, 1000, 8, 3), dtype=int)
    f_eval.create_dataset('nCluster8', (1,), dtype=int)
    f_eval['nCluster8'][0] = 0    
    f_eval.close()
        

    #loop over annotated clusters
    for cluster in clusters:
        idx_begin = cluster[0]
        idx_end = cluster[1]
        x_cen = cluster[2]
        y_cen = cluster[3]
        radius = cluster[4]

        parameter_preprocessing = []
        parameter_preprocessing.append('--preprocessing-idx_begin')
        parameter_preprocessing.append(str(idx_begin))
        parameter_preprocessing.append('--preprocessing-idx_end')
        parameter_preprocessing.append(str(idx_end))
        parameter_preprocessing.append('--preprocessing-x_start')
        parameter_preprocessing.append(str(x_cen - radius))
        parameter_preprocessing.append('--preprocessing-x_end')
        parameter_preprocessing.append(str(x_cen + radius))
        parameter_preprocessing.append('--preprocessing-y_start')
        parameter_preprocessing.append(str(y_cen - radius))
        parameter_preprocessing.append('--preprocessing-y_end')
        parameter_preprocessing.append(str(y_cen + radius))


        parameter_autocontext = []
        parameter_autocontext.append('--global-x_center')
        parameter_autocontext.append(str(radius))
        parameter_autocontext.append('--global-y_center')
        parameter_autocontext.append(str(radius))
        

        number_objects = cluster[5]
        parameter_mht = []
        parameter_mht = parameter_mht + parameter_autocontext
        
        parameter_mht.append('--preprocessing-idx_begin')
        parameter_mht.append(str(idx_begin))
        parameter_mht.append('--preprocessing-idx_end')
        parameter_mht.append(str(idx_end))
        parameter_mht.append('--preprocessing-x_start')
        parameter_mht.append(str(x_cen - radius))
        parameter_mht.append('--preprocessing-y_start')
        parameter_mht.append(str(y_cen - radius))


        parameter_mht.append('--mht-number_objects')
        parameter_mht.append(str(number_objects))
        if number_objects>=1:
            parameter_mht.append('--mht-centers_start_x1')
            parameter_mht.append(str(cluster[6]))
            parameter_mht.append('--mht-centers_start_y1')
            parameter_mht.append(str(cluster[7]))
            parameter_mht.append('--mht-centers_end_x1')
            parameter_mht.append(str(cluster[8]))
            parameter_mht.append('--mht-centers_end_y1')
            parameter_mht.append(str(cluster[9]))
        if number_objects>=2:
            parameter_mht.append('--mht-centers_start_x2')
            parameter_mht.append(str(cluster[10]))
            parameter_mht.append('--mht-centers_start_y2')
            parameter_mht.append(str(cluster[11]))
            parameter_mht.append('--mht-centers_end_x2')
            parameter_mht.append(str(cluster[12]))
            parameter_mht.append('--mht-centers_end_y2')
            parameter_mht.append(str(cluster[13]))
        if number_objects>=3:
            parameter_mht.append('--mht-centers_start_x3')
            parameter_mht.append(str(cluster[14]))
            parameter_mht.append('--mht-centers_start_y3')
            parameter_mht.append(str(cluster[15]))
            parameter_mht.append('--mht-centers_end_x3')
            parameter_mht.append(str(cluster[16]))
            parameter_mht.append('--mht-centers_end_y3')
            parameter_mht.append(str(cluster[17]))
        if number_objects>=4:
            parameter_mht.append('--mht-centers_start_x4')
            parameter_mht.append(str(cluster[18]))
            parameter_mht.append('--mht-centers_start_y4')
            parameter_mht.append(str(cluster[19]))
            parameter_mht.append('--mht-centers_end_x4')
            parameter_mht.append(str(cluster[20]))
            parameter_mht.append('--mht-centers_end_y4')
            parameter_mht.append(str(cluster[21]))
        if number_objects>=5:
            parameter_mht.append('--mht-centers_start_x5')
            parameter_mht.append(str(cluster[22]))
            parameter_mht.append('--mht-centers_start_y5')
            parameter_mht.append(str(cluster[23]))
            parameter_mht.append('--mht-centers_end_x5')
            parameter_mht.append(str(cluster[24]))
            parameter_mht.append('--mht-centers_end_y5')
            parameter_mht.append(str(cluster[25]))
        if number_objects>=6:
            parameter_mht.append('--mht-centers_start_x6')
            parameter_mht.append(str(cluster[26]))
            parameter_mht.append('--mht-centers_start_y6')
            parameter_mht.append(str(cluster[27]))
            parameter_mht.append('--mht-centers_end_x6')
            parameter_mht.append(str(cluster[28]))
            parameter_mht.append('--mht-centers_end_y6')
            parameter_mht.append(str(cluster[29]))
        if number_objects>=7:
            parameter_mht.append('--mht-centers_start_x7')
            parameter_mht.append(str(cluster[30]))
            parameter_mht.append('--mht-centers_start_y7')
            parameter_mht.append(str(cluster[31]))
            parameter_mht.append('--mht-centers_end_x7')
            parameter_mht.append(str(cluster[32]))
            parameter_mht.append('--mht-centers_end_y7')
            parameter_mht.append(str(cluster[33]))
        if number_objects>=8:
            parameter_mht.append('--mht-centers_start_x8')
            parameter_mht.append(str(cluster[34]))
            parameter_mht.append('--mht-centers_start_y8')
            parameter_mht.append(str(cluster[35]))
            parameter_mht.append('--mht-centers_end_x8')
            parameter_mht.append(str(cluster[36]))
            parameter_mht.append('--mht-centers_end_y8')
            parameter_mht.append(str(cluster[37]))


        """
        Preprocessing
        """
        #print options.config_file
        #cropp rawdata
        check_call(["python", os.path.abspath("../Preprocessing/cropp_rawdata.py"), "--config", 'config_pred.ini'] + parameter_preprocessing)

        """
        Autocontext
        """
        #predict probmaps of cropped rawdata
        check_call(["python", os.path.abspath("../Autocontext/predict_for_dpm.py"), "--config", 'config_pred.ini'] + parameter_autocontext)

        """
        DPM
        """
        #calc scoremaps out of probmaps
        check_call([os.path.abspath("../DPM/bin/dpmdetect"), os.path.abspath("../DPM/samples/pred/predfile.txt"), os.path.abspath("../DPM/samples/model/model"), os.path.abspath("../DPM/samples/pred/probmap")])

        """
        MultiHypoTracking
        """
        
        #perform tracking
        check_call(["python", os.path.abspath("../MultiHypoTracking/tracker.py"), "--config", 'config_pred.ini'] + parameter_mht)
