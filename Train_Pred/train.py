"""
Script for executing training the entire pipeline
"""

import os
import sys
import configargparse
from subprocess import check_call



if __name__=='__main__':

    """
    parse config file
    """

    #config arg parse
    p = configargparse.ArgParser()
    p.add('-c', '--config', is_config_file=True, help='config file path')
    p.add('--dpm-n_parts', default=3, type=int, help='number of model parts')
    p.add('--dpm-negative_count', default=500, type=int, help='number of negative examples for initializing filters with linear SVM')
    
    options, unknown = p.parse_known_args()
    #parse parameters
    n_parts = options.dpm_n_parts
    negative_count = options.dpm_negative_count    

    
    """
    Autocontext
    """
    """
    #preprocessing step: put labelmaps together, put trainimgs together, sample trainpxl 
    check_call(["python", os.path.abspath("../Autocontext/crossValSets.py"), "--config", 'config_train.ini'])
    #calc features on rawdata
    check_call(["python", os.path.abspath("../Autocontext/features.py")])
    #train autocontext
    check_call(["python", os.path.abspath("../Autocontext/train_for_dpm.py"), "--config", 'config_train.ini'])
    
    """
    """
    DPM
    """
    
    #create files (indices_pos.h5, indices_neg.h5) with indices within rawdata of positive and negative examples and filenames of probability maps
    check_call(["python", os.path.abspath("../DPM/create_trainingsdata.py"), "--config", 'config_train.ini'])
    #predict prob maps of trainimg for DPM
    check_call(["python", os.path.abspath("../Autocontext/predict_trainimg_for_dpm.py"), "--config", 'config_train.ini'])
    
    #train DPM model
    check_call([os.path.abspath("../DPM/bin/dpmcreate"), "--positive-list", os.path.abspath("../DPM/samples/train/posfile.txt"), "--background-list", os.path.abspath("../DPM/samples/train/negfile.txt"),  "--negative-count", str(negative_count), "--model-component", "1", "--model-part", str(n_parts), "--working-dir", os.path.abspath("../DPM/samples/model"), "--base-dir", os.path.abspath("../DPM/samples/train/pos")])
    
