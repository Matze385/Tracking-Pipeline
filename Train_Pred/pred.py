"""
script for prediction of tracks
"""

import os
import sys
#sys.path.insert(0, os.path.abspath('..'))

from subprocess import check_call
import configargparse


if __name__=='__main__':

    """
    Preprocessing
    """
    #print options.config_file
    #cropp rawdata
    check_call(["python", os.path.abspath("../Preprocessing/cropp_rawdata.py"), "--config", 'config_pred.ini'])
    
    """
    Autocontext
    """
    #predict probmaps of cropped rawdata
    check_call(["python", os.path.abspath("../Autocontext/predict_for_dpm.py"), "--config", 'config_pred.ini'])

    """
    DPM
    """
    #calc scoremaps out of probmaps
    check_call([os.path.abspath("../DPM/bin/dpmdetect"), os.path.abspath("../DPM/samples/pred/predfile.txt"), os.path.abspath("../DPM/samples/model/model"), os.path.abspath("../DPM/samples/pred/probmap")])

    """
    MultiHypoTracking
    """
    #perform tracking
    check_call(["python", os.path.abspath("../MultiHypoTracking/tracker.py"), "--config", 'config_pred.ini'])
    
