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
