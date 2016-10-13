"""
Script for executing training the entire pipeline
"""

import os
import sys

from subprocess import check_call



if __name__=='__main__':
    """
    parse config file
    """
    """
    p = configargparse.ArgParser(default_config_files=['config_train.ini'])
    p.add('-c', '--config', dest='config_file', is_config_file=True, help='config file path')
    options = p.parse_args()
    """
    
    """
    Autocontext
    """
    #preprocessing step: put labelmaps together, put trainimgs together, sample trainpxl 
    check_call(["python", os.path.abspath("../Autocontext/crossValSets.py"), "--config", 'config_train.ini'])
    
    """
    DPM
    """
    
