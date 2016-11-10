import numpy as np
import h5py
import configargparse


#count labels, deduce maximal number of labels
#need config_train.ini as config file


if __name__ == '__main__':
    #fixed parameters
    N_IMG = 5
    relative_path_labelmaps = 'Autocontext/' #relative path from labelCounting.py to Autocontext working dir
    
    #default parameters when no command line parameter is used
    N_CLASS = 4                             #number of classes
    filename_labelmap_1 = 'Labels_0.h5'
    filename_labelmap_2 = 'Labels_33.h5'
    filename_labelmap_3 = 'Labels_66.h5'
    filename_labelmap_4 = 'Labels_97.h5'
    labelmap_dataset = 'data'

    #config arg parse
    p = configargparse.ArgParser()#(default_config_files=['config.ini'])
    p.add('-c', '--config', is_config_file=True, help='config file path')
    p.add('--autocontext-n_classes', default=N_CLASS, type=int, help='number of classes to train the autocontext')
    p.add('--autocontext-labelmap_1', default=filename_labelmap_1, type=str, help='filename of partially labeled labelmaps for training autocontext as hdf5 files in format [x, y, ch=1]')
    p.add('--autocontext-labelmap_2', default=filename_labelmap_2, type=str, help='filename of partially labeled labelmaps for training autocontext as hdf5 files in format [x, y, ch=1]')
    p.add('--autocontext-labelmap_3', default=filename_labelmap_3, type=str, help='filename of partially labeled labelmaps for training autocontext as hdf5 files in format [x, y, ch=1]')
    p.add('--autocontext-labelmap_4', default=filename_labelmap_4, type=str, help='filename of partially labeled labelmaps for training autocontext as hdf5 files in format [x, y, ch=1]')
    p.add('--autocontext-labelmap_dataset', default=labelmap_dataset, type=str, help='name of dataset in labelmaps')    
    options, unknown = p.parse_known_args()
    
    #parse parameter
    N_CLASS = options.autocontext_n_classes
    filename_labelmap_1 = options.autocontext_labelmap_1
    filename_labelmap_2 = options.autocontext_labelmap_2
    filename_labelmap_3 = options.autocontext_labelmap_3
    filename_labelmap_4 = options.autocontext_labelmap_4
    labelmap_dataset = options.autocontext_labelmap_dataset

    #deduced paramters
    filenames_labels = [filename_labelmap_1, filename_labelmap_2, filename_labelmap_3, filename_labelmap_4]
    
    print 'start label counting'
    class_count = np.zeros((N_IMG-1, N_CLASS + 1,))
    for i_img, filename_label in enumerate(filenames_labels):
        f = h5py.File(relative_path_labelmaps + filename_label, 'r')
        img = f[labelmap_dataset][0,:,:,0]
        f.close()
        x_dim = img.shape[0]
        y_dim = img.shape[1]
        img = img.reshape((x_dim*y_dim))
        for cl in np.arange(N_CLASS + 1):
            length = len(filter(lambda x: x==cl, img))
            class_count[i_img, cl] = length
    for cl in np.arange(N_CLASS + 1):
        print 'class: ', cl, 'min length: ', class_count[:,cl].min(), 'in labelmap: ', np.argmin(class_count[:,cl])+1 
    print 'maximal possible number of trainingspixels: ', class_count.min()*N_CLASS*(N_IMG-1)

