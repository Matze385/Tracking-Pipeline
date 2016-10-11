import numpy as np
#from skimage.transform import rotate
import h5py 
import configargparse

"""
take every i th image of rawdata 
"""
if __name__  == '__main__':
    #default parameters used without config file
    filename_raw ='RawData.h5'
    dataset_raw = 'data'
    time_interval=False #if time cropping is True take only images with idx between i_img_begin and i_img_end 
    idx_begin = 200
    idx_end = 400
    every_i_img = 4 #must be >=4
    filename_new = 'rawdata.h5'
    dataset_new = 'data'
    spatial_cropping=False #if spatial_cropping=True cropp certain area out specified below [x_start,x_end), [y_start, y_end)
    x_start = 0 
    x_end = 10 
    y_start = 0
    y_end = 10
    
    
    #config arg parse
    p = configargparse.ArgParser()#(default_config_files=['config.ini'])
    p.add('-c', '--my-config', is_config_file=True, help='config file path')
    p.add('--preprocessing-filename_rawdata', default=filename_raw, type=str, help='filename of rawdata for preprocessing')
    p.add('--preprocessing-datasetname_rawdata', default=dataset_raw, type=str, help='filename of dataset of rawdata for preprocessing')
    p.add('--preprocessing-time_interval', default=time_interval, action='store_true', help='if true only img with idx between idx_begin and idx_end are chosen')
    p.add('--preprocessing-idx_begin', default=idx_begin, type=int, help='start idx for cropping in time')
    p.add('--preprocessing-idx_end', default=idx_end, type=int, help='end idx for cropping in time')
    p.add('--preprocessing-every_i_img_rawdata', default=every_i_img, type=int, help='selecting every i th image for tracking (preprocessing)')
    p.add('--preprocessing-filename_new_rawdata', default=filename_new, type=str, help='filename of rawdata after preprocessing')
    p.add('--preprocessing-datasetname_new_rawdata', default=dataset_new, type=str, help='filename of dataset for rawdata after preprocessing')
    p.add('--preprocessing-spatial_cropping', default=spatial_cropping, action='store_true', help='if true spatial cropping according to [x_start,x_end) [y_start,y_end)')
    p.add('--preprocessing-x_start', default=x_start, type=int, help='limits of interval for spatial cropping [x_start, x_end)')
    p.add('--preprocessing-x_end', default=x_end, type=int, help='limits of interval for spatial cropping [x_start, x_end)')
    p.add('--preprocessing-y_start', default=y_start, type=int, help='limits of interval for spatial cropping [y_start, y_end)')
    p.add('--preprocessing-y_end', default=y_end, type=int, help='limits of interval for spatial cropping [y_start, y_end)')
    
    options = p.parse_args()
 
    #parse parameters 
    filename_raw = options.preprocessing_filename_rawdata
    dataset_raw = options.preprocessing_datasetname_rawdata
    time_interval = options.preprocessing_time_interval
    idx_begin = options.preprocessing_idx_begin
    idx_end = options.preprocessing_idx_end
    every_i_img = options.preprocessing_every_i_img_rawdata
    spatial_cropping = options.preprocessing_spatial_cropping
    x_start = options.preprocessing_x_start
    x_end = options.preprocessing_x_end
    y_start = options.preprocessing_y_start
    y_end = options.preprocessing_y_end
    filename_new = options.preprocessing_filename_new_rawdata
    dataset_new = options.preprocessing_datasetname_new_rawdata

    #main program
    rawdata_f = h5py.File(filename_raw,'r')
    shape = rawdata_f[dataset_raw].shape #[x]
    n_img = shape[0]
    x_dim = shape[1]
    y_dim = shape[2]
    new_n_img = 0
    new_x_dim = x_dim
    new_y_dim = y_dim
    #selection for time idx
    selected_img = np.arange(1)
    if time_interval==True:
        assert(idx_end>idx_begin),'idx_end must bigger than idx_end'
        selected_img = np.arange(idx_begin, idx_end, every_i_img)
        new_n_img = len(selected_img)
    else:
        selected_img = np.arange(0, n_img, every_i_img)
        new_n_img = len(selected_img)
    #selection of x, y interval, and setting of new_x_dim, new_y_dim
    if spatial_cropping==True:
        assert(x_start<x_end),'x_start must be smaller than x_end'
        assert(x_start>=0),'x_start must be greater/equal than 0'
        assert(x_end<=x_dim),'x_end must be smaller/equal than x_dim of rawdata'
        assert(y_start<y_end),'y_start must be smaller than y_end'
        assert(y_start>=0),'y_start must be greater/equal than 0'
        assert(y_end<=y_dim),'y_end must be smaller/equal than y_dim of rawdata'
        new_x_dim = x_end-x_start
        new_y_dim = y_end-y_start
    else:
        x_start = 0
        x_end = x_dim
        y_start = 0
        y_end = y_dim
        
        
    newdata_f = h5py.File(filename_new,'w')
    newdata_f.create_dataset(dataset_new, (new_n_img, new_x_dim, new_y_dim, shape[3]) , dtype = rawdata_f[dataset_raw].dtype)
    
    print('cropping rawdata')
    for i,idx in enumerate(selected_img):
        #print progress
        progress = float(i)/float(len(selected_img))
        print 'progress: {0}%\r'.format(int(progress*100.)),
        unrotated = rawdata_f[dataset_raw][idx,x_start:x_end,y_start:y_end,0]
        newdata_f[dataset_new][i,:,:,0] = unrotated
        #for i_rot in np.arange(3):
            #i_rot += 1
            #augdata_f[dataset_aug][i+i_rot*n_img/every_i_img,:,:,0] = np.rot90(unrotated, i_rot) 
            # augdata_f[dataset_aug][i*4+i_rot,:,:,0] = rotate(unrotated, 90.*i_rot, resize=False, center=(shape[2], shape[1]), preserve_range=True )
    
    newdata_f.close()
    rawdata_f.close()
