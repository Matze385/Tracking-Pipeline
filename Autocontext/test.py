import h5py
import numpy as np

"""
with h5py.File('indices_pos.h5','w') as f:
    f.create_dataset('data', (1000,))
    f.create_dataset('n_img', (1,))
    f['n_img'][0] =7
    f['data'][0] = 1750
    f['data'][1] = 400
    f['data'][2] = 50
    f['data'][3] = 1400
    f['data'][4] = 300
    f['data'][5] = 1300
    f['data'][6] = 200
"""

#change dataset name of 
"""
with h5py.File('80_raw_cropped.h5', 'r+') as f:
    print f['volume/data'].dtype
    #save_data = f['volume/data'][:,:,:,:]
    #f.create_dataset('data', data=f['/volume/data'])
    #f.__delitem__('volume')
"""

with h5py.File('CrossValFolds.h5', 'r') as f:
    print f['folds'].shape
