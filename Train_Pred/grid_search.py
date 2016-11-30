import numpy as np

"""
this is an implementation of some functions used for grid search 
"""
class grid_search_helper(object):
    def __init__(self):
        #one weight stays the same because optimal solution depends only on ratio of weights
        self.weights_trans_move = [5.38]
        self.weights_trans_angle = [1., 1.4, 1.96, 2.74, 3.84, 5.38, 7.53, 10.54, 14.76, 20.66, 28.93] 
        self.weights_det = [1., 1.4, 1.96, 2.74, 3.84, 5.38, 7.53, 10.54, 14.76, 20.66, 28.93]
        """
        self.weights_trans_move = [1., 1.8, 3.2, 5.8, 10.5]
        self.weights_trans_angle = [1., 1.8, 3.2, 5.8, 10.5]
        self.weights_det = [1., 1.8, 3.2, 5.8, 10.5]
        """

        self.n_idx = len(self.weights_trans_move) * len(self.weights_trans_angle) * len(self.weights_det)
        #lookup table for weights corresponding to index, shape(n_idx, 3) for each idex an triple of weights is saved in the following order [weight_trans_move, weight_trans_angle, weight_det]
        self.look_up_table = np.zeros((self.n_idx, 3), dtype=float)
        idx = 0
        for weight_trans_move in self.weights_trans_move:
            for weight_trans_angle in self.weights_trans_angle:
                for weight_det in self.weights_det:
                    self.look_up_table[idx,0] = weight_trans_move
                    self.look_up_table[idx,1] = weight_trans_angle
                    self.look_up_table[idx,2] = weight_det
                    idx += 1
                          
        
    #get range of indices for grid search
    def get_idx_range(self):
        return np.arange(self.n_idx)
   
    #get weights corresponding to index
    def get_weights_out_of_idx(self, idx):
        assert(idx<self.n_idx)
        assert(0<=idx)
        return self.look_up_table[idx,:]
    
    def get_weight_trans_move(self, idx):
        assert(idx<self.n_idx)
        assert(0<=idx)
        return self.look_up_table[idx,0]

    def get_weight_trans_angle(self, idx):
        assert(idx<self.n_idx)
        assert(0<=idx)
        return self.look_up_table[idx,1]

    def get_weight_det(self, idx):
        assert(idx<self.n_idx)
        assert(0<=idx)
        return self.look_up_table[idx,2]


        

if __name__ == '__main__':
    grid_search_helper = grid_search_helper()
    print grid_search_helper.get_idx_range()
    print grid_search_helper.get_weight_det(85)
    

