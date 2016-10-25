import numpy as np
import multiHypoTracking_with_gurobi as mht
import json

#write a dictionary in a .json file with name filename.json
def write_json(dictionary, filename):
    f = open(filename + '.json', 'w')
    json.dump(dictionary, f)
    f.close()

#read in file with name filename.json and return dictionary
def read_json(filename):
    f = open(filename + '.json', 'r')
    dictionary = json.load(f)
    f.close()
    return dictionary

if __name__ == "__main__":
    max_move_per_frame = 10.
    #load model
    model = read_json('model')
    #weights = {"weights": [1./float(max_move_per_frame), 3./180., 7., 1., 1.]}
    #print type(model)
    #load groundtruth
    gt = read_json('ground_truth')
    #print type(ground_truth)
    weights = mht.train(model, gt)    
    print( weights )
    #write_json(weights, 'weights')

