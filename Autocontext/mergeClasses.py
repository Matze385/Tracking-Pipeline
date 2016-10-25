import numpy as np

#merge classes of hard segementation to one class 
def merge_classes(y_label, merger):
#y_lable: array of shape (samples) with labels
#merger:1-D array with indices of classes that should be merged, eg. [1,2,3] if classes 1,2,3 should be merged
    v_merge = np.vectorize(lambda x: merger.min() if x in merger else x)    
    return v_merge(y_label)

#add prob maps in merger to soft segmentation :
def merge_probs(prob_map, merger):
#prob_map: [x,y,n_classes]
    prob_map_new = np.zeros((prob_map.shape[0], prob_map.shape[1]), dtype=np.float32)
    for merged_class in merger:
        #-1: class begin with 1 array indices with 0
        prob_map_new[:,:] += prob_map[:,:,merged_class-1] 
    return prob_map_new


#mergers: list of lists, lists in list contain indices that should be merged to one class
#returns soft segmentation channel 0 agglomerate all class probabilities that are not used for merging
def merge(prob_map, mergers):
    #count number of classes
    n_merged_classes = 0
    for merger in mergers:
        if len(merger) > 0:
            n_merged_classes += 1
    prob_map_merged = np.zeros((prob_map.shape[0], prob_map.shape[1], n_merged_classes), dtype=prob_map.dtype)
    i_merged_class = 0
    for merger in mergers:
        if len(merger) > 0:
            for cl in merger:
                ch = cl - 1 #relation channel and class in prob_map
                prob_map_merged[:,:,i_merged_class] += prob_map[:,:,ch]
            i_merged_class += 1
    return prob_map_merged
    
#normalized: if normalized=True probabilities add to one, if False there exist a class that is
#returns hard seg out of prob map: winner takes it all
def hard_seg(prob_map, normalized=True):
    if normalized==False:
        prob_map_new = np.zeros((prob_map.shape[0], prob_map.shape[1], prob_map.shape[2]+1), dtype=prob_map.dtype)
        prob_map_new[:,:,0] = 1.
        for ch in np.arange(prob_map.shape[2]):
            prob_map_new[:,:,ch+1] = prob_map[:,:,ch]
            prob_map_new[:,:,0] -= prob_map[:,:,ch]
        return np.argmax(prob_map_new, axis=2)
    else:
        return np.argmax(prob_map, axis=2) +1 
    
     
