import numpy as np
import h5py
import multiHypoTracking_with_gurobi as mht
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scorestack import ScoreStack
from graph_as_dict import GraphDict

import configargparse

"""
save Hypotheses (t,x,y,alpha,score) in datastructure of following format: [[t1, several_detections_t1], [t2, several_detections_t2], ...]
with t1<t2<t3,... and detections of timestep ti several_detections_ti = [samples, attributes] attributes: x,y,alpha, score
"""
class Hypotheses:
    def __init__ (self):
        self.frames = []
        
    #t: time index
    #several_detections: numpy array with shape (n_samples, 4) 4: x, y, alpha, score
    def add_frame(self, t, several_detections):
        new_item = [t, several_detections] 
        insert_idx = 0
        last_idx = len(self.frames) - 1 
        t_exist = 0
        #insert new item at correct position and remove existing item when time is equal
        if len(self.frames)==0:       #if no elements are in self.frames or new item time is bigger than all others just append 
            self.frames.append(new_item)
        elif self.frames[-1][0]<t:
            self.frames.append(new_item)
        else:
            for item in self.frames:
                if t==item[0]:
                    t_exist = 1
                    del self.frames[insert_idx]
                    self.frames.insert(insert_idx, new_item)
                    break
                if t<item[0]:
                    self.frames.insert(insert_idx, new_item)
                    break
                insert_idx += 1
        if t_exist == 1:     
            print ('Note: timestep {0} exist already, overwritten'.format(t))

    #find maximal score above threshold_abs
    def max_score(self, threshold_abs):
        max_score = threshold_abs
        for t_item in self.frames:
            for detection in t_item[1]:
                if detection[3] > max_score:
                    max_score = detection[3]
        return max_score

    def print_frames(self):
        print self.frames

    def print_times(self):
        for item in self.frames:
            print item[0]

"""
get detection hypotheses (x,y,t,alpha,score) as input and build the underlying graphical model that is saved in a dicitionary format and handled via the class GraphDict. More explicitly it normalize the score, computes transition features and do some crosschecks for consistency
"""

class Tracker:
    #hypotheses: object of class Hypotheses with hypotheses information
    #max_move_per_frame: only hypotheses with this or lower distance get possible connections
    #threshold_abs: minimal value of scores, important for detection features that no detection energy is always higher than detection energy
    #arrow_orientation: if true back and front exist, if false only axis exist
    def __init__(self, hypotheses, threshold_abs, max_move_per_frame = 25, optimizerEpGap=0.05, arrow_orientation=True):
        self.graph = GraphDict(optimizerEpGap=optimizerEpGap)
        self.threshold_abs = threshold_abs
        self.max_move_per_frame= max_move_per_frame
        #save hypotheses input in own datastructure
        self.hypotheses = hypotheses.frames
        #save maximal score for normalization of detection features
        self.max_score = hypotheses.max_score(threshold_abs)
        #check that no time frame is missing
        self.complete = False
        #save first and last time as attribute
        self.first_t = 0
        self.last_t = 0
        self.check_complete_frames()
        #save ids corresponding to hypotheses: [[t1, ids], ...] ids: array [n_samples] 
        self.ids = self.zero_ids()
        #save conflict set ids in format [[id_1, id_2, ...], [...],...]
        self.exclusion_sets = []
        #save normalized angle_weights
        self.angle_weights = []
        #check that all detection hypotheses have id
        self.graph_complete = False
        #attributes for initialization of first and last frame
        self.neutral_feature=[[0.],[0.]] 
        self.detection_reward = -9.*10**3.
        self.no_detection_reward = -9.*10**3
        #current id for enumerate hypothesis
        self.current_id = 1 
        #ids of true objects in first frame  
        self.start_ids = []
        #number of tracks in total
        self.n_tracks = 0
        #dictionaryfor tracking results
        self.track_result = {}
        self.print_status()        
        self.arrow_orientation=arrow_orientation
 
    #iterate over items in hypotheses and check if times are consecutive
    def check_complete_frames(self):
        last_t = 0
        complete = True
        current_item = 0
        for item in self.hypotheses:
            if current_item == 0:
                self.first_t = item[0]
                last_t = item[0]
                current_item = 1
            else:
                if last_t + 1 == item[0]:
                    last_t += 1
                else:
                    complete = False
                    print('time frame {0} is missing'.format(last_t+1))
        self.last_t = last_t
        if complete==True:
            self.complete = True
        else:
            self.complete = False

    def zero_ids(self):
        ids = []
        idx = 0
        for t in np.arange(self.first_t, self.last_t+1):
            n_samples = self.hypotheses[idx][1].shape[0]
            zero_ids = np.zeros((n_samples) ,dtype = int)
            ids.append([t,zero_ids])
            idx += 1
        return ids

    #checks that all detection hypotheses have an id
    #return complete: True if all detection hypotheses have id
    def check_ids_complete(self):
        complete = True
        n_frames = self.last_t - self.first_t + 1
        #iterate over time frames
        for idx in np.arange(n_frames): 
            n_samples = self.hypotheses[idx][1].shape[0]
            #iterate over detection hypotheses in one frame
            for i_sample in np.arange(n_samples):
                if self.ids[idx][1][i_sample] == 0:
                    complete = False
                    print('missing id in frame {0} i_sample {1}'.format(idx+self.first_t, i_sample))
                    break
            if complete==False:
                break
        return complete
    
    def print_status(self):
        print( 'number of tracks: ', self.n_tracks)
        print( 'maximal movement per frame:', self.max_move_per_frame)
        print( 'frames without gap:', self.complete)
        print( 'start frame:', self.first_t)
        print( 'final frame:', self.last_t)
        print( 'graph complete:', self.graph_complete)
        print( 'threshold for scores:', self.threshold_abs)
    
    def print_json(self, filename, dictionary):
        f = open(filename + '.json', 'w')
        json.dump(dictionary, f)
        f.close()    

    #calculate detection features out of scores:
    # E(X=0) = -threshold + 0.01
    # E(X=1) = -score -> E(X=0) > E(X=1)
    #detection: array [x,y,alpha, score]
    def calc_det_features(self, detection):
        return [ [ float(-self.threshold_abs+0.01)/self.max_score ],[ float(-detection[3])/self.max_score ] ]  #[[no_detection],[detection]]

    #calculate detection features out of scores:
    # E(X=0) = 1/0.005
    # E(X=1) = 1/(score+0.01-threshold) -> E(X=0) > E(X=1)
    #detection: array [x,y,alpha, score]
    def calc_det_features_old(self, detection):
        return [ [ float(1./0.005) ],[ float(1./(detection[3]+0.005-self.threshold_abs)) ] ]  #[[no_detection],[detection]]
    
    #returns for given angle appropriate angle_weight
    #number of angle_weights determine how many angle intervals are used
    def angle_weight(self, angle, angle_weights):
        if len(self.angle_weights)==0:
            #normalization factor
            norm_factor = angle_weights[-1]
            for angle_weight in angle_weights:
                self.angle_weights.append(angle_weight/norm_factor)
        complete_interval = 180.
        assert(angle<=180.), 'transition angle must be smaller than 180. degree'
        if self.arrow_orientation==False:
            complete_interval = 90.
            assert(angle<=90.), 'transition angle must be smaller than 90. degree'
        angle_interval_len = complete_interval/(len(self.angle_weights)-1.)
        idx = int(angle/angle_interval_len + 0.5)
        return self.angle_weights[idx]

    #calculate transition features out of distance and angle:
    #the energies are normalized later on by deviding with E(0)
    #movement/distance:
    # E(Y=0) = max_distance + 0.1
    # E(Y=1) = distance
    #angle 
    # E(Y=0) = 180.1
    # E(Y=1) = angle_weight(angle)
    #detection: array [x,y,alpha, score]
    #consecutive detections: t1 < t_2 
    #angle_weights = list of weights for different orientations [w_0, w_45, w_90, w_135, w_180]
    def calc_trans_features(self, detection_t1, detection_t2, angle_weights):
        x_1 = detection_t1[0]
        y_1 = detection_t1[1]
        alpha_1 = detection_t1[2]
        x_2 = detection_t2[0]
        y_2 = detection_t2[1]
        alpha_2 = detection_t2[2] 
        
        #movement feature
        distance = np.sqrt((x_1-x_2)**2 + (y_1-y_2)**2)
        #trans_feature = np.exp(-distance/self.alpha)-0.0000
        move_feature = [[(np.float32(self.max_move_per_frame)+0.1)/np.float32(self.max_move_per_frame)], [distance/np.float32(self.max_move_per_frame)]]
        
        #angle feature
        delta_alpha = 0.
        if self.arrow_orientation==False:
            delta_alpha = (alpha_1 - alpha_2) % 180.
            if delta_alpha > 90.:
                delta_alpha = 180. - delta_alpha
        else:
            delta_alpha = (alpha_1 - alpha_2) % 360.
            if delta_alpha > 180.:
                delta_alpha = 360. - delta_alpha
        angle_feature = [[180.1/180.],[self.angle_weight(delta_alpha, angle_weights)]]
        
        return([ [ float(move_feature[0][0]), float(angle_feature[0][0]) ], [ float(move_feature[1][0]), float(angle_feature[1][0]) ] ])

    #find in detections_t2 those detections with lower distance than max_move_per_frame when use_radius= False else radius is used
    #detection_t1: array [x,y,alpha, score]
    #detection_t2: array [n_samples, attributes]
    #use_radius: decide if you want use radius or max_move_per_frame
    #return links: list of indices of detections_t2 that are in vicinity of detection_t1
    def find_nearest_neighbours(self, detection_t1, detections_t2, use_radius=False, radius=3.):
        if use_radius==False:
            radius=self.max_move_per_frame
        links = []
        x = detection_t1[0]
        y = detection_t1[1]
        n_samples = detections_t2.shape[0]
        for i_sample in np.arange(n_samples):
            current_det = detections_t2[i_sample,:]
            distance = np.sqrt((x-current_det[0])**2 + (y-current_det[1])**2)
            if distance < radius:
                links.append(i_sample)
        return links
    
    #find maximal hypos around centers
    def find_hypo_idc_auto(self, frame, centers, radius):
        idc = []
        for center in centers:
            #maximal score of hypos
            max_score = 0.
            max_idx = -1
            hypos = self.find_hypo_idc(frame, center, radius, print_hypos=False)
            for idx, hypo in enumerate(hypos):
                score = hypo[1][3]
                if idx==0:      
                    max_score = score
                    max_idx = hypo[0]
                else:
                    if score > max_score: 
                        max_score = score
                        max_idx = hypo[0]
            idc.append(max_idx)
        return idc
                           
        

    #for initializing first and last frame the indices for the correct hypotheses must be selected, choose in a frame the center pos of circle with given length
    #return hypotheses with indices in circle as list [[idx, hypo], ...]
    #frame: frame idx
    #center: center pos of circle np.array([x, y])
    #radius: radius for circle for looking for hypotheses
    #print_hypos: if true print hypos in bbox in nice format
    def find_hypo_idc(self, frame, center, radius, print_hypos=True):
        selec_hypos = []
        assert(frame >= self.first_t)
        assert(frame <= self.last_t)
        frame_idx = frame - self.first_t
        hypos = self.hypotheses[frame_idx][1] 
        for idx, hypo in enumerate(hypos):
            dist_squared = (hypo[0]-center[0])**2 + (hypo[1]-center[1])**2
            if dist_squared<=radius**2:
                selec_hypos.append([idx, hypo])
        if print_hypos==True:
            for item in selec_hypos:
                print([item[0], item[1][0], item[1][1], item[1][2], item[1][3]])
        return selec_hypos
 

    #set detections in first frame with indices in true_det_idc to one, and add all detection hypotheses with ids in first frame
    #true_det_idc: list with indices of true detections, indices are not ids, ids are given automatically by class
    def initialize_first_frame(self, true_det_idc):
        self.n_tracks = len(true_det_idc)
        #number of hypotheses in first frame  
        n_samples = self.hypotheses[0][1].shape[0]  
        for idx in np.arange(n_samples):
            detection = self.hypotheses[0][1][idx, :]  # array [x, y, alpha, score]        
            self.ids[0][1][idx] = self.current_id
            #true_detections
            if idx in true_det_idc: 
                self.graph.add_seg_hypo(self.current_id, self.true_detection(), x=int(detection[0]), y=int(detection[1]), alpha=int(detection[2]), t=int(self.first_t), appearanceFeatures=self.neutral_feature) 
                self.start_ids.append(self.current_id)
            #wrong_detections
            else:
                self.graph.add_seg_hypo(self.current_id, self.false_detection(), x=int(detection[0]), y=int(detection[1]), alpha=int(detection[2]), t=int(self.first_t) ) #appearance_feature    
            self.current_id += 1        
    
    #set detections in last frame with indices in true_det_idc to one, and add all detection hypotheses with ids in last frame
    #true_det_idc: list with indices of true detections 
    def initialize_last_frame(self, true_det_idc):  
        if not self.n_tracks == len(true_det_idc):
            print('error: first frame initialized with {0} tracks and last frame with {1} tracks'.format(self.n_tracks,len(true_det_idc)))
        #number of hypotheses in last frame
        n_samples = self.hypotheses[-1][1].shape[0]  
        for idx in np.arange(n_samples):
            detection = self.hypotheses[-1][1][idx, :]  # array [x, y, alpha, score]        
            self.ids[-1][1][idx] = self.current_id
            #true_detections
            if idx in true_det_idc: 
                self.graph.add_seg_hypo(self.current_id, self.true_detection(), x=int(detection[0]), y=int(detection[1]), alpha=int(detection[2]), t=int(self.last_t), disappearanceFeatures=self.neutral_feature) 
            #wrong_detections
            else:
                self.graph.add_seg_hypo(self.current_id, self.false_detection(), x=int(detection[0]), y=int(detection[1]), alpha=int(detection[2]), t=int(self.last_t) ) #appearance_feature    
            self.current_id += 1       

    #add intermediate segmentation hypothesis (i.e. detections in frames that are not first and last one)
    def add_intermediate_det_hypos(self):
        time_interval = self.last_t - self.first_t
        #iterate over indices of hypotheses according to intermediat timesteps
        for idx in np.arange(1, time_interval): #1,2,...,time_interval-1
            t = self.hypotheses[idx][0]
            detections = self.hypotheses[idx][1]
            n_samples = detections.shape[0]
            #iterate over detection hypothesis
            for i_sample in np.arange(n_samples):
                self.ids[idx][1][i_sample] = self.current_id
                detection = detections[i_sample, :]
                self.graph.add_seg_hypo(self.current_id, self.calc_det_features(detection), x=int(detection[0]), y=int(detection[1]), alpha=int(detection[2]), t=int(t))
                self.current_id += 1
    

    #add transition hypothesis, each detection hypothesis in t is connected with all detection hypos in t+1 when the distance is smaller than self.max_move_per_frame
    def add_trans_hypos(self, angle_weights):
        time_interval = self.last_t - self.first_t
        #iterate over all timeframes except the last one and build links
        for idx in np.arange(time_interval):
            t = self.hypotheses[idx][0]
            detections = self.hypotheses[idx][1]
            n_samples = detections.shape[0]
            #iterate over all hypotheses in current time frame
            for i_sample in np.arange(n_samples):
                detection = detections[i_sample,:]
                detection_id = self.ids[idx][1][i_sample]
                links = self.find_nearest_neighbours(detection, self.hypotheses[idx+1][1])
                #iterate over all links
                for link in links:
                    link_id = self.ids[idx+1][1][link]
                    link_detection = self.hypotheses[idx+1][1][link]
                    self.graph.add_link_hypo(detection_id, link_id, self.calc_trans_features(detection, link_detection, angle_weights))
   
    #add conflict sets, i.e. hypotheses within one frame with a distance smaller than radius are not allowed to be turned on at the same time  
    def add_conflict_sets(self, radius): 
        for t_idx, t_item in enumerate(self.hypotheses):
            for hypo in t_item[1]:
                conflict_set_ids = []
                conflict_set_idc = self.find_nearest_neighbours(hypo, t_item[1], use_radius=True, radius=radius)
                for conflict_set_idx in conflict_set_idc:
                    conflict_id = self.ids[t_idx][1][conflict_set_idx]
                    conflict_set_ids.append(conflict_id)
                if len(conflict_set_ids)>1:
                    if not conflict_set_ids in self.exclusion_sets: 
                        self.exclusion_sets.append(conflict_set_ids)
                        self.graph.add_exclusions(conflict_set_ids)
                    

    #add weights to graphical model
    #weights: list [transition_move, transition_angle , detection, appearance, disappearance]
    def add_weights(self, weights):
        self.graph.add_weights(weights)       
                  
    #build complete graph and check that all detection hypotheses have ids  
    #true_det_first_t/true_det_last_t: list with indices of hypotheses that are true
    #weights: list [transition, detection, appearance, disappearance]
    def build_graph(self, true_det_first_t, true_det_last_t, weights, angle_weights, conflict_radius=3.):
        #add detection hypothesis of all time frames
        self.initialize_first_frame(true_det_first_t)
        self.add_intermediate_det_hypos()
        self.initialize_last_frame(true_det_last_t)
        #add transition hypothesis
        self.add_trans_hypos(angle_weights)
        #add conflict sets if conflict_radius=0 ,i.e. hypos in the same frame with distance smaller than given radius
        if not conflict_radius==0.:
            self.add_conflict_sets(conflict_radius)
        #add weights
        self.add_weights(weights)
        if self.check_ids_complete()==True:
            self.graph_complete = True
        else:
            print('error: not all hypotheses have id')
    
    #build graph and performes tracking and save result as dictionary in self.track_result
    #true_det_first_t/true_det_last_t: list with indices of hypotheses that are true
    #weights: list [transition, detection, appearance, disappearance]
    #angle_weights: weight different angles in transition features [w_0, w_45, w_90, w_135, w_180], 
    #len(angle_weights) corresponds to the number of different intervals between 0 and 180 degree 
    def track(self, true_det_first_t, true_det_last_t, weights, conflict_radius, angle_weights, print_model=False, print_result=False):
        self.build_graph(true_det_first_t, true_det_last_t, weights, angle_weights, conflict_radius)
        result = mht.track(self.graph.model, self.graph.weights)
        self.track_result = result
        if print_model == True:
            self.print_json('model', self.graph.model)
        if print_result == True:
            self.print_json('ground_truth', result)

    def true_detection(self):
        return [[-self.detection_reward],[self.detection_reward]]

    def false_detection(self):
        return [[self.no_detection_reward],[-self.no_detection_reward]]

    #given an id of a true hypothesis (src_id) find the follower 
    #return: dest_id, if not found 0 is returned
    def find_dest(self, src_id):
        dest_id = 0
        for linking in self.track_result['linkingResults']:
            if linking["src"] == src_id:
                dest_id = linking["dest"]
                break
        return dest_id
        
    #return list of ids of hypotheses belonging to a track that start in first frame with track_start_id
    def get_track(self, track_start_id):
        time_interval = self.last_t - self.first_t
        track_ids = [track_start_id]
        current_src = track_start_id
        for idx in np.arange(time_interval):
            dest_id = self.find_dest(current_src)
            track_ids.append(dest_id)
            current_src = dest_id
        return track_ids

    #get sequence of coordinates array: [x,y,alpha, score]
    #track: list of ids, result of get_track
    def get_coordinates(self, track):
        coordinates = []
        n_frames = len(track)
        for idx in np.arange(len(track)):
            current_id = track[idx]
            n_ids = self.ids[idx][1].shape[0]
            for idx_id in np.arange(n_ids):
                if current_id == self.ids[idx][1][idx_id]:
                    coordinates.append(self.hypotheses[idx][1][idx_id,:])
                    break
            #if no matching hypothesis is found add array with zeros as hypotheses
            #if not len(coordinates) == idx+1:
                #zero = np.zeros((4), dtype=float)
                #coordinates.append(zero)
        return coordinates

    def rotate_coord(self, x, y, angle):
        angle = np.pi/180.*angle
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        vector = np.array([x,y])
        return np.dot(rot_matrix, vector)

    #saves the distance and angle transitions of the objects in the format [dx, dy, dalpha]
    #all_tracks: if true all tracks are used, if false only selected tracks in track_start_ids are used
    #track_start_ids: list of start_ids that should be used for tracking
    #add_to_h5_file: if True look if file with name filename exist and add transitions to it otherwise create new file file f['data'] array with transition
    #return transitions: list of transition objects [dx, dy, dalpha]
    def object_transition_analysis(self, all_tracks=True, track_start_ids=[], add_to_h5_file=False, filename='transitions.h5'):
        if all_tracks==True:
            track_start_ids = self.start_ids
        else:
            #if selected tracks are used check that start indices are correct
            for track_id in track_start_ids:
                assert(track_id in self.start_ids)
        transitions = []
        for start_id in track_start_ids:
            track_ids = self.get_track(start_id)
            coordinates = self.get_coordinates(track_ids)
            idx = 0
            for idx_coord in np.arange(1, len(coordinates)):
                coord = coordinates[idx_coord]
                coord_before = coordinates[idx_coord-1]
                alpha_2 = coord[2]
                alpha_1 = coord_before[2] 
                delta_alpha = (alpha_2 - alpha_1) % 360.
                dx = coord[0] - coord_before[0]
                dy = coord[1] - coord_before[1]  
                trans_vector = self.rotate_coord(dx, dy, alpha_1)
                trans_angle = alpha_2 - alpha_1
                transitions.append([trans_vector[1], -trans_vector[0], trans_angle])
        if add_to_h5_file==True:
            #number of existing transitions
            n_ex_trans = 0
            f = h5py.File(filename,'a')
            if 'data' in f.keys():
                n_ex_trans = f['data'].shape[0]
                assert (f['data'].shape[1]==3)
                trans_arr_old = f['data'][:,:]
                del f['data']
                f.create_dataset('data', (len(transitions)+n_ex_trans, 3))
                f['data'][:n_ex_trans, :] = trans_arr_old
                for idx, transition in enumerate(transitions):
                    f['data'][n_ex_trans+idx, 0] = transition[0]   
                    f['data'][n_ex_trans+idx, 1] = transition[1]   
                    f['data'][n_ex_trans+idx, 2] = transition[2]
            else:
                f.create_dataset('data', (len(transitions), 3))
                for idx, transition in enumerate(transitions):
                    f['data'][idx, 0] = transition[0]   
                    f['data'][idx, 1] = transition[1]   
                    f['data'][idx, 2] = transition[2]  
            f.close()
        return transitions

    #draw track history in images
    #path_imgs_clean: relative path to raw images without tracks, raw images should have name in format RawData_2970.png
    #path_imgs_tracks: relative path to raw images with tracks, track images have name in format Track_2970.png
    #length_arrow: length of arrow in pixels for visualization of orientation of object
    #rescale_factor: factor for rescaling from scoremap to orginal img
    #all_tracks: decide if you want all tracks drawn or only selected ones
    #track_start_ids: if all_tracks is False here are the selected start ids of the chosen tracks  
    def draw_tracks(self, path_imgs_tracks, length_arrow, rescale_factor=2., all_tracks=True, track_start_ids=[]):
        if all_tracks==True:
            track_start_ids = self.start_ids
        else:
            #if selected tracks are used check that start indices are correct
            for track_id in track_start_ids:
                assert(track_id in self.start_ids)
        time_interval = self.last_t - self.first_t
        f = h5py.File(relative_path_cropped_rawdata + filename_cropped_rawdata, 'r')
        for dt in np.arange(0, time_interval+1):
            #img = mpimg.imread(path_imgs_clean + 'RawData_' + str(self.first_t+dt).zfill(zfill) + '.png')
            img = f[dataset_cropped_rawdata][self.first_t+dt,:,:,0]
            img = np.transpose(img)
            fig = plt.figure()
            #fig.set_size_inches(float(img.shape[1])/float(img.shape[0]), 1., forward=False)
            #ax = plt.Axes(fig, [0., 0., 1., 1.])
            #ax.set_axis_off()
            #fig.add_axes(ax)
            ax = fig.add_subplot(1,1,1)
            ax.imshow(img, cmap=matplotlib.cm.Greys_r)
            #extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            
            for start_id in track_start_ids:
                track_ids = self.get_track(start_id)
                coordinates = self.get_coordinates(track_ids)  
                ax.scatter(rescale_factor*coordinates[0][0],rescale_factor*coordinates[0][1], marker='x', color='r', s=0.5, alpha=0.7)
                #for coordinate in coordinates:
                   # print( coordinate[0])  
                #print('length of coordinates', len(coordinates))
                line_x = []
                line_y = []
                track_length = min(dt+1, len(coordinates))
                if track_length==1:
                    x_id = 2*coordinates[0][0]
                    y_id = 2*coordinates[0][1]
                    ax.annotate(str(start_id),size=3,color='g', xy=(x_id, y_id), xytext=(x_id ,y_id-15), arrowprops=dict(arrowstyle='->', color='g'))#arrowprops=dict(facecolor='black', shrink=0.005))
                for step in np.arange(track_length):
                    line_x.append(rescale_factor * coordinates[step][0])
                    line_y.append(rescale_factor * coordinates[step][1])
                ax.plot(line_x, line_y, linewidth=1., color='r', alpha=0.6)
                arrow_x_start = line_x[-1]
                arrow_y_start = line_y[-1]
                arrow_delta_x = np.sin(np.pi/180.*coordinates[track_length-1][2])*length_arrow
                arrow_delta_y = -np.cos(np.pi/180.*coordinates[track_length-1][2])*length_arrow
                if not len(line_x) == 0:
                    if self.arrow_orientation==True:
                        ax.arrow(arrow_x_start, arrow_y_start, arrow_delta_y , arrow_delta_x , head_width=4, head_length=8, fc='r', ec='r', alpha=0.7, width=0.8)
                    else:
                        ax.arrow(arrow_x_start, arrow_y_start, arrow_delta_y/2. , arrow_delta_x/2. , head_width=4, head_length=8, fc='r', ec='r', alpha=0.7, width=0.8)
                if self.arrow_orientation==False:
                    ax.arrow(arrow_x_start, arrow_y_start, -arrow_delta_y/2. , -arrow_delta_x/2. , head_width=4, head_length=8, fc='r', ec='r', alpha=0.7, width=0.8)
                    
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            #ax.axis('off')
            #ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2])))
            #ax.set_aspect('equal')
            #fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            #fig.tight_layout()
            fig.savefig(path_imgs_tracks + 'Track_' + str(self.first_t+dt) + '.png', cmap=matplotlib.cm.Greys_r, bbox_inches='tight', pad_inches=-0.6, dpi=330)
        f.close()
        


#function to find true_det_idc when using new lower threshold
#hypos_old: hypos of old threshold array [samples, attributes]
#true_det_old: indices of true detection with old threshold
#hypos new: hypos of new threshold array [samples, attributes]
def find_hypo_idx(hypos_new, true_det_old, hypos_old):
    true_det_new = []
    for old_det_idx in true_det_old:
        old_detection = hypos_old[old_det_idx,:] 
        n_samples_new = hypos_new.shape[0]
        for idx in np.arange(n_samples_new):
            if (old_detection == hypos_new[idx,:]).all():
                true_det_new.append(idx)
                break
    return true_det_new
    
       

if __name__ == "__main__":
    
    
    #fixed parameters
    relative_path_cropped_rawdata = '../Preprocessing/'
    filename_cropped_rawdata = 'rawdata.h5'
    dataset_cropped_rawdata = 'data'
    path_scoremaps = '../DPM/samples/pred/scoremap/'
    
    #parameters set by config file
    #rotational paramater
    use_config = True
    use_LoG = False
    sigma_LoG = 1.
    arrow_orientation = True
    x_center = 510/2
    y_center = 514/2
    n_rot = 8
    energy_gap_graphical_model = 0.01
    threshold_abs = 0.3
    max_move_per_frame = 10.
    conflict_radius = 5.
    #radius for finding hypotheses with highest score as initialization
    radius_ini = 4.
    weight_trans_move=1.
    weight_trans_angle=3.
    weight_det=1.
    angle_weights = [0., 20., 135., 180., 180.]
    start_idx = 0
    n_idx = 51
    
    #derived parameters
    weights = [weight_trans_move, weight_trans_angle, weight_det, 1., 1. ] #[trans_move, trans_angle, detection, appearance, disappearance ]

    #configargparse
    p = configargparse.ArgParser()
    p.add('-c', '--config', is_config_file=True, help='config file path')
    p.add('--mht-use_LoG', default=False, action='store_true', help='if true: Laplacian of Gaussian is calculated on Scoremap for blob detection')    
    p.add('--mht-sigma_LoG', default=False, type=float, help='sigma for Laplacian of Gaussian')    
    p.add('--global-arrow_orientation', default=False, action='store_true', help='if true: arrow orientation exist, if wrong: only axis orientation exist')    
    p.add('--global-n_rot', default=n_rot, type=int, help='number of rotations/different orientations')    
    p.add('--global-x_center', default=x_center, type=int, help='coordinates of center for rotations')    
    p.add('--global-y_center', default=y_center, type=int, help='coordinates of center for rotations') 
    p.add('--mht-energy_gap_graphical_model', default=energy_gap_graphical_model, type=float, help='tolerated energy gap when solving graphical model')
    p.add('--mht-threshold_abs', default=threshold_abs, type=float, help='threshold for local maxima in scoremap used for generating hypotheses') 
    p.add('--mht-max_move_per_frame', default=max_move_per_frame, type=float, help='maximal possible transition distance')
    p.add('--mht-conflict_radius', default=conflict_radius, type=float, help='hypotheses within one frame with a smaller distance than this conflict radius are never used for two different tracks')
    p.add('--mht-radius_ini', default=radius_ini, type=float, help='radius for finding hypotheses with highest score as initialization in first and last frame')
    p.add('--mht-weight_trans_move', default=weight_trans_move, type=float, help='relative weights for energy components')    
    p.add('--mht-weight_trans_angle', default=weight_trans_angle, type=float, help='relative weights for energy components')    
    p.add('--mht-weight_det', default=weight_det, type=float, help='relative weights for energy components')    
    p.add('--mht-angle_weight_1', default=-1, type=float, help='energies for change in orientation')    
    p.add('--mht-angle_weight_2', default=-1, type=float, help='energies for change in orientation')    
    p.add('--mht-angle_weight_3', default=-1, type=float, help='energies for change in orientation')    
    p.add('--mht-angle_weight_4', default=-1, type=float, help='energies for change in orientation')    
    p.add('--mht-angle_weight_5', default=-1, type=float, help='energies for change in orientation')    
    p.add('--mht-angle_weight_6', default=-1, type=float, help='energies for change in orientation')    
    p.add('--mht-angle_weight_7', default=-1, type=float, help='energies for change in orientation')    
    p.add('--mht-angle_weight_8', default=-1, type=float, help='energies for change in orientation')    
    p.add('--mht-angle_weight_9', default=-1, type=float, help='energies for change in orientation')    
    p.add('--mht-angle_weight_10', default=-1, type=float, help='energies for change in orientation')    
    p.add('--mht-angle_weight_11', default=-1, type=float, help='energies for change in orientation')    
    p.add('--mht-angle_weight_12', default=-1, type=float, help='energies for change in orientation')    
    p.add('--mht-angle_weight_13', default=-1, type=float, help='energies for change in orientation')    
    p.add('--mht-angle_weight_14', default=-1, type=float, help='energies for change in orientation')    
    p.add('--mht-angle_weight_15', default=-1, type=float, help='energies for change in orientation')    
    p.add('--mht-angle_weight_16', default=-1, type=float, help='energies for change in orientation')    
    p.add('--preprocessing-idx_begin', default=0, type=int, help='start idx for cropping in time')
    p.add('--preprocessing-idx_end', default=1, type=int, help='end idx for cropping in time')
    p.add('--preprocessing-every_i_img_rawdata', default=1, type=int, help='selecting every i th image for tracking (preprocessing)')


    options, unknown = p.parse_known_args()
           
    #parse parameter
    if use_config==True:
        use_LoG = options.mht_use_LoG
        sigma_LoG = options.mht_sigma_LoG
        arrow_orientation = options.global_arrow_orientation
        n_rot = options.global_n_rot
        #divided by 2 to get distances in downsamples scoremap
        x_center = int(options.global_x_center/2)
        y_center = int(options.global_y_center/2)
        energy_gap_graphical_model = options.mht_energy_gap_graphical_model
        threshold_abs = options.mht_threshold_abs
        #divided by 2 to get distances in downsamples scoremap
        max_move_per_frame = options.mht_max_move_per_frame/2
        #divided by 2 to get distances in downsamples scoremap
        conflict_radius = options.mht_conflict_radius/2
        start_idx = 0
        n_idx = int((options.preprocessing_idx_end - options.preprocessing_idx_begin)/options.preprocessing_every_i_img_rawdata)
        #divided by 2 to get distances in downsamples scoremap
        radius_ini = options.mht_radius_ini/2
        weight_trans_move = options.mht_weight_trans_move
        weight_trans_angle = options.mht_weight_trans_angle
        weight_det = options.mht_weight_det
        
        weights = [weight_trans_move, weight_trans_angle, weight_det, 1., 1. ] #[trans_move, trans_angle, detection, appearance, disappearance ]
        angle_weights = []
        if options.mht_angle_weight_1>0:
            angle_weights.append(options.mht_angle_weight_1)
        if options.mht_angle_weight_2>0:
            angle_weights.append(options.mht_angle_weight_2)
        if options.mht_angle_weight_3>0:
            angle_weights.append(options.mht_angle_weight_3)
        if options.mht_angle_weight_4>0:
            angle_weights.append(options.mht_angle_weight_4)
        if options.mht_angle_weight_5>0:
            angle_weights.append(options.mht_angle_weight_5)
        if options.mht_angle_weight_6>0:
            angle_weights.append(options.mht_angle_weight_6)
        if options.mht_angle_weight_7>0:
            angle_weights.append(options.mht_angle_weight_7)
        if options.mht_angle_weight_8>0:
            angle_weights.append(options.mht_angle_weight_8)
        if options.mht_angle_weight_9>0:
            angle_weights.append(options.mht_angle_weight_9)
        if options.mht_angle_weight_10>0:
            angle_weights.append(options.mht_angle_weight_10)
        if options.mht_angle_weight_11>0:
            angle_weights.append(options.mht_angle_weight_11)
        if options.mht_angle_weight_12>0:
            angle_weights.append(options.mht_angle_weight_12)
        if options.mht_angle_weight_13>0:
            angle_weights.append(options.mht_angle_weight_13)
        if options.mht_angle_weight_14>0:
            angle_weights.append(options.mht_angle_weight_14)
        if options.mht_angle_weight_15>0:
            angle_weights.append(options.mht_angle_weight_15)
        if options.mht_angle_weight_16>0:
            angle_weights.append(options.mht_angle_weight_16)


    """
    #generate hypotheses
    """  

    hypotheses = Hypotheses()

    for idx in np.arange(start_idx, start_idx + n_idx):
        stack = ScoreStack(idx, n_rot, x_center=x_center, y_center=y_center, arrow_orientation=arrow_orientation, path_scoremaps=path_scoremaps, LoG=use_LoG, sigma_LoG=1.)
        hypotheses.add_frame(idx, stack.extract_hypotheses(threshold_abs=threshold_abs))
    
    """
    #perform tracking
    """
    
    tracker = Tracker(hypotheses, threshold_abs, max_move_per_frame=max_move_per_frame, optimizerEpGap=energy_gap_graphical_model, arrow_orientation=arrow_orientation)
    
    x_start = 551.
    y_start = 234.
    #automatic start_idc search
    centers_start = np.array([[(637.-x_start)/2, (331.-y_start)/2], [(666.-x_start)/2, (318.-y_start)/2], [(666.-x_start)/2, (375.-y_start)/2] ])
    centers_end = np.array([[(682.-x_start)/2, (312.-y_start)/2], [(652.-x_start)/2, (396.-y_start)/2], [(853.-x_start)/2, (417.-y_start)/2]])
        
    true_det_first_t = tracker.find_hypo_idc_auto(start_idx, centers_start, radius_ini)   
    true_det_last_t = tracker.find_hypo_idc_auto(start_idx + n_idx-1, centers_end, radius_ini)
    #put all energies on one scale
    tracker.track(true_det_first_t, true_det_last_t, weights, conflict_radius, angle_weights, print_model=True, print_result=True)
    tracker.print_status()
    tracker.draw_tracks('../MultiHypoTracking/images_with_tracks/', length_arrow=15, all_tracks=True)
    
    
