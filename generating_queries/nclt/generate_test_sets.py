import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import gputransform
import config as cfg

#####For training and test data split#####
cfg.submap_interval = 1.5

x_width = 150
y_width = 150

# For Oxford
p1 = [5735712.768124,620084.402381]
p2 = [5735611.299219,620540.270327]
p3 = [5735237.358209,620543.094379]
p4 = [5734749.303802,619932.693364]

# For University Sector
p5 = [363621.292362,142864.19756]
p6 = [364788.795462,143125.746609]
p7 = [363597.507711,144011.414174]

# For Residential Area
p8 = [360895.486453,144999.915143]
p9 = [362357.024536,144894.825301]
p10 = [361368.907155,145209.663042]

p_dict = {"oxford":[p1,p2,p3,p4], "university":[
    p5,p6,p7], "residential": [p8,p9,p10], "business":[]}

p = [-50.0, 150.0, -250.0, 150.0]


# check if the location is in the test set
def check_in_test_set(northing, easting, points):
    in_test_set = False
    if(points[0] < northing and northing < points[1] and points[2] < easting and easting < points[3]):
        in_test_set = True
    return in_test_set


# check if it's a new place
def check_submap(northing, easting, prev_northing, prev_easting):
    is_submap = False
    euclidean = np.abs(np.sqrt((prev_northing-northing)**2 + (prev_easting-easting)**2))
    
    if(euclidean < cfg.submap_interval and euclidean >= (cfg.submap_interval - 0.5)):
        is_submap = True
    
    return is_submap


# find closest place timestamp with index returned
def find_closest_timestamp(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx


# dump the tuples to pickle files for training
def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


# construct evaluation tuples
def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename, p, output_name):
    
    database_trees = []
    test_trees = []

    for folder in folders:
        print(folder)

        # construct dataframe
        velo_file = []
        df_database = pd.DataFrame(columns=['file','northing','easting','yaw'])
        df_test = pd.DataFrame(columns=['file','northing','easting','yaw'])
        df_velo = pd.DataFrame(columns=['file','northing','easting','yaw'])

        # read groundtruth in each folder
        gt_filename = "ground_truth/groundtruth_" + folder + '.csv'
        df_locations = pd.read_csv(os.path.join(
            base_path,runs_folder,folder,gt_filename), header=0, names = ['timestamp','northing','easting', 'height','roll','pitch','yaw'],low_memory=False)
        all_filenames = sorted(os.listdir(os.path.join(base_path,runs_folder,folder+pointcloud_fols)))
        
        # get the file name
        for names in all_filenames:
            names = os.path.splitext(names)[0]
            velo_file.append(names)

        # convert data type
        df_locations['timestamp'] = df_locations['timestamp'].astype(int)
        df_velo['file'] = velo_file

        # save all relative info into df_velo
        for idx in range(len(df_velo)):
            loc_idx = find_closest_timestamp(df_locations['timestamp'].values, int(df_velo['file'][idx]))
            df_velo['yaw'][idx] = df_locations['yaw'][loc_idx]
            df_velo['northing'][idx] = df_locations['northing'][loc_idx]
            df_velo['easting'][idx] = df_locations['easting'][loc_idx]
        
        df_velo['file'] = runs_folder+folder + \
            pointcloud_fols+df_velo['file'].astype(str)+'.bin'

        first_flag=False

        for index, row in df_velo.iterrows():
            # entire business district is in the test set
            if np.isnan(float(row['northing'])) or np.isnan(float(row['easting'])):
                continue
            elif not first_flag :
                prev_northing, prev_easting = float(row['northing']), float(row['easting'])
                first_flag = True
            
            if(check_submap(float(row['northing']), float(row['easting']), float(prev_northing), float(prev_easting))):
                if(check_in_test_set(float(row['northing']), float(row['easting']), p)):
                    df_test = df_test.append(row, ignore_index=True)
                df_database = df_database.append(row, ignore_index=True)
                prev_northing, prev_easting = float(row['northing']), float(row['easting'])

        # construct KDTree for each folder
        database_tree = KDTree(df_database[['northing','easting']])
        test_tree = KDTree(df_test[['northing','easting']])
        database_trees.append(database_tree)
        test_trees.append(test_tree)

    test_sets = []
    database_sets = []

    # iter all folders for building datasets
    for folder in folders:
        
        database = {}
        test = {}
        velo_file = []

        # read groundtruth in each folder
        df_velo = pd.DataFrame(columns=['file','northing','easting','yaw'])
        gt_filename = "ground_truth/groundtruth_" + folder + '.csv'
        df_locations = pd.read_csv(os.path.join(
            base_path,runs_folder,folder,gt_filename), header=0, names = ['timestamp','northing','easting', 'height','roll','pitch','yaw'], low_memory=False)
        all_filenames = sorted(os.listdir(os.path.join(base_path,runs_folder,folder+pointcloud_fols)))
        
        # get the file name
        for names in all_filenames:
            names = os.path.splitext(names)[0]
            velo_file.append(names)
        
        # convert data type
        df_locations['timestamp'] = df_locations['timestamp'].astype(int)
        df_velo['file'] = velo_file

        for idx in range(len(df_velo)):
            loc_idx = find_closest_timestamp(df_locations['timestamp'].values, int(df_velo['file'][idx]))
            df_velo['yaw'][idx] = df_locations['yaw'][loc_idx]
            df_velo['northing'][idx] = df_locations['northing'][loc_idx]
            df_velo['easting'][idx] = df_locations['easting'][loc_idx]
        
        # change the file dir
        df_velo['file'] = runs_folder+folder + \
            pointcloud_fols+df_velo['file'].astype(str)+'.bin'

        first_flag = False
        
        # construct datasets
        for index,row in df_velo.iterrows():
            if np.isnan(float(row['northing'])) or np.isnan(float(row['easting'])):
                continue
            elif not first_flag :
                prev_northing, prev_easting = float(row['northing']), float(row['easting'])
                first_flag = True
            
            # add all info into database set
            if(check_submap(float(row['northing']), float(row['easting']), float(prev_northing), float(prev_easting))):
                filename = row['file'].replace('.bin','.npy')
                filename = filename.replace(pointcloud_fols,'/velo_trans/')
                if(check_in_test_set(float(row['northing']), float(row['easting']), p)):
                    test[len(test.keys())] = {
                        'query':filename,'northing':row['northing'],'easting':row['easting'],'heading':row['yaw']}

                database[len(database.keys())] = {
                    'query':filename,'northing':row['northing'],'easting':row['easting'],'heading':row['yaw']}
                prev_northing, prev_easting = float(row['northing']), float(row['easting'])
        
        database_sets.append(database)
        test_sets.append(test)
    
    # use kd tree to find evaluation sets
    for i in range(len(database_sets)):
        tree = database_trees[i]

        # pair all database and test sets
        for j in range(len(test_sets)):
            if(i == j):
                continue
            for key in range(len(test_sets[j].keys())):
                # get test set coordination and find its neighbors
                coor = np.array(
                    [[test_sets[j][key]["northing"],test_sets[j][key]["easting"]]])
                
                # get pointcloud index within radius    
                index = tree.query_radius(coor, r=3)
                
                # indices of the positive matches in database i of each query (key) in test set j
                test_sets[j][key][i] = index[0].tolist()
    
    # dump all sets to pickle files for evaluation
    output_to_file(database_sets, output_name+'_evaluation_database.pickle')
    output_to_file(test_sets, output_name+'_evaluation_query.pickle')


# Building database and query files for evaluation
BASE_DIR = "/media/mav-lab/1T/Data/Datasets/NCLT/NCLT/"
base_path = "/media/mav-lab/1T/Data/Datasets/NCLT/NCLT/"

# For Oxford
folders = []
velo_file = []
runs_folder = "/media/mav-lab/1T/Data/Datasets/NCLT/NCLT/"
all_folders = sorted(os.listdir(os.path.join(BASE_DIR,base_path,runs_folder)))

for index in range(len(all_folders)):
    if index == 0:
        continue
    folders.append(all_folders[index])

# folders.append(all_folders[1])

print("folders",folders)
# construct query and database sets
construct_query_and_database_sets(base_path, runs_folder, folders, "/velodyne_sync/",
                                  "ground_truth.csv", p, "nclt")
