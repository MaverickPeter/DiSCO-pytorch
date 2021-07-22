import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

import config as cfg

#####For training and test data split#####
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


# check if the location is in the test set
def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    for point in points:
        if(point[0]-x_width < northing and northing < point[0]+x_width and point[1]-y_width < easting and easting < point[1]+y_width):
            in_test_set = True
            break
    return in_test_set


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


# construct training tuples
def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename, gps_filename, p, output_name):
    
    database_trees = []
    test_trees = []

    # iter all folders for building KDTree
    for folder in folders:
        print(folder)

        # construct dataframe
        df_database = pd.DataFrame(columns=['file','northing','easting','yaw'])
        df_test = pd.DataFrame(columns=['file','northing','easting','yaw'])

        # read csv in each folder
        df_locations = pd.read_csv(os.path.join(
            base_path,runs_folder,folder,filename),sep=',')
        
        # construct database and test set 
        for index, row in df_locations.iterrows():
            # entire business district is in the test set
            if(output_name == "business"):
                df_test = df_test.append(row, ignore_index=True)
            elif(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
                df_test = df_test.append(row, ignore_index=True)
            df_database = df_database.append(row, ignore_index=True)

        # build KDTree for neighborhood finding
        database_tree = KDTree(df_database[['northing','easting']])
        test_tree = KDTree(df_test[['northing','easting']])
        
        # different folder hold different kdtree
        database_trees.append(database_tree)
        test_trees.append(test_tree)

    test_sets = []
    database_sets = []

    # iter all folders for building datasets
    for folder in folders:
        
        database = {}
        test = {}
        
        # read csv in each folder
        df_locations = pd.read_csv(os.path.join(
            base_path,runs_folder,folder,filename),sep=',')

        # read gps in each folder gps folder exist
        df_gps = pd.read_csv(os.path.join(
            base_path,runs_folder,folder,gps_filename),sep=',')

        # convert data type and initialize yaw. !!Note that the gps file is to determine the yaw
        df_locations['timestamp'] = df_locations['timestamp'].astype(str)
        df_locations['yaw'] = 0.0
        
        # for dataset with no timestamp sync process, we find closest timestamp by ourselves
        for idx in range(len(df_locations)):
            loc_idx = find_closest_timestamp(df_gps['timestamp'].values, int(df_locations['timestamp'][idx]))
            df_locations['yaw'][idx] = df_gps['yaw'][loc_idx]
        
        # save pointcloud bin file use 'timestamp' frame and change the column name to 'file'
        df_locations['timestamp'] = runs_folder+folder + \
            pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
        df_locations = df_locations.rename(columns={'timestamp':'file'})
        
        # construct datasets
        for index,row in df_locations.iterrows():
            # entire business district is in the test set
            if(output_name == "business"):
                test[len(test.keys())] = {
                    'query':row['file'],'northing':row['northing'],'easting':row['easting'],'heading':row['yaw']}
            elif(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
                # if in test region, add all info to the test set
                test[len(test.keys())] = {
                    'query':row['file'],'northing':row['northing'],'easting':row['easting'],'heading':row['yaw']}
            # add all info into database set
            database[len(database.keys())] = {
                'query':row['file'],'northing':row['northing'],'easting':row['easting'],'heading':row['yaw']}
        
        # build database and test sets for all folders
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
                index = tree.query_radius(coor, r=25)
                
                # indices of the positive matches in database i of each query (key) in test set j
                test_sets[j][key][i] = index[0].tolist()

    # dump all sets to pickle files for evaluation
    output_to_file(database_sets, output_name+'_evaluation_database.pickle')
    output_to_file(test_sets, output_name+'_evaluation_query.pickle')


# Building database and query files for evaluation
BASE_DIR = "./pointnetvlad_dataset/"
base_path = "./pointnetvlad_dataset/benchmark_datasets/"

# For Oxford
folders = []
runs_folder = "oxford_test/"
all_folders = sorted(os.listdir(os.path.join(BASE_DIR,base_path,runs_folder)))
index_list = [5,6,7,9,10,11,12,13,14,15,16,17,18,19,22,24,31,32,33,38,39,43,44]
print(len(index_list))
print(all_folders)

for index in index_list:
    folders.append(all_folders[index])

print(folders)

# construct query and database sets
construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_20m/",
                                  "pointcloud_locations_20m.csv", "gps/ins.csv", p_dict["oxford"], "oxford")


