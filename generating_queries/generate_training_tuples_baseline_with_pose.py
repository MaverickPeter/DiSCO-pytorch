import os
import pickle
import random
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from icp import *
import math

BASE_DIR = "./pointnetvlad_dataset/benchmark_datasets/"
base_path = "./pointnetvlad_dataset/benchmark_datasets/"

runs_folder = "oxford_test/"
filename = "pointcloud_locations_20m_10overlap.csv"
gps_filename = "ins.csv"

pointcloud_fols = "/pointcloud_20m_10overlap/"

all_folders = sorted(os.listdir(os.path.join(base_path,runs_folder)))

folders = []

# All runs are used for training (both full and partial)
index_list = range(len(all_folders)-1)
print("Number of runs: "+str(len(index_list)))
for index in index_list:
    folders.append(all_folders[index])
print(folders)

#####For training and test data split#####
x_width = 150
y_width = 150
p1 = [5735712.768124,620084.402381]
p2 = [5735611.299219,620540.270327]
p3 = [5735237.358209,620543.094379]
p4 = [5734749.303802,619932.693364]
p = [p1,p2,p3,p4]

icp_num_tests = 15

def isRotationMatrix(R) :

    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)

    return n < 1e-6

def rotationMatrixToEulerAngles(R) :
    # assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def load_pc_file(filename):
    # returns Nx3 matrix
    pc = np.fromfile(os.path.join("./pointnetvlad_dataset/benchmark_datasets/", filename), dtype=np.float64)

    if(pc.shape[0] != 4096*3):
        print("Error in pointcloud shape")
        return np.array([])

    pc = np.reshape(pc,(pc.shape[0]//3, 3))
    return pc


def load_pc_files(filenames):
    pcs = []
    for filename in filenames:
        # print(filename)
        pc = load_pc_file(filename)
        if(pc.shape[0] != 4096):
            continue
        pcs.append(pc)
    pcs = np.array(pcs)
    return pcs


def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    for point in points:
        if(point[0]-x_width < northing and northing < point[0]+x_width and point[1]-y_width < easting and easting < point[1]+y_width):
            in_test_set = True
            break
    return in_test_set
##########################################

def construct_query_dict(df_centroids, filename):
    tree = KDTree(df_centroids[['northing','easting']])
    ind_nn = tree.query_radius(df_centroids[['northing','easting']],r=10)
    ind_r = tree.query_radius(df_centroids[['northing','easting']], r=50)
    queries = {}
    for i in range(len(ind_nn)):

        query = df_centroids.iloc[i]["file"]

        query_yaw = df_centroids.iloc[i]["yaw"]
        heading = []
        positives = np.setdiff1d(ind_nn[i],[i]).tolist()
        
        negatives = np.setdiff1d(
            df_centroids.index.values.tolist(),ind_r[i]).tolist()
        random.shuffle(negatives)
        queries[i] = {"query":query, "heading":query_yaw,
                      "positives":positives,"negatives":negatives}
    # print(query)
    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


# Initialize pandas DataFrame
df_train = pd.DataFrame(columns=['file','northing','easting','yaw'])
df_test = pd.DataFrame(columns=['file','northing','easting','yaw'])

for folder in folders:
    df_locations = pd.read_csv(os.path.join(
        base_path,runs_folder,folder,filename),sep=',')
    df_locations['timestamp'] = runs_folder+folder + \
        pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
    df_locations = df_locations.rename(columns={'timestamp':'file'})

    for index, row in df_locations.iterrows():
        if(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
            df_test = df_test.append(row, ignore_index=True)
        else:
            df_train = df_train.append(row, ignore_index=True)

print("Number of training submaps: "+str(len(df_train['file'])))
print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))
construct_query_dict(df_train,"./training_queries_baseline.pickle")
construct_query_dict(df_test,"./test_queries_baseline.pickle")
