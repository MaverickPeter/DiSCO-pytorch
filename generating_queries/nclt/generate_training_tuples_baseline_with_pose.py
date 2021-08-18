import os
import pickle
import random
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import struct
import config as cfg
import gputransform


BASE_DIR = "/media/mav-lab/1T/Data/Datasets/NCLT/NCLT/"
base_path = "/media/mav-lab/1T/Data/Datasets/NCLT/NCLT/"
runs_folder = "/media/mav-lab/1T/Data/Datasets/NCLT/NCLT/"
pointcloud_fols = "/velodyne_data/velodyne_sync/"

all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))

folders = []
velo_file = []

# All runs are used for training (both full and partial)
index_list = range(len(all_folders)-1)
print("Number of runs: " + str(len(index_list)))

for index in index_list:
    if index == 0:
        folders.append(all_folders[index])
print(folders)

p = [-50.0, 150.0, -250.0, 150.0]

# check if the location is in the test set
def check_in_test_set(northing, easting, points):
    in_test_set = False
    if(points[0] < northing and northing < points[1] and points[2] < easting and easting < points[3]):
        in_test_set = True
    return in_test_set


# check if it's a new place in test set
def check_submap_test(northing, easting, prev_northing, prev_easting):
    is_submap = False
    euclidean = np.abs(np.sqrt((prev_northing-northing)**2 + (prev_easting-easting)**2))
    if(euclidean < cfg.SUBMAP_INTERVAL_TEST + 0.5 and euclidean >= cfg.SUBMAP_INTERVAL_TEST):
        is_submap = True
    return is_submap


# check if it's a new place in train set
def check_submap_train(northing, easting, prev_northing, prev_easting):
    is_submap = False
    euclidean = np.abs(np.sqrt((prev_northing-northing)**2 + (prev_easting-easting)**2))
    if(euclidean < cfg.SUBMAP_INTERVAL_TRAIN + 1.0 and euclidean >= cfg.SUBMAP_INTERVAL_TRAIN):
        is_submap = True
    return is_submap


# find closest place timestamp with index returned
def find_closest_timestamp(A, target):
    # A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx


# nclt pointcloud utils
def convert(x_s, y_s, z_s):
    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z
    

# load lidar file in nclt dataset
def load_lidar_file_nclt(file_path):
    n_vec = 4
    f_bin = open(file_path,'rb')

    hits = []

    while True:

        x_str = f_bin.read(2)
        if x_str == b"": # eof
            break

        x = struct.unpack('<H', x_str)[0]
        y = struct.unpack('<H', f_bin.read(2))[0]
        z = struct.unpack('<H', f_bin.read(2))[0]
        i = struct.unpack('B', f_bin.read(1))[0]
        l = struct.unpack('B', f_bin.read(1))[0]

        x, y, z = convert(x, y, z)
        s = "%5.3f, %5.3f, %5.3f, %d, %d" % (x, y, z, i, l)

        # filter and normalize the point cloud to -1 ~ 1
        if np.abs(x) < 70. and z > -20. and z < -2. and np.abs(y) < 70. and not(np.abs(x) < 5. and np.abs(y) < 5.):
            hits += [[x/70., y/70., z/20.]]

    f_bin.close()
    hits = np.asarray(hits)
    hits[:, 2] = -hits[:, 2]

    return hits



# load pointcloud and process it using CUDA accelerate 
def load_pc_file(filename):
    # returns Nx3 matrix
    # scale the original pointcloud 
    pc = load_lidar_file_nclt(os.path.join("/media/mav-lab/1T/Data/Datasets/NCLT/NCLT/", filename))
    pc[:,0] = pc[:,0] / np.max(pc[:,0] + 1e-15) - 0.0001
    pc[:,1] = pc[:,1] / np.max(pc[:,1] + 1e-15) - 0.0001
    pc[:,2] = pc[:,2] / np.max(pc[:,2] + 1e-15) - 0.0001

    # !Debug
    # x = pc[...,0]
    # y = pc[...,1]
    # z = pc[...,2]
    # fig2 = plt.figure()
    # ax2 = Axes3D(fig2)
    # ax2.scatter(x, y, z)
    # plt.show()

    size = pc.shape[0]
    pc_img = np.zeros([cfg.num_height * cfg.num_ring * cfg.num_sector])
    pc = pc.transpose().flatten().astype(np.float32)

    transer = gputransform.GPUTransformer(pc, size, cfg.max_length, cfg.max_height, cfg.num_ring, cfg.num_sector, cfg.num_height, 1)
    transer.transform()
    point_t = transer.retreive()
    point_t = point_t.reshape(-1, 3)
    point_t = point_t[...,2]
    pc_img = point_t.reshape(cfg.num_height, cfg.num_ring, cfg.num_sector)

    pc = np.sum(pc_img, axis=0)
    # plt.imshow(pc)
    # plt.show()
    return pc_img


# construct query dict for training
def construct_query_dict(df_centroids, filename, pickle_flag):
    tree = KDTree(df_centroids[['northing','easting']])
    
    # get neighbors pair
    ind_nn = tree.query_radius(df_centroids[['northing','easting']],r=1.5)
    
    # get far away pairs
    ind_r = tree.query_radius(df_centroids[['northing','easting']], r=2)
    queries = {}
    print("ind_nn",len(ind_nn))
    print("ind_r",len(ind_r))

    for i in range(len(ind_nn)):
        print("index",i,' / ',len(ind_nn))
        
        # get query info
        query = df_centroids.iloc[i]["file"]

        # get yaw info of this query
        query_yaw = df_centroids.iloc[i]["yaw"]

        # get positive filename and shuffle
        positives = np.setdiff1d(ind_nn[i],[i]).tolist()
        random.shuffle(positives)
        # positives = positives[0:2]

        # get negative filename and shuffle
        negatives = np.setdiff1d(
            df_centroids.index.values.tolist(),ind_r[i]).tolist()
        random.shuffle(negatives)
        # negatives = negatives[0:50]
        
        # add all info to query dict
        queries[i] = {"query":query, "heading":query_yaw,
                      "positives":positives,"negatives":negatives}

    # dump all queries into pickle file for training
    if pickle_flag:
        with open(filename, 'wb') as handle:
            pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Done ", filename)


# Initialize pandas DataFrame
df_train = pd.DataFrame(columns=['file','northing','easting','yaw'])
df_test = pd.DataFrame(columns=['file','northing','easting','yaw'])
df_all = pd.DataFrame(columns=['file','northing','easting','yaw'])


for folder in folders:
    print(folder)

    velo_file = []

    if folder == folders[0]:
        save_flag = True
    else:
        save_flag = False

    df_velo = pd.DataFrame(columns=['file','northing','easting','yaw'])

    # get groundtruth file and load it
    gt_filename = "ground_truth/groundtruth_" + folder + '.csv'
    df_locations = pd.read_csv(os.path.join(
        base_path,runs_folder,folder,gt_filename), header=0, names = ['timestamp','northing','easting', 'height','roll','pitch','yaw'], low_memory=False)
    all_filenames = sorted(os.listdir(os.path.join(base_path, runs_folder, folder + pointcloud_fols)))
    
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
    
    # get full path of the point cloud files
    df_velo['file'] = runs_folder+folder + \
        pointcloud_fols+df_velo['file'].astype(str)+'.bin'

    # x = df_velo['northing']
    # y = df_velo['easting']
    # plt.scatter(x,y)
    # plt.show()
    first_flag = False


    for index, row in df_velo.iterrows():
        print(row['file'])
        print("index", index, ' / ', len(df_velo))
        # for not nan value and very first ones (which often wrong)
        if np.isnan(float(row['northing'])) or np.isnan(float(row['easting'])):
            continue
        elif not first_flag :
            prev_northing, prev_easting = float(row['northing']), float(row['easting'])
            first_flag = True          

        if save_flag:
            if(check_submap_train(float(row['northing']), float(row['easting']), float(prev_northing), float(prev_easting))):
                # process point cloud and save
                velo = load_pc_file(row['file'])
                save_name = row['file'].replace('.bin','.npy')
                row['file'] = row['file'].replace('.bin','.npy')
                save_name = save_name.replace(pointcloud_fols, cfg.TRAIN_FOLDER)
                row['file'] = row['file'].replace(pointcloud_fols, cfg.TRAIN_FOLDER)
                np.save(save_name, velo)

                if(check_in_test_set(float(row['northing']), float(row['easting']), p)):
                    df_test = df_test.append(row, ignore_index=True)
                else:
                    df_train = df_train.append(row, ignore_index=True)

                prev_northing, prev_easting = float(row['northing']), float(row['easting'])

        else:
            if(check_submap_test(float(row['northing']), float(row['easting']), float(prev_northing), float(prev_easting))):
                # process point cloud and save
                velo = load_pc_file(row['file'])
                save_name = row['file'].replace('.bin','.npy')
                row['file'] = row['file'].replace('.bin','.npy')
                save_name = save_name.replace(pointcloud_fols, cfg.TEST_FOLDER)
                row['file'] = row['file'].replace(pointcloud_fols, cfg.TEST_FOLDER)
                np.save(save_name, velo)
                
                if(check_in_test_set(float(row['northing']), float(row['easting']), p)):
                    df_test = df_test.append(row, ignore_index=True)
                else:
                    df_train = df_train.append(row, ignore_index=True)

                prev_northing, prev_easting = float(row['northing']), float(row['easting'])

    if save_flag:
        print("Number of training submaps: "+str(len(df_train['file'])))
        print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))
        construct_query_dict(df_train, "./training_queries_baseline_" + cfg.EXPERIMENT_NAME + ".pickle", pickle_flag=save_flag)
        construct_query_dict(df_test, "./test_queries_baseline_" + cfg.EXPERIMENT_NAME + ".pickle", pickle_flag=save_flag)

        gt_train_filename = "gt_" + cfg.EXPERIMENT_NAME + "_0.5m.csv"
        df_train.to_csv(os.path.join(base_path,runs_folder,folder,gt_train_filename))
        gt_test_filename = "gt_" + cfg.EXPERIMENT_NAME + "_test_0.5m.csv"
        df_test.to_csv(os.path.join(base_path,runs_folder,folder,gt_test_filename))
    
    else:          
        gt_train_filename = "gt_" + cfg.EXPERIMENT_NAME + "_3m.csv"
        df_train.to_csv(os.path.join(base_path,runs_folder,folder,gt_train_filename))
        gt_test_filename = "gt_" + cfg.EXPERIMENT_NAME + "_test_3m.csv"
        df_test.to_csv(os.path.join(base_path,runs_folder,folder,gt_test_filename))
