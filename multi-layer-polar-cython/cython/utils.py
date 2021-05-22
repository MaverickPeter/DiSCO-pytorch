import numpy as np
import torch
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from dot_cython import pt2rs
# import sys
# sys.setrecursionlimit(1000000)

def xy2theta(x, y):
    (b,c) = x.shape
    theta = np.zeros((b,c))

    for i in range(b):
        if (x[i,...] >= 0 and y[i,...] >= 0): # 1 1 
            theta[i,...] = 180/np.pi * np.arctan(y[i,...]/x[i,...])
        if (x[i,...] < 0 and y[i,...] >= 0): # -1 1 
            theta[i,...] = 180 - ((180/np.pi) * np.arctan(y[i,...]/(-x[i,...])))
        if (x[i,...] < 0 and y[i,...] < 0): # -1 -1 
            theta[i,...] = 180 + ((180/np.pi) * np.arctan(y[i,...]/x[i,...]))
        if (x[i,...] >= 0 and y[i,...] < 0): # 1 -1 
            theta[i,...] = 360 - ((180/np.pi) * np.arctan((-y[i,...])/x[i,...]))
    return theta

def pt2rs(point, gap_ring, gap_sector, num_ring, num_sector):
    # print("begin", type(point), type(gap_ring),type(gap_sector),type(num_ring),type(num_sector))

    (b,c,d) = point.shape
    x = point[...,0]
    y = point[...,1]
    z = point[...,2]
    print("x:",x)
    print("y:",y)
    print("z:",z)
    for i in range(b):
        if(x[i,...] == 0.0):
            x[i,...] = 0.001
        if(y[i,...] == 0.0):
            y[i,...] = 0.001
    
    theta = xy2theta(x, y)
    faraway = np.sqrt(x**2 + y**2)
    # idx_ring = np.divmod(faraway, gap_ring)[0]       
    # idx_sector = np.divmod(theta, gap_sector)[0]
    idx_ring = np.floor_divide(faraway, gap_ring)
    idx_sector = np.floor_divide(theta, gap_sector)
    
    for i in range(b):
        if(idx_ring[i,...] >= num_ring):
            idx_ring[i,...] = num_ring-1 # python starts with 0 and ends with N-1
    
    idx_ring = idx_ring.astype(np.int32)
    idx_sector = idx_sector.astype(np.int32)
    # print("idx_ring", idx_ring.shape, idx_sector.shape)
    return idx_ring, idx_sector

def point2gridmap(ptcloud, max_length, num_ring, num_sector, enough_large):
    
    # ptcloud = ptcloud.cpu().numpy()
    # ptcloud = np.asarray(ptcloud)
    (b,c,num_points,d) = ptcloud.shape
    # print("batch size: ",ptcloud.shape)
    
    gap_ring = max_length/num_ring
    gap_sector = 360/num_sector
    
    # storage = []
    counter = np.zeros([b, num_ring*num_sector])
    counter.astype(np.int32)
    # grid = np.zeros([b, num_points])-1
    grid = np.zeros([b, num_ring*num_sector, enough_large])
    mask = np.zeros([b, num_ring*num_sector, enough_large])

    
    for pt_idx in range(num_points):
        point = ptcloud[:,:,pt_idx,:]

        idx_ring, idx_sector = pt2rs(point, gap_ring, gap_sector, num_ring, num_sector)
        # idx_ring, idx_sector = pt2rs(point, gap_ring, gap_sector, num_ring, num_sector)
        print("idx_ring",idx_ring)
        print("idx_sector",idx_sector)
        # for i in range(b):
        #     grid[i, pt_idx] = idx_ring[i,...] * idx_sector[i,...] + idx_ring[i,...]
        idx_ring = idx_ring.astype(np.int32)
        idx_sector = idx_sector.astype(np.int32)
        for i in range(b):
            if counter[i, idx_ring[i,...] * idx_sector[i,...] + idx_ring[i,...]] >= enough_large:
                continue
            pt_count = int(counter[i, idx_ring[i,...] * idx_sector[i,...] + idx_ring[i,...]])
            grid[i, idx_ring[i,...] * idx_sector[i,...] + idx_ring[i,...], pt_count] = pt_idx
            mask[i, idx_ring[i,...] * idx_sector[i,...] + idx_ring[i,...], pt_count] = 1
            counter[i, idx_ring[i,...] * idx_sector[i,...] + idx_ring[i,...]] += 1
            # print("loc",idx_ring[i,...] * idx_sector[i,...] + idx_ring[i,...], pt_count)
    # pyplot_draw_point_cloud(grid)
    return grid, mask

def pyplot_draw_point_cloud(points, output_filename=None):
    """ points is a Nx3 numpy array """
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

# def xy2theta(x, y):
#     (b,c) = x.shape
#     theta = torch.zeros((b,c))
#     for i in range(b):
#         if (x[i,...] >= 0 and y[i,...] >= 0): 
#             theta[i,...] = 180/np.pi * torch.atan(y[i,...]/x[i,...])
#         if (x[i,...] < 0 and y[i,...] >= 0): 
#             theta[i,...] = 180 - ((180/np.pi) * torch.atan(y[i,...]/(-x[i,...])))
#         if (x[i,...] < 0 and y[i,...] < 0): 
#             theta[i,...] = 180 + ((180/np.pi) * torch.atan(y[i,...]/x[i,...]))
#         if (x[i,...] >= 0 and y[i,...] < 0):
#             theta[i,...] = 360 - ((180/np.pi) * torch.atan((-y[i,...])/x[i,...]))

#     return theta

# def pt2rs(point, gap_ring, gap_sector, num_ring, num_sector):
#     (b,c,d) = point.shape
#     x = point[...,0]
#     y = point[...,1]
#     z = point[...,2]

#     for i in range(b):
#         if(x[i,...] == 0.0):
#             x[i,...] = 0.001
#         if(y[i,...] == 0.0):
#             y[i,...] = 0.001
    
#     theta = xy2theta(x, y)
#     faraway = torch.sqrt(x*x + y*y)

#     # idx_ring = np.divmod(faraway, gap_ring)[0]       
#     # idx_sector = np.divmod(theta, gap_sector)[0]
#     idx_ring = torch.fmod(faraway, gap_ring)[0]       
#     idx_sector = torch.fmod(theta, gap_sector)[0]

#     for i in range(b):
#         if(idx_ring[i,...] >= num_ring):
#             idx_ring[i,...] = num_ring-1 # python starts with 0 and ends with N-1
    
#     idx_ring = idx_ring.long()
#     idx_sector = idx_sector.long()

#     return idx_ring, idx_sector

# def point2gridmap(ptcloud, max_length, num_ring, num_sector, enough_large):
#     ptcloud = ptcloud.cuda()
#     (b,c,num_points,d) = ptcloud.shape
    
#     gap_ring = max_length/num_ring
#     gap_sector = 360/num_sector
    
#     # storage = []
#     counter = torch.zeros([b, num_ring*num_sector]).cuda().long()
#     # grid = np.zeros([b, num_points])-1
#     grid = torch.zeros([b, num_ring*num_sector, enough_large]).cuda().long()
#     mask = torch.zeros([b, num_ring*num_sector, enough_large]).cuda().long()

#     for pt_idx in range(num_points):
#         point = ptcloud[:,:,pt_idx,:]
#         point_height = point[...,2]

#         idx_ring, idx_sector = pt2rs(point, gap_ring, gap_sector, num_ring, num_sector)
#         # for i in range(b):
#         #     grid[i, pt_idx] = idx_ring[i,...] * idx_sector[i,...] + idx_ring[i,...]
        
#         for i in range(b):
#             if counter[i, idx_ring[i,...] * idx_sector[i,...] + idx_ring[i,...]] >= enough_large:
#                 continue
#             pt_count = int(counter[i, idx_ring[i,...] * idx_sector[i,...] + idx_ring[i,...]])
#             grid[i, idx_ring[i,...] * idx_sector[i,...] + idx_ring[i,...], pt_count] = pt_idx
#             mask[i, idx_ring[i,...] * idx_sector[i,...] + idx_ring[i,...], pt_count] = 1
#             counter[i, idx_ring[i,...] * idx_sector[i,...] + idx_ring[i,...]] += 1
#             # grid[i, idx_ring[i,...], idx_sector[i,...]] = pt_idx

#     return grid, mask


# num_points = 4096
# np.random.seed(666)
# sim_data = np.random.random(size=(1,num_points,3)).astype(np.float32) *100
# sim_data = torch.from_numpy(sim_data)
# print(sim_data)
# sim_data = sim_data.unsqueeze(1)
# # sim_data = Variable(torch.randn(1, 1, num_points, 3)*100)
# sim_data = sim_data.cuda()

# num_sector = 60
# num_ring = 20
# max_length = 20
# enough_large = 2
# time_start=time.time()
# grid, mask = point2gridmap(sim_data, max_length, num_ring, num_sector, enough_large)
# time_netend=time.time()
# print('forward time cost',time_netend-time_start,'s')
# print(grid.reshape(-1))
# print(np.max(grid))
# print(grid.requires_grad)
