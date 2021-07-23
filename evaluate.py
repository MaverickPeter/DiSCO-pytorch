import argparse
import math
import numpy as np
import socket
import importlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
import cv2
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from torchvision import transforms, utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from loading_pointclouds import *
import models.DiSCO as SC
from tensorboardX import SummaryWriter
import loss.loss_function
import gputransform
import config as cfg
import scipy.io as scio

cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():
    model = SC.DiSCO(output_dim = cfg.FEATURE_OUTPUT_DIM)
    corr2soft = SC.Corr2Softmax(200., 0.)
    corr2soft = corr2soft.to(device)
    model = model.to(device)

    resume_filename = cfg.LOG_DIR + cfg.MODEL_FILENAME
    print("Resuming From ", resume_filename)
    checkpoint = torch.load(resume_filename)
    saved_state_dict = checkpoint['state_dict']
    saved_corr2soft_dict = checkpoint['corr2soft']

    model.load_state_dict(saved_state_dict)
    corr2soft.load_state_dict(saved_corr2soft_dict)
    model = nn.DataParallel(model)

    print(evaluate_model(model, corr2soft))


def evaluate_model(model, corr2soft):
    DATABASE_SETS = get_sets_dict(cfg.EVAL_DATABASE_FILE)
    QUERY_SETS = get_sets_dict(cfg.EVAL_QUERY_FILE)
    print("database set: ",len(DATABASE_SETS))
    print("query set: ",len(QUERY_SETS))

    if not os.path.exists(cfg.RESULTS_FOLDER):
        os.mkdir(cfg.RESULTS_FOLDER)

    recall = np.zeros(25)
    count = 0
    similarity = []
    yaw_err_mean = []
    yaw_err_std = []
    one_percent_recall = []
    fail_total = 0
    total = 0

    DATABASE_VECTORS = []
    QUERY_VECTORS = []
    FFT_DATABASE_VECTORS = []
    FFT_QUERY_VECTORS = []
    YAW_GT_DATABASE = []
    YAW_GT_QUERY = []

    for i in range(len(DATABASE_SETS)):
        q_out, fft_out, yaw, sig_output = get_latent_vectors(model, DATABASE_SETS[i])
        DATABASE_VECTORS.append(q_out)
        FFT_DATABASE_VECTORS.append(fft_out)
        YAW_GT_DATABASE.append(yaw)

    for j in range(len(QUERY_SETS)):
        q_out, fft_out, yaw, sig_output = get_latent_vectors(model, QUERY_SETS[j])
        QUERY_VECTORS.append(q_out)
        FFT_QUERY_VECTORS.append(fft_out)
        YAW_GT_QUERY.append(yaw)

    for m in range(len(QUERY_SETS)):
        for n in range(len(QUERY_SETS)):
            if (m == n):
                continue
            pair_recall, pair_similarity, pair_opr, err_mean, err_std, fail_count, eva_count = get_recall(
                m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS, FFT_DATABASE_VECTORS, FFT_QUERY_VECTORS, YAW_GT_DATABASE, YAW_GT_QUERY, corr2soft)
            yaw_err_mean.append(err_mean)
            yaw_err_std.append(err_std)
            
            recall += np.array(pair_recall)
            fail_total += fail_count
            count += 1
            total += eva_count

            print("process", count/len(QUERY_SETS)/len(DATABASE_SETS) * 100, flush=True)
            one_percent_recall.append(pair_opr)
            
            for x in pair_similarity:
                similarity.append(x)

    print()
    ave_recall = recall / count
    ave_err_mean = np.mean(yaw_err_mean)
    ave_err_std = np.mean(yaw_err_std)

    average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)

    fail_total /= total
    fail_total = 1 - fail_total

    with open(cfg.OUTPUT_FILE, "w") as output:
        output.write("Average Recall @N:\n")
        output.write(str(ave_recall))
        output.write("\n\n")
        output.write("Average Similarity:\n")
        output.write(str(average_similarity))
        output.write("\n\n")
        output.write("Average Top 1% Recall:\n")
        output.write(str(ave_one_percent_recall))
        output.write("\n\n")
        output.write("Average Err Mean:\n")
        output.write(str(ave_err_mean))
        output.write("\n\n")
        output.write("Average Err Std:\n")
        output.write(str(ave_err_std))
        output.write("\n\n")
        output.write("Succes rate:\n")
        output.write(str(fail_total))
    return ave_one_percent_recall


def get_latent_vectors(model, dict_to_process):

    model.eval()
    is_training = False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = cfg.EVAL_BATCH_SIZE * \
        (1 + cfg.EVAL_POSITIVES_PER_QUERY + cfg.EVAL_NEGATIVES_PER_QUERY)
    q_output = []
    fft_output = []
    yaw_output = []
    sig_output = []
    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = []
        heading = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
            heading.append(dict_to_process[index]["heading"])
        queries = load_pc_files(file_names)
        queries = np.array(queries, dtype=np.float32)
        heading = np.array(heading)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.to(device)
            feed_tensor = feed_tensor.view((-1, cfg.num_height, cfg.num_ring, cfg.num_sector))
            out, outfft, fft_result, unet_out = model(feed_tensor)

        fft_result = fft_result.detach().cpu().numpy()
        fft_result = np.squeeze(fft_result)
        outfft = outfft.detach().cpu().numpy()
        outfft = np.squeeze(outfft)
        out = out.detach().cpu().numpy()
        out = np.squeeze(out)
        sig_output.append(outfft)
        fft_output.append(fft_result)
        q_output.append(out)
        yaw_output.append(heading)

    fft_output = np.array(fft_output)
    sig_output = np.array(sig_output)
    yaw_output = np.array(yaw_output)
    q_output = np.array(q_output) #[25,17,40,120]

    if(len(q_output) != 0):
        sig_output = sig_output.reshape(-1, cfg.num_ring, cfg.num_sector)
        fft_output = fft_output.reshape(-1, cfg.num_ring, cfg.num_sector, 2)
        q_output = q_output.reshape(-1, cfg.FEATURE_OUTPUT_DIM)
        yaw_output = yaw_output.reshape(-1,1)

    # handle edge case
    index_edge = len(train_file_idxs) // batch_num * batch_num
    if index_edge < len(dict_to_process.keys()):
        file_indices = train_file_idxs[index_edge:len(dict_to_process.keys())]
        file_names = []
        heading = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
            heading.append(dict_to_process[index]["heading"])

        queries = load_pc_files(file_names)
        queries = np.array(queries, dtype=np.float32)
        heading_edge = np.array(heading)
        heading_edge = heading_edge.reshape(-1,1)

        randomYaw = (np.random.rand(heading_edge.shape[0],1) - 0.5) * 90.
        heading_edge = heading_edge / np.pi * 180.
        heading_edge += randomYaw

        for b in range(queries.shape[0]):
            for dims in range(queries.shape[1]):
                queries[b,dims,...] = rotation_on_SCI(queries[b,dims,...], randomYaw[b,0])

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.to(device)
            feed_tensor = feed_tensor.view((-1, cfg.num_height, cfg.num_ring, cfg.num_sector))
            o1, outfft, fft_result, unet_out = model(feed_tensor)
        
        fft_result = fft_result.detach().cpu().numpy()
        fft = fft_result.squeeze(1)
        outfft = outfft.detach().cpu().numpy()
        outsig = outfft.squeeze(1)
        output = o1.detach().cpu().numpy()
        output = output.reshape(output.shape[0],-1)

        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
            fft_output = np.vstack((fft_output, fft))
            sig_output = np.vstack((sig_output,outsig))
            yaw_output = np.vstack((yaw_output, heading_edge))
        else:
            q_output = output
            fft_output = fft
            sig_output = outsig
            yaw_output = heading_edge

    model.train()

    return q_output, fft_output, yaw_output, sig_output


def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image, cmap='jet')
    plt.show()


def get_recall(m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS, FFT_DATABASE_VECTORS, FFT_QUERY_VECTORS, YAW_GT_DATABASE, YAW_GT_QUERY, corr2soft):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    database_output = DATABASE_VECTORS[m]
    database_fft = FFT_DATABASE_VECTORS[m]
    gt_yaw_database = YAW_GT_DATABASE[m]
    gt_yaw_query = YAW_GT_QUERY[n]
    query_fft = FFT_QUERY_VECTORS[n]

    queries_output = QUERY_VECTORS[n]
    database_nbrs = KDTree(database_output)
    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    yaw_err = []
    one_percent_retrieved = 0
    yaw_err_mean = 0
    yaw_err_std = 0
    count = 0
    fail_count = 0
    
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):

        true_neighbors = QUERY_SETS[n][i][m]
        if(len(true_neighbors) == 0):
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if(j == 0):
                    count += 1
                    gt_angle = GT_sc_angle_convert(-gt_yaw_query[i] + gt_yaw_database[indices[0][j]], cfg.num_sector)
                    angle, _ = phase_corr(database_fft[indices[0][j]], query_fft[i], device, corr2soft)
                    angle = angle.detach().cpu().numpy()
                    
                    error = np.min([np.abs(gt_angle - angle), np.abs(np.abs((gt_angle - angle))-cfg.num_sector)])
                    error = error / cfg.num_sector * 360.
                    
                    yaw_err_mean += error
                    yaw_err.append(error)

                    similarity = np.dot(
                        queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    if count!=0:
        yaw_err_mean /= count
    else:
        yaw_err_mean = 0.1
    std = 0.
    
    for k in range(len(yaw_err)):
        std += np.power((yaw_err[k] - yaw_err_mean), 2)
    if len(yaw_err)!=0:
        yaw_err_std = np.sqrt( std / len(yaw_err))
    else:
        yaw_err_std = 0.1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100

    return recall, top1_similarity_score, one_percent_recall, yaw_err_mean, yaw_err_std, fail_count, count


def GT_angle_convert(this_gt, size):
    for batch_num in range(this_gt.shape[0]):
        if this_gt[batch_num] < 0:
            this_gt[batch_num] = -this_gt[batch_num]

        if this_gt[batch_num] >= 90:
            this_gt[batch_num] = cfg.num_ring - (this_gt[batch_num] - 90.) / 180 * size
        elif this_gt[batch_num] >= 0 and this_gt[batch_num] < 90:
            this_gt[batch_num] = (90 - this_gt[batch_num]) * size / 180
        
        this_gt[batch_num] = this_gt[batch_num] // 1 + (this_gt[batch_num] % 1 + 0.5)//1
        if this_gt[batch_num] == size:
            this_gt[batch_num] = this_gt[batch_num] - 1
    return this_gt


def GT_sc_angle_convert(gt_yaw, size):
    gt_yaw = gt_yaw % 360
    if gt_yaw > 180:
        gt_yaw -= 360
    elif gt_yaw < -180:
        gt_yaw += 360
    
    gt_angle = gt_yaw
    for batch_num in range(gt_angle.shape[0]):            
        if gt_angle[batch_num] <= -180.:
            gt_angle[batch_num] = gt_angle[batch_num] + 540.
        elif gt_angle[batch_num] >= 180.:
            gt_angle[batch_num] = gt_angle[batch_num] - 180.
        else:
            gt_angle[batch_num] = gt_angle[batch_num] + 180.
    gt_angle = np.ceil(gt_angle * float(cfg.num_sector) / 360.) - 1.
    return gt_angle


def fftshift2d(x):
    for dim in range(1, len(x.size())):
        n_shift = x.size(dim)//2
        if x.size(dim) % 2 != 0:
            n_shift = n_shift + 1  # for odd-sized images
        x = roll_n(x, axis=dim, n=n_shift)
    return x  # last dim=2 (real&imag)


def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def phase_corr(a, b, device, corr2soft):
    # a: template; b: source
    # imshow(a.squeeze(0).float())
    # [B, 1, cfg.num_ring, cfg.num_sector, 2]
    eps = 1e-15

    real_a = torch.from_numpy(a[...,0]).to(device)
    real_b = torch.from_numpy(b[...,0]).to(device)
    imag_a = torch.from_numpy(a[...,1]).to(device)
    imag_b = torch.from_numpy(b[...,1]).to(device)

    # compute a * b.conjugate; shape=[B,H,W,C]
    R = torch.FloatTensor(1, 1, cfg.num_ring, cfg.num_sector, 2).to(device)
    R[...,0] = real_a * real_b + imag_a * imag_b
    R[...,1] = real_a * imag_b - real_b * imag_a

    r0 = torch.sqrt(real_a ** 2 + imag_a ** 2 + eps) * torch.sqrt(real_b ** 2 + imag_b ** 2 + eps).to(device)
    R[...,0] = R[...,0].clone()/(r0 + eps).to(device)
    R[...,1] = R[...,1].clone()/(r0 + eps).to(device)

    corr = torch.ifft(R, 2)
    corr_real = corr[...,0]
    corr_imag = corr[...,1]
    corr = torch.sqrt(corr_real ** 2 + corr_imag ** 2 + eps)
    corr = fftshift2d(corr)

    corr = corr.squeeze(1)
    corr_wb = corr2soft(corr)
    corr_ang = torch.sum(corr_wb, 1, keepdim=False)

    angle = torch.argmax(corr)
    angle = angle % cfg.num_sector

    return angle, corr


def rotation_on_SCI(sc, rotation):
    # rotation to translation [-180:180] -> [-cfg.num_sector//2:cfg.num_sector//2]
    if rotation > 0:
        t = rotation / 180. * (cfg.num_sector // 2)
        t = np.floor(t).astype(int)
        patch = sc[:, (cfg.num_sector-t):cfg.num_sector]
        col, row = cfg.num_sector, cfg.num_ring
        center = (col // 2, row // 2)
        t_x, t_y = t, 0.

        M = cv2.getRotationMatrix2D(center, 0.0, 1.0)
        sc = cv2.warpAffine(sc, M, (col, row))

        N = np.float32([[1,0,t_x],[0,1,t_y]])
        sc = cv2.warpAffine(sc, N, (col, row))
        sc[:, 0:t] = patch
    else:
        t = -rotation / 180. * (cfg.num_sector // 2)
        t = np.floor(t).astype(int)
        patch = sc[:, 0:t]
        col, row = cfg.num_sector, cfg.num_ring
        center = (col // 2, row // 2)
        t_x, t_y = -t, 0.

        M = cv2.getRotationMatrix2D(center, 0.0, 1.0)
        sc = cv2.warpAffine(sc, M, (col, row))

        N = np.float32([[1,0,t_x],[0,1,t_y]])
        sc = cv2.warpAffine(sc, N, (col, row))
        sc[:, (cfg.num_sector-t):cfg.num_sector] = patch
        # plt.imshow(sc)
        # plt.show()
    return sc


if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--positives_per_query', type=int, default=4,
                        help='Number of potential positives in each training tuple [default: 2]')
    parser.add_argument('--negatives_per_query', type=int, default=12,
                        help='Number of definite negatives in each training tuple [default: 20]')
    parser.add_argument('--eval_batch_size', type=int, default=2,
                        help='Batch Size during training [default: 1]')
    parser.add_argument('--dimension', type=int, default=256)
    parser.add_argument('--decay_step', type=int, default=200000,
                        help='Decay step for lr decay [default: 200000]')
    parser.add_argument('--decay_rate', type=float, default=0.7,
                        help='Decay rate for lr decay [default: 0.8]')
    parser.add_argument('--results_dir', default='results/',
                        help='results dir [default: results]')
    parser.add_argument('--dataset_folder', default='../../dataset/',
                        help='PointNetVlad Dataset Folder')
    FLAGS = parser.parse_args()


    cfg.EVAL_BATCH_SIZE = FLAGS.eval_batch_size
    cfg.EVAL_POSITIVES_PER_QUERY = FLAGS.positives_per_query
    cfg.EVAL_NEGATIVES_PER_QUERY = FLAGS.negatives_per_query
    cfg.DECAY_STEP = FLAGS.decay_step
    cfg.DECAY_RATE = FLAGS.decay_rate
    cfg.RESULTS_FOLDER = FLAGS.results_dir

    cfg.LOG_DIR = 'log/'
    cfg.OUTPUT_FILE = cfg.RESULTS_FOLDER + 'results.txt'
    cfg.MODEL_FILENAME = "model.ckpt"
    cfg.DATASET_FOLDER = FLAGS.dataset_folder

    evaluate()
