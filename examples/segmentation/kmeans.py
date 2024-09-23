# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch
import numpy as np
import glob
import time
import argparse
import pykeops
from pykeops.torch import LazyTensor
pykeops.clean_pykeops()
from IPython.core.debugger import set_trace

def parse_args():
    """parse input arguments"""
    parser = argparse.ArgumentParser(description='label_recommendation')
    parser.add_argument('--point_data', type=str, default='/data/s3dis/train/')
    parser.add_argument('--feat_data', type=str, default = '/userhome/LiM/processed/s3dis/feat/')
    parser.add_argument('--num_points', type=int, default=20)
    parser.add_argument('--num_iters', type=int, default=50)
    parser.add_argument('--output', type=str, default='/userhome/LiM/processed/s3dis/output')
    return parser.parse_args()


def kmeans(pointcloud, k=10, iterations=10, noise_filter=0, verbose=True):
    n, dim = pointcloud.shape  # Number of samples, dimension of the ambient space
    start = time.time()
    k_inds = np.random.choice(n,k,replace=False)
    clusters = pointcloud[k_inds, :].clone()  # Simplistic random initialization
    pointcloud_cuda = LazyTensor(pointcloud[:, None, :])  # (Npoints, 1, D)

    # K-means loop:
    for _ in range(iterations):
        clusters_previous = clusters.clone()
        clusters_gpu = LazyTensor(clusters[None, :, :])  # (1, Nclusters, D)
        distance_matrix = ((pointcloud_cuda - clusters_gpu) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cloest_clusters = distance_matrix.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # #points for each cluster
        clusters_count = torch.bincount(cloest_clusters, minlength=k).float()  # Class weights
        for d in range(dim):  # Compute the cluster centroids with torch.bincount:
            clusters[:, d] = torch.bincount(cloest_clusters, weights=pointcloud[:, d], minlength=k) / clusters_count
        
        # for clusters that have no points assigned
        mask = clusters_count == 0
        clusters[mask] = clusters_previous[mask]
    
    clusters = clusters[clusters_count > noise_filter]

    end = time.time()

    if verbose:
        print("K-means example with {:,} points in dimension {:,}, K = {:,}:".format(n, dim, k))
        print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format(
                iterations, end - start, iterations, (end-start) / iterations))
    
    # nearest neighbouring search for each cluster
    cloest_points_to_centers = distance_matrix.argmin(dim=0).long().view(-1)
    return cloest_points_to_centers, clusters, cloest_clusters

def kmeans_sampling(args):
    pointcloud_names = glob.glob(os.path.join(args.raw_data, "*.pth"))

    sampled_inds = {}
    for idx, pointcloud_name in enumerate(pointcloud_names):
        print('{}/{}: {}'.format(idx, len(pointcloud_names), pointcloud_name))
        pointcloud = torch.load(pointcloud_name)
        scene_name = os.path.basename(pointcloud_name).split('.')[0]

        coords = pointcloud[0].astype(np.float32)
        colors = pointcloud[1].astype(np.int32)

        candidates = []
        candidates.append(coords)
        candidates.append(colors)
        feats = torch.load(os.path.join(args.feat_data, scene_name))
        candidates.append(feats)
        candidates = torch.from_numpy(np.concatenate(candidates,1)).cuda().float()

        K = args.num_points
        sampled_inds_per_scene = kmeans(candidates, K, args.num_iters).cpu().numpy()
        sampled_inds[scene_name] = sampled_inds_per_scene

    return sampled_inds


if __name__ == "__main__":
    args = parse_args()
    sampled_inds = kmeans_sampling(args)
    torch.save(sampled_inds, args.output)