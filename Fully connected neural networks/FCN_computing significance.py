#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:42:35 2024

@author: willredman
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from wasserstein_distance import wass_dist

# Globals 
n_shuffles = 100
random_seeds = np.array([2, 3, 6, 7, 11, 16, 17, 22, 23, 31, 33, 34, 40, 41, 42, 51, 56, 57, 61, 68, 69, 71, 87, 88, 95]) # seeds evaluated in experiments
n_seeds = len(random_seeds)
k_mode = 10
results_path = '/Users/willredman/Documents/AIMdyn/Identifying Equivalent Optimization Algorithms/Github/Fully connected neural networks/Results/'

# Loading eigenvalues
eigs_h5 = np.load(results_path + 'Koopman_eigenvalues_FCN_MNIST_h=5.npy')
eigs_h10 = np.load(results_path + 'Koopman_eigenvalues_FCN_MNIST_h=10.npy')
eigs_h40 = np.load(results_path + 'Koopman_eigenvalues_FCN_MNIST_h=40.npy')

# Computing true distance
wD_true = np.zeros((n_seeds, 3))
assignment_true = np.zeros((n_seeds, 3, 2, k_mode))
for ii in range(n_seeds):
    wD_true[ii, 0], assignment_true[ii, 0, :, :] = wass_dist(eigs_h5[ii, :], eigs_h10[ii, :])
    wD_true[ii, 1], assignment_true[ii, 1, :, :] = wass_dist(eigs_h5[ii, :], eigs_h40[ii, :])
    wD_true[ii, 2], assignment_true[ii, 2, :, :] = wass_dist(eigs_h10[ii, :], eigs_h40[ii, :])

wD_shuffle = np.zeros((n_seeds, 3, n_shuffles))

# Computing shuffled distance between h = 5 and h = 10
for ii in range(n_seeds):
    for jj in range(n_shuffles):
        e0 = copy.deepcopy(eigs_h5[ii, :])
        e1 = copy.deepcopy(eigs_h10[ii, :])
       
        e0_shuff = []
        e1_shuff = [] 
       
        for kk in range(k_mode):
            coin_flip = np.random.rand()
            if coin_flip > 0.5:
                e0_shuff.append(e1[int(assignment_true[ii, 0, 1, kk])])
                e1_shuff.append(e0[kk])
            
            else:
                e0_shuff.append(e0[kk])
                e1_shuff.append(e1[int(assignment_true[ii, 0, 1, kk])])
        
        wD_shuffle[ii, 0, jj], _ = wass_dist(e0_shuff, e1_shuff)
        
# Computing shuffled distance between h = 5 and h = 40
for ii in range(n_seeds):
    for jj in range(n_shuffles):
        e0 = copy.deepcopy(eigs_h5[ii, :])
        e1 = copy.deepcopy(eigs_h40[ii, :])

        e0_shuff = []
        e1_shuff = []
    
        for kk in range(k_mode):
            coin_flip = np.random.rand()
            if coin_flip > 0.5:
                e0_shuff.append(e1[int(assignment_true[ii, 1, 1, kk])])
                e1_shuff.append(e0[kk])
            
            else:
                e0_shuff.append(e0[kk])
                e1_shuff.append(e1[int(assignment_true[ii, 1, 1, kk])])
        
        wD_shuffle[ii, 1, jj], _ = wass_dist(e0_shuff, e1_shuff)

# Computing shuffled distance between h = 10 and h = 40
for ii in range(n_seeds):
    for jj in range(n_shuffles):
        e0 = copy.deepcopy(eigs_h10[ii, :])
        e1 = copy.deepcopy(eigs_h40[ii, :])
    
        e0_shuff = []
        e1_shuff = []
    
        for kk in range(k_mode):
            coin_flip = np.random.rand()
            if coin_flip > 0.5:
                e0_shuff.append(e1[int(assignment_true[ii, 2, 1, kk])])
                e1_shuff.append(e0[kk])
           
            else:
                e0_shuff.append(e0[kk])
                e1_shuff.append(e1[int(assignment_true[ii, 2, 1, kk])])
       
        wD_shuffle[ii, 2, jj], _ = wass_dist(e0_shuff, e1_shuff)
   
# Computing how many of the shuffles satisfy the null hypothesis (that the true Wasserstein distance is greater than or equal to the true Wasserstein distance)
null_hypothesis_wass_dist = np.zeros((n_seeds, 3))
for ii in range(n_seeds):
    null_hypothesis_wass_dist[ii, 0] += np.nansum(wD_true[ii, 0] <= wD_shuffle[ii, 0, :])
    null_hypothesis_wass_dist[ii, 1] += np.nansum(wD_true[ii, 1] <= wD_shuffle[ii, 1, :])
    null_hypothesis_wass_dist[ii, 2] += np.nansum(wD_true[ii, 2] <= wD_shuffle[ii, 2, :])
    
print(str(np.nansum(null_hypothesis_wass_dist[:, 0]) / (n_seeds * n_shuffles)))
print(str(np.nansum(null_hypothesis_wass_dist[:, 1]) / (n_seeds * n_shuffles)))
print(str(np.nansum(null_hypothesis_wass_dist[:, 2]) / (n_seeds * n_shuffles)))

