#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 19:23:51 2024

@author: willredman
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from wasserstein_distance import wass_dist

# Globals 
n_shuffles = 100
k_mode = 10
results_path = '/Users/willredman/Documents/AIMdyn/Identifying Equivalent Optimization Algorithms/Github/Online mirror online gradient descent/Results/'

# Loading eigenvalues
mirror_descent_eigs = np.load(results_path + 'mirror_descent_eigs.npy')
gradient_descent_eigs = np.load(results_path + 'gradient_descent_eigs.npy')
bisection_method_eigs = np.load(results_path + 'bisection_method_eigs.npy')

# Computing true distance
wD_true = np.zeros((3))
assignment_true = np.zeros((3, 2, k_mode))
wD_true[0], assignment_true[0, :, :] = wass_dist(mirror_descent_eigs, gradient_descent_eigs)
wD_true[1], assignment_true[1, :, :] = wass_dist(gradient_descent_eigs, bisection_method_eigs)
wD_true[2], assignment_true[2, :, :] = wass_dist(mirror_descent_eigs, bisection_method_eigs)

# Computing shuffled distance between mirror descent and gradient descent
wD_shuffle = np.zeros((3, n_shuffles))

for ii in range(n_shuffles):
    e0 = copy.deepcopy(mirror_descent_eigs)
    e1 = copy.deepcopy(gradient_descent_eigs)
    
    e0_shuff = []
    e1_shuff = []
    
    for kk in range(k_mode):
        coin_flip = np.random.rand()
        if coin_flip > 0.5:
            e0_shuff.append(e1[int(assignment_true[0, 1, kk])])
            e1_shuff.append(e0[kk])
        
        else:
            e0_shuff.append(e0[kk])
            e1_shuff.append(e1[int(assignment_true[0, 1, kk])])
        
    wD_shuffle[0, ii], _ = wass_dist(e0_shuff, e1_shuff)
    
# Computing shuffled distance between  gradient descent and bisection method
for ii in range(n_shuffles):
    e1 = copy.deepcopy(gradient_descent_eigs)
    e2 = copy.deepcopy(bisection_method_eigs)
    
    e1_shuff = []
    e2_shuff = []
    
    for kk in range(k_mode):
        coin_flip = np.random.rand()
        if coin_flip > 0.5:
            e0_shuff.append(e1[int(assignment_true[1, 1, kk])])
            e1_shuff.append(e0[kk])
        
        else:
            e0_shuff.append(e0[kk])
            e1_shuff.append(e1[int(assignment_true[1, 1, kk])])
        
    wD_shuffle[1, ii], _ = wass_dist(e1_shuff, e2_shuff)  

# Computing shuffled distance between mirror descent and bisection method
for ii in range(n_shuffles):
    e0 = copy.deepcopy(mirror_descent_eigs)
    e2 = copy.deepcopy(bisection_method_eigs)
    
    e0_shuff = []
    e2_shuff = []
    
    for kk in range(k_mode):
        coin_flip = np.random.rand()
        if coin_flip > 0.5:
            e0_shuff.append(e1[int(assignment_true[2, 1, kk])])
            e1_shuff.append(e0[kk])
        
        else:
            e0_shuff.append(e0[kk])
            e1_shuff.append(e1[int(assignment_true[2, 1, kk])])   
        
    wD_shuffle[2, ii], _ = wass_dist(e0_shuff, e2_shuff)    
    
# Statistics 
print(str(np.sum(wD_true[0] <= wD_shuffle[0, :]) / n_shuffles))
print(str(np.sum(wD_true[1] <= wD_shuffle[1, :]) / n_shuffles))
print(str(np.sum(wD_true[2] <= wD_shuffle[2, :]) / n_shuffles))


