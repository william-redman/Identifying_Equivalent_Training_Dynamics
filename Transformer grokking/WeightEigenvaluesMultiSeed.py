#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:47:22 2022

@author: wtredman
"""
import numpy as np
import scipy.optimize
import scipy.special
import matplotlib.pyplot as plt
import scipy.spatial
import scipy.stats
from dynamic_mode_decomposition import time_delay, data_matrices, dmd, dim_reduction
from wasserstein_distance import wass_dist

# Globals for saved weight trajectories
random_seeds = np.array([16, 17, 22, 23, 31, 33, 34, 40, 41, 42,]) # seeds evaluated in experiments
N = np.shape(random_seeds)[0]
M = 10 # number of initializations per random seed
alpha = 0.8

exp_path = '/Users/willredman/Documents/AIMdyn/Identifying Equivalent Optimization Algorithms/Grokking/Experiments/'
save_path = '/Users/willredman/Documents/AIMdyn/Identifying Equivalent Optimization Algorithms/Grokking/Figures/'
datapath = exp_path + 'grokking_mod_addition'

rrr_flag = True
save_flag = True
seed_plot = 5
init_plot = 0
k_mode = 10 # number of eigenvalues to keep
n_dim = 10 # number of dimensions to project data onto

# Assembling data matrices and saving losses
T = 100 # number of training iterations
D = 128 * 512

train_acc = np.zeros([M, N, 2, T])
test_acc = np.zeros([M, N, 2, T])
wasserstein_dist = np.zeros(N)
eigs = []
eigs_constrained = []

for ii in range(N):
    print(ii)
    
    X = np.zeros((T, D, M))
    X_constrained = np.zeros((T, D, M))

    for jj in range(M):
        tr_a = np.load(datapath + '_seed_' + str(random_seeds[ii]) + '_init_' + str(jj) + '_train_accuracies.npy')
        train_acc[jj, ii, 0, :] = tr_a[:T]
        te_a = np.load(datapath + '_seed_' + str(random_seeds[ii]) + '_init_' + str(jj) + '_test_accuracies.npy')
        test_acc[jj, ii, 0, :] = te_a[:T]
        W = np.load(datapath + '_seed_' + str(random_seeds[ii]) + '_init_' + str(jj) + '_weights.npy')
        X[:,:, jj] = W[:T, :]

        tr_a_constrained = np.load(datapath + '_constrained_alpha_' + str(alpha) + '_seed_' + str(random_seeds[ii]) + '_init_' + str(jj) + '_train_accuracies.npy')
        train_acc[jj, ii, 1, :] = tr_a_constrained[:T]
        te_a_constrained = np.load(datapath + '_constrained_alpha_' + str(alpha) + '_seed_'  + str(random_seeds[ii]) + '_init_' + str(jj) + '_test_accuracies.npy')
        test_acc[jj, ii, 1, :] = te_a_constrained[:T]
        W_constrained = np.load(datapath + '_constrained_alpha_' + str(alpha) + '_seed_' + str(random_seeds[ii]) + '_init_' + str(jj) + '_weights.npy')
        X_constrained[:,:, jj] = W_constrained[:T, :]

    # Reducing dimension of the data matrix
    X = dim_reduction(X, n_dim)
    X_constrained = dim_reduction(X_constrained, n_dim)

    # Computing the Koopman eigenvalues and their Wasserstein distance
    n_delays = 32

    Z = time_delay(X, int(n_delays))
    Z, Z_prime = data_matrices(Z)  
    Z_constrained = time_delay(X_constrained, int(n_delays))
    Z_constrained, Z_prime_constrained = data_matrices(Z_constrained)   


    e, _ = dmd(Z.T, Z_prime.T)
    e_constrained, _ = dmd(Z.T, Z_prime_constrained.T)
        
    eigs.append(e)
    eigs_constrained.append(e_constrained)

    wasserstein_dist[ii], _ = wass_dist(e, e_constrained)

#-----------PLOTTING---------------------------#
unit_circle_x = np.sin(np.arange(0, 2 * np.pi, 0.01))
unit_circle_y = np.cos(np.arange(0, 2 * np.pi, 0.01))
        
# Plotting all eigenvalues
fig, ax = plt.subplots()
plt.plot(np.real(eigs[seed_plot]), np.imag(eigs[seed_plot]), 'ko', label = 'Mod. addition') 
plt.plot(np.real(eigs_constrained[seed_plot]), np.imag(eigs_constrained[seed_plot]), 'r^', label = 'Constr. mod. addition') 
plt.xlabel('Real($\lambda$)')
plt.ylabel('Imag($\lambda$)')
plt.legend()
#ax.set_aspect('equal', 'box')
plt.show()
if save_flag:
    fig.savefig(save_path + '/Koopman_eigenvalues_grokking.svg', format = 'svg', dpi = 1200)
 

# Plotting the Wasserstein distance between spectra
fig, ax = plt.subplots() 
plt.hist(wasserstein_dist)
plt.xlabel('Wasserstein distance')
plt.show() 
if save_flag:
    fig.savefig(save_path + '/Wasserstein_distance_grokking.svg', format = 'svg', dpi = 1200)               


# Plotting the training loss
train_acc = np.reshape(train_acc, (N * M, 2, T))
test_acc = np.reshape(test_acc, (N * M, 2, T))

fig, ax = plt.subplots()  
plt.fill_between(np.arange(0, T, 1), np.percentile(train_acc[:, 0, :], 25, axis = 0), np.percentile(train_acc[:, 0, :], 75, axis = 0), alpha = 0.5, color = 'k')
plt.plot(np.mean(train_acc[:, 0, :], axis = 0), 'k-', label='Unconstr. train')
plt.fill_between(np.arange(0, T, 1), np.percentile(train_acc[:, 1, :], 25, axis = 0), np.percentile(train_acc[:, 1, :], 75, axis = 0), alpha = 0.5, color = 'r')
plt.plot(np.mean(train_acc[:, 1, :], axis = 0), 'r-', label='Constr. train')
plt.fill_between(np.arange(0, T, 1), np.percentile(test_acc[:, 0, :], 25, axis = 0), np.percentile(test_acc[:, 0, :], 75, axis = 0), alpha = 0.5, color = 'k')
plt.plot(np.mean(test_acc[:, 0, :], axis = 0), 'k--', label='Unconstr. test')
plt.fill_between(np.arange(0, T, 1), np.percentile(test_acc[:, 1, :], 25, axis = 0), np.percentile(test_acc[:, 1, :], 75, axis = 0), alpha = 0.5, color = 'r')
plt.plot(np.mean(test_acc[:, 1, :], axis = 0), 'r--', label='Constr. test')
#plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()
if save_flag:
    fig.savefig(save_path + '/Loss_grokking.svg', format = 'svg', dpi = 1200)        

