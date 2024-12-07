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
from dynamic_mode_decomposition import time_delay, data_matrices, dmd
from wasserstein_distance import wass_dist

# Globals for saved weight trajectories
random_seeds = np.array([2, 3, 6, 7, 11, 16, 17, 22, 23, 31, 33, 34, 40, 41, 42, 51, 56, 57, 61, 68, 69, 71, 87, 88, 95]) # seeds evaluated in experiments
N = np.shape(random_seeds)[0]
M = 10 # number of initializations per random seed
exp_path = '/Users/willredman/Documents/AIMdyn/Identifying Equivalent Optimization Algorithms/Github/Fully connected neural networks/Results/'
save_path = '/Users/willredman/Documents/AIMdyn/Identifying Equivalent Optimization Algorithms/Github/Fully connected neural networks/Figures/'
datapath = exp_path + 'SGD_MNIST_'
save_flag = False
n_weights_plot = 5
seed_plot = 1
init_plot = 5
k_mode = 10

# Assembling data matrices and saving losses
T = 1000 # number of training iterations
D0 = 5
D1 = 10 
D2 = 40 

manipulations = ['h='+str(D0), 'h='+str(D1), 'h='+str(D2)]
n_manipulations = np.shape(manipulations)[0]

X0 = np.zeros((T, D0 * 10, M, N))
X1 = np.zeros((T, D1 * 10, M, N))
X2 = np.zeros((T, D2 * 10, M, N))
train_loss = np.zeros([M, N, n_manipulations, T])
test_loss = np.zeros([M, N, n_manipulations])

for ii in range(N):
    for jj in range(M):
        tr_l0 = np.load(datapath + manipulations[0] + '_seed' + str(random_seeds[ii]) + '_initialization' + str(jj) + '_Train_Loss.npy')
        train_loss[jj, ii, 0, :] = tr_l0[:][0]
        te_l0 = np.load(datapath + manipulations[0] + '_seed' + str(random_seeds[ii]) + '_initialization' + str(jj) + '_Loss.npy')
        test_loss[jj, ii, 0] = te_l0[:][0]
        W0 = np.load(datapath + manipulations[0] + '_seed' + str(random_seeds[ii]) + '_initialization' + str(jj) + '_weights.npy')
        X0[:,:, jj, ii] = W0

        tr_l1 = np.load(datapath + manipulations[1] + '_seed' + str(random_seeds[ii]) + '_initialization' + str(jj) + '_Train_Loss.npy')
        train_loss[jj, ii, 1, :] = tr_l1[:][0] 
        te_l1 = np.load(datapath + manipulations[1] + '_seed' + str(random_seeds[ii]) + '_initialization' + str(jj) + '_Loss.npy')
        test_loss[jj, ii, 1] = te_l1[:][0]
        W1 = np.load(datapath + manipulations[1] + '_seed' + str(random_seeds[ii]) + '_initialization' + str(jj) + '_weights.npy')
        X1[:,:, jj, ii] = W1
    
        tr_l2 = np.load(datapath + manipulations[2] + '_seed' + str(random_seeds[ii]) + '_initialization' + str(jj) + '_Train_Loss.npy')
        train_loss[jj, ii, 2, :] = tr_l2[:][0]
        te_l2 = np.load(datapath + manipulations[2] + '_seed' + str(random_seeds[ii]) + '_initialization' + str(jj) + '_Loss.npy')
        test_loss[jj, ii, 2] = te_l2[:][0]
        W2 = np.load(datapath + manipulations[2] + '_seed' + str(random_seeds[ii]) + '_initialization' + str(jj) + '_weights.npy')
        X2[:,:, jj, ii] = W2

# Computing the Koopman eigenvalues and their Wasserstein distance
n_delays = 32
wasserstein_dist = np.zeros((N, 3))
eigs0 = []
eigs1 = []
eigs2 = []

for ii in range(N):
    Z0 = time_delay(X0[:, :, :, ii], int(n_delays * D1 / D0))
    Z0, Z0_prime = data_matrices(Z0)  
    Z1 = time_delay(X1[:, :, :, ii], int(n_delays * D1 / D1))
    Z1, Z1_prime = data_matrices(Z1)
    Z2 = time_delay(X2[:, :, :, ii], int(n_delays * D1 / D2))
    Z2, Z2_prime = data_matrices(Z2)    

    if rrr_flag:
        e0, _ = dmd(Z0.T, Z0_prime.T, k_mode=k_mode)
        e1, _ = dmd(Z1.T, Z1_prime.T, k_mode=k_mode)
        e2, _ = dmd(Z2.T, Z2_prime.T, k_mode=k_mode)
        
        eigs0.append(e0)
        eigs1.append(e1)
        eigs2.append(e2)
        
    else: 
        e0, _ = dmd(Z0.T, Z0_prime.T)
        e1, _ = dmd(Z1.T, Z1_prime.T)
        e2, _ = dmd(Z2.T, Z2_prime.T)
        
        eigs0.append(e0)
        eigs1.append(e1)
        eigs2.append(e2)

    wasserstein_dist[ii, 0], _ = wass_dist(e0, e1)
    wasserstein_dist[ii, 1], _  = wass_dist(e0, e2)
    wasserstein_dist[ii, 2], _ = wass_dist(e1, e2)
    
# Computing the Wasserstein distance between seeds of h = 40
between_seeds_wasserstein_dist = np.zeros((N, N)) * np.nan    
for ii in range(N):
    for jj in range(N):
        if ii < jj:
            between_seeds_wasserstein_dist[ii, jj], _ = wass_dist(eigs2[ii], eigs2[jj])
        
# Saving eigenvalues and Wasserstein distance 
if save_flag:
   np.save(exp_path + '/Koopman_eigenvalues_FCN_MNIST_' + manipulations[0] + '.npy', eigs0) 
   np.save(exp_path + '/Koopman_eigenvalues_FCN_MNIST_' + manipulations[1] + '.npy', eigs1)   
   np.save(exp_path + '/Koopman_eigenvalues_FCN_MNIST_' + manipulations[2] + '.npy', eigs2) 
   np.save(exp_path + '/Wasserstein_distance_FCN_MNIST_' + manipulations[0] + manipulations[1] + manipulations[2] + '.npy', wasserstein_dist)
   np.save(exp_path + '/between_seeds_Wasserstein_distance_FCN_MNIST_' + manipulations[2] + '.npy', between_seeds_wasserstein_dist)         
    
#-----------PLOTTING---------------------------#
unit_circle_x = np.sin(np.arange(0, 2 * np.pi, 0.01))
unit_circle_y = np.cos(np.arange(0, 2 * np.pi, 0.01))
        
# Plotting all eigenvalues
fig, ax = plt.subplots()
plt.plot(unit_circle_x, unit_circle_y, 'k--', label = 'Unit circle')
plt.plot(np.real(eigs0[seed_plot]), np.imag(eigs0[seed_plot]), 'ko', label = manipulations[0]) 
plt.plot(np.real(eigs1[seed_plot]), np.imag(eigs1[seed_plot]), 'r^', label = manipulations[1]) 
plt.plot(np.real(eigs2[seed_plot]), np.imag(eigs2[seed_plot]), 'b*', label = manipulations[2]) 
plt.xlabel('Real($\lambda$)')
plt.ylabel('Imag($\lambda$)')
plt.xlim([0.95, 1.025])
plt.ylim([-0.05, 0.05])
plt.legend()
#ax.set_aspect('equal', 'box')
plt.show()
if save_flag:
    fig.savefig(save_path + '/Koopman_eigenvalues_FCN_MNIST.svg', format = 'svg', dpi = 1200)
 
# Plotting eigenvalues of two manipulations (zoomed in)
fig, ax = plt.subplots(figsize=(6,2))
plt.plot(unit_circle_x, unit_circle_y, 'k--', label = 'Unit circle')
plt.plot(np.real(eigs1[seed_plot]), np.imag(eigs1[seed_plot]), 'r^', label = manipulations[1]) 
plt.plot(np.real(eigs2[seed_plot]), np.imag(eigs2[seed_plot]), 'b*', label = manipulations[2]) 
plt.xlabel('Real($\lambda$)')
plt.ylabel('Imag($\lambda$)')
plt.xlim([0.75, 1.025])
plt.ylim([-0.05, 0.05])
plt.show()
if save_flag:
    fig.savefig(save_path + '/Koopman_eigenvalues_FCN_MNIST_zoomed_h=10_h=40.svg', format = 'svg', dpi = 1200)

fig, ax = plt.subplots(figsize=(6,2))
plt.plot(unit_circle_x, unit_circle_y, 'k--', label = 'Unit circle')
plt.plot(np.real(eigs0[seed_plot]), np.imag(eigs0[seed_plot]), 'ko', label = manipulations[0]) 
plt.plot(np.real(eigs2[seed_plot]), np.imag(eigs2[seed_plot]), 'b*', label = manipulations[2]) 
plt.xlabel('Real($\lambda$)')
plt.ylabel('Imag($\lambda$)')
plt.xlim([0.75, 1.025])
plt.ylim([-0.05, 0.05])
plt.show()
if save_flag:
    fig.savefig(save_path + '/Koopman_eigenvalues_FCN_MNIST_zoomed_h=5_h=40.svg', format = 'svg', dpi = 1200)

# Plotting the Wasserstein distance between spectra
fig, ax = plt.subplots() 
xlabels = [manipulations[0] + '-' + manipulations[1], manipulations[0] + '-' + manipulations[2], manipulations[1] + '-' + manipulations[2]]       
plt.bar(xlabels, np.mean(wasserstein_dist, axis = 0))
plt.plot([xlabels[0], xlabels[0]], [np.mean(wasserstein_dist[:, 0]) - np.std(wasserstein_dist[:, 0]), np.mean(wasserstein_dist[:, 0]) + np.std(wasserstein_dist[:, 0])], 'k-')
plt.plot([xlabels[1], xlabels[1]], [np.mean(wasserstein_dist[:, 1]) - np.std(wasserstein_dist[:, 1]), np.mean(wasserstein_dist[:, 1]) + np.std(wasserstein_dist[:, 1])], 'k-')
plt.plot([xlabels[2], xlabels[2]], [np.mean(wasserstein_dist[:, 2]) - np.std(wasserstein_dist[:, 2]), np.mean(wasserstein_dist[:, 2]) + np.std(wasserstein_dist[:, 2])], 'k-')
plt.ylabel('Wasserstein distance')
plt.show() 
if save_flag:
    fig.savefig(save_path + '/Wasserstein_distance_FCN_MNIST.svg', format = 'svg', dpi = 1200)               

# Plotting the distribution of end losses (relative to the first initialization)
relative_loss = [[], [], []]

for ii in range(N): 
    rel_l0 = test_loss[:, ii, 0] / test_loss[0, ii, 0]
    relative_loss[0].append(rel_l0)
    rel_l1 = test_loss[:, ii, 1] / test_loss[0, ii, 1]
    relative_loss[1].append(rel_l1)
    rel_l2 = test_loss[:, ii, 2] / test_loss[0, ii, 2]
    relative_loss[2].append(rel_l2)
    
fig, ax = plt.subplots(1, 3, figsize = (15, 3))
ax[0].hist(np.array(relative_loss[0][:]).flatten())
ax[0].set_xlabel('Relative test loss')
ax[0].set_ylabel('Count')
ax[0].set_title('h = ' + str(D0))
ax[1].hist(np.array(relative_loss[1][:]).flatten())
ax[1].set_xlabel('Relative test loss')
ax[1].set_ylabel('Count')
ax[1].set_title('h = ' + str(D1))
ax[2].hist(np.array(relative_loss[2][:]).flatten())
ax[2].set_xlabel('Relative test loss')
ax[2].set_ylabel('Count')
ax[2].set_title('h = ' + str(D2))
plt.show()
if save_flag:
    fig.savefig(save_path + '/Relative_loss.svg', format = 'svg', dpi = 1200)


# Plotting the training loss
train_loss = np.reshape(train_loss, (N * M, 3, T))
fig, ax = plt.subplots()  
plt.fill_between(np.linspace(0, 60000, int(T / 10)), np.percentile(train_loss[:, 0, np.arange(0, T, 10)], 25, axis = 0), np.percentile(train_loss[:, 0, np.arange(0, T, 10)], 75, axis = 0), alpha = 0.5, color = 'k')
plt.plot(np.linspace(0, 60000, int(T / 10)), np.mean(train_loss[:, 0, np.arange(0, T, 10)], axis = 0), 'k-')
plt.fill_between(np.linspace(0, 60000, int(T / 10)), np.percentile(train_loss[:, 1, np.arange(0, T, 10)], 25, axis = 0), np.percentile(train_loss[:, 1, np.arange(0, T, 10)], 75, axis = 0), alpha = 0.5, color = 'r')
plt.plot(np.linspace(0, 60000, int(T / 10)), np.mean(train_loss[:, 1, np.arange(0, T, 10)], axis = 0), 'r-')
plt.fill_between(np.linspace(0, 60000, int(T / 10)), np.percentile(train_loss[:, 2, np.arange(0, T, 10)], 25, axis = 0), np.percentile(train_loss[:, 2, np.arange(0, T, 10)], 75, axis = 0), alpha = 0.5, color = 'b')
plt.plot(np.linspace(0, 60000, int(T / 10)), np.mean(train_loss[:, 2, np.arange(0, T, 10)], axis = 0), 'b-')
plt.yscale('log')
plt.xlabel('Training images')
plt.ylabel('Training loss')
plt.legend(manipulations)
plt.show()
if save_flag:
    fig.savefig(save_path + '/Loss_FCN_MNIST.svg', format = 'svg', dpi = 1200)        

# Plotting example weight evolutions
W0 = np.load(datapath + manipulations[0] + '_seed' + str(random_seeds[seed_plot]) +  '_initialization' + str(init_plot) + '_weights.npy')
W1 = np.load(datapath + manipulations[1] + '_seed' + str(random_seeds[seed_plot]) +  '_initialization' + str(init_plot) + '_weights.npy')
W2 = np.load(datapath + manipulations[2] + '_seed' + str(random_seeds[seed_plot]) +  '_initialization' + str(init_plot) + '_weights.npy')

time = np.linspace(0, 60000, int(T/10) + 1)
time = time.astype(int)
time = time[:-1]
weight_time = np.linspace(0, T, int(T/10) + 1)
weight_time = weight_time.astype(int)
weight_time = weight_time[:-1]
weight0_ids = np.random.permutation(np.shape(W0)[1])[:n_weights_plot]
weight1_ids = np.random.permutation(np.shape(W1)[1])[:n_weights_plot]
weight2_ids = np.random.permutation(np.shape(W2)[1])[:n_weights_plot]


fig, ax = plt.subplots()
for ii in range(len(weight0_ids)):
    plt.plot(time, W0[weight_time, weight0_ids[ii]], 'k-')
    plt.plot(time, W2[weight_time, weight2_ids[ii]], 'b-')
plt.ylim([-1.0, 1.0])
plt.xlabel('Training images')
plt.ylabel('w')
if save_flag:
    fig.savefig(save_path + '/Weight_trajectories_FCN_MNIST_h=5_h=40.svg', format = 'svg', dpi = 1200)  

fig, ax = plt.subplots()
for ii in range(len(weight0_ids)):
    plt.plot(time, W1[weight_time, weight1_ids[ii]], 'r-')
    plt.plot(time, W2[weight_time, weight2_ids[ii]], 'b-')
plt.ylim([-1.0, 1.0])
plt.xlabel('Training images')
plt.ylabel('w')
if save_flag:
    fig.savefig(save_path + '/Weight_trajectories_FCN_MNIST_h=10_h=40.svg', format = 'svg', dpi = 1200)  

