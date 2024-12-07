#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 10:46:53 2024

@author: willredman
"""

import numpy as np
import matplotlib.pyplot as plt
from wasserstein_distance import wass_dist

# Globals
eigs_path = '/Users/willredman/Documents/AIMdyn/Identifying Equivalent Optimization Algorithms/Github/Fully connected neural networks/Results/'
save_path =  '/Users/willredman/Documents/AIMdyn/Identifying Equivalent Optimization Algorithms/Github/Fully connected neural networks/Figures/'
random_seeds = np.array([2, 3, 6, 7, 11, 16, 17, 22, 23, 31, 33, 34, 40, 41, 42, 51, 56, 57, 61, 68, 69, 71, 87, 88, 95]) # seeds evaluated in experiments
n_seeds = len(random_seeds)
h = np.array([5, 10, 40])
n_h = len(h)
k_mode = 10
seed_plot = 1
init_plot = 5
save_flag = True

# Loading saved eigenvalues
eigs0 = np.load(eigs_path + 'Koopman_eigenvalues_FCN_MNIST_h=5.npy')
eigs1 = np.load(eigs_path + 'Koopman_eigenvalues_FCN_MNIST_h=10.npy')
eigs2 = np.load(eigs_path + 'Koopman_eigenvalues_FCN_MNIST_h=40.npy')

# Computing the Koopman eigenvalues
wasserstein_dist = np.zeros((n_seeds, 3))

for ss in range(n_seeds):
    e0 = eigs0[ss, :]
    e1 = eigs1[ss, :]
    e2 = eigs2[ss, :]
    
    wasserstein_dist[ss, 0], _ = wass_dist(e0, e1)
    wasserstein_dist[ss, 1], _  = wass_dist(e0, e2)
    wasserstein_dist[ss, 2], _ = wass_dist(e1, e2)
        
# Plotting all eigenvalues
unit_circle_x = np.sin(np.arange(0, 2 * np.pi, 0.01))
unit_circle_y = np.cos(np.arange(0, 2 * np.pi, 0.01))
        
fig, ax = plt.subplots()
plt.plot(unit_circle_x, unit_circle_y, 'k--', label = 'Unit circle')
plt.plot(np.real(eigs0[seed_plot]), np.imag(eigs0[seed_plot]), 'ko', label = 'h = 5') 
plt.plot(np.real(eigs1[seed_plot]), np.imag(eigs1[seed_plot]), 'r^', label = 'h = 10') 
plt.plot(np.real(eigs2[seed_plot]), np.imag(eigs2[seed_plot]), 'b*', label = 'h = 40') 
plt.xlabel('Real($\lambda$)')
plt.ylabel('Imag($\lambda$)')
plt.xlim([0.95, 1.025])
plt.ylim([-0.05, 0.05])
plt.legend()
plt.show()
if save_flag:
    plt.savefig(save_path + '/Koopman_eigenvalues_FCN_MNIST.png')

# Plotting eigenvalues of two manipulations (zoomed in)
fig, ax = plt.subplots(figsize=(6,2))
plt.plot(unit_circle_x, unit_circle_y, 'k--', label = 'Unit circle')
plt.plot(np.real(eigs1[seed_plot]), np.imag(eigs1[seed_plot]), 'r^', label = 'h = 10') 
plt.plot(np.real(eigs2[seed_plot]), np.imag(eigs2[seed_plot]), 'b*', label = 'h = 40') 
plt.xlabel('Real($\lambda$)')
plt.ylabel('Imag($\lambda$)')
plt.xlim([0.75, 1.025])
plt.ylim([-0.05, 0.05])
plt.show()
if save_flag:
    fig.savefig(save_path + '/Koopman_eigenvalues_FCN_MNIST_zoomed_h=10_h=40.png')

fig, ax = plt.subplots(figsize=(6,2))
plt.plot(unit_circle_x, unit_circle_y, 'k--', label = 'Unit circle')
plt.plot(np.real(eigs0[seed_plot]), np.imag(eigs0[seed_plot]), 'ko', label = 'h = 5') 
plt.plot(np.real(eigs2[seed_plot]), np.imag(eigs2[seed_plot]), 'b*', label = 'h = 40') 
plt.xlabel('Real($\lambda$)')
plt.ylabel('Imag($\lambda$)')
plt.xlim([0.75, 1.025])
plt.ylim([-0.05, 0.05])
plt.show()
if save_flag:
    fig.savefig(save_path + '/Koopman_eigenvalues_FCN_MNIST_zoomed_h=5_h=40.png')

# Plotting the Wasserstein distance between spectra
fig, ax = plt.subplots() 
xlabels = ['h = 5 - h = 10', 'h = 5 - h = 40', 'h = 10 - h = 40']       
plt.bar(xlabels, np.mean(wasserstein_dist, axis = 0))
plt.plot([xlabels[0], xlabels[0]], [np.mean(wasserstein_dist[:, 0]) - np.std(wasserstein_dist[:, 0]), np.mean(wasserstein_dist[:, 0]) + np.std(wasserstein_dist[:, 0])], 'k-')
plt.plot([xlabels[1], xlabels[1]], [np.mean(wasserstein_dist[:, 1]) - np.std(wasserstein_dist[:, 1]), np.mean(wasserstein_dist[:, 1]) + np.std(wasserstein_dist[:, 1])], 'k-')
plt.plot([xlabels[2], xlabels[2]], [np.mean(wasserstein_dist[:, 2]) - np.std(wasserstein_dist[:, 2]), np.mean(wasserstein_dist[:, 2]) + np.std(wasserstein_dist[:, 2])], 'k-')
plt.ylabel('Wasserstein distance')
plt.show() 
if save_flag:
    fig.savefig(save_path + '/Wasserstein_distance_FCN_MNIST.png')               

