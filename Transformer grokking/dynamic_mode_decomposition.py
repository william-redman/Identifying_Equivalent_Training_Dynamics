#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 14:28:01 2024

@author: willredman
"""

import numpy as np
import scipy
from sklearn.decomposition import PCA

def time_delay(X, n_delays):
    # X is assumed to be T x D x N, where T is the number of time points the 
    # algorithm is run, D is the number of dimensions of the problem, and N is 
    # the number of initial conditions sampled

    T, D, N = np.shape(X)
    X_delayed = np.zeros([T - n_delays, (n_delays + 1) * D, N])

    for nn in range(N):
        for dd in range(n_delays + 1):
            X_delayed[:, (dd * D):((dd + 1) * D), nn] = X[dd:(T - n_delays + dd), :, nn]
        
    return X_delayed

def data_matrices(X): 
    # X is assumed to be T x D x N, where T is the number of time points the 
    # algorithm is run, D is the number of dimensions of the problem, and N is 
    # the number of initial conditions sampled
    
    T, D, N = np.shape(X)
    
    Z = X[:-1, :, 0]
    Z_prime = X[1:, :, 0]
    
    for nn in range(1, N):
        Z = np.concatenate((Z, X[:-1, :, nn]), axis = 0)
        Z_prime = np.concatenate((Z_prime, X[1:, :, nn]), axis = 0)
        
    return Z, Z_prime

def dmd(X, Y):
    U, S, Vh = np.linalg.svd(X, full_matrices = False)
    V = Vh.T
    Sinv = 1./S
    A = (U.T @ Y @ V) * Sinv 
    eigs, v = scipy.linalg.eig(A)
    modes = U @ v
    print('DMD')
    return eigs, modes

def dim_reduction(X, n_dim):
    X_dim_reduced = np.zeros((int(np.shape(X)[0]), int(n_dim), int(np.shape(X)[2])))
    Z, _ = data_matrices(X)
    pca = PCA(n_components=n_dim)
    pca.fit(Z)
    components = pca.components_
        
    for jj in range(np.shape(X)[2]):
        X_dim_reduced[:, :, jj] = X[:, :, jj] @ components.T
    
    return X_dim_reduced
    
    