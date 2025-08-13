#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 10:06:26 2024

@author: willredman
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def wass_dist(e1, e2):
    # e1 and e2 are assumed to be two sets of complex eigenvalues
    
    s1 = np.vstack([np.real(e1), np.imag(e1)])
    s2 = np.vstack([np.real(e2), np.imag(e2)])
    d = cdist(s1.T, s2.T)
    assignment = linear_sum_assignment(d)
    return d[assignment].mean(), assignment 