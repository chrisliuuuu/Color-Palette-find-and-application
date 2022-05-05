#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 20:09:35 2022
"""

import numpy as np
from scipy.spatial.distance import euclidean

# target_plt: np.array
# output_plt: np.array


def get_closest(target_vec, vectors_array):
    '''return a vector, which is closest to the target vector, from in an array of arrays;
    Also returns the distance between the two vectors.'''
    min_dist = float('inf')
    closest_vec = None

    for vec in vectors_array:
        dist = euclidean(target_vec, vec)
        if dist < min_dist:
            closest_vec = vec
            min_dist = dist

    return closest_vec, min_dist


def getRewardFromPalettes(target_plt, output_plt) -> float:
    '''return the sum of distance between target color pallette and the output color pallette'''
    ''''Target_plt and output_plt must by a numpy array of arrays, and they must have in the same format'''
    dist_sum = 0
    for target_color in target_plt:
        cloest, min_dist = get_closest(target_color, output_plt)
        dist_sum += min_dist
    
    return dist_sum
