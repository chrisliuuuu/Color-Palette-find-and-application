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
    min_dist = 100000000000
    closest_vec = None

    for vec in vectors_array:
        print(vec)
        dist = euclidean(target_vec, vec)
        if dist < min_dist:
            closest_vec = vec
            min_dist = dist

    return closest_vec, min_dist


def reward(target_plt, output_plt):
    dist_sum = 0
    for target_color in target_plt():
        cloest, min_dist = get_closest(target_color, output_plt)
        dist_sum += min_dist
    
    return dist_sum