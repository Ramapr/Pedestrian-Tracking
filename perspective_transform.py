# -*- coding: utf-8 -*-
"""
Created on Thu May 19 01:16:04 2022

@author: Mi
"""

import numpy as np


def calc_transform_matrix(X, U):
    """
    X - source coordinate [4 x 2]
    U - target coordinate [4 x 2]
    output:  [a0, a1, a2, b0, b1, b2, c0, c1]
    # check dims \n
    """
    zer = np.zeros((X.shape[0], 3))
    on = np.ones((X.shape[0], 1))
    b = np.concatenate([U[:, :1], U[:, 1:]])

    half_A_up = np.concatenate([X, on, zer], axis=1)
    half_A_bottom = np.concatenate([zer, X, on], axis=1)
    rigth_part = (-1. * np.vstack([X, X])) * np.hstack([b, b]) 
    A = np.hstack([np.concatenate([half_A_up, half_A_bottom], axis=0), rigth_part])
    return np.linalg.solve(A, b)

#M = np.concatenate([out, [[1.]]]).reshape((3, 3))\