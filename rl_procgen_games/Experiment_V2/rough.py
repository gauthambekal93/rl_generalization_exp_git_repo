# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 19:10:24 2024

@author: gauthambekal93
"""

import numpy as np

a = np.array([2,3])

b= np.array([4,5])

np.stack([a, b], axis = 0)


arr1 = np.array([[1, 2, 3], [3, 4, 4]])
arr2 = np.array([[5, 6, 10], [7,12,  8]])


X =np.stack( [ [arr1, arr2] ], axis = 0)

Y =np.stack([arr1, arr2], axis = 1)


Z =np.stack([arr1, arr2], axis = 2)

'''
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7,12]])


X =np.stack([arr1, arr2], axis = 0)

Y =np.stack([arr1, arr2], axis = 1)


Z =np.stack([arr1, arr2], axis = 2)
'''