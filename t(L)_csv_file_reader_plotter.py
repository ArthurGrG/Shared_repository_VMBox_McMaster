#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 14:24:57 2022

@author: arthur
"""

"""Import section """
import numpy as np 
import matplotlib.pyplot as plt 

"""Reading the dsv file with numpy"""
# path file 
path_file = 't(L)_0-1_15_N350.csv'
# reading the file 
array = np.genfromtxt(path_file, delimiter=',')


"""Plotting the t(L) curve"""
plt.figure(figsize=(9, 9))
plt.plot(array[0, :], array[1, :], color='red', label='t(L)')
plt.grid()
plt.legend()

