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
square_file = './tL_square_0-1_15_N350.csv'
hex_file = './tL_hex_0-1_15_N350.csv'
# reading the file 
array_square = np.genfromtxt(square_file, delimiter=',')
array_hex = np.genfromtxt(hex_file, delimiter=',')


"""Plotting the t(L) curves"""
plt.figure(figsize=(9, 9))
plt.plot(array_square[0, :], array_square[1, :], color='red', label='t(L) square')
plt.plot(array_hex[0, :], array_hex[1, :], color='blue', label='t(L) hex')
plt.grid()
plt.legend()
plt.show()

