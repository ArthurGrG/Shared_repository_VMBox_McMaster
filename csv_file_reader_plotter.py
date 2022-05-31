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
# path file for t(L)
square_file_tL = './results_csv/tL_square_0-01_15_N500.csv'
hex_file_tL = './results_csv/tL_hex_0-01_15_N500.csv'
# path file for DE(t, L(t)) 
square_file_DE = './results_csv/DE_square_0-01_15_N500.csv'
hex_file_DE = './results_csv/DE_hex_0-01_15_N500.csv'

# reading files
array_square_tL = np.genfromtxt(square_file_tL, delimiter=',')
array_hex_tL = np.genfromtxt(hex_file_tL, delimiter=',')
array_square_DE = np.genfromtxt(square_file_DE, delimiter=',')
array_hex_DE = np.genfromtxt(hex_file_DE, delimiter=',')


"""Plotting the t(L) curves"""
plt.figure(figsize=(9, 9))
plt.xlabel("t"); plt.ylabel("L(t)")
plt.plot(array_square_tL[1, :], array_square_tL[0, :], color='red', label='L(t) square')
plt.plot(array_hex_tL[1, :], array_hex_tL[0, :], color='blue', label='L(t) hex')
plt.grid()
plt.legend()
plt.show()


"""Plotting the DE(t, L(t)) curves"""
plt.figure(figsize=(9, 9))
plt.xlabel("t"); plt.ylabel("DE(t, L(t))")
plt.plot(array_square_DE[0, :], array_square_DE[1, :], color='purple', label='DE(t, L(t)) square')
plt.plot(array_hex_DE[0, :], array_hex_DE[1, :], color='darkorange', label='DE(t, L(t)) hex')
plt.grid()
plt.legend()
plt.show()

"""Computation of F(l(t)) to plot it"""
F_Lt_square = ((array_square_tL[0, :]**2)*array_square_DE[1, :] - array_square_tL[0, :]*4)/(array_square_tL[1, :]**2)
F_Lt_hex = ((array_hex_tL[0, :]**2)*(3*np.sqrt(3)/2)*array_hex_DE[1, :] - array_hex_tL[0, :]*4)/(array_hex_tL[1, :]**2)
plt.figure(figsize=(9, 9))
plt.xlabel("t"); plt.ylabel("F(L(t))")
plt.plot(array_square_DE[0, :], F_Lt_square, color='green', label='F(L(t)) square')
plt.plot(array_hex_DE[0, :], F_Lt_hex, color='darkblue', label='F(L(t)) hex')
plt.grid()
plt.legend()
plt.show()

