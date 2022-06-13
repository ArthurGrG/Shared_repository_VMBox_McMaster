#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 14:24:57 2022

@author: arthur
"""

"""Import section """
import numpy as np 
import matplotlib.pyplot as plt 

"""Reading the csv file with numpy"""
# path file for t(L) with adj 
square_file_tL_adj = './results_csv/tL_square_adj_0-1_10_N200.csv'
hex_file_tL_adj = './results_csv/tL_hex_adj_0-1_10_N200.csv'
# path file for DE(t, L(t)) with adj 
square_file_DE_adj = './results_csv/DE_square_adj_0-1_10_N200.csv'
hex_file_DE_adj = './results_csv/DE_hex_adj_0-1_10_N200.csv'
# path file for t(L) with adj 
square_file_tL_df = './results_csv/tL_square_0-1_10_N1500.csv'
hex_file_tL_df = './results_csv/tL_hex_0-1_10_N1500.csv'
# path file for DE(t, L(t)) with adj 
square_file_DE_df = './results_csv/DE_square_0-1_10_N1500.csv'
hex_file_DE_df = './results_csv/DE_hex_0-1_10_N1500.csv'


# reading files adj
array_square_tL_adj = np.genfromtxt(square_file_tL_adj, delimiter=',')
array_hex_tL_adj = np.genfromtxt(hex_file_tL_adj, delimiter=',')
array_square_DE_adj = np.genfromtxt(square_file_DE_adj, delimiter=',')
array_hex_DE_adj = np.genfromtxt(hex_file_DE_adj, delimiter=',')
# reading files df 
array_square_tL_df = np.genfromtxt(square_file_tL_df, delimiter=',')
array_hex_tL_df = np.genfromtxt(hex_file_tL_df, delimiter=',')
array_square_DE_df = np.genfromtxt(square_file_DE_df, delimiter=',')
array_hex_DE_df = np.genfromtxt(hex_file_DE_df, delimiter=',')


"""Plotting the t(L) curves"""
plt.figure(figsize=(9, 9))
plt.xlabel("t"); plt.ylabel("L(t)")
plt.plot(array_square_tL_adj[1, :], array_square_tL_adj[0, :], color='red', label='L(t) square adj')
plt.plot(array_hex_tL_adj[1, :], array_hex_tL_adj[0, :], color='blue', label='L(t) hex adj')
plt.plot(array_square_tL_df[1, :], array_square_tL_df[0, :], color='green', label='L(t) square df')
plt.plot(array_hex_tL_df[1, :], array_hex_tL_df[0, :], color='purple', label='L(t) hex df')
plt.grid()
plt.legend()
plt.show()


"""Plotting the DE(t, L(t)) curves"""
plt.figure(figsize=(9, 9))
plt.xlabel("t"); plt.ylabel("DE(t, L(t))")
plt.plot(array_square_DE_adj[0, :], array_square_DE_adj[1, :], color='red', label='DE(t, L(t)) square adj')
plt.plot(array_hex_DE_adj[0, :], array_hex_DE_adj[1, :], color='blue', label='DE(t, L(t)) hex adj')
plt.plot(array_square_DE_df[0, :], array_square_DE_df[1, :], color='green', label='DE(t, L(t)) square df')
plt.plot(array_hex_DE_df[0, :], array_hex_DE_df[1, :], color='purple', label='DE(t, L(t)) hex df')
plt.grid()
plt.legend()
plt.show()


