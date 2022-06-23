"""Import section """
import numpy as np 
import matplotlib.pyplot as plt 

"""Reading the csv file with numpy"""
# path file for t(L) 
file_FL = './results_csv/results_DE_given_t/FL_square_0-1_10_N150_iso.csv'
file_FpL = './results_csv/results_DE_given_t/FpL_square_0-1_10_N150_iso.csv'
file_denom = './results_csv/results_DE_given_t/denom_square_0-1_10_N150_iso.csv'
# reading
array_FL = np.genfromtxt(file_FL, delimiter=',')
array_FpL = np.genfromtxt(file_FpL, delimiter=',') 
array_denom = np.genfromtxt(file_denom, delimiter=',') 

"""Computation of DE(t, L) and dDE(t, L)/dL for a given t"""
t = 1.
DE_square = (t**2)*(array_FL[1, :]/(array_FL[0, :]**2)) + 4/array_FL[0, :]
dL_DE_square = (t**2)*(array_FpL[1, :]/(array_FpL[0, :]**2) - 2*array_FL[1, :]/(array_FL[0, :]**3)) - 4/(array_FL[0, :]**2)

"""Plot of DE(t, L) and dDE(t, L)/dL for a given t"""
plt.figure(figsize=(9, 9))
plt.xlabel("L"); plt.ylabel("DE(t, L)")
plt.plot(array_FL[0, :], DE_square, color='red', label='DE(t, L) square')
plt.plot(array_FpL[0, :], dL_DE_square, color='blue', label='dDE/dL(t, L) square')
plt.grid()
plt.legend()
plt.show()

"""Plot of F'(L)L - 2F(L) with the analytical form"""
plt.figure(figsize=(9, 9))
plt.xlabel("L"); plt.ylabel("F'(L)L - 2F(L)")
plt.plot(array_denom[0, :], array_denom[1, :], color='red', label='F\'(L)L - 2F(L) square')
plt.grid()
plt.legend()
plt.show()

