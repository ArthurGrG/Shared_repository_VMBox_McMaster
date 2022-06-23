"""Import section """
import numpy as np 
import matplotlib.pyplot as plt 

"""Reading the csv file with numpy"""
# path file for t(L) 
square_file_tL = './results_csv/tL_square_aniso_rho1-3_c1-5_0-1_10_N150.csv'
hex_file_tL = './results_csv/tL_hex_aniso_rho1-3_c1-5_0-1_10_N150.csv'
# path file for DE(t, L(t)) with adj 
square_file_DE = './results_csv/DE_square_aniso_rho1-3_c1-5_0-1_10_N150.csv'
hex_file_DE = './results_csv/DE_hex_aniso_rho1-3_c1-5_0-1_10_N150.csv'

# reading files
array_square_tL = np.genfromtxt(square_file_tL, delimiter=',')
array_square_DE = np.genfromtxt(square_file_DE, delimiter=',')
array_hex_tL = np.genfromtxt(hex_file_tL, delimiter=',')
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
plt.plot(array_square_DE[0, :], array_square_DE[1, :], color='red', label='DE(t, L(t)) square')
plt.plot(array_hex_DE[0, :], array_hex_DE[1, :], color='blue', label='DE(t, L(t)) hex')
plt.grid()
plt.legend()
plt.show()
