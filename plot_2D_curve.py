import matplotlib.pyplot as plt 
import numpy as np 


file_path = './results_csv/results_isotropic_juillet/tL_square_0-003_0-3_micrometres_N150.csv'
file_path2 = './results_csv/results_isotropic_juillet/DE_square_adj_0-01_100_N150.csv'
tL = np.genfromtxt(file_path, delimiter=',')
DE = np.genfromtxt(file_path2, delimiter=',')

plt.figure()
plt.plot(tL[1, :], tL[0, :], color='red', label='L(t)')
#plt.plot(np.linspace(0.1, 10, 150), DE2, color='green')
plt.legend()
plt.show()



