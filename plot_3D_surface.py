# imports 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt

# values to plot
path_file = './results_csv/results_DE_anisotropic/DE_square_t1-8_c2-0_rho0-2_4-0_L0-2-4-0_N50.csv'
vect_DE = np.genfromtxt(path_file, delimiter=',')

# number of points
N = 50
# inf/sup boundaries
b_inf = 0.2
b_sup = 4
# discretization step
h = (b_sup-b_inf)/(N-1)
# discretization vector
vect_L = np.arange(start=b_inf, stop=b_sup+(h-1e-7), step=h)[:]
vect_rho = np.arange(start=b_inf, stop=b_sup+(h-1e-7), step=h)[:]

# grid 
Y, X = np.meshgrid(vect_L, vect_rho)
Z = vect_DE.reshape((N, N))[:, :]

# plot 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
ax.set_xlabel('L')
ax.set_ylabel('rho')
ax.set_zlabel('DE(L, rho)')
plt.show()

a = 2; b = 27
c = 14; d = 39
vect_L = np.arange(start=b_inf, stop=b_sup+(h-1e-7), step=h)[a:b]
vect_rho = np.arange(start=b_inf, stop=b_sup+(h-1e-7), step=h)[c:d]
Y, X = np.meshgrid(vect_L, vect_rho)
Z = vect_DE.reshape((N, N))[c:d, a:b]

plt.figure()
plt.contour(X, Y, Z, 200)
plt.colorbar()
plt.show()


