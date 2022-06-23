"""Import section"""
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from firedrake.petsc import PETSc


"""General parameters"""
const_k = 1.
const_nu = 0.25
cell = "square"
t = 2.5
eps = 5e-5
step = 0.3
MAX_ITER = 40
plot_iter = False


"""Curve of dLDE(t, L)"""
if(plot_iter == True):
    file_FL = './results_csv/results_DE_given_t/FL_square_0-5_5_N200_iso.csv'
    file_FpL = './results_csv/results_DE_given_t/FpL_square_0-5_5_N200_iso.csv'
    array_FL = np.genfromtxt(file_FL, delimiter=',')
    array_FpL = np.genfromtxt(file_FpL, delimiter=',') 
    vect_DE = (t**2)*(array_FL[1, :]/(array_FL[0, :]**2)) + 4/array_FL[0, :]


"""Functions for the variational formulation"""
def sym_grad(v): 
    return fd.sym(fd.grad(v))

def sigma(v, nu):
    return (nu/((1+nu)*(1-2*nu)))*fd.tr(sym_grad(v))*fd.Identity(2) + (1/(1+nu))*sym_grad(v)


"""Definition of the unique mesh"""
PETSc.Sys.Print('Creation of the mesh ...')
# number of discretization points
M = 36
# definition of the mesh 
if(cell == "square"):
    mesh = fd.UnitSquareMesh(M, M)
if(cell == "hexagonal"):
    mesh = fd.Mesh('./built_meshes/UnitHexagonal.msh')
# space of vector functions on the mesh
V = fd.VectorFunctionSpace(mesh, "Lagrange", degree=2)
# definition of the loading 
f = fd.Function(V) 
x = fd.SpatialCoordinate(mesh)
# declaration of the test and trial functions to compute F(L)
u = fd.TrialFunction(V)
v = fd.TestFunction(V)

"""Gradient algorithm"""
PETSc.Sys.Print('Start of the gradient algorithm...')
# init point
L0 = 1.5
L = float(np.copy(L0))
for k in range(MAX_ITER): 
    if(cell == "square"):   
        f = fd.interpolate(L*(x - fd.Constant((1/2,1/2))), V)
    if(cell == "hexagonal"):
        f = fd.interpolate(L*x, V)
    a = (fd.inner(sigma(u, const_nu), sym_grad(v)) + const_k*(L**2)*fd.dot(u, v))*fd.dx
    l = -const_k*(L**2)*fd.dot(f, v)*fd.dx
    w = fd.Function(V, name="Displacement")
    fd.solve(a == l, w, solver_parameters={'ksp_type': 'cg'})
    FL = fd.assemble((1/2)*(fd.inner(sigma(w, const_nu), sym_grad(w)) + const_k*(L**2)*fd.dot(w + f, w + f))*fd.dx)
    FpL = fd.assemble(const_k*L*(fd.dot(w, w) + 3*fd.dot(w, f) + 2*fd.dot(f, f))*fd.dx)
    if(cell == "square"):
        grad_DE = (t**2)*((FpL*L - 2*FL)/(L**3)) - 4/(L**2)
    if(cell == 'hexagonal'): 
        grad_DE = 0 # A FAIRE
    PETSc.Sys.Print('Iteration: %d -> Gradient norm: %f -> Value of L: %f' % (k, np.abs(grad_DE), L))
    if(np.abs(grad_DE) < eps): 
        break
    if(plot_iter == True): 
        plt.figure(figsize=(9, 9))
        plt.plot(array_FpL[0, :], vect_DE, color='red', label='DE(t, L)')
        plt.axvline(x=L, linestyle='dotted', color='black')
        DE = (t**2)*(FL/(L**2)) + 4/L
        plt.plot(np.linspace(0, 5, 300), DE + grad_DE*(np.linspace(0, 5, 300) - L), color='blue')
        plt.grid()
        plt.legend()
        plt.show()
    L = float(L - step*grad_DE)
    """if(np.abs(grad_DE) < 0.02 and np.abs(grad_DE) > 1e-4): 
        step = step  + 1.0"""

