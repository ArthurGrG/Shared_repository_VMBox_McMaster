"""Import section"""
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from firedrake.petsc import PETSc


"""General parameters"""
const_k = 1.
const_nu = 0.25
cell = "square"
t = 2.
eps = 1e-5
MAX_ITER = 40


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
# declaration of the test and trial functions to compute F(L)
u_adj = fd.TrialFunction(V)
v_adj = fd.TestFunction(V)


"""Curve of dLDE(t, L)"""
file_FL = './results_csv/results_DE_given_t/FL_square_0-5_5_N200_iso.csv'
file_FpL = './results_csv/results_DE_given_t/FpL_square_0-5_5_N200_iso.csv'
array_FL = np.genfromtxt(file_FL, delimiter=',')
array_FpL = np.genfromtxt(file_FpL, delimiter=',') 
vect_dLDE = (t**2)*(array_FpL[1, :]/(array_FpL[0, :]**2) - 2*array_FL[1, :]/(array_FL[0, :]**3)) - 4/(array_FL[0, :]**2)


"""Gradient algorithm"""
PETSc.Sys.Print('Start of the Newton algorithm...')
# init point
L0 = 1.5
L = float(np.copy(L0))
# loop for the Newton iterations 
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
        dL_DE = (t**2)*((FpL*L - 2*FL)/(L**3)) - 4/(L**2)
    if(cell == 'hexagonal'): 
        dL_DE = 0 # A FAIRE
    # stop condition dLDE = 0
    PETSc.Sys.Print('Iteration: %d -> dLDE(t, L) = %f -> Value of L: %f' % (k, dL_DE, L))
    if(np.abs(dL_DE) < eps): 
        break
    # computation of the adjoint state 
    a_adj = (fd.inner(sigma(u_adj, const_nu), sym_grad(v_adj)) + const_k*(L**2)*fd.dot(u_adj, v_adj))*fd.dx
    l_adj = -const_k*L*fd.dot(v_adj, 2*w + 3*f)*fd.dx
    adj = fd.Function(V, name="Adjoint")
    fd.solve(a_adj == l_adj, adj, solver_parameters={'ksp_type': 'cg'})
    # computation F''(L) 
    FppL = fd.assemble(const_k*(fd.dot(w, w) + 6*fd.dot(w, f) + 6*fd.dot(f, f) + L*fd.dot(adj, 2*w + 3*L*f))*fd.dx)
    if(cell == 'square'):
        d2L_DE = (1/(L**3))*((t**2)*(FppL*L - FpL) + 8 - (3*(t**2)/L)*(FpL*L - 2*FL))
        #d2L_DE = (1/(L**3))*((t**2)*(FppL*L - FpL) + 8) - ((3*(t**2))/(L**4))*(FpL*L - 2*FL)
    if(cell == 'hexagonal'):
        d2L_DE = 0 # A FAIRE

    plt.figure(figsize=(9, 9))
    plt.plot(array_FpL[0, :], vect_dLDE, color='red', label='dDE/dL(t, L)')
    #plt.plot(array_FL[0, :], array_FL[1, :], color='red', label='F(L)')
    plt.axvline(x=L, linestyle='solid', color='black')
    #plt.axhline(y=FL, linestyle='solid', color='black')
    plt.plot(np.linspace(0, 5, 300), dL_DE + d2L_DE*(np.linspace(0, 5, 300) - L), color='blue')
    plt.grid()
    plt.legend()
    plt.show()
    
    L = float(L - dL_DE/d2L_DE)