"""Import section"""
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from firedrake.petsc import PETSc


"""General parameters"""
const_k = 1.
const_nu = 0.25
cell = "square"
eps = 5e-5
step = 0.3
MAX_ITER = 40
L_MIN = 0.05
L_MAX = 15
write_result = True; path_result = './results_csv/results_dichotomy/Lt_iso_square_0-1_10_N50.csv'


"""Values of t"""
path_t = './results_csv/results_adjoint/tL_square_adj_0-1_10_N200.csv'
t_values = np.genfromtxt(path_t, delimiter=',')[1, :]
vect_t = np.zeros(50)
cmp = 0
for j in range(0, t_values.size):
    if(j%4 == 0): 
        vect_t[cmp] = t_values[j]
        cmp = cmp + 1



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

"""Dichotomy algorithm"""
PETSc.Sys.Print('Start of the dichotomy algorithm...')
vect_Lt = np.zeros(vect_t.size)
for i in range(0, vect_t.size):
    PETSc.Sys.Print('Index %d / %d ----------------------------------------------------------' % (i, vect_t.size))
    # init point
    L_inf = np.copy(L_MIN)
    L_sup = np.copy(L_MAX)
    t = float(vect_t[i])
    for k in range(MAX_ITER): 
        L = float((L_sup + L_inf)/2)
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
        PETSc.Sys.Print('Iteration: %d -> Gradient: %f -> Value of L_inf: %f // Value of L_sup: %f // Value of L: %f' % (k, grad_DE, L_inf, L_sup, L))
        if(np.abs(grad_DE) < eps): 
            break
        if(grad_DE > 0): 
            L_sup = np.copy(L)
        else: 
            L_inf = np.copy(L)
    vect_Lt[i] = L

if(write_result == True):
    PETSc.Sys.Print("Writing the L(t) results in csv file...")
    np.savetxt(path_result, np.c_[vect_t, vect_Lt].T, delimiter=',')

