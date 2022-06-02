"""Import section"""
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from firedrake.petsc import PETSc

"""General parameters"""
array_square_tL = np.genfromtxt('./results_csv/tL_square_0-1_10_N1500.csv', delimiter=',')
plot_mesh = False
const_k = 1.
const_nu = 0.25
cell = "square"
index = 0
L = float(array_square_tL[0, index])

"""Functions for the variational formulation"""
def sym_grad(v): 
    return fd.sym(fd.grad(v))

def sigma(v, nu):
    return (nu/((1+nu)*(1-2*nu)))*fd.tr(sym_grad(v))*fd.Identity(2) + (1/(1+nu))*sym_grad(v)


"""Definition of the unique mesh"""
PETSc.Sys.Print('Creation of the mesh ...')
# number of discretization points
M = 30
# definition of the mesh 
if(cell == "square"):
    mesh = fd.UnitSquareMesh(M, M)
if(cell == "hexagonal"):
    mesh = fd.Mesh('./built_meshes/UnitHexagonal.msh')
# plot of the mesh (optional)
if(plot_mesh is True):
    fig, axes = plt.subplots(figsize=(9, 9))
    fd.triplot(mesh, axes=axes)
    axes.legend()
    plt.show(block=True)
# space of vector functions on the mesh
V = fd.VectorFunctionSpace(mesh, "Lagrange", degree=2)
# definition of the loading 
f = fd.Function(V) 
x = fd.SpatialCoordinate(mesh)
# declaration of the test and trial functions
u = fd.TrialFunction(V)
v = fd.TestFunction(V)
lbda = fd.TrialFunction(V)
vp = fd.TestFunction(V)
    
"""Computation of F(L)"""
PETSc.Sys.Print('Computation of F(L)...')
# variable for L²
L2 = L**2
# update the function f wrt the used cell
if(cell == "square"):   
    f = fd.interpolate(float(L)*(x - fd.Constant((1/2,1/2))), V)
if(cell == "hexagonal"):
    f = fd.interpolate(float(L)*x, V)
# bilinear and linear forms
a = (fd.inner(sigma(u, const_nu), sym_grad(v)) + const_k*L2*fd.dot(u, v))*fd.dx
l = -const_k*L2*fd.dot(f, v)*fd.dx
# solution
w = fd.Function(V, name="Displacement")
fd.solve(a == l, w, solver_parameters={'ksp_type': 'cg'})
# computation of F(L)
energy = (1/2)*(fd.inner(sigma(w, const_nu), sym_grad(w)) + const_k*L2*fd.dot(w + f, w + f))*fd.dx
F_L = fd.assemble(energy)
# for the adjoint
ap = (fd.inner(sigma(lbda, const_nu), sym_grad(vp)) + const_k*L2*fd.dot(lbda, vp))*fd.dx
lp = np.dot(fd.Constant((0., 0.)), v)*fd.dx
wp = fd.Function(V, name="Adjoint")
fd.solve(ap == lp, wp)
energy_adjoint = const_k*L*(fd.dot(w + 2*L*f,w + L*f) + fd.dot(wp, 2*w + 3*L*f))*fd.dx
Fp_L = fd.assemble(energy_adjoint)
    
    
"""Computation of the t(L)"""
PETSc.Sys.Print('Computation of t(L)...')
# number of edges for the cell considered
if cell == "square": 
    nb_edges = 4
if(cell == "hexagonal"):
    nb_edges = 6
denom_tL = Fp_L*L - 2*F_L
t_L = np.sqrt((L*nb_edges)/np.abs(denom_tL))

print(L)
print(t_L)
print(array_square_tL[1, index])