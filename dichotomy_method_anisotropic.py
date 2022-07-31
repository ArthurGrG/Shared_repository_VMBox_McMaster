"""Import section"""
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from firedrake.petsc import PETSc


"""General parameters"""
const_k = 1.
const_nu = 0.25
MAX_ITER = 40
L_MIN = 0.05; rho_MIN = 0.05
L_MAX = 15; rho_MAX = 15
t = 1.8 
c2 = 2.0
eps = 1e-5


"""Functions for the variational formulation"""
def sym_grad(v): 
    return fd.sym(fd.grad(v))

def sigma(u, v, nu, rho): 
    lbda_nu = (nu/((1+nu)*(1-2*nu)))
    mu_nu = (1/(2*(1+nu)))
    A = np.zeros((2, 2, 2, 2))
    A[0, 0, 0, 0] =  (rho**3)*(lbda_nu + 2*mu_nu)
    A[1, 1, 1, 1] = (1/rho)*(lbda_nu + 2*mu_nu)
    A[0, 1, 0, 1] = rho*mu_nu; A[1, 0, 0, 1] = rho*mu_nu; A[0, 1, 1, 0] = rho*mu_nu; A[1, 0, 1, 0] = rho*mu_nu
    return fd.inner(fd.as_tensor(A), fd.outer(sym_grad(u), sym_grad(v)))

def sigma_R(u, v, nu, rho): 
    lbda_nu = (nu/((1+nu)*(1-2*nu)))
    mu_nu = (1/(2*(1+nu)))
    A = np.zeros((2, 2, 2, 2))
    A[0, 0, 0, 0] =  rho*(lbda_nu + 2*mu_nu)
    A[1, 1, 1, 1] = -(1/(rho**3))*(lbda_nu + 2*mu_nu)
    return fd.inner(fd.as_tensor(A), fd.outer(sym_grad(u), sym_grad(v)))


"""Definition of the unique mesh"""
PETSc.Sys.Print('Creation of the mesh ...')
# number of discretization points
M = 36
# definition of the mesh 
mesh = fd.UnitSquareMesh(M, M)
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
L_inf = np.copy(L_MIN); rho_inf = np.copy(rho_MIN)
L_sup = np.copy(L_MAX); rho_sup = np.copy(rho_MAX)
for k in range(0, MAX_ITER):
    # float variables for L and L²
    L = float((L_sup + L_inf)/2)
    rho = float((rho_sup + rho_inf)/2)
    L2 = L**2
    vec_f = fd.as_vector((x[0] - 1/2, c2*(x[1] - 1/2))) 
    f = fd.interpolate(L*vec_f, V)
    # bilinear and linear forms
    mat_bil = fd.as_matrix([[rho, 0], [0, 1]])
    a = (sigma(u, v, const_nu, rho) + const_k*rho*L2*fd.dot(mat_bil*u, mat_bil*v))*fd.dx
    l = -const_k*(rho**2)*L2*fd.dot(f, v)*fd.dx
    # solution
    w = fd.Function(V, name="Displacement")
    fd.solve(a == l, w, solver_parameters={'ksp_type': 'cg'})
    # computation of F(rho, L)
    f_energy = fd.interpolate(L*fd.as_vector((x[0] - 1/2, c2*rho*(x[1] - 1/2))), V)
    F = fd.assemble((1/2)*(sigma(w, w, const_nu, rho) + const_k*L2*rho*fd.dot(mat_bil*w + f_energy, mat_bil*w + f_energy))*fd.dx)
    # computation of dLF(rho, L)
    dLF = fd.assemble(const_k*rho*L*(fd.dot(mat_bil*w, mat_bil*w) + 3*fd.dot(mat_bil*w, f_energy) + 2*fd.dot(f_energy, f_energy))*fd.dx)
    # computation of drhoF(rho, L)
    vec_x = fd.as_vector((1, 0))
    vec_y = fd.as_vector((0, 1))
    drhoF_ = fd.assemble((sigma_R(w, w, const_nu, rho) + const_k*L2*(fd.dot(w, f) + rho*(fd.dot(fd.dot(vec_x, w), fd.dot(vec_x, w)) + fd.dot(fd.dot(vec_y, f), fd.dot(vec_y, f)))))*fd.dx)
    # gradient computation (peut être optimisé)
    grad_DE = np.array([((t**2)/(L**2))*drhoF_ - 2/((rho**2)*L), ((t**2)/(rho*(L**3)))*(dLF*L - 2*F) - 2/(L**2) - 2/(rho*(L**2))])
    PETSc.Sys.Print('Iteration: %d ----------------------------------' % (k))
    PETSc.Sys.Print('Value of L_inf: %f // Value of L_sup: %f // Value of L: %f' % (L_inf, L_sup, L))
    PETSc.Sys.Print('Value of rho_inf: %f // Value of rho_sup: %f // Value of rho: %f' % (rho_inf, rho_sup, rho))
    PETSc.Sys.Print('Gradient: [%f, %f]' % (grad_DE[0], grad_DE[1]))
    if(np.linalg.norm(grad_DE) < eps): 
        break 
    if(grad_DE[0] > 0): 
        rho_sup = np.copy(rho)
    if(grad_DE[0] < 0):
        rho_inf = np.copy(rho)
    if(grad_DE[1] > 0): 
        L_sup = np.copy(L)
    if(grad_DE[1] < 0):
        L_inf = np.copy(L)
    