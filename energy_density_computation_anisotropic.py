"""Import section"""
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from firedrake.petsc import PETSc


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


"""General parameters"""
write_tL_csv = True; path_file = "./results_csv/results_DE_anisotropic/DE_square_t1-8_c2-0_L1-9_rho0-1_10-0_N200.csv"
const_k = 1.
const_nu = 0.25
t = 1.8
c2 = 2.


"""Discretization for the values of L and rho"""
# number of points
N = 200
# inf/sup boundaries
b_inf = 0.1
b_sup = 10
# discretization step
h = (b_sup-b_inf)/(N-1)
# discretization vector
vect_L = np.array([1.9])#np.arange(start=b_inf, stop=b_sup+(h-1e-7), step=h)
vect_rho = np.arange(start=b_inf, stop=b_sup+(h-1e-7), step=h)


"""Definition of the square mesh"""
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


"""Computation of the density for a grid"""
PETSc.Sys.Print('Computation of the values of DE(t, c2, rho, L)...')
# initialization of the F(Li) vector
vect_DE = np.zeros(vect_L.size*vect_rho.size)
# loop over the values of L 
for i in range(0, 1): 
    for j in range(0, N): 
        PETSc.Sys.Print('Index %d / %d' % (i*N + j, N))
        # float variables for L and L²
        L = float(vect_L[i])
        rho = float(vect_rho[j])
        L2 = L**2
        # update the function f 
        vec_f = fd.as_vector((x[0] - 1/2, c2*(x[1] - 1/2))) 
        f = fd.interpolate(L*vec_f, V)
        # bilinear and linear forms
        mat_bil = fd.as_matrix([[rho, 0], [0, 1]])
        a = (sigma(u, v, const_nu, rho) + const_k*rho*L2*fd.dot(mat_bil*u, mat_bil*v))*fd.dx
        l = -const_k*(rho**2)*L2*fd.dot(f, v)*fd.dx
        # solution
        w = fd.Function(V, name="Displacement")
        fd.solve(a == l, w, solver_parameters={'ksp_type': 'cg'})
        # computation of F(L)
        f_energy = fd.interpolate(L*fd.as_vector((x[0] - 1/2, c2*rho*(x[1] - 1/2))), V)
        energy = (1/2)*(sigma(w, w, const_nu, rho) + const_k*L2*rho*fd.dot(mat_bil*w + f_energy, mat_bil*w + f_energy))*fd.dx
        F_L = fd.assemble(energy)
        vect_DE[i*N + j] = (t**2)*(F_L/(rho*L2)) + 2/L + 2/(rho*L)


if(write_tL_csv == True):
    PETSc.Sys.Print("Writing the density in csv file...")
    np.savetxt(path_file, vect_DE, delimiter=',')