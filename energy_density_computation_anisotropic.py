"""Import section"""
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from firedrake.petsc import PETSc


"""Functions for the variational formulation"""
def sym_grad(v): 
    return fd.sym(fd.grad(v))

def sigma(u, v, nu, h, rho): 
    lbda_nu = ((nu*h)/((1+nu)*(1-nu)))
    mu_nu = (h/(2*(1+nu)))
    A = np.zeros((2, 2, 2, 2))
    A[0, 0, 0, 0] =  (rho**3)*(lbda_nu + 2*mu_nu)
    A[1, 1, 1, 1] = (1/rho)*(lbda_nu + 2*mu_nu)
    A[1, 1, 0, 0] = rho*lbda_nu; A[0, 0, 1, 1] = rho*lbda_nu
    A[0, 1, 0, 1] = rho*mu_nu; A[1, 0, 0, 1] = rho*mu_nu; A[0, 1, 1, 0] = rho*mu_nu; A[1, 0, 1, 0] = rho*mu_nu
    return fd.inner(fd.as_tensor(A), fd.outer(sym_grad(u), sym_grad(v)))

def sigma_R(u, v, nu, h, rho): 
    lbda_nu = ((h*nu)/((1+nu)*(1-nu)))
    mu_nu = (h/(2*(1+nu)))
    A = np.zeros((2, 2, 2, 2))
    A[0, 0, 0, 0] =  rho*(lbda_nu + 2*mu_nu)
    A[1, 1, 1, 1] = -(1/(rho**3))*(lbda_nu + 2*mu_nu)
    return fd.inner(fd.as_tensor(A), fd.outer(sym_grad(u), sym_grad(v)))


"""General parameters"""
write_tL_csv = True; path_file = "./results_csv/results_DE_anisotropic/DE&grad_square_t11147_c1_rho0-009_1-1_L0-0001_0-5_N50.csv"
const_nu = 0.45
hf = 2.6e-4
const_k = 2e2
c2 = 1.0
t = 11147.77891255


"""Discretization for the values of L and rho"""
# number of points
N = 40
# inf/sup boundaries
b_inf_rho = 0.01
b_sup_rho = 1.1
b_inf_L = 0.0001
b_sup_L = 0.01
# discretization step
h_rho = (b_sup_rho-b_inf_rho)/(N-1)
h_L = (b_sup_L-b_inf_L)/(N-1)
# discretization vector
vect_rho = np.arange(start=b_inf_rho, stop=b_sup_rho+(h_rho-1e-7), step=h_rho)
vect_L = np.arange(start=b_inf_L, stop=b_sup_L+(h_L-1e-7), step=h_L)



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
vect_DE = np.zeros((vect_L.size*vect_rho.size, 3))
# loop over the values of L 
for i in range(0, N): 
    for j in range(0, N): 
        PETSc.Sys.Print('Index %d / %d' % (i*N + j, N*N))
        # float variables for L and L²
        L = float(vect_L[i])
        rho = float(vect_rho[j])
        L2 = L**2
        # update the function f 
        vec_f = fd.as_vector((x[0] - 1/2, c2*(x[1] - 1/2))) 
        f = fd.interpolate(L*vec_f, V)
        # bilinear and linear forms
        mat_bil = fd.as_matrix([[rho, 0], [0, 1]])
        a = (sigma(u, v, const_nu, hf, rho) + const_k*rho*L2*fd.dot(mat_bil*u, mat_bil*v))*fd.dx
        l = -const_k*(rho**2)*L2*fd.dot(f, v)*fd.dx
        # solution
        w = fd.Function(V, name="Displacement")
        fd.solve(a == l, w, solver_parameters={'ksp_type': 'cg'})
        # computation of F(L)
        f_energy = fd.interpolate(L*fd.as_vector((x[0] - 1/2, c2*rho*(x[1] - 1/2))), V)
        energy = (1/2)*(sigma(w, w, const_nu, hf, rho) + const_k*L2*rho*fd.dot(mat_bil*w + f_energy, mat_bil*w + f_energy))*fd.dx
        F_L = fd.assemble(energy)
        vect_DE[i*N + j, 0] = (t**2)*(F_L/(rho*L2)) + 1/L + 1/(rho*L)
        # for the gradient
        dLF = fd.assemble(const_k*rho*L*(fd.dot(mat_bil*w, mat_bil*w) + 3*fd.dot(mat_bil*w, f_energy) + 2*fd.dot(f_energy, f_energy))*fd.dx)
        vec_x = fd.as_vector((1, 0))
        vec_y = fd.as_vector((0, 1))
        drhoF_ = fd.assemble((sigma_R(w, w, const_nu, hf, rho) + const_k*L2*(fd.dot(w, f) + rho*(fd.dot(fd.dot(vec_x, w), fd.dot(vec_x, w)) + fd.dot(fd.dot(vec_y, f), fd.dot(vec_y, f)))))*fd.dx)
        vect_DE[i*N + j, 1] = ((t**2)/(L**2))*drhoF_ - 1/((rho**2)*L)
        vect_DE[i*N + j, 2] = ((t**2)/(rho*(L**3)))*(dLF*L - 2*F_L) - 1/(L**2) - 1/(rho*(L**2))

if(write_tL_csv == True):
    PETSc.Sys.Print("Writing the density in csv file...")
    np.savetxt(path_file, vect_DE, delimiter=',')
