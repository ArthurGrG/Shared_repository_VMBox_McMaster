"""Import section"""
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from firedrake.petsc import PETSc

"""General parameters"""
write_FL_csv = True ; path_file_FL = "./results_csv/results_DE_given_t/FL_square_0-01_25_N200_iso.csv"
write_FpL_csv = True ; path_file_FpL = "./results_csv/results_DE_given_t/FpL_square_0-5_5_N200_iso.csv"
write_denom_csv = False ; path_file_denom = "./results_csv/results_DE_given_t/denom_square_0-1_10_N150_iso.csv"
const_k = 1.
const_nu = 0.25
cell = "square"

"""Functions for the variational formulation"""
def sym_grad(v): 
    return fd.sym(fd.grad(v))

def sigma(v, nu):
    return (nu/((1+nu)*(1-2*nu)))*fd.tr(sym_grad(v))*fd.Identity(2) + (1/(1+nu))*sym_grad(v)


"""(1) Discretization for the values of L"""
# number of points
N = 200
# inf/sup boundaries
b_inf = 0.5
b_sup = 5
# discretization step
h = (b_sup-b_inf)/(N-1)
# discretization vector
vect_L = np.arange(start=b_inf, stop=b_sup+(h-1e-7), step=h)


"""(2) Definition of the unique mesh"""
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
    

"""(3) Computation of the F(Li) and F'(Li) values with the adjoint state"""
PETSc.Sys.Print('Computation of the values of F(L) and F\'(L)...')
# initialization of the F(Li) vector
vect_FL = np.zeros(N)
vect_Fp_L = np.zeros(N)
check_denom = np.zeros(N)
# loop over the values of L 
for i in range(0, N): 
    PETSc.Sys.Print('Index %d / %d' % (i, N))
    # float variables for L and L²
    L = float(vect_L[i])
    L2 = L**2
    # update the function f wrt the used cell
    if(cell == "square"):   
        f = fd.interpolate(L*(x - fd.Constant((1/2,1/2))), V)
    if(cell == "hexagonal"):
        f = fd.interpolate(L*x, V)
    # bilinear and linear forms
    a = (fd.inner(sigma(u, const_nu), sym_grad(v)) + const_k*L2*fd.dot(u, v))*fd.dx
    l = -const_k*L2*fd.dot(f, v)*fd.dx
    # solution
    w = fd.Function(V, name="Displacement")
    fd.solve(a == l, w, solver_parameters={'ksp_type': 'cg'})
    # computation of F(L)
    energy = (1/2)*(fd.inner(sigma(w, const_nu), sym_grad(w)) + const_k*L2*fd.dot(w + f, w + f))*fd.dx
    F_L = fd.assemble(energy)
    vect_FL[i] = F_L
    # computation of F'(L)
    FpL = const_k*L*(fd.dot(w, w) + 3*fd.dot(w, f) + 2*fd.dot(f, f))*fd.dx 
    vect_Fp_L[i] = fd.assemble(FpL)
    # computation of F'(L)L - 2F(L) which is the denominator for t(L) 
    denom = (-fd.inner(sigma(w, const_nu), sym_grad(w)) + const_k*L2*(fd.dot(w, f) + fd.dot(f, f)))*fd.dx
    check_denom[i] = fd.assemble(denom)


if(write_FL_csv == True):
    PETSc.Sys.Print("Writing the F(L) results in csv file...")
    np.savetxt(path_file_FL, np.c_[vect_L, vect_FL].T, delimiter=',')
if(write_FpL_csv == True):
    PETSc.Sys.Print("Writing the F'(L) results in csv file...")
    np.savetxt(path_file_FpL, np.c_[vect_L, vect_Fp_L].T, delimiter=',')
if(write_denom_csv == True):
    PETSc.Sys.Print("Writing the denominator F'(L)L - 2F(L) (analytic computation) results in csv file...")
    np.savetxt(path_file_denom, np.c_[vect_L, check_denom].T, delimiter=',')
