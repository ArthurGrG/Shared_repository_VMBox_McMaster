"""Import section"""
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from firedrake.petsc import PETSc


"""General parameters"""
plot_mesh = False
plot_displacement = False
write_FpL_csv = True; path_file_tL = "./results_csv/FpL_square_adj_0-1_5_N150.csv"
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
N = 150
# inf/sup boundaries
b_inf = 0.1
b_sup = 5.
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
# declaration of the test and trial functions to compute F(L)
u = fd.TrialFunction(V)
v = fd.TestFunction(V)
# declaration of the test and trial functions to compute F'(L) (adjoint) 
lbda = fd.TrialFunction(V)
vp = fd.TestFunction(V)

"""(3) Computation of the F(Li) and F'(Li) values with the adjoint state"""
PETSc.Sys.Print('Computation of the values of F(L) and F\'(L)...')
# initialization of the F(Li) vector
vect_FL = np.zeros(N)
vect_Fp_L = np.zeros(N)
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
    if(i == 0 and plot_displacement == True): 
        fig, axes = plt.subplots()
        contours = fd.trisurf(w)
        fig.colorbar(contours, ax=axes)
        plt.show(block=True)
    # computation of F(L)
    energy = (1/2)*(fd.inner(sigma(w, const_nu), sym_grad(w)) + const_k*L2*fd.dot(w + f, w + f))*fd.dx
    F_L = fd.assemble(energy)
    vect_FL[i] = F_L
    # computation of F'(L)
    FpL = L*(fd.dot(w, w) + 3*fd.dot(w, f) + 2*fd.dot(f, f))*fd.dx 
    vect_Fp_L[i] = fd.assemble(FpL)


if(write_FpL_csv == True):
    PETSc.Sys.Print("Writing the t(L) result in csv file...")
    np.savetxt(path_file_tL, np.c_[vect_L, vect_Fp_L].T, delimiter=',')