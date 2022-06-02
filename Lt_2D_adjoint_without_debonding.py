"""Import section"""
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from firedrake.petsc import PETSc

"""General parameters"""
plot_mesh = False
plot_displacement = False
plot_denom_tL = False
write_tL_csv = True; path_file_tL = "./results_csv/tL_hex_adj_0-1_10_N200.csv"
write_DE_csv = True; path_file_DE = "./results_csv/DE_hex_adj_0-1_10_N200.csv"
const_k = 1.
const_nu = 0.25
cell = "hexagonal"

"""Functions for the variational formulation"""
def sym_grad(v): 
    return fd.sym(fd.grad(v))

def sigma(v, nu):
    return (nu/((1+nu)*(1-2*nu)))*fd.tr(sym_grad(v))*fd.Identity(2) + (1/(1+nu))*sym_grad(v)


"""(1) Discretization for the values of L"""
# number of points
N = 200
# inf/sup boundaries
b_inf = 0.1
b_sup = 10
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
    # bilinear and linear forms for the adjoint state
    ap = (fd.inner(sigma(lbda, const_nu), sym_grad(vp)) + const_k*L2*fd.dot(lbda, vp))*fd.dx
    lp = np.dot(fd.Constant((0., 0.)), v)*fd.dx
    # solution for adjoint state
    wp = fd.Function(V, name="Adjoint")
    fd.solve(ap == lp, wp)
    # computation of F'(L)
    energy_adjoint = const_k*L*(fd.dot(w + 2*L*f,w + L*f) + fd.dot(wp, 2*w + 3*L*f))*fd.dx
    Fp_L = fd.assemble(energy_adjoint)
    vect_Fp_L[i] = Fp_L

    
"""(4) Computation of the t(Li), i = 0, ..., N and wrinting in csv file"""
PETSc.Sys.Print('Computation of t(L)...')
# number of edges for the cell considered
if cell == "square": 
    nb_edges = 4
if(cell == "hexagonal"):
    nb_edges = 6
# initialization of the t(Li) vector 
vect_tL = np.zeros(N)
# initialization of the vector of the denominator of t(L) (optional)
if(plot_denom_tL == True):
    vect_denom_tL = np.zeros(N)
# loop over the values of L 
for i in range(0, N): 
    L = vect_L[i]
    FL = vect_FL[i]
    FpL = vect_Fp_L[i]
    denom_tL = FpL*L - 2*FL
    vect_tL[i] = np.sqrt((L*nb_edges)/np.abs(denom_tL))
    if(plot_denom_tL == True): 
        vect_denom_tL[i] = denom_tL
# writing the result of t(L) in a file 
if(write_tL_csv == True):
    PETSc.Sys.Print("Writing the t(L) result in csv file...")
    np.savetxt(path_file_tL, np.c_[vect_L, vect_tL].T, delimiter=',')
# writing the result of the energy density DE(t, L(t))
if(write_DE_csv == True): 
    if(cell == "square"):
        DE = ((vect_tL**2)*vect_FL + nb_edges*vect_L)/(vect_L**2)
    if(cell == "hexagonal"):
        DE = (2/(np.sqrt(3)*3))*(((vect_tL**2)*vect_FL + nb_edges*vect_L)/(vect_L**2))
    PETSc.Sys.Print("Writing the DE(t, L(t)) result in csv file...")
    np.savetxt(path_file_DE, np.c_[vect_tL, DE].T, delimiter=',')
# plot denominator of t(L) (optional)
if(plot_denom_tL == True):
    plt.figure()
    plt.plot(vect_L, vect_denom_tL, color='red', marker='o', label='denominator of t(L)')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.legend()
    plt.show(block=True)
