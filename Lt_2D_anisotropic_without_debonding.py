"""Import section"""
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from firedrake.petsc import PETSc

"""General parameters"""
plot_mesh = False
plot_displacement = False
plot_denom_tL = False
write_tL_csv = False; path_file_tL = "./results_csv/tL_hex_aniso_rho1-3_c1-5_0-1_10_N150.csv"
write_DE_csv = True; path_file_DE = "./results_csv/DE_hex_aniso_rho1-3_c1-5_0-1_10_N150.csv"
const_k = 2e-4
const_nu = 0.45
hf = 260
rho = 1.0
c2 = 1.0
cell = "square"

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


"""(1) Discretization for the values of L"""
# number of points
N = 1
# inf/sup boundaries
b_inf = 0.1
b_sup = 10
# discretization step
#h = (b_sup-b_inf)/(N-1)
# discretization vector
vect_L = np.array([4000.0])#np.arange(start=b_inf, stop=b_sup+(h-1e-7), step=h)


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
        vec_f = fd.as_vector((x[0] - 1/2, c2*(x[1] - 1/2))) 
        f = fd.interpolate(L*vec_f, V)
    if(cell == "hexagonal"):
        vec_f = fd.as_vector((x[0], c2*x[1]))
        f = fd.interpolate(L*vec_f, V)
    # bilinear and linear forms
    mat_bil = fd.as_matrix([[rho, 0], [0, 1]])
    a = (sigma(u, v, const_nu, hf, rho) + const_k*rho*L2*fd.dot(mat_bil*u, mat_bil*v))*fd.dx
    l = -const_k*(rho**2)*L2*fd.dot(f, v)*fd.dx
    # solution
    w = fd.Function(V, name="Displacement")
    fd.solve(a == l, w, solver_parameters={'ksp_type': 'cg'})
    if(i == 0 and plot_displacement == True): 
        fig, axes = plt.subplots()
        contours = fd.trisurf(w)
        fig.colorbar(contours, ax=axes)
        plt.show(block=True)
    # computation of F(L)
    f_energy = fd.interpolate(L*fd.as_vector((x[0] - 1/2, c2*rho*(x[1] - 1/2))), V)
    energy = (1/2)*(sigma(w, w, const_nu, hf, rho) + const_k*L2*rho*fd.dot(mat_bil*w + f_energy, mat_bil*w + f_energy))*fd.dx
    F_L = fd.assemble(energy)
    vect_FL[i] = F_L
    # computation of F'(L)
    FpL = const_k*rho*L*(fd.dot(mat_bil*w, mat_bil*w) + 3*fd.dot(mat_bil*w, f_energy) + 2*fd.dot(f_energy, f_energy))*fd.dx 
    vect_Fp_L[i] = fd.assemble(FpL)

    
"""(4) Computation of the t(Li), i = 0, ..., N and wrinting in csv file"""
PETSc.Sys.Print('Computation of t(L)...')
# number of edges for the cell considered
if cell == "square": 
    weight_frac = (rho + 1)
if(cell == "hexagonal"):
    weight_frac = 2*(rho + np.sqrt(rho**2 + 3))
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
    vect_tL[i] = np.sqrt((L*weight_frac)/np.abs(denom_tL))
    if(plot_denom_tL == True): 
        vect_denom_tL[i] = denom_tL
# writing the result of t(L) in a file 
if(write_tL_csv == True):
    PETSc.Sys.Print("Writing the t(L) result in csv file...")
    np.savetxt(path_file_tL, np.c_[vect_L, vect_tL].T, delimiter=',')
# writing the result of the energy density DE(t, L(t))
if(write_DE_csv == True): 
    if(cell == "square"):
        vect_tL = vect_tL = 0.0021022
        DE = (vect_tL**2)*(vect_FL/(rho*(vect_L**2))) + weight_frac/(rho*vect_L)
        print(DE)
    if(cell == "hexagonal"):
        DE = (2/(np.sqrt(3)*3))*((vect_tL**2)*(vect_FL/(rho*(vect_L**2))) + weight_frac/(rho*vect_L))
    PETSc.Sys.Print("Writing the DE(t, L(t)) result in csv file...")
    #np.savetxt(path_file_DE, np.c_[vect_tL, DE].T, delimiter=',')
# plot denominator of t(L) (optional)
if(plot_denom_tL == True):
    plt.figure()
    plt.plot(vect_L, vect_denom_tL, color='red', marker='o', label='denominator of t(L)')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.legend()
    plt.show(block=True)