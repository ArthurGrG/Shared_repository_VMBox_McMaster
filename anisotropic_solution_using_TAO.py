"""derivative of firedrake
setValues from PETCs """

"""Import section"""
import numpy as np 
import matplotlib.pyplot as plt 
import firedrake as fd
from petsc4py import PETSc


"""General parameters"""
cell = "square"
const_nu = 0.45
hf = 2.6e-4
const_k = 2e2
c2 = 1.0


"""Definition of the square mesh"""
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


"""Function definitions"""
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

# rho = y[0] // L = y[1]
def FormObjGrad(tao, y, G): 
    print(y.array)
    # parameters to be optimize
    rho = float(y[0])
    L = float(y[1])
    L2 = float(L**2)
    # computation of the objective function
    if(cell=='square'):
        vec_f = fd.as_vector((x[0] - 1/2, c2*(x[1] - 1/2))) 
        f = fd.interpolate(L*vec_f, V)
        f_energy = fd.interpolate(L*fd.as_vector((x[0] - 1/2, c2*rho*(x[1] - 1/2))), V)
    if(cell=='hexagonal'):
        vec_f = fd.as_vector((x[0], c2*x[1]))
        f = fd.interpolate(L*vec_f, V)
        f_energy = fd.interpolate(L*fd.as_vector((x[0], c2*rho*x[1])), V)
    mat_bil = fd.as_matrix([[rho, 0], [0, 1]])
    a = (sigma(u, v, const_nu, hf, rho) + const_k*rho*L2*fd.dot(mat_bil*u, mat_bil*v))*fd.dx
    l = -const_k*(rho**2)*L2*fd.dot(f, v)*fd.dx
    w = fd.Function(V, name="Displacement")
    fd.solve(a == l, w, solver_parameters={'ksp_type': 'cg'})
    F = fd.assemble((1/2)*(sigma(w, w, const_nu, hf, rho) + const_k*L2*rho*fd.dot(mat_bil*w + f_energy, mat_bil*w + f_energy))*fd.dx)
    # computation of the gradient
    G.zeroEntries()
    dLF = fd.assemble(const_k*rho*L*(fd.dot(mat_bil*w, mat_bil*w) + 3*fd.dot(mat_bil*w, f_energy) + 2*fd.dot(f_energy, f_energy))*fd.dx)
    vec_x = fd.as_vector((1, 0))
    vec_y = fd.as_vector((0, 1))
    drhoF_ = fd.assemble((sigma_R(w, w, const_nu, hf, rho) + const_k*L2*(fd.dot(w, f) + rho*(fd.dot(fd.dot(vec_x, w), fd.dot(vec_x, w)) + fd.dot(fd.dot(vec_y, f), fd.dot(vec_y, f)))))*fd.dx)
    if(cell=='square'):
        G[0] = ((t**2)/(L**2))*drhoF_ - 1/((rho**2)*L)
        G[1] = ((t**2)/(rho*(L**3)))*(dLF*L - 2*F) - 1/(L**2) - 1/(rho*(L**2))
        FF = (t**2)*(F/(rho*L2)) + (rho*L + L)/(rho*L2)
    if(cell=='hexagonal'):
        G[0] = (2/(np.sqrt(3)*3))*(((t**2)/(L**2))*drhoF_ - 3/(L*(rho**2)*np.sqrt(rho**2 + 3)))
        G[1] = (2/(np.sqrt(3)*3))*(((t**2)/(rho*(L**3)))*(dLF*L - 2*F) - (1/L2)*(1 + np.sqrt(rho**2 + 3)/rho))
        FF = (2/(np.sqrt(3)*3))*((t**2)*(F/(rho*L2)) + (1/L)*(1 + np.sqrt(rho**2 + 3)/rho))
    #print(G.array)
    return FF


"""Start of the loop over t if wanted"""
nb_discr = 1
vect_rho = np.zeros(nb_discr)
vect_L = np.zeros(nb_discr)
vect_t = np.array([11147.77891255])
cmpt = 0
for val in vect_t:
    t = val
    print("Loop index = " + str(cmpt) + "/" + str(nb_discr))
    print("Value of t = " + str(t))
    print("Value of c2 = "+ str(c2))
    print("Value of k = " + str(const_k))
    """Create solution vector"""
    y = PETSc.Vec().create(PETSc.COMM_SELF)
    y.setSizes(int(2))
    y.setFromOptions()
    y.set(1.0) # initial value 
    """Boundaries"""
    lb = PETSc.Vec().create(PETSc.COMM_SELF)
    lb.setSizes(int(2))
    lb.setFromOptions()
    lb.set(1e-4)
    ub = PETSc.Vec().create(PETSc.COMM_SELF)
    ub.setSizes(int(2))
    ub.setFromOptions()
    ub.set(1e8)
    """TAO section"""
    tao = PETSc.TAO().create(PETSc.COMM_SELF)
    tao.setType('blmvm')
    tao.setFromOptions()
    tao.setObjectiveGradient(FormObjGrad, None)
    tao.setVariableBounds(lb, ub)
    tao.setSolution(y)
    tao.solve()
    print(y.array)
    vect_rho[cmpt] = y.array[0]
    vect_L[cmpt] = y.array[1]
    tao.destroy()
    cmpt = cmpt + 1
