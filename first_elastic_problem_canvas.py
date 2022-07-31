"""Import section"""
import numpy as np 
import matplotlib.pyplot as plt 
import firedrake as fd
from petsc4py import PETSc


"""Function definitions"""
def eps(v):
    return fd.sym(fd.grad(v))

def sigma(v, lmbda, mu):
    return lmbda*fd.tr(eps(v))*fd.Identity(2) + 2.0*mu*eps(v)


"""General parameters"""
E = 200 # doesn't have any impact here
nu = 0.3
lmbda = 1e-4*(E*nu)/((1+nu)*(1-nu))
mu = 1e-4*E/(2*(1+nu))
t = 1.0

"""Definition of the square mesh"""
PETSc.Sys.Print('Creation of the mesh ...')
# number of discretization points
M = 30
# definition of the mesh 
mesh = fd.Mesh('./built_meshes/square.msh')#fd.UnitSquareMesh(M, M)
# space of vector functions on the mesh
V = fd.VectorFunctionSpace(mesh, "Lagrange", degree=2)
# declaration of the test and trial functions to compute F(L)
u = fd.TrialFunction(V)
v = fd.TestFunction(V)


"""Boundary conditions"""
bc_1 = fd.DirichletBC(V, fd.Constant((0.5*t, 0.5*t)), 12)
bc_2 = fd.DirichletBC(V, fd.Constant((0.5*t, -0.5*t)), 13)
bc_3 = fd.DirichletBC(V, fd.Constant((-0.5*t, -0.5*t)), 14)
bc_4 = fd.DirichletBC(V, fd.Constant((-0.5*t, 0.5*t)), 15)
bc = [bc_1, bc_2, bc_3, bc_4]

"""Variational solution"""
a = fd.inner(sigma(u, lmbda, mu), eps(v))*fd.dx(2)
l = fd.dot(fd.Constant((0, 0)), v)*fd.dx(2)
w = fd.Function(V, name="Displacement")
fd.solve(a==l, w, bcs=bc, solver_parameters={'ksp_type': 'cg'})

"""Vizualization of the weak solution
fig, axes = plt.subplots()
contours = fd.trisurf(w)
fig.colorbar(contours, ax=axes)
plt.show(block=True)"""

fd.File("./pvd_files/displacement_first_elastic_problem_canvas.pvd").write(w)

"""Numpy array for w and the mesh points"""
x = fd.SpatialCoordinate(mesh)
coord_mesh = fd.interpolate(fd.as_vector((x[0], x[1])), V)

w_array = w.vector().array()
coord_array = coord_mesh.vector().array()

size = int(w_array.size/2)

w_array = w_array.reshape(size, 2)
coord_array = coord_array.reshape(size, 2)

amplitude_w = np.linalg.norm(w_array, axis=1)

"""Vizualisation of the mesh points
plt.figure(figsize=(8, 8))coord_array ay[:, 1], amplitude_w, levels=14, linewidths=0.5, colors='k')
cntr = ax.tricontourf(coord_array[:, 0], coord_array[:, 1], amplitude_w, levels=14, cmap="RdBu_r")
fig.colorbar(cntr, ax=ax)
plt.show()"""


"""Numpy array for e(w)"""
V_tensor = fd.TensorFunctionSpace(mesh, "Lagrange", degree=2)

eps_w = fd.project(eps(w), V_tensor)

#just to check that it's the same as coord_array (verified)
coord_mesh_tensor = fd.interpolate(fd.as_matrix([[x[0], x[1]],[0, 0]]), V_tensor) 
coord_array_tensor = coord_mesh_tensor.vector().array().reshape(size, 2, 2)

eps_w_array = eps_w.vector().array().reshape(size, 2, 2)
#print(eps_w_array)


"""Restriction to the top right frame of the rectangle (loadings same sign)"""
indices = []
for i in range(0, size): 
    c = coord_array[i, :]
    if(c[0] > 0 and c[1] > 0):
        indices.append(i)


"""Clustering the grid points wrt to the eigenvalues of e(w)"""
ind_same_sign_pos = []
ind_same_sign_neg = [] # empty a priori
ind_diff_sign = []
for i in range(0, size): 
    mat = eps_w_array[i]
    eig_val, _ = np.linalg.eig(mat)
    sign = np.sign(eig_val)
    if(sign.sum() == 2): 
        ind_same_sign_pos.append(i)
    if(sign.sum() == -2):
        ind_same_sign_neg.append(i)
    if(sign.sum() == 0): 
        ind_diff_sign.append(i)


"""plt.figure(figsize=(8, 8))
plt.scatter(coord_array[ind_same_sign_pos, 0], coord_array[ind_same_sign_pos, 1], color='red', s=5, label='same signs > 0')
plt.scatter(coord_array[ind_diff_sign, 0], coord_array[ind_diff_sign, 1], color='blue', s=5, label='different signs')
plt.legend()
plt.show()"""

j = 1987
print("Coordinates of the mesh point = " + str(coord_array[ind_same_sign_pos[j]]))
print("Eigenvalues = " + str(np.linalg.eig(eps_w_array[ind_same_sign_pos[j]])[0]))
print("Eigenvectors = \n" + str(np.linalg.eig(eps_w_array[ind_same_sign_pos[j]])[1]))


print(lmbda)
print(mu)
