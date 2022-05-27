#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 09:39:22 2022

@author: arthur
"""


import firedrake as fd 
import matplotlib.pyplot as plt
import numpy as np 

# adimensional size of the film 
L = 2.5

# load value 
t = 10.

# Poisson coefficient
nu = 0.25

# discretization of the mesh 
N = 10

# mesh a squared cell 
mesh_square = fd.SquareMesh(N, N, L)

# plot of the mesh
"""fig, axes = plt.subplots(figsize=(9, 9))
fd.triplot(mesh_square, axes=axes)
axes.legend()
fig.show()"""

# space of vector functions
V = fd.VectorFunctionSpace(mesh_square, "Lagrange", degree=2)

# definition of the trial and test spaces
u = fd.TrialFunction(V)
v = fd.TestFunction(V)

# definition of the right hand side function 
f = fd.Function(V, name='function_f') 
x = fd.SpatialCoordinate(mesh_square)
f = fd.interpolate(-t * (x - fd.Constant((L/2,L/2))), V)


# definition of the bilinear/linear forms 
def sym_grad(v): 
    return fd.sym(fd.grad(v))

def sigma(v):
    return (nu/((1+nu)*(1-2*nu)))*fd.tr(sym_grad(v))*fd.Identity(2) + (1/(1+nu))*sym_grad(v)

a = (fd.inner(sigma(u), sym_grad(v)) + fd.dot(u, v))*fd.dx
L = fd.dot(f, v)*fd.dx

# solution
w = fd.Function(V, name="Displacement")
fd.solve(a == L, w)

# writing the result in a paraview file
"""fd.File("test_f.pvd").write(f)

fig, axes = plt.subplots()
contours = fd.trisurf(w)
fig.colorbar(contours, ax=axes)
fig.show()"""

energy = (1/2)*(fd.inner(sigma(w), sym_grad(w)) + fd.dot(w - f, w - f))*fd.dx

E = fd.assemble(energy)





