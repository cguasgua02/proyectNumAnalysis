"""
Proyecto Análisis Numérico
MAT 3001

Implementación para la solución numérica de las ecuaciones de Navier-Stokes
Solución numérica de las ecuaciones de Navier-Stokes para un flujo laminar entre
dos placas usando una alternativa del método de Chorin (IPCS).

"""

from __future__ import print_function
from fenics import *
import numpy as np

# Defino algunas constantes
T = 10.0           # tiempo final
numPasos = 500    # número de pasos de tiempo
dt = T / numPasos # paso de tiempo
mu = 1             # viscosidad cinemática
rho = 1             # densidad

# primero defino el dominio en el que voy a resolver la ecuación diferencial crear una malla y definir un espacio funcional
malla = UnitSquareMesh(16, 16)
V = VectorFunctionSpace(malla, 'P', 2)
Q = FunctionSpace(malla, 'P', 1)

# Defino los valores y condiciones de frontera
inflow  = 'near(x[0], 0)'
outflow = 'near(x[0], 1)'
walls   = 'near(x[1], 0) || near(x[1], 1)'
bcu_noslip  = DirichletBC(V, Constant((0, 0)), walls)
bcp_inflow  = DirichletBC(Q, Constant(8), inflow)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_noslip]
bcp = [bcp_inflow, bcp_outflow]

# Definir funciones de prueba
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Definir funciones para las soluciones en instante previo y actual
u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)

# Definir expresiones usadas en formas variacionales
U   = 0.5*(u_n + u)
n   = FacetNormal(malla)
f   = Constant((0, 0))
k   = Constant(dt)
mu  = Constant(mu)
rho = Constant(rho)

# Definir tensor deformación
def epsilon(u):
    return sym(nabla_grad(u))

# Defino el tensor de esfuerzo
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Defino primer problema variacional
F1 = rho*dot((u - u_n) / k, v)*dx + \
     rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Defino segundo problema variacional
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

# Defino tercer problema variacional
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

# Ensamblo las matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Aplico condiciones de borde a matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# Configurando el paso de tiempo
t = 0
for n in range(numPasos):

    t += dt

    # Primer paso: Paso de velocidad tentativa
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1)

    # Segundo paso: Paso para corrección de presión
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2)

    # Tercer paso: Paso para correción de velocidad
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3)

    # Genero la gráfica de la solución
    plot(u_)

    # Calculo el error
    u_e = Expression(('4*x[1]*(1.0 - x[1])', '0'), degree=2)
    u_e = interpolate(u_e, V)
    error = np.abs(u_e.vector().array() - u_.vector().array()).max()
    print('t = %.2f: error = %.3g' % (t, error))
    print('max u:', u_.vector().array().max())

    # Actualizo la solución previa
    u_n.assign(u_)
    p_n.assign(p_)

#Finalmente muestro la gráfica interactiva
interactive()