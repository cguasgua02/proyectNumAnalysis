"""
# Proyecto de Análisis Numérico MAT 3001
# Resolución del problema de Poiseuille con método de Chorin modificado usando la librería FEniCS

## Cristian Guasgua
#### 07 de noviembre de 2023

El caso del flujo de Poiseulle que muestra el flujo laminar entre dos placas finitas, el cual se puede modelar con una ecuación de Navier stokes. 
Por lo tanto, el proyecto trata de una solución numérica de las ecuaciones de Navier-Stokes usando el esquema numérico  de corrección incremental de presión(IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
"""

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt

T = 10.0           # tiempo final
num_steps = 500    # número de pasos de tiempo
dt = T / num_steps # paso de tiempo
mu = 1             # viscosidad cinemática. Recordemos que está escalado unitariamente
rho = 1            # densidad

# Creando malla y definiendo espacio funcional
mesh = UnitSquareMesh(16, 16) #malla cuadrada de 16x16
V = VectorFunctionSpace(mesh, 'P', 2) #espacio funcional de vectores de funciones continuas de orden 2 en la malla definida para la velocidad
Q = FunctionSpace(mesh, 'P', 1) #espacio funcional de funciones continuas de orden 1 en la malla definida para la presión

# Definiendo fronteras
inflow  = 'near(x[0], 0)' # Fondo. la función booleana 'near' me dice si x[0] está cerca de 0
outflow = 'near(x[0], 1)' # Techo
walls   = 'near(x[1], 0) || near(x[1], 1)'# Paredes

# Definiendo condiciones de frontera
bcu_noslip  = DirichletBC(V, Constant((0, 0)), walls)
bcp_inflow  = DirichletBC(Q, Constant(8), inflow)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_noslip] #condiciones de frontera para velocidad en lista
bcp = [bcp_inflow, bcp_outflow] #condiciones de frontera para presión en lista

# Defino las funciones de prueba
u = TrialFunction(V) #velocidad a calcular
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Defino funciones para las soluciones en instante previo y actual
u_n = Function(V) #paso previo
u_  = Function(V) #paso actual
p_n = Function(Q) #paso previo
p_  = Function(Q) #paso actual

# Defino expresiones usadas en formas variacionales
U   = 0.5*(u_n + u) #velocidad promedio 
n   = FacetNormal(mesh)
f   = Constant((0, 0))
k   = Constant(dt)
mu  = Constant(mu)
rho = Constant(rho)

# Defino el tensor deformación
def epsilon(u):
    return sym(nabla_grad(u))

# Defino tensor de esfuerzo
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Defino primer problema variacional: obtengo velocidad tentativa
F1 = rho*dot((u - u_n) / k, v)*dx + \
     rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(f, v)*dx
a1 = lhs(F1) #funcion que obtiene la forma bilineal de la ecuación variacional F1 
L1 = rhs(F1) #funcion que obtiene la forma lineal de la ecuación variacional F1

# Defino segundo problema variacional: variación de la presión
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

# Defino tercer problema variacional: Resolución problema de Poisson
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

# Ensamblando matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Aplicando condiciones de borde a matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# Loop temporal de paso de tiempo
t = 0
for n in range(num_steps):

    t += dt

    # Aplico primer paso: Paso de velocidad tentativa
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1)

    # Aplico segundo paso: Paso para corrección de presión
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2)

    # Aplico 3 paso: Paso para correción de velocidad
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3)

    # Graficar solución
    plot(u_)

    #vtkfile_u = File("resultado_u_{0}.pvd".format(n))
    #vtkfile_u << (u_, n)

    # Calculando el error
    u_e = Expression(('4*x[1]*(1.0 - x[1])', '0'), degree=2) #solución exacta
    u_e = interpolate(u_e, V)
    error = np.abs(u_e.vector().get_local() - u_.vector().get_local()).max()
    print('t = %.2f: error = %.3g' % (t, error))
    print('max u:', u_.vector().max())

    # Actualizar solución previa
    u_n.assign(u_)
    p_n.assign(p_)




# Muestro el gráfico
plt.show() 
