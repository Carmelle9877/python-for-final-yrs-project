import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


x_0 = [1,1]
a0 = 5 # x0
b0 = np.sqrt(69) # px0
y_0 = [1,4]
c0 = 5 # y0
d0 = 1 # py0

# setting up the differential equation for x
t=np.linspace(0,20, 1000)

def vdp_derivativesx(t,y):
    x = y[0]
    v = y[1] # v = dxdt
    return [v, -x] 

solx = solve_ivp(fun = vdp_derivativesx, t_span = [t[0], t[-1]], y0=x_0, t_eval = t)

a = solx.y[0]

def vdp_derivativesy(t,y):
    z = y[0] # z = y
    v = y[1] # v = dydt
    return [v, -2*z] # insert equation for y here

soly = solve_ivp(fun = vdp_derivativesy, t_span = [t[0], t[-1]], y0=y_0, t_eval =t)

b = soly.y[0]

def V(x,y):
    return (x**2)/2+y**2

#a,b = np.meshgrid(a,b)
c = V(a,b)

limit = 5

x = np.linspace(-limit, limit, 200)
y = np.linspace(-limit, limit, 200)

X, Y = np.meshgrid(x, y)
Z = V(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(a,b,c, 'red', linewidth = 3)
ax.plot_surface(X, Y, Z)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#ax.set_zlim(0,2)

plt.show()