# attempt to make trajectories for Oakley plot without obstacles

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# defining V(x,y) function
def V(x,y):
    return (1-y*x)**2/(x*(np.arctan((x-y)/4)/np.pi+1/2)+y*(np.arctan((y-x)/4)/np.pi+1/2))**2

# setting initial conditions from Hamiltonian equaitons
E = 1
x = 10
y = 0.5
py = 1.1
px = np.sqrt(2*E - 2*(V(x,y)) - py**2)
initial = [x, y, px, py]

t = np.linspace(0,5, 1000)

def derivatives(x,y):
    dvdx = (2*y*(1-y*x))/(x*(np.arctan((x-y)/4)/np.pi+1/2)+y*(np.arctan((y-x)/4)/np.pi+1/2))**2-(2*(1-y*x)**2*(np.arctan((x-y)/4)/np.pi+x/(4*np.pi*((x-y)**2/16+1))-y/(4*np.pi*((y-x)**2/16+1))+1/2))/(x*(np.arctan((x-y)/4)/np.pi+1/2)+y*(np.arctan((y-x)/4)/np.pi+1/2))**3
    dvdy = (2*x*(1-x*y))/(y*(np.arctan((y-x)/4)/np.pi+1/2)+x*(np.arctan((x-y)/4)/np.pi+1/2))**2-(2*(1-x*y)**2*(np.arctan((y-x)/4)/np.pi+y/(4*np.pi*((y-x)**2/16+1))-x/(4*np.pi*((x-y)**2/16+1))+1/2))/(y*(np.arctan((y-x)/4)/np.pi+1/2)+x*(np.arctan((x-y)/4)/np.pi+1/2))**3
    return [dvdx, dvdy]

# setting up differential equations for x
def vdp_derivatives(t,a):
    x  = a[0]
    y  = a[1]
    px = a[2]
    py = a[3]
    return [px, py, derivatives(x,y)[0], derivatives(x,y)[1]]

solx = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=initial, t_eval = t)

coords = solx.y[:2]

plt.plot(coords[0], coords[1], 'r-', linewidth = 1)

#setting x and y
x = np.linspace(-10,10,1000)
y = np.linspace(-10,10,1000)

#setting z
X, Y = np.meshgrid(x, y)
Z = V(X, Y)

#plotting
cs = plt.contour(X, Y, Z, levels = 20)
plt.xlabel('x')
plt.ylabel('y')
plt.show()