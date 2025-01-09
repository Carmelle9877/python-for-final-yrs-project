# trajectories for x^2 +2*y^2

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def V(x,y):
    return (x**2)/2+y**2

#initial conditions
E = 7
x = 1 # x0
y = 1 # px0
py = 3 # y0

if x>0 : px = -np.sqrt(2*E - 2*(V(x,y)) - py**2)
else   : px =  np.sqrt(2*E - 2*(V(x,y)) - py**2)

initial = [x, y, px, py]


# setting up the differential equation for x
t=np.linspace(0,50, 1000)

def derivative(x, y):
    dvdx = -x
    dvdy = -2*y
    return [dvdx, dvdy]

def vdp_derivatives(t,a):
    x  = a[0]
    y  = a[1]
    px = a[2]
    py = a[3]
    return [px, py, derivative(x,y)[0], derivative(x,y)[1]] 

sol = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=initial, t_eval = t)

coords = [sol.y[0], sol.y[1]]


#setting x and y values
xlimit = 4
ylimit = 4
a = np.linspace(-xlimit, xlimit, 100)
b = np.linspace(-ylimit, ylimit, 100)


#setting and plotting contours
X, Y = np.meshgrid(a, b)
Z = V(X, Y)
plt.contour(X, Y, Z, levels = 20,)

# plotting trajectories
plt.plot(coords[0],coords[1],'red', linewidth = 1.5)

# aesthetics
plt.xlabel('x')
plt.ylabel('y')

plt.show()