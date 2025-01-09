# my plot in 3D with trajectories layed over

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# x^2, x^3, x^4, x^5, x^6
xco = [-60,3,1,0,0]
# y^2, y^3, y^4, y^5, y^6
yco = [-60,3,1,0,0]
# xy, x^2y, xy^2, x^2y^2, x^3y, xy^3, x^3y^2, x^2y^3, x^3y^3, 
xyco = [0,3,-3,0,0,0,0,0,0]

def V(x,y):

    # defining V(x,y) function
    xterms = xco[0]*x**2 + xco[1]*x**3 + xco[2]*x**4 + xco[3]*x**5 + xco[4]*x**6
    yterms = yco[0]*y**2 + yco[1]*y**3 + yco[2]*y**4 + yco[3]*y**5 + yco[4]*y**6
    xyterms = xyco[0]*x*y + xyco[1]*(x**2)*y + xyco[2]*x*y**2 + xyco[3]*(x**2)*y**2 + xyco[4]*(x**3)*y + xyco[5]*x*y**3 + xyco[6]*(x**3)*y**2 + xyco[7]*(x**2)*y**3 + xyco[8]*(x**3)*y**3

    return (xterms + yterms + xyterms)

# setting initial conditions from Hamiltonian equations
E = -100
x = -7
y = -7
py = 0

if x>0:
    px = -np.sqrt(2*E - 2*(V(x,y)) - py**2)
else :
    px = np.sqrt(2*E - 2*(V(x,y)) - py**2)

y0 = [x,y,px,py]


t = np.linspace(0,1,1000)


def derivative(x,y):

    # negative derivative function for V(x,y)
    dvdx = -np.dot(xco, [2*x, 3*x**2, 4*x**3, 5*x**4, 6*x**5])-np.dot(xyco,[y, 2*x*y, y**2, 2*x*y**2, 3*(x**2)*y, y**3, 3*(x**2)*y**2, 2*x*y**3, 3*(x**2)*y**3])
    dvdy = -np.dot(yco, [2*y, 3*y**2, 4*y**3, 5*y**4, 6*y**5] )-np.dot(xyco, [x, x**2, 2*x*y, 2*(x**2)*y, x**3, 2*x*y**2, 2*(x**3)*y, 3*(x**2)*y**2, 3*(x**3)*y**2])

    return [dvdx, dvdy]

# setting up differential equaitons
def vdp_derivatives(t,a):
    x, y, px, py  = a
    return [px, py ,derivative(x,y)[0], derivative(x,y)[1]]

sol = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=y0, t_eval = t)
coords = sol.y[:2]

c = V(coords[0], coords[1])

# defining and plotting contours
x1 = np.linspace(-12,10.5,1000)
y1 = np.linspace(-12,9,1000)

X, Y = np.meshgrid(x1, y1)
Z = V(X, Y)

ax = plt.axes(projection='3d')

Z[Z>0]= np.nan
Z[Z<-6000]= np.nan


ax.plot3D(coords[0], coords[1],c+200, 'red', linewidth = 1.2)
#ax.plot_wireframe(X, Y, Z, rstride=30, cstride=30, colors='black')
#ax.contour3D(X, Y, Z, 50, cmap = 'binary',zorder = 1)
ax.plot_surface(X, Y, Z, cmap = 'viridis_r', alpha = 0.7)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#ax.set_xlim(-1000,250)
#ax.set_ylim(-1,1)
ax.set_zlim(-6000,1000)
plt.show()