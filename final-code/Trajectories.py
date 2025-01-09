# trajectories of my plot

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams["contour.linewidth"] = (1)
fig = plt.figure()
ax = fig.add_subplot()
ax.set_aspect('equal', adjustable='box')

# set coefficient variables for V(x,y)

# x^2, x^3, x^4, x^5, x^6
# y^2, y^3, y^4, y^5, y^6
# xy, x^2y, xy^2, x^2y^2, x^3y, xy^3, x^3y^2, x^2y^3, x^3y^3

xco = [-40,-2,1,0,0]
yco = [-40,2,1,0,0]
xyco = [0,1,1,0,0.1,0.1,0.05,0,0]

limits = -7.1,8,-8.2,7,500

levels = 25

t = np.linspace(0,1.5,1000)

def V(x,y):

    # defining V(x,y) function
    xterms = xco[0]*x**2 + xco[1]*x**3 + xco[2]*x**4 + xco[3]*x**5 + xco[4]*x**6
    yterms = yco[0]*y**2 + yco[1]*y**3 + yco[2]*y**4 + yco[3]*y**5 + yco[4]*y**6
    xyterms = xyco[0]*x*y + xyco[1]*(x**2)*y + xyco[2]*x*y**2 + xyco[3]*(x**2)*y**2 + xyco[4]*(x**3)*y + xyco[5]*x*y**3 + xyco[6]*(x**3)*y**2 + xyco[7]*(x**2)*y**3 + xyco[8]*(x**3)*y**3

    return (xterms + yterms + xyterms)


# setting initial conditions from Hamiltonian equations
x = -6
y = 4
E = -200
py = 2  

px = np.sqrt(2*E - 2*(V(x,y)) - py**2)
y0 = [x,y,px,py]



def vdp_derivatives(t,a):

    # setting up differential equations
    [x,y,px,py] = a

    dvdx = -np.dot(xco, [2*x, 3*x**2, 4*x**3, 5*x**4, 6*x**5])-np.dot(xyco,[y, 2*x*y, y**2, 2*x*y**2, 3*(x**2)*y, y**3, 3*(x**2)*y**2, 2*x*y**3, 3*(x**2)*y**3])
    dvdy = -np.dot(yco, [2*y, 3*y**2, 4*y**3, 5*y**4, 6*y**5] )-np.dot(xyco, [x, x**2, 2*x*y, 2*(x**2)*y, x**3, 2*x*y**2, 2*(x**3)*y, 3*(x**2)*y**2, 3*(x**3)*y**2])

    return [px, py, dvdx, dvdy]


soly = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=y0, t_eval = t)
coords = soly.y[:2]
momentum = soly.y[2:]

def contours(limits):
    
    xmax, xmin, ymax, ymin, zlim = limits
    # defining and plotting contours
    x1 = np.linspace(xmin,xmax,1000)
    y1 = np.linspace(ymin,ymax,1000)

    X, Y = np.meshgrid(x1, y1)
    Z = V(X, Y)

    # limit height of conotour lines
    if zlim != "none":
        for i in range(len(Z)):
            for j in range(len(Z[0])):
                    if Z[i,j]>zlim:
                        Z[i,j] = zlim
    
    cs = plt.contour(X, Y, Z, levels)
    plt.clabel(cs, cs.levels[2:], inline=True, fontsize=7)
    
    return

contours(limits)


# aesthetics
plt.xlabel('XY bond length')
plt.ylabel('YZ bond length')


# plotting trajectories
plt.plot(coords[0],coords[1], 'r-', linewidth = 1.2, label = r'E=%i, x=%i, y=%i, $p_{y}$=%i' %(E, x, y, py)) # updates automatically
#plt.plot(-coords[0],coords[1], 'r-', linewidth = 1.5)
plt.legend(loc = 'best')

plt.show()