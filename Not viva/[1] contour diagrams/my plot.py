# contour plot for my plot

import numpy as np
import matplotlib.pyplot as plt
from sympy import *

#plt.rcParams["figure.figsize"] = (5,5)
#plt.rcParams["contour.linewidth"] = (0.9)
#plt.rcParams["contour.linestyles"] = ('-')

fig = plt.figure()
ax = fig.add_subplot()
ax.set_aspect('equal', adjustable='box')

# set coefficient variables for V(x,y)

# x^2, x^3, x^4, x^5, x^6
# y^2, y^3, y^4, y^5, y^6
# xy, x^2y, xy^2, x^2y^2, x^3y, xy^3, x^3y^2, x^2y^3, x^3y^3


xco = [-40,-1,1,0,0]
yco = [-40,3,1,0,0]
xyco = [0,1,1,0,0.1,0.1,0.05,0,0]

limits = -7.1,8,-8.5,6,500

xco = [-60,3,1,0,0]
yco = [-60,5,1,0,0]
xyco = [0,-4,-2,0,0,0,0,0,0]

limits = -10,8,-10.5,7.5,1500

xco = [-40,-1,1,0,0]
yco = [-40,3,1,0,0]
xyco = [0,1,1,0,0.1,0.1,0,0,0]

limits = -7.1,8,-8.5,6,500


levels = 20

def V(x,y):

    # defining V(x,y) function
    xterms = xco[0]*x**2 + xco[1]*x**3 + xco[2]*x**4 + xco[3]*x**5 + xco[4]*x**6
    yterms = yco[0]*y**2 + yco[1]*y**3 + yco[2]*y**4 + yco[3]*y**5 + yco[4]*y**6
    xyterms = xyco[0]*x*y + xyco[1]*(x**2)*y + xyco[2]*x*y**2 + xyco[3]*(x**2)*y**2 + xyco[4]*(x**3)*y + xyco[5]*x*y**3 + xyco[6]*(x**3)*y**2 + xyco[7]*(x**2)*y**3 + xyco[8]*(x**3)*y**3

    return (xterms + yterms + xyterms)

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

plt.xlabel('x')
plt.ylabel('y')

plt.show()