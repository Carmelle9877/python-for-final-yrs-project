import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

plt.close()

plt.rcParams["contour.linewidth"] = (1)
fig = plt.figure()
ax = fig.add_subplot()
ax.set_aspect('equal', adjustable='box')

#=================================================================================================

# x^2, x^3, x^4, x^5, x^6
# y^2, y^3, y^4, y^5, y^6
# xy, x^2y, xy^2, x^2y^2, x^3y, xy^3, x^3y^2, x^2y^3, x^3y^3,

# xmax, xmin, ymax, ymin, zlim

# initial guess for turning point and period of periodic orbit
# decide which energy level you want

xco = [-40,-1,1,0,0]
yco = [-40,3,1,0,0]
xyco = [0,1,1,0,0.1,0.1,0,0,0]

limits = -7.1,8,-8.5,6,500



approx_xyp = [[-5.6, 0, 0.36], # left saddle
[0.3, -7.8, 0.3], # bottom saddle
[6.6, 0.3, 0.3], # right saddle
[0.3, 4.6, 0.27], # top saddle
[-5.5, 5.2, 0.36], 
[-6.5, -7.4, 0.3], 
[7.4, -7.6, 0.3],
[5.7, 4.3, 0.27],
[-5.8, 2, 0.36], 
[-6.3, -2.2, 0.3], 
[6.9, -1.1, 0.3],
[2.6, 5, 0.27]] 

E = -100



#=================================================================================================

# defining V(x,y) function
def V(x,y):

    xterms = xco[0]*x**2 + xco[1]*x**3 + xco[2]*x**4 + xco[3]*x**5 + xco[4]*x**6
    yterms = yco[0]*y**2 + yco[1]*y**3 + yco[2]*y**4 + yco[3]*y**5 + yco[4]*y**6
    xyterms = xyco[0]*x*y + xyco[1]*(x**2)*y + xyco[2]*x*y**2 + xyco[3]*(x**2)*y**2 + xyco[4]*(x**3)*y + xyco[5]*x*y**3 + xyco[6]*(x**3)*y**2 + xyco[7]*(x**2)*y**3 + xyco[8]*(x**3)*y**3

    return (xterms + yterms + xyterms)

#print(V(3.35,3.1))

# negative derivative function for V(x,y)
def derivative(x,y):
    
    dvdx = -np.dot(xco, [2*x, 3*x**2, 4*x**3, 5*x**4, 6*x**5])-np.dot(xyco,[y, 2*x*y, y**2, 2*x*y**2, 3*(x**2)*y, y**3, 3*(x**2)*y**2, 2*x*y**3, 3*(x**2)*y**3])
    dvdy = -np.dot(yco, [2*y, 3*y**2, 4*y**3, 5*y**4, 6*y**5] )-np.dot(xyco, [x, x**2, 2*x*y, 2*(x**2)*y, x**3, 2*x*y**2, 2*(x**3)*y, 3*(x**2)*y**2, 3*(x**3)*y**2])

    return [dvdx, dvdy]

# setting up differential equations
def vdp_derivatives(t,a):

    [x,y,px,py] = a

    return [px, py, derivative(x,y)[0], derivative(x,y)[1]]

#=================================================================================================

# function that used to find periodic orbits
def func(approx_xyp):
    x,y,p = approx_xyp # can ony have one input so we put all vars into one

    t = np.linspace(0,p,1000)
    soly = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=[x,y,0,0], t_eval = t)
    
    coords = soly.y[:2]
    momentum = soly.y[2:]

    energy0 = V(coords[0,-1],coords[1,-1])
    energy1 = V(coords[0,0],coords[1,0])

    # outputs the last value for momentum and energy (E-V(x,y)) looking for all to be 0
    return [momentum[0,-1], momentum[1,-1], energy1+energy0-2*E]

# solves for periodic orbits and plots them given initial conditions
def periodic_orbits(approx_xyp):
    initial_x,initial_y,p = fsolve(func,approx_xyp, xtol=1e-12)

    t = np.linspace(0,2*p,1000)
    soly = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=[initial_x,initial_y,0,0], t_eval = t)
    coords = soly.y[:2]
    momentum = soly.y[2:]

    plt.plot(coords[0],coords[1], 'r-', linewidth = 1, alpha = 1)
    return [initial_x,initial_y,p], coords


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

    levels = 20

    cs = ax.contour(X, Y, Z, levels)
    plt.clabel(cs, cs.levels[:], inline=True, fontsize=5)   

    return

contours(limits)
for i in range(len(approx_xyp)):
    periodic_orbits(approx_xyp[i])
    

#=================================================================================================

plt.xlabel('x')
plt.ylabel('y')

plt.show()

