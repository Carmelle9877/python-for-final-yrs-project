import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

plt.rcParams["contour.linewidth"] = (1)
fig = plt.figure()
ax = fig.add_subplot()
ax.set_aspect('equal', adjustable='box')

#=================================================================================================




# x^2, x^3, x^4, x^5, x^6
xco = [-60,3,1,0,0]
# y^2, y^3, y^4, y^5, y^6
yco = [-60,5,1,0,0]
# xy, x^2y, xy^2, x^2y^2, x^3y, xy^3, x^3y^2, x^2y^3, x^3y^3
xyco = [0,-4,-2,0,0,0,0,0,0]

# xmax, xmin, ymax, ymin, zlim
limits = -10,8,-10.5,7.5,1500


# initial guess for turning point and period of periodic orbit
approx_xyp = [3.6, 2.8, 0.5]

# decide which energy level you want
E = -1000

levels = 20

#==========# setting up ODE for trajectories #============================================================================

# defining V(x,y) function
def V(x,y):

    xterms = xco[0]*x**2 + xco[1]*x**3 + xco[2]*x**4 + xco[3]*x**5 + xco[4]*x**6
    yterms = yco[0]*y**2 + yco[1]*y**3 + yco[2]*y**4 + yco[3]*y**5 + yco[4]*y**6
    xyterms = xyco[0]*x*y + xyco[1]*(x**2)*y + xyco[2]*x*y**2 + xyco[3]*(x**2)*y**2 + xyco[4]*(x**3)*y + xyco[5]*x*y**3 + xyco[6]*(x**3)*y**2 + xyco[7]*(x**2)*y**3 + xyco[8]*(x**3)*y**3

    return (xterms + yterms + xyterms)

# setting up differential equations
def vdp_derivatives(t,a):

    [x,y,px,py] = a

    dvdx = -np.dot(xco, [2*x, 3*x**2, 4*x**3, 5*x**4, 6*x**5])-np.dot(xyco,[y, 2*x*y, y**2, 2*x*y**2, 3*(x**2)*y, y**3, 3*(x**2)*y**2, 2*x*y**3, 3*(x**2)*y**3])
    dvdy = -np.dot(yco, [2*y, 3*y**2, 4*y**3, 5*y**4, 6*y**5] )-np.dot(xyco, [x, x**2, 2*x*y, 2*(x**2)*y, x**3, 2*x*y**2, 2*(x**3)*y, 3*(x**2)*y**2, 3*(x**3)*y**2])

    return [px, py, dvdx, dvdy]

#==========# solving for periodic orbits #=======================================================================================

# function that used to find periodic orbits
def func(approx_xyp):
    x,y,p = approx_xyp # can ony have one input so we put all vars into one

    t = np.linspace(0,p,1000)
    soly = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=[x,y,0,0], t_eval = t)
    
    coords = soly.y[:2]
    momentum = soly.y[2:]

    energy = V(coords[0,-1],coords[1,-1])

    # outputs the last value for momentum and energy (E-V(x,y)) looking for all to be 0
    return [momentum[0,-1], momentum[1,-1], E-energy]

x_P,y_P,period = fsolve(func,approx_xyp) # coords for a periodic orbit 

#=================================================================================================

t = np.linspace(0,period*2,1000)

soly = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=[x_P,y_P,0,0], t_eval = t)
coords = soly.y[:2]
momentum = soly.y[2:]

#=================================================================================================

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
    plt.clabel(cs, cs.levels[:], inline=True, fontsize=7)

    return

contours(limits)


plt.xlabel('x')
plt.ylabel('y')
plt.plot(coords[0],coords[1], 'r-', linewidth = 1, alpha = 1, label = r'E=%i, x=%i, y=%i, $p_{y}$=%i' %(E, x_P, y_P, 0)) # updates automatically
plt.legend(loc = 'best')


plt.show()

