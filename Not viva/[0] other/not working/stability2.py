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
# y^2, y^3, y^4, y^5, y^6
# xy, x^2y, xy^2, x^2y^2, x^3y, xy^3, x^3y^2, x^2y^3, x^3y^3,

# xmax, xmin, ymax, ymin, zlim

# initial guess for turning point and period of periodic orbit
# decide which energy level you want



xco = [-80,3,1,0,0]
yco = [-80,3,1,0,0]
xyco = [0,3,-3,0,0,0,0,0,0]

limits = -12,10,-12,9,"none"

approx_xyp = [-0.7,-8.7,0.3] # [x,y,p]
E = -2000

points = 10000

#=================================================================================================

# defining V(x,y) function
def V(x,y):

    xterms = xco[0]*x**2 + xco[1]*x**3 + xco[2]*x**4 + xco[3]*x**5 + xco[4]*x**6
    yterms = yco[0]*y**2 + yco[1]*y**3 + yco[2]*y**4 + yco[3]*y**5 + yco[4]*y**6
    xyterms = xyco[0]*x*y + xyco[1]*(x**2)*y + xyco[2]*x*y**2 + xyco[3]*(x**2)*y**2 + xyco[4]*(x**3)*y + xyco[5]*x*y**3 + xyco[6]*(x**3)*y**2 + xyco[7]*(x**2)*y**3 + xyco[8]*(x**3)*y**3

    return (xterms + yterms + xyterms)

# setting up differential equations
def vdp_derivatives(t,a):

    [x,y,px,py] = a

    # negative derivative function for V(x,y)
    dvdx = -np.dot(xco, [2*x, 3*x**2, 4*x**3, 5*x**4, 6*x**5])-np.dot(xyco,[y, 2*x*y, y**2, 2*x*y**2, 3*(x**2)*y, y**3, 3*(x**2)*y**2, 2*x*y**3, 3*(x**2)*y**3])
    dvdy = -np.dot(yco, [2*y, 3*y**2, 4*y**3, 5*y**4, 6*y**5] )-np.dot(xyco, [x, x**2, 2*x*y, 2*(x**2)*y, x**3, 2*x*y**2, 2*(x**3)*y, 3*(x**2)*y**2, 3*(x**3)*y**2])

    return [px, py, dvdx, dvdy]

def dxy(x,y):

    # negative derivative function for V(x,y)
    dvdx = -np.dot(xco, [2*x, 3*x**2, 4*x**3, 5*x**4, 6*x**5])-np.dot(xyco,[y, 2*x*y, y**2, 2*x*y**2, 3*(x**2)*y, y**3, 3*(x**2)*y**2, 2*x*y**3, 3*(x**2)*y**3])
    dvdy = -np.dot(yco, [2*y, 3*y**2, 4*y**3, 5*y**4, 6*y**5] )-np.dot(xyco, [x, x**2, 2*x*y, 2*(x**2)*y, x**3, 2*x*y**2, 2*(x**3)*y, 3*(x**2)*y**2, 3*(x**3)*y**2])

    return np.array([dvdx, dvdy])

#=================================================================================================

# function that used to find periodic orbits
def func(approx_xyp):
    x,y,p = approx_xyp # can ony have one input so we put all vars into one

    t = np.linspace(0,p,points)
    soly = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=[x,y,0,0], t_eval = t)
    
    coords = soly.y[:2]
    momentum = soly.y[2:]

    energy = V(coords[0,-1],coords[1,-1])

    # outputs the last value for momentum and energy (E-V(x,y)) looking for all to be 0
    return [momentum[0,-1], momentum[1,-1], E-energy]

# solves for periodic orbits and plots them given initial conditions
def periodic_orbits(approx_xyp):
    x,y,p = fsolve(func,approx_xyp, xtol=1e-12)

    print(fsolve(func,approx_xyp))

    t = np.linspace(0,p,points)
    soly = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=[x,y,0,0], t_eval = t)
    coords = soly.y[:2]
    momentum = soly.y[2:]

    #print(momentum[:,-1], V(coords[0,-1],coords[1,-1]), V(coords[0,0],coords[1,0]))

    plt.plot(coords[0],coords[1], 'r-', linewidth = 1, alpha = 1)
    return x,y,p


#=================================================================================================


def compute_floquet_multipliers(xyp):

    # For given PO
    x,y,p = xyp

    # Linearize the system around the periodic orbit
    A = np.zeros((2, 2))
    dz = 0.001
    A[:,0] = (dxy(x,y) - dxy(x+dz,y))/dz
    A[:,1] = (dxy(x,y) - dxy(x,y+dz))/dz

    print((dxy(x,y) - dxy(x+dz,y))/dz)

    # Compute the fundamental matrix solution
    phi = np.eye(2)
    for ti in range(points-1):
        dt = 2*p / points
        phi = phi + (A * dt)

    # Compute the Floquet multipliers (eigenvalues of the fundamental matrix)
    floquet_multipliers, _ = np.linalg.eig(phi)

    print(floquet_multipliers)

    return floquet_multipliers


x,y,p = periodic_orbits(approx_xyp)
xyp = x,y,p

compute_floquet_multipliers(xyp)

#=================================================================================================

t = np.linspace(0,p,points)

soly = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=[x,y,0,0], t_eval = t)
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

    cs = plt.contour(X, Y, Z, levels = 30)
    plt.clabel(cs, cs.levels[:], inline=True, fontsize=7)

    return

contours(limits)

plt.xlabel('x')
plt.ylabel('y')
#plt.legend(loc = 'best')

#plt.savefig('[5] images/PO,cubic+3xy,E=-500.png', format = 'png', dpi = 1200)

plt.show()

