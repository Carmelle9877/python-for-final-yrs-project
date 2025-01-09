# attempting to make periodic orbits again

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


plt.rcParams["contour.linewidth"] = (1)
fig = plt.figure()
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
ax = fig.add_subplot()
ax.set_aspect('equal', adjustable='box')

# set coefficient variables for V(x,y)

# x^2, x^3, x^4, x^5, x^6
xco = [-40,0,1,0,0]
# y^2, y^3, y^4, y^5, y^6
yco = [-40,0,1,0,0]
# xy, x^2y, xy^2, x^2y^2, x^3y, xy^3, x^3y^2, x^2y^3, x^3y^3, 
xyco = [0,0,0,0,0,0,0,0,0]


def V(x,y):

    # defining V(x,y) function
    xterms = xco[0]*x**2 + xco[1]*x**3 + xco[2]*x**4 + xco[3]*x**5 + xco[4]*x**6
    yterms = yco[0]*y**2 + yco[1]*y**3 + yco[2]*y**4 + yco[3]*y**5 + yco[4]*y**6
    xyterms = xyco[0]*x*y + xyco[1]*(x**2)*y + xyco[2]*x*y**2 + xyco[3]*(x**2)*y**2 + xyco[4]*(x**3)*y + xyco[5]*x*y**3 + xyco[6]*(x**3)*y**2 + xyco[7]*(x**2)*y**3 + xyco[8]*(x**3)*y**3

    return (xterms + yterms + xyterms)


# set initial conditions for trajectory
# my guess for periodic orbit
x = 2
y = 0


# intial conditions for a periodic orbit
E = V(x,y)
py = 0
px = 0

y0 = [x,y,px,py]


def derivative(x,y):

    # negative derivative function for V(x,y)
    dvdx = -np.dot(xco, [2*x, 3*x**2, 4*x**3, 5*x**4, 6*x**5])-np.dot(xyco,[y, 2*x*y, y**2, 2*x*y**2, 3*(x**2)*y, y**3, 3*(x**2)*y**2, 2*x*y**3, 3*(x**2)*y**3])
    dvdy = -np.dot(yco, [2*y, 3*y**2, 4*y**3, 5*y**4, 6*y**5] )-np.dot(xyco, [x, x**2, 2*x*y, 2*(x**2)*y, x**3, 2*x*y**2, 2*(x**3)*y, 3*(x**2)*y**2, 3*(x**3)*y**2])

    return [dvdx, dvdy]


def vdp_derivatives(t,a):

    # setting up differential equations
    [x,y,px,py] = a

    return [px, py, derivative(x,y)[0], derivative(x,y)[1]]


# how long and how many points are on the trajectory
length = 0.3
points = 1000

xmax, xmin, ymax, ymin, zlim = -10,10,-10,10,3000


def ODE_solver(x,y,length, points):

    y0 = [x,y,0,0]

    t = np.linspace(0,length,points)

    soly = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=y0, t_eval = t)
    coords = soly.y[:2]
    momentum = soly.y[2:]

    return coords, momentum

coords, momentum = ODE_solver(x,y,length, points)

# periodic orbits

# function that outputs final momentum after period
def func(a):
    x,y,p = a

    y0 = [x,y,0,0]
    t = np.linspace(0,p,2)
    soly = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=y0, t_eval = t)
    
    coords = soly.y[:2]
    momentum = soly.y[2:]

    energy = V(coords[0,1],coords[1,1])

    return [momentum[0,1], momentum[1,1],-144-energy]

a = [x,y,length]

sol = fsolve(func,a)
print(sol)

def contours(xmax, xmin, ymax, ymin, zlim):
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

    cs = plt.contour(X, Y, Z, levels = 20)
    plt.clabel(cs, cs.levels[:10], inline=True, fontsize=7)

    return

contours(xmax, xmin, ymax, ymin, zlim)



# aesthetics
#plt.title(r'$-%i\frac{x^2}{2} -%i\frac{y^2}{2} + %i\frac{x^3}{3} + %i\frac{y^3}{3} + %i\frac{y^4}{4} + %i\frac{y^4}{4}$' %(x2, y2, x3, y3, x4, y4)) #updates automatically
plt.xlabel('x')
plt.ylabel('y')


# trajectories
plt.plot(coords[0],coords[1], 'r-', linewidth = 1, alpha = 0.9, label = r'E=%i, x=%i, y=%i, $p_{y}$=%i' %(E, x, y, py)) # updates automatically
# symmetrical trajectory
#plt.plot(-coords[0],coords[1], 'r-', linewidth = 1.5)
plt.legend(loc = 'best')

plt.show()