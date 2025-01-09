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

xco = [-40,3,1,0,0]
yco = [-40,3,1,0,0]
xyco = [0,0,0,0,0,0,0,0,0]

limits = -9,6,-9,6,1000



approx_xyp = [[0, 3.8, 0.6],
[-7.7, 0, 0.8],
[0, -2.1, 0.55],
[3.8, 0, 0.8],
[4, 4, 0.8],
[-3.8, 5, 0.47],
[-7.5, -7.5, 0.77],
[3, -5.7, 0.4]]

E = -1000

approx_xyp9 = [3.4, 4.6, 0.96]
approx_xyp10 = [4.7, 3.5, 0.8]
approx_xyp11 = [4.3, 2.4, 0.8]
approx_xyp12 = [-7.7, 0, 0.5]


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
def periodic_orbits(approx_xyp,col):
    initial_x,initial_y,p = fsolve(func,approx_xyp, xtol=1e-12)

    #print(fsolve(func,approx_xyp, xtol=1e-12))

    t = np.linspace(0,2*p,1000)
    soly = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=[initial_x,initial_y,0,0], t_eval = t)
    coords = soly.y[:2]
    momentum = soly.y[2:]

    #print(momentum[:,-1], V(coords[0,-1],coords[1,-1]), V(coords[0,0],coords[1,0]))

    #plt.plot(coords[0],coords[1], col, linewidth = 1, alpha = 1)
    return [initial_x,initial_y,p], coords

#fig, ((ax1, ax2), (ax3, ax4))  = plt.subplots(2, 2)

#fig.suptitle('Sharing x per column, y per row')

while E<0:
    for i in range(len(approx_xyp)):
        xi,yi,_ = approx_xyp[i]
        if V(xi,yi)<=E:
            approx_xyp[i], coords = periodic_orbits(approx_xyp[i], 'r-')
            ax.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
    E += 100

'''
approx_xyp1, coords1 = periodic_orbits(approx_xyp1, 'r-')
ax.plot(coords1[0], coords1[1], 'r-', linewidth = 0.8, alpha = 1)
'''
'''
approx_xyp2, coords2 = periodic_orbits(approx_xyp2, 'r-')
ax.plot(coords2[0], coords2[1], 'r-', linewidth = 0.8, alpha = 1)
approx_xyp3, coords3 = periodic_orbits(approx_xyp3, 'r-')
ax.plot(coords3[0], coords3[1], 'r-', linewidth = 0.8, alpha = 1)


approx_xyp4, coords4 = periodic_orbits(approx_xyp4, 'r-')
ax.plot(coords4[0], coords4[1], 'r-', linewidth = 0.8, alpha = 1)

approx_xyp5, coords5 = periodic_orbits(approx_xyp5, 'r-')
ax.plot(coords5[0], coords5[1], 'r-', linewidth = 0.8, alpha = 1)

approx_xyp6, coords6 = periodic_orbits(approx_xyp6, 'r-')
ax.plot(coords6[0], coords6[1], 'r-', linewidth = 0.8, alpha = 1)
approx_xyp7, coords7 = periodic_orbits(approx_xyp7, 'r-')
ax.plot(coords7[0], coords7[1], 'r-', linewidth = 0.8, alpha = 1)
approx_xyp8, coords8 = periodic_orbits(approx_xyp8, 'r-')
ax.plot(coords8[0], coords8[1], 'r-', linewidth = 0.8, alpha = 1)
'''
'''
approx_xyp1, coords1 = periodic_orbits(approx_xyp1, 'r-')
ax1.plot(coords1[0], coords1[1], 'r-', linewidth = 0.8, alpha = 1)
approx_xyp2, coords2 = periodic_orbits(approx_xyp2, 'r-')
ax1.plot(coords2[0], coords2[1], 'r-', linewidth = 0.8, alpha = 1)
approx_xyp3, coords3 = periodic_orbits(approx_xyp3, 'r-')
ax1.plot(coords3[0], coords3[1], 'r-', linewidth = 0.8, alpha = 1)
approx_xyp4, coords4 = periodic_orbits(approx_xyp4, 'r-')
ax1.plot(coords4[0], coords4[1], 'r-', linewidth = 0.8, alpha = 1)

approx_xyp5, coords5 = periodic_orbits(approx_xyp5, 'r-')
ax1.plot(coords5[0], coords5[1], 'r-', linewidth = 0.8, alpha = 1)
approx_xyp6, coords6 = periodic_orbits(approx_xyp6, 'r-')
ax1.plot(coords6[0], coords6[1], 'r-', linewidth = 0.8, alpha = 1)
approx_xyp7, coords7 = periodic_orbits(approx_xyp7, 'r-')
ax1.plot(coords7[0], coords7[1], 'r-', linewidth = 0.8, alpha = 1)
approx_xyp8, coords8 = periodic_orbits(approx_xyp8, 'r-')
ax1.plot(coords8[0], coords8[1], 'r-', linewidth = 0.8, alpha = 1)


periodic_orbits(approx_xyp5, 'r-')
periodic_orbits(approx_xyp6, 'r-') #-900
periodic_orbits(approx_xyp7, 'r-') #-1200
periodic_orbits(approx_xyp8, 'r-') #-900
 


periodic_orbits(approx_xyp9, 'b-')
periodic_orbits(approx_xyp10, 'g-')
periodic_orbits(approx_xyp11, 'r-')
#periodic_orbits(approx_xyp12, 'r-')

need to plot periodic orbits at multiple different energy levels to see how they merge when they get high enough
need to save each plot
need to use the xyp from the previous for the next one
dont plot them for ones that are too low or dont work

'''
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

    levels = 20

    cs = ax.contour(X, Y, Z, levels)
    plt.clabel(cs, cs.levels[:], inline=True, fontsize=5)   

    '''
    cs1 = ax1.contour(X, Y, Z, levels)
    plt.clabel(cs1, cs1.levels[:], inline=True, fontsize=5)
    cs2 = ax2.contour(X, Y, Z, levels)
    plt.clabel(cs2, cs2.levels[:], inline=True, fontsize=5)
    cs3 = ax3.contour(X, Y, Z, levels)
    plt.clabel(cs3, cs3.levels[:], inline=True, fontsize=5)
    cs4 = ax4.contour(X, Y, Z, levels)
    plt.clabel(cs4, cs4.levels[:], inline=True, fontsize=5)
    '''

    return

contours(limits)

plt.xlabel('x')
plt.ylabel('y')
#plt.legend(loc = 'best')

#plt.savefig('[5] images/test5.png', format = 'png', dpi = 1200)

plt.show()

