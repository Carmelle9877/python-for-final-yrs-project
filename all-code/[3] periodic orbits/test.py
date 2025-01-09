import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


plt.rcParams["contour.linewidth"] = (1)

fig = plt.figure()
ax = fig.add_subplot()
ax.set_aspect('equal', adjustable='box')
'''
fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.set_aspect('equal', adjustable='box')

fig3 = plt.figure()
ax3 = fig3.add_subplot()
ax3.set_aspect('equal', adjustable='box')

fig4 = plt.figure()
ax4 = fig4.add_subplot()
ax4.set_aspect('equal', adjustable='box')

fig5 = plt.figure()
ax5 = fig5.add_subplot()
ax5.set_aspect('equal', adjustable='box')
'''

# ========================================== #

# x^2, x^3, x^4, x^5, x^6
# y^2, y^3, y^4, y^5, y^6
# xy, x^2y, xy^2, x^2y^2, x^3y, xy^3, x^3y^2, x^2y^3, x^3y^3

xco = [-60,3,1,0,0]
yco = [-60,5,1,0,0]
xyco = [0,-4,-2,0,0,0,0,0,0]

limits = -10,8,-10.5,7.5,1500

approx_xyp = [
    [-6.6, -1.4, 0.3], # left saddle 0
    [-1.74, -7.34, 0.3], # bottom saddle 1
    [4.3, 0.6, 0.3], # right saddle 2
    [0.2, 3.8, 0.3], # top saddle 3

    [-7.53, 4.86, 0.6], # TL min 4
    [-7.8, 4.2, 0.32], # TL min 5 

    [-4.4, -6.3, 0.18], # BL min 6 
    [-4.27, -7.1, 0.34], # BL min 7 

    [4.6, -7.84, 0.24], # BR min 8 
    [3.75, -7.15, 0.33], # BR min 9 

    [5.1, 5.13, 0.33], # TR min 10 -1500
    [5.05, 4.4, 0.21]] # TR min 11 -1500

E = -2750

target_E = -1000
step = 1
# ========================================================== #

def V(x,y):

    xterms = xco[0]*x**2 + xco[1]*x**3 + xco[2]*x**4 + xco[3]*x**5 + xco[4]*x**6
    yterms = yco[0]*y**2 + yco[1]*y**3 + yco[2]*y**4 + yco[3]*y**5 + yco[4]*y**6
    xyterms = xyco[0]*x*y + xyco[1]*(x**2)*y + xyco[2]*x*y**2 + xyco[3]*(x**2)*y**2 + xyco[4]*(x**3)*y + xyco[5]*x*y**3 + xyco[6]*(x**3)*y**2 + xyco[7]*(x**2)*y**3 + xyco[8]*(x**3)*y**3

    return (xterms + yterms + xyterms)

def vdp_derivatives(t,a):

    [x,y,px,py] = a

    dvdx = -np.dot(xco, [2*x, 3*x**2, 4*x**3, 5*x**4, 6*x**5])-np.dot(xyco,[y, 2*x*y, y**2, 2*x*y**2, 3*(x**2)*y, y**3, 3*(x**2)*y**2, 2*x*y**3, 3*(x**2)*y**3])
    dvdy = -np.dot(yco, [2*y, 3*y**2, 4*y**3, 5*y**4, 6*y**5] )-np.dot(xyco, [x, x**2, 2*x*y, 2*(x**2)*y, x**3, 2*x*y**2, 2*(x**3)*y, 3*(x**2)*y**2, 3*(x**3)*y**2])

    return [px, py, dvdx, dvdy]

def func(approx_xyp):
    x,y,p = approx_xyp

    t = np.linspace(0,p,1000)
    soly = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=[x,y,0,0], t_eval = t)
    
    coords = soly.y[:2]
    momentum = soly.y[2:]

    energy0 = V(coords[0,-1],coords[1,-1])

    return [momentum[0,-1], momentum[1,-1], energy0-E]

def contours(limits):
    xmax, xmin, ymax, ymin, zlim = limits

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

while E<=target_E:

    for i in range(len(approx_xyp)):

        xi, yi, p_i = approx_xyp[i]

        if V(xi,yi)<=E:
            
            approx_xyp[i] = fsolve(func,approx_xyp[i], xtol=1e-15)
            
            if E == target_E:
                xi, yi, p_i = approx_xyp[i]
                t = np.linspace(0,p_i*2,1000)
                soly = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=[xi,yi,0,0], t_eval = t)
                coords = soly.y[:2]
                plt.plot(coords[0],coords[1], 'r-', linewidth = 1, alpha = 1)
            
            if E % 500==0:
                print(approx_xyp[i],i, E)

    E+=step



plt.xlabel('x')
plt.ylabel('y')
plt.show()