import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

#====================# Setting Initial Conditions #====================#

# cubic with x^2y terms
xco = [-80,3,1,0,0]
yco = [-80,3,1,0,0]
xyco = [0,3,-3,0,0,0,0,0,0]

approx_xyp = [-0.7,-8.7,0.3] # [x,y,p]
E = -2000

points = 1000

#====================# Solving ODEs for Trajectories #====================#

def V(x,y):

    xterms = xco[0]*x**2 + xco[1]*x**3 + xco[2]*x**4 + xco[3]*x**5 + xco[4]*x**6
    yterms = yco[0]*y**2 + yco[1]*y**3 + yco[2]*y**4 + yco[3]*y**5 + yco[4]*y**6
    xyterms = xyco[0]*x*y + xyco[1]*(x**2)*y + xyco[2]*x*y**2 + xyco[3]*(x**2)*y**2 + xyco[4]*(x**3)*y + xyco[5]*x*y**3 + xyco[6]*(x**3)*y**2 + xyco[7]*(x**2)*y**3 + xyco[8]*(x**3)*y**3

    return (xterms + yterms + xyterms)

def Trajectory_derivatives(t,a):

    [x,y,px,py] = a

    dvdx = -np.dot(xco, [2*x, 3*x**2, 4*x**3, 5*x**4, 6*x**5])-np.dot(xyco,[y, 2*x*y, y**2, 2*x*y**2, 3*(x**2)*y, y**3, 3*(x**2)*y**2, 2*x*y**3, 3*(x**2)*y**3])
    dvdy = -np.dot(yco, [2*y, 3*y**2, 4*y**3, 5*y**4, 6*y**5] )-np.dot(xyco, [x, x**2, 2*x*y, 2*(x**2)*y, x**3, 2*x*y**2, 2*(x**3)*y, 3*(x**2)*y**2, 3*(x**3)*y**2])

    return [px, py, dvdx, dvdy]


#====================# Periodic Orbits #====================#

def func(approx_xyp):
    x,y,p = approx_xyp
    t = np.linspace(0,p,1000)

    soly = solve_ivp(fun = Trajectory_derivatives, t_span = [t[0], t[-1]], y0=[x,y,0,0], t_eval = t)
    
    coords = soly.y[:2]
    momentum = soly.y[2:]

    # outputs the last value for momentum and energy (E-V(x,y)) looking for all to be 0
    return [momentum[0,-1], momentum[1,-1], E-V(coords[0,-1],coords[1,-1])]

x_PO,y_PO,half_period = fsolve(func,approx_xyp) # z0 coords for a periodic orbit 
period = 2*half_period


#====================# Solving the ODEs for delta_z #====================#

#=====#                d(delta_z)/dt = A(t)*delta_z                #=====#

t = np.linspace(0,period,points)
soly = solve_ivp(fun = Trajectory_derivatives, t_span = [t[0], t[-1]], y0=[x_PO,y_PO,0,0], t_eval = t)
coords = soly.y[:2]
momentum = soly.y[2:]

def gradient(x,y):

    dvdx = -np.dot(xco, [2*x, 3*x**2, 4*x**3, 5*x**4, 6*x**5])-np.dot(xyco,[y, 2*x*y, y**2, 2*x*y**2, 3*(x**2)*y, y**3, 3*(x**2)*y**2, 2*x*y**3, 3*(x**2)*y**3])
    dvdy = -np.dot(yco, [2*y, 3*y**2, 4*y**3, 5*y**4, 6*y**5] )-np.dot(xyco, [x, x**2, 2*x*y, 2*(x**2)*y, x**3, 2*x*y**2, 2*(x**3)*y, 3*(x**2)*y**2, 3*(x**3)*y**2])

    dydx = dvdx/dvdy
    
    return dydx

def perturbation(x, y):

    dydx = gradient(x,y)
    u = [1,dydx]/la.norm([1,dydx])

    return u

def second_derivatives(x,y):

    dvdxx = np.dot(xco, [2, 6*x, 12*x**2, 20*x**3, 30*x**4]) + np.dot(xyco, [0, 2*y, 0, 2*y**2, 6*x*y, 0, 6*x*y**2, 2*y**3, 6*x*y**3])
    dvdxy = np.dot(xyco, [1, 2*x, 2*y, 4*x*y, 3*x**2, 3*y**2, 6*(x**2)*y, 6*x*y**2, 9*(x**2)*y**2])
    dvdyy = np.dot(yco, [2, 6*y, 12*y**2, 20*y**3, 30*y**4]) + np.dot(xyco, [0, 0, 2*x, 2*x**2, 0, 6*x*y, 2*x**3, 6*(x**2)*y, 6*(x**3)*y])

    return dvdxx, dvdxy, dvdyy

def A(n):

    n = int((n/period)*999)

    dvdxx, dvdxy, dvdyy = second_derivatives(coords[0,n], coords[1,n])

    return np.array([[0,0,1,0],[0,0,0,1],[-dvdxx,-dvdxy,0,0],[-dvdxy,-dvdyy,0,0]])

def stability_derivatives(t,a):

    delta_x, delta_y, delta_px, delta_py = a
    delta_z = np.array([delta_x, delta_y, delta_px, delta_py])

    A_t = A(t)
    
    return A_t@delta_z

delta0 = perturbation(x_PO,y_PO)

t = np.linspace(0,period,1000)
solz = solve_ivp(fun = stability_derivatives, t_span = [t[0], t[-1]], y0=delta0, t_eval = t, jac = A)
delta_coords = solz.y[:2]
delta_momentum = solz.y[2:]


#====================# Monodromy matrix for stability #====================#

# need to calculate the monodromy matrix which is the matrix with dot products of perturbation directions at start of trajectory and at end on one period
# need to redefine the directions of perturbations



# M = [[d_z1(T).d_z1(0), d_z2(T).d_z1(0)], 
#      [d_z1(T).d_z2(0), d_z2(T).d_z1(0)]]