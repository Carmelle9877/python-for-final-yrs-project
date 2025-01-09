import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

xco = [-80,3,1,0,0]
yco = [-80,3,1,0,0]
xyco = [0,3,-3,0,0,0,0,0,0]

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

def perturbation(x, y):

    delta = 1e-6
    t = np.linspace(0, delta, 2)
    soly = solve_ivp(fun = Trajectory_derivatives, t_span = [t[0], t[-1]], y0=[x,y,0,0], t_eval = t) 

    x0,y0 = soly.y[:2]
    vector_along_trajectory = [x0[0]-x0[1], y0[0]-y0[1], V(x0[0], y0[0])-V(x0[1],y0[1])]/(np.sqrt((x0[0]-x0[1])**2+(y0[0]-y0[1])**2+(V(x0[0], y0[0])-V(x0[1],y0[1]))**2))
    #gradient = abs((y0[0]-y0[1])/(x0[0]-x0[1]))

    dvdx = -np.dot(xco, [2*x, 3*x**2, 4*x**3, 5*x**4, 6*x**5])-np.dot(xyco,[y, 2*x*y, y**2, 2*x*y**2, 3*(x**2)*y, y**3, 3*(x**2)*y**2, 2*x*y**3, 3*(x**2)*y**3])
    dvdy = -np.dot(yco, [2*y, 3*y**2, 4*y**3, 5*y**4, 6*y**5] )-np.dot(xyco, [x, x**2, 2*x*y, 2*(x**2)*y, x**3, 2*x*y**2, 2*(x**3)*y, 3*(x**2)*y**2, 3*(x**3)*y**2])

    normal_to_surface = [-dvdx, -dvdy, 1]/(np.sqrt(1+dvdx**2+dvdy**2)) 

    # cross product of direction of trajectory and normal to the surface to get a vector perpendicular to both
    pert_direction = np.cross(normal_to_surface, vector_along_trajectory)

    delta_z = delta*pert_direction

    return delta_z




A = [[0,0,1,0],[0,0,0,1],[-2,-3,0,0],[-5,-7,0,0]]
b = np.array([1,1,2,5])
col = ['r', 'r', 'r', 'r', 'b', 'b', 'b', 'b', 'g', 'g', 'g', 'g', 'p','p','p','p']
print(col[1])