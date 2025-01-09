import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# cubic with x^2y terms
xco = [-80,3,1,0,0]
yco = [-80,3,1,0,0]
xyco = [0,3,-3,0,0,0,0,0,0]

approx_xyp = [-0.7,-8.7,0.3] # [x,y,p]
E = -2000

points = 1000

# plotting and defining trajectories
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

# finding periodic orbits
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

def perturbation(x, y):

    delta = 1
    t = np.linspace(0, delta, 2)
    soly = solve_ivp(fun = Trajectory_derivatives, t_span = [t[0], t[-1]], y0=[x,y,0,0], t_eval = t) 

    x0,y0 = soly.y[:2]
    px0, py0 = soly.y[2:]
    E0 = V(x0[0], y0[0])
    E1 = V(x0[1],y0[1]) + 0.5*((px0[1])**2+(py0[1])**2)
    vector_along_trajectory = [x0[0]-x0[1], y0[0]-y0[1], E0-E1]/la.norm([x0[0]-x0[1], y0[0]-y0[1], E0-E1])


    dvdx = -np.dot(xco, [2*x, 3*x**2, 4*x**3, 5*x**4, 6*x**5])-np.dot(xyco,[y, 2*x*y, y**2, 2*x*y**2, 3*(x**2)*y, y**3, 3*(x**2)*y**2, 2*x*y**3, 3*(x**2)*y**3])
    dvdy = -np.dot(yco, [2*y, 3*y**2, 4*y**3, 5*y**4, 6*y**5] )-np.dot(xyco, [x, x**2, 2*x*y, 2*(x**2)*y, x**3, 2*x*y**2, 2*(x**3)*y, 3*(x**2)*y**2, 3*(x**3)*y**2])

    normal_to_surface = [dvdx, dvdy, V(x,y)]/la.norm([dvdx, dvdy, V(x,y)])

    pert_direction = np.cross(normal_to_surface, vector_along_trajectory)



    momentum_along_trajectory = [px0[0]-px0[1], py0[0]-py0[1]]/la.norm([px0[0]-px0[1], py0[0]-py0[1]])
    momentum_perp_traj = np.empty_like(momentum_along_trajectory)
    momentum_perp_traj[0] = -momentum_along_trajectory[1]
    momentum_perp_traj[1] = momentum_along_trajectory[0]

    delta_pz = delta*momentum_perp_traj
    delta_px, delta_py = delta_pz
    #print(pert_direction)

    delta_z = delta*pert_direction
    delta_x, delta_y, _ = delta_z

    return [delta_x, delta_y, delta_px, delta_py]


# outputs the value of A for a certain point n
def A(n):

    n = int((n/period)*999)
    t = np.linspace(0,period, points)
    soly = solve_ivp(fun = Trajectory_derivatives, t_span = [t[0], t[-1]], y0=[x_PO,y_PO,0,0], t_eval = t[n:n+1])
    
    coords = soly.y[:2]
    x = coords[0,0]
    y = coords[1,0]

    dvdxx = np.dot(xco, [2, 6*x, 12*x**2, 20*x**3, 30*x**4]) + np.dot(xyco, [0, 2*y, 0, 2*y**2, 6*x*y, 0, 6*x*y**2, 2*y**3, 6*x*y**3])
    dvdxy = np.dot(xyco, [1, 2*x, 2*y, 4*x*y, 3*x**2, 3*y**2, 6*(x**2)*y, 6*x*y**2, 9*(x**2)*y**2])
    dvdyy = np.dot(yco, [2, 6*y, 12*y**2, 20*y**3, 30*y**4]) + np.dot(xyco, [0, 0, 2*x, 2*x**2, 0, 6*x*y, 2*x**3, 6*(x**2)*y, 6*(x**3)*y])

    return np.array([[0,0,1,0],[0,0,0,1],[-dvdxx,-dvdxy,0,0],[-dvdxy,-dvdyy,0,0]])

# since t gives weird values, i could add an extra input and loop for each value in t
def stability_derivatives(t,a):

    delta_z = np.array(a)

    # outputs the values of small perturbations at each point along the orbit 
    return (A(t))@delta_z

# initial perturbation needs to be in a direction not on the periodic orbit so we need to find the gradient of the PO at the initial point



def monodromy(x_PO, y_PO):

    delta0 = (A(0))@(perturbation(x_PO,y_PO))

    t = np.linspace(0,period,points)
    solz = solve_ivp(fun = stability_derivatives, t_span = [t[0], t[-1]], y0=delta0, t_eval = t)
    delta_coords = solz.y[:2]
    delta_momentum = solz.y[2:]

    d_z1_0 = [delta_coords[0,0], delta_coords[1,0],0,0]
    #/la.norm(delta_coords[:,0])
    d_z1_T = [delta_coords[0,-1], delta_coords[1,-1],0,0]
    #/la.norm(delta_coords[:,-1])
    d_z2_0 = [0,0,delta_momentum[0,0], delta_momentum[1,0]]
    #/la.norm(delta_momentum[:,0])
    d_z2_T = [0,0,delta_momentum[0,-1], delta_momentum[1,-1]]
    #/la.norm(delta_momentum[:,-1])

    M = [[np.dot(d_z1_T, d_z1_0), np.dot(d_z2_T, d_z1_0)],[np.dot(d_z1_T, d_z2_0), np.dot(d_z2_T, d_z2_0)]]
    print(M)

    eigvalues, _ = la.eig(M)
    return eigvalues

a, b = monodromy(x_PO, y_PO)
print(a,b,a*b)

# M = [[d_z1(T).d_z1(0), d_z2(T).d_z1(0)], 
#      [d_z1(T).d_z2(0), d_z2(T).d_z2(0)]]
