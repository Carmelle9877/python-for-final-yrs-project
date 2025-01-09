import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import pdb

# cubic with x^2y terms
xco = [-80,3,1,0,0]
yco = [-80,3,1,0,0]
xyco = [0,3,-3,0,0,0,0,0,0]

approx_xyp = [-0.7,-8.7,0.3] # [x,y,p]
E = -2000

points = 1000

# setting up V(x,y) for the surface
def V(x,y):

    xterms = xco[0]*x**2 + xco[1]*x**3 + xco[2]*x**4 + xco[3]*x**5 + xco[4]*x**6
    yterms = yco[0]*y**2 + yco[1]*y**3 + yco[2]*y**4 + yco[3]*y**5 + yco[4]*y**6
    xyterms = xyco[0]*x*y + xyco[1]*(x**2)*y + xyco[2]*x*y**2 + xyco[3]*(x**2)*y**2 + xyco[4]*(x**3)*y + xyco[5]*x*y**3 + xyco[6]*(x**3)*y**2 + xyco[7]*(x**2)*y**3 + xyco[8]*(x**3)*y**3

    return (xterms + yterms + xyterms)

# partial derivatives of V(x,y)
def first_derivative(x,y):

    dvdx = -np.dot(xco, [2*x, 3*x**2, 4*x**3, 5*x**4, 6*x**5])-np.dot(xyco,[y, 2*x*y, y**2, 2*x*y**2, 3*(x**2)*y, y**3, 3*(x**2)*y**2, 2*x*y**3, 3*(x**2)*y**3])
    dvdy = -np.dot(yco, [2*y, 3*y**2, 4*y**3, 5*y**4, 6*y**5] )-np.dot(xyco, [x, x**2, 2*x*y, 2*(x**2)*y, x**3, 2*x*y**2, 2*(x**3)*y, 3*(x**2)*y**2, 3*(x**3)*y**2])

    return dvdx, dvdy

# second partial derivatives of V(x,y)
def second_derivatives(x,y):

    dvdxx = np.dot(xco, [2, 6*x, 12*x**2, 20*x**3, 30*x**4]) + np.dot(xyco, [0, 2*y, 0, 2*y**2, 6*x*y, 0, 6*x*y**2, 2*y**3, 6*x*y**3])
    dvdxy = np.dot(xyco, [1, 2*x, 2*y, 4*x*y, 3*x**2, 3*y**2, 6*(x**2)*y, 6*x*y**2, 9*(x**2)*y**2])
    dvdyy = np.dot(yco, [2, 6*y, 12*y**2, 20*y**3, 30*y**4]) + np.dot(xyco, [0, 0, 2*x, 2*x**2, 0, 6*x*y, 2*x**3, 6*(x**2)*y, 6*(x**3)*y])

    return dvdxx, dvdxy, dvdyy

#print(second_derivatives(-0.87, -7.5))
# setting up the ivp that solves for teh hamiltonian system of the trajectories
def Trajectory_derivatives(t,a):

    [x,y,px,py] = a
    dvdx, dvdy = first_derivative(x,y)

    #pdb.set_trace()

    return [px, py, dvdx, dvdy]

# fuction that solves for periodic orbits by finding a point where px, py and E-energy are 0
def func(approx_xyp):
    x,y,p = approx_xyp
    t = np.linspace(0,p,1000)

    soly = solve_ivp(fun = Trajectory_derivatives, t_span = [t[0], t[-1]], y0=[x,y,0,0], t_eval = t)
    
    coords = soly.y[:2]
    momentum = soly.y[2:]

    # outputs the last value for momentum and energy (E-V(x,y)) looking for all to be 0
    return [momentum[0,-1], momentum[1,-1], E-V(coords[0,-1],coords[1,-1])]

# solve for a periodic orbits for an initial prediction to then use x_PO and y_PO for stability analysis
x_PO,y_PO,half_period = fsolve(func,approx_xyp) # z0 coords for a periodic orbit 
period = 2*half_period
#print(x_PO, y_PO, period)

# direction of the perturbations. perpendicular to the direction of the Periodic
def perturbation(x, y):

    # derivatives of V wrt x and y
    dvdx, dvdy = first_derivative(x,y)
    
    # gradient of the curve of the potential energy surface (perpendicular to PO)
    dydx = dvdx/dvdy

    # normalised vector equivalent of above gradient
    u = np.array([1,dydx]/la.norm([1,dydx]))
    #print(u)
    delta_z1 = np.array([u[0],u[1],0,0])
    delta_z2 = np.array([0,0,u[0],u[1]])

    return delta_z1, delta_z2

# setting up th ivp to solve for the PO and the displacement of two trajectories that near teh PO 
def stability_derivatives(t,a):

    # unpacking a into periodic points and delta points in two directions
    z0 = a[:4]
    x,y,px,py = z0
    dvdx, dvdy = first_derivative(x,y)

    # perturbations in two directions
    delta_z1 = a[4:8]
    delta_z2 = a[8:]
    dvdxx, dvdxy, dvdyy = second_derivatives(x,y)

    # Jacobian at points along the orbit, used to solve for perturbations
    A = np.array([[0,0,1,0],[0,0,0,1],[-dvdxx,-dvdxy,0,0],[-dvdxy,-dvdyy,0,0]])
    #print(A)
    A_z1 = A@delta_z1
    A_z2 = A@delta_z2

    #print(A_z1, A_z2)

    return [px, py, dvdx, dvdy, A_z1[0],A_z1[1],A_z1[2],A_z1[3], A_z2[0], A_z2[1], A_z2[2], A_z2[3]]

P = [[-211,39],[39,385]]
print(la.eig(P))

def monodromy(x_PO, y_PO):

    #perturbations in 2 directions
    delta0_z1, delta0_z2 = perturbation(x_PO,y_PO)

    delta0 = [x_PO,y_PO,0,0,delta0_z1[0],delta0_z1[1],delta0_z1[2],delta0_z1[3], delta0_z2[0], delta0_z2[1], delta0_z2[2], delta0_z2[3]]

    t = np.linspace(0,period,points)

    solz = solve_ivp(fun = stability_derivatives, t_span = [t[0], t[-1]], y0=delta0, t_eval=t)

    z = solz.y
    #print(z[:4,1])
    z0 = solz.y[:4] # periodic orbit
    z1 = solz.y[4:8] #
    z2 = solz.y[8:]

    #print(z0[:,-1], z1[:,-1])


    M = [[np.dot(z1[:,-1], z1[:,0]), np.dot(z2[:,-1], z1[:,0])],
         [np.dot(z1[:,-1], z2[:,0]), np.dot(z2[:,-1], z2[:,0])]]

    #print(M)

    eigvalues, _ = la.eig(M)
    return eigvalues, z0,z1,z2

eigvalues, z0,z1,z2 = monodromy(x_PO, y_PO)
lambda_1, lambda_2 = eigvalues
print(lambda_1,lambda_2, lambda_1*lambda_2)

plt.plot(z0[0],z0[1], 'r-')
#plt.plot(z1[0]+z0[0],z1[1]+z0[1], 'b-')
plt.plot(z2[0]+z0[0],z2[1]+z0[1], 'g-')


#print([x_PO, y_PO, 0,0])

#print(soly.y[:,1])



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
limits = -12,10,-12,9,"none"
contours(limits)

#plt.show()