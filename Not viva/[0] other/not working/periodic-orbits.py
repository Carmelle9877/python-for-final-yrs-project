# attemp to find periodic orbits, not good

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# setting up graph 
plt.rcParams["contour.linewidth"] = (1)
fig = plt.figure()
ax = fig.add_subplot()
ax.set_aspect('equal', adjustable='box')

#================================================================================================================================

# set coefficient variables for V(x,y)
y2 = x2 = 40
y3 = x3 = 3
y4 = x4 = 1
y5 = x5 = 0
y6 = x6 = 0

xy = 15 
x2y = 5
xy2 = 5
x2y2 = 2.5

x3y = 1
xy3 = 1
x3y2 = 0
x2y3 = 0
x3y3 = 0

# defining V(x,y) function
def V(x,y):
    x_y   =  xy  * x*y
    quadratic = -x2*(x**2)/2 + x2y*(x**2)*y + x2y2*(x**2)*(y**2) + xy2*x*(y**2) - y2*(y**2)/2
    cubic     =  x3 * (x**3)/3 + x3y*(x**3)*y + x3y2*(x**3)*(y**2) + x3y3*(x**3)*(y**3) + x2y3*(x**2)*(y**3)+ xy3*x*(y**3)+ y3 * (y**3)/3
    quartic   =  x4 * (x**4)/4 + y4 * (y**4)/4
    quintic   =  x5 * (x**5)/5 + y5 * (y**5)/5
    sextic    =  x6 * (x**6)/6 + y6 * (y**6)/6

    return (quadratic + cubic + quartic + quintic + sextic + x_y)

#======================================================================================================================

# setting initial conditions from Hamiltonian equations
x = 0
y = -3.5
E = V(x,y)
py = 0

def ODE_solver(x,y,py):

    E = V(x,y)

    if x>0:
        px = -np.sqrt(2*E - 2*(V(x,y)) - py**2)
    else :
        px = np.sqrt(2*E - 2*(V(x,y)) - py**2)

    y0 = [x,y,px,py]


    # calculating trajectories
    t = np.linspace(0,10,100)

    # negative derivative function for V(x,y)
    def derivative(x,y):
        dvdx = x2*x - x3*(x**2) - x4*(x**3) - x5*(x**4) - x6*(x**5) - xy*y - xy2*y**2 - 2*x2y*x*y - 2*x2y2*x*(y**2) - xy3*(y**3) - 3*x3y*(x**2)*y - 2*x2y3*(y**3)*x - 3*x3y2*(x**2)*(y**2) - 3*x3y3*(x**2)*(y**3)
        dvdy = y2*y - y3*(y**2) - y4*(y**3) - y5*(y**4) - y6*(y**5) - xy*x - x2y*x**2 - 2*xy2*x*y - 2*x2y2*(x**2)*y - x3y*(x**3) - 3*xy3*x*(y**2) - 2*x3y2*(x**3)*y - 3*x2y3*(x**2)*(y**2) - 3*x3y3*(x**3)*(y**2)
        return [dvdx, dvdy]

    # setting up differential equaitons
    def vdp_derivatives(t,a):
        x  = a[0]
        y  = a[1]
        px = a[2]
        py = a[3]
        return [px, py ,derivative(x,y)[0], derivative(x,y)[1]]

    soly = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=y0, t_eval = t)

    solution = soly.y[:]

    return solution

#================================================================================================================================

momentum = ODE_solver(x,y,py)[2:]
coords = ODE_solver(x,y,py)[:2]

#================================================================================================================================

x1 = np.linspace(7.9,10.6,28)
#y1 = np.linspace(-14,14,29)

# finding critical points

#for i in range(len(x1)):
#    coords = ODE_solver(x1[i],y,py)[:2]
#    if [x,y]==coords:
#        print([x,y])


#critical = []
#momentum_sum = (ODE_solver(x1[1],y,py)[2]+ ODE_solver(x1[1],y,py)[3])[1:]
#min_momentum = min(momentum_sum)
#index_min_sum = np.where(momentum_sum == min_momentum)
#print(min_momentum)
#for i in range(len(x1)):
#    momentum = ODE_solver(x1[i],y,py)[2:]
#    coords = ODE_solver(x1[i],y,py)[:2]
#    momentum_sum = (abs(momentum[0])+abs(momentum[1]))[1:]
#    if min(momentum_sum)<min_momentum:
#        min_momentum = min(momentum_sum)
#        index_min_sum = np.where(momentum_sum == min_momentum)
#        coords_minimum = [coords[0,index_min_sum],coords[1,index_min_sum]]
#
#    for k in range(1, len(momentum)):
#        if momentum[0,k]==momentum[1,k]==0:
#            print(coords[:,i])
#            
#            #critical.append([coords[:,k]])
#print(min_momentum)
#print(coords_minimum)         
#
#print(critical)

# defining and plotting contours
x1 = np.linspace(-14,14,100)
y1 = np.linspace(-14,14,100)


X, Y = np.meshgrid(x1, y1)
Z = V(X, Y)

# limit heiht of conotour lines
for i in range(len(Z)):
    for j in range(len(Z[0])):
            if Z[i,j]>7500:
                Z[i,j] = 7500

cs = plt.contour(X, Y, Z, levels = 30)

# aesthetics
plt.clabel(cs, cs.levels[:10], inline=True, fontsize=7) # label contour lines
#plt.title(r'$-%i\frac{x^2}{2} -%i\frac{y^2}{2} + %i\frac{x^3}{3} + %i\frac{y^3}{3} + %i\frac{y^4}{4} + %i\frac{y^4}{4}$' %(x2, y2, x3, y3, x4, y4)) #updates automatically
plt.xlabel('x')
plt.ylabel('y')

# plottimg trajectories
plt.plot(coords[0],coords[1], 'r-', linewidth = 0.9, label = r'E=%i, x=%i, y=%i, $p_{y}$=%i' %(E, x, y, py)) # updates automatically
plt.legend(loc = 'best')

#plt. savefig(r'a=%i,b=%i,c=%i_E=%i,x=%i_y=%i,py=%i.png' %(x2, x3, x4, E, x, y, py), format = 'png', dpi = 1200) # updates automatically as long as twox=twoy etc
#plt. savefig(r'xy=%i,x2y=%i,x2y2=%i,x3y=%i_E=%i,x=%i_y=%i,py=%i.png' %(xy, x2y, x2y2,x3y, E, x, y, py), format = 'png', dpi = 1200)

plt.show()