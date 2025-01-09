from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.integrate import solve_ivp

plt.rcParams["contour.linewidth"] = (1)

y2 = x2 = 40
y3 = x3 = 0
y4 = x4 = 1
y5 = x5 = 0
y6 = x6 = 0

xy = 0 
x2y = 1
xy2 = 1
x2y2 = 0

x3y = 0
xy3 = 0
x3y2 = 0
x2y3 = 0
x3y3 = 0


# defining V(x,y) function
def V(x,y):
    x_y   =  xy  * x*y
    quadratic = -x2*(x**2)/2 + x2y*(x**2)*y + x2y2*(x**2)*(y**2) + xy2*x*(y**2) - y2*(y**2)/2
    cubic     =  x3 * (x**3)/3 + x3y*(x**3)*y + x3y2*(x**3)*(y**2) + x3y3*(x**3)*(y**3) + x2y3*(x**2)*(y**3)+ xy3*x*(y**3)+ y3 * (y**3)/3
    quartic   =  x4  * (x**4)/4 + y4  * (y**4)/4
    quintic   =  x5  * (x**5)/5 + y5  * (y**5)/5
    sextic    =  x6   * (x**6)/6 + y6   * (y**6)/6

    return (quadratic + cubic + quartic + quintic + sextic + x_y)

# setting initial conditions from Hamiltonian equations
E = 250
x = 1
y = 1
py = 0

if x>0:
    px = -np.sqrt(2*E - 2*(V(x,y)) - py**2)
else :
    px = np.sqrt(2*E - 2*(V(x,y)) - py**2)

y0 = [x,y,px,py]

t = np.linspace(0,10,10000)

# negative derivative function for V(x,y)
def derivative(x,y):
    dvdx = x2*x - x3*(x**2) - x4*(x**3) - x5*(x**4) - x6*(x**5) - xy*y - 2*x2y*x*y - xy2*y**2 - 2*x2y2*x*y**2  - 3*x3y*(x**2)*y - xy3*(y**3) - 3*x3y2*(x**2)*(y**2) - 2*x2y3*x*(y**3) - 3*x3y3*(x**2)*(y**3)
    dvdy = y2*y - y3*(y**2) - y4*(y**3) - y5*(y**4) - y6*(y**5) - xy*x - x2y*x**2 - 2*xy2*x*y - 2*x2y2*(x**2)*y - x3y*(x**3) - 3*xy3*x*(y**2) - 2*x3y2*(x**3)*y - 3*x2y3*(x**2)*(y**2) - 3*x3y3*(x**3)*(y**2)
    return [dvdx, dvdy]

# setting up differential equaitons
def vdp_derivatives(t,a):
    x  = a[0]
    y  = a[1]
    px = a[2]
    py = a[3]
    return [px, py ,derivative(x,y)[0], derivative(x,y)[1]]

sol = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=y0, t_eval = t)
coords = sol.y[:2]

c = V(coords[0], coords[1])

x = np.arange(-12,10,0.1)
y = np.arange(-12,10,0.1)
X,Y = np.meshgrid(x,y)
Z = V(X,Y)


fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection='3d')


surf = ax.plot_surface(X, Y, Z, alpha=0.8, cmap = 'viridis')
cset = ax.contour(X, Y, Z, zdir='z', offset=np.min(Z)-500, levels = 20)
#cset = ax.contour(X, Y, Z, zdir='x', offset=-10, levels = 20)
#cset = ax.contour(X, Y, Z, zdir='y', offset=10, levels = 20)


fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

ax.plot3D(coords[0], coords[1],c, 'black', linewidth = 1.5)
ax.plot3D(coords[0],coords[1],np.min(Z)-500, 'r-', linewidth = 0.7)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_zlim(np.min(Z) -500, np.max(Z))
ax.set_title('3D surface with 2D contour plot projections')


plt.show()