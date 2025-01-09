# 3D version of my plot

import numpy as np
import matplotlib.pyplot as plt

y2 = x2 = 40
y3 = x3 = 0
y4 = x4 = 1
y5 = x5 = 0
y6 = x6 = 0

xy = 0 
x2y = 0
xy2 = 0
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

limit = 10
x = np.linspace(-12,12,1000)
y = np.linspace(-12,12,1000)

X, Y = np.meshgrid(x, y)
Z = V(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.contour3D(X, Y, Z, 100, cmap = 'binary')
ax.plot_surface(X, Y, Z)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(-1200,500)

plt.savefig('3D-plot.png', format = 'png', dpi = 150)

plt.show()