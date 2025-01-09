#working 3D Oakley plot without extra obstacle

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits import mplot3d

def f(x, y):
    return ((1-x*y)**2)/(x*(0.5+(np.arctan(0.25*(x-y)))/(np.pi))+y*(0.5+(np.arctan(0.25*(y-x)))/(np.pi)))**2

#((1-x*y)**2/(x*(0.5+(np.arctan((x-y)/4))/np.pi)))
            
#+y*(0.5+(np.arctan((y-x)/4))/np.pi)


x = np.linspace(0.01, 2, 200)
y = np.linspace(0.01, 2, 200)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

zlim = 2

if zlim != "none":
    for i in range(len(Z)):
        for j in range(len(Z[0])):
                if Z[i,j]>zlim:
                    Z[i,j] = zlim

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(0,zlim)

plt.show()