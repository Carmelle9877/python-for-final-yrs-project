# not working Oakley trajectories, quite wrong
# looks like i plotted px vs py using hamiltonian equations rather than integrating

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# setting variables
a = 1
theta = 0
sigma1 = 0.2
sigma2 = 0.2
points = 20
limit = 5
# defining T, D and M
T = np.matrix([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
D =  np.matrix([[sigma1**2, 0], [0, sigma2**2]])
M = np.dot(np.dot(T, D), np.linalg.inv(T))

# fixed values
E = 1
x = 10

# chosed values for trajectories
py = np.linspace(-limit, limit, points)
y = np.linspace(-0.5, 0.5, points)

# Oakleys equation
def V(x,y):
    top = (1-x*y)**2
    bottomx = 1/2 + (np.arctan(0.25*(x-y)))/(np.pi)
    bottomy = 1/2 + (np.arctan(0.25*(y-x)))/(np.pi)
    return top/(x*bottomx + y*bottomy)**2

# hamiltonian function for px
px = np.sqrt(2*E - 2*(V(x,y)) - py**2)

plt.plot(px,py, 'r')

# initialise x and y to plot v(x,y)
x = np.linspace(0.1,limit,points)
y = np.linspace(0.1,limit,points)

# setting Z and plotting contours
X, Y = np.meshgrid(x, y)
Z = V(X, Y)
plt.contour(X, Y, Z, colors='black', levels = 50)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()