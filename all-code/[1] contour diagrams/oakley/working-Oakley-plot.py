import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#setting variables
a = 1
theta = 0
sigma1 = 0.2
sigma2 = 0.2
points = 200
limit = 3

#defining T, D and M
T = np.matrix([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
D =  np.matrix([[sigma1**2, 0], [0, sigma2**2]])
M = np.dot(np.dot(T, D), np.linalg.inv(T))

#initialise x and y
x = np.linspace(0.2,limit,points)
y = np.linspace(0.2,limit,points) 

#Oakleys equation
def V(x,y):
    top = (1-x*y)**2
    bottomx = 1/2 + (np.arctan(0.25*(x-y)))/(np.pi)
    bottomy = 1/2 + (np.arctan(0.25*(y-x)))/(np.pi)
    return top/(x*bottomx + y*bottomy)**2
#+a*np.exp((-(T2)*np.dot((np.linalg.inv(M)),[x-1,y-1]))/2)

#((1-x*y)**2)/(x*(0.5+(np.arctan(0.25*(x-y)))/(np.pi))+y*(0.5+(np.arctan(0.25*(y-x)))/(np.pi)))**2

#setting Z and plotting contours
X, Y = np.meshgrid(x, y)
Z = V(X, Y)
plt.contour(X, Y, Z, levels = 50)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()