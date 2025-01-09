# working conout plot for x^2 - 2y^2

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def V(x,y):
    return (x**2)/2-y**2

x = np.linspace(-20,20,100)
y = np.linspace(-20,20,100)

X, Y = np.meshgrid(x, y)
Z = V(X, Y)

plt.contour(X, Y, Z, colors='black')
plt.show()
