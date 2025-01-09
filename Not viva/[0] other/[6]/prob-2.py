import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

t = np.linspace(1,40)

def model(y,t):
    if t<10:
        dydt = (1/5)*(-y)
    if t>=10:
        dydt = (1/5)*(-y+2)
    return dydt

y0 = 1

y = odeint(model,y0,t)

plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()