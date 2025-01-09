import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def model(y,t):
    dydt = -y + 1
    return dydt

y0 = 0

t = np.linspace(0,20)

y = odeint(model,y0,t)


plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.legend()
plt.show()