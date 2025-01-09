import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

x0=1
y0=1

def modelx(x,t):
    dxdt = -x
    return dxdt

def modely(y,t):
    dydt = -2*y
    return dydt

t=np.linspace(0,5)

x = odeint(modelx, x0, t)
y = odeint(modely, y0, t)

#plt.plot(t, x, 'b-', label = ('x(t)'))
#plt.plot(t, y, 'g--', label = ('y(t)'))
plt.plot(x, y, 'g--')
plt.xlabel('px')
plt.ylabel('py')
plt.legend()
plt.show()

