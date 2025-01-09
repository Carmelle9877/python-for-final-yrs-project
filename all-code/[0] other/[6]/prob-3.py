import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#dont know why this doesnt work :(

def modelx(x,t):
    dxdt = 3 * np.exp(-t)
    print('dxdt for t=', t ,'is',dxdt)
    return dxdt

x0 = 0

def modely(y,t):
    dydt = -y+3
    print('dydt for t=' ,t, 'is', dydt)
    return dydt

y0 = 0 

t=np.linspace(0,5)

x = odeint(modelx, x0, t)
y = odeint(modely, y0, t)

plt.plot(t, x, 'b-', label = ('x(t)'))
plt.plot(t, y, 'g--', label = ('y(t)'))
plt.xlabel('time')
plt.ylabel('values')
plt.legend()
plt.show()

