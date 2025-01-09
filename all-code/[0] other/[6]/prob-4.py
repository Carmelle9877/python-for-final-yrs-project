import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#function
def model(z, t):
    x = z[0]
    y = z[1]
    if t<5:
        dxdt = 1/2(-x)
    else:
        dxdt = 1/2(-x + 2)
    dydt = 1/5(-y+x)
    dzdt = [dxdt,dydt]
    return dzdt

#initial conditions
z0=[0,0]

#time points
t=np.linspace(0,10)

#solve ODE
z = odeint(model, z0, t)

#plot points

plt.plot(t, z[:,0],'b-',label = 'dxdt')
plt.plot(t,z[:,1],'r--', label = 'dydt')
plt.xlabel('time')
plt.ylabel('values')
plt.legend()
plt.show()