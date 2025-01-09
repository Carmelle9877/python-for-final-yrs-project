import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# funtion for dzdt
def model(z,t,u):
    x = z[0]
    y = z[1]
    dxdt = (-x+u)/2
    dydt = (-y+x)/5
    dzdt = [dxdt,dydt]
    return dzdt

# initial conditions
z0 = [0,0]

# number of time points
n=401

# time points
t = np.linspace(0,40,n)

# step intput
u = np.zeros(n)
# change to 2 at time = 5
u[51:] = 2

# store solutions
x = np.empty_like(t)
y = np.empty_like(t)

# record initial conditions
x[0] = z0[0]
y[0] = z0[1]

# solve ODEs
#doing each i individually is basically spolitting th eODE into lots of little pieces, i dont know why this works better but it does
for i in range(1,n):
    # span for next time step
    tspan = [t[i-1],t[i]]
    #solve for next step
    z = odeint(model,z0,tspan,args=(u[i],))
    # store solutions for plotting
    x[i] = z[1][0]
    y[i] = z[1][1]
    # next initial condition
    z0 = z[1]

# plot results
plt.plot(t,u,'g:',label='u(t)')
plt.plot(t,x,'b-',label='x(t)')
plt.plot(t,y,'r--',label='y(t)')
plt.ylabel('values')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()