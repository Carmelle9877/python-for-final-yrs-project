import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

plt.rcParams['figure.figsize'] = [10,5]

def vdp_derivatives(t,y):
    x = y[0]
    v = y[1]
    return [v, mu*(1-x*x)*v - x]

mu = 2
t = np.linspace(0,10,500)

sol = solve_ivp( fun=vdp_derivatives, t_span = [t[0], t[-1]], y0 = [1,0], t_eval=t )

plt.subplot(1,2,1)
plt.plot(sol.y[0], sol.y[1])
plt.xlabel('Position, x')
plt.ylabel('Velocity, v=dx/dy')

plt.subplot(1,2,2)
plt.plot(sol.t, sol.y[0])
plt.ylabel('Position, x')
plt.xlabel('time')

plt.tight_layout()