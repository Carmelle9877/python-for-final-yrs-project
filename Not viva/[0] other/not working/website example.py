import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
plt.rcParams['figure.figsize'] = [10, 5]

def lorenz(t, state, sigma, beta, rho):
    x, y, z = state
     
    dx = 10 * (y - x)
    dy = x * (28 - z) - y
    dz = x * y - (8/3) * z
     
    return [dx, dy, dz]
 
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0
 
p = (sigma, beta, rho)  # Parameters of the system
 
y0 = [1.0, 1.0, 1.0]  # Initial state of the system

t_span = (0.0, 40.0)
t = np.arange(0.0, 40.0, 0.01)

result_solve_ivp = solve_ivp(lorenz, t_span  = [t[0], t[-1]], y0=y0, args=p,method='LSODA', t_eval=t)
 
 
plt.plot(result_solve_ivp.y[0, :],
        result_solve_ivp.y[1, :],
        result_solve_ivp.y[2, :],
        linewidth = 1)
plt.set_title("solve_ivp LSODA")

plt.show()