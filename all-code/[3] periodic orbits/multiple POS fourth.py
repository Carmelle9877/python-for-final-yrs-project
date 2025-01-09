import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

plt.close()

plt.rcParams["contour.linewidth"] = (1)
fig = plt.figure()
ax = fig.add_subplot()
ax.set_aspect('equal', adjustable='box')
'''
fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.set_aspect('equal', adjustable='box')

fig3 = plt.figure()
ax3 = fig3.add_subplot()
ax3.set_aspect('equal', adjustable='box')

fig4 = plt.figure()
ax4 = fig4.add_subplot()
ax4.set_aspect('equal', adjustable='box')

fig5 = plt.figure()
ax5 = fig5.add_subplot()
ax5.set_aspect('equal', adjustable='box')

fig6 = plt.figure()
ax6 = fig6.add_subplot()
ax6.set_aspect('equal', adjustable='box')

fig7 = plt.figure()
ax7 = fig7.add_subplot()
ax7.set_aspect('equal', adjustable='box')

fig8 = plt.figure()
ax8 = fig8.add_subplot()
ax8.set_aspect('equal', adjustable='box')

fig9 = plt.figure()
ax9 = fig9.add_subplot()
ax9.set_aspect('equal', adjustable='box')
fig10 = plt.figure()
ax10 = fig10.add_subplot()
ax10.set_aspect('equal', adjustable='box')
fig11 = plt.figure()
ax11 = fig11.add_subplot()
ax11.set_aspect('equal', adjustable='box')
fig12 = plt.figure()
ax12 = fig12.add_subplot()
ax12.set_aspect('equal', adjustable='box')
fig13 = plt.figure()
ax13 = fig13.add_subplot()
ax13.set_aspect('equal', adjustable='box')
fig14 = plt.figure()
ax14 = fig14.add_subplot()
ax14.set_aspect('equal', adjustable='box')
fig15 = plt.figure()
ax15 = fig15.add_subplot()
ax15.set_aspect('equal', adjustable='box')


fig16 = plt.figure()
ax16 = fig16.add_subplot()
ax16.set_aspect('equal', adjustable='box')
fig17 = plt.figure()
ax17 = fig17.add_subplot()
ax17.set_aspect('equal', adjustable='box')
fig18 = plt.figure()
ax18 = fig18.add_subplot()
ax18.set_aspect('equal', adjustable='box')

fig19 = plt.figure()
ax19 = fig19.add_subplot()
ax19.set_aspect('equal', adjustable='box')
fig20 = plt.figure()
ax20 = fig20.add_subplot()
ax20.set_aspect('equal', adjustable='box')
fig21 = plt.figure()
ax21 = fig21.add_subplot()
ax21.set_aspect('equal', adjustable='box')
'''






#=================================================================================================

# x^2, x^3, x^4, x^5, x^6
# y^2, y^3, y^4, y^5, y^6
# xy, x^2y, xy^2, x^2y^2, x^3y, xy^3, x^3y^2, x^2y^3, x^3y^3

# initial guess for turning point and period of periodic orbit
# decide which energy level you want

xco = [-40,-2,1,0,0]
yco = [-40,2,1,0,0]
xyco = [0,1,1,0,0.1,0.1,0.05,0,0]

limits = -9.5,8,-9.5,7,1500



approx_xyp = [
    [-9, -1, 0.36], # left saddle 0
    [-1, -7.4, 0.3], # bottom saddle 1
    [5.4, 0.6, 0.3], # right saddle 2
    [0.1, 4, 0.27], # top saddle 3

    [-9.4, 4.4, 0.3], # TL min 4
    [-3.7, 6.4, 0.2], # TL min 5

    [-8.75, -6.4, 0.3], # BL min 6
    [-9.1, -4.5, 0.2], # BL min 7 

    [6.7, -8.11, 0.2], # BR min 8
    [2.2, -8.2, 0.2], # BR min 9

    [5.7, 4.8, 0.3], # TR min 10
    [2.5, 5.3, 0.3]] # TR min 11

approx_xyp = [
    [-4.5, -1, 0.36], # left saddle 0
    [-1, -7.4, 0.3], # bottom saddle 1
    [5.4, 0.6, 0.3], # right saddle 2
    [0.1, 4, 0.27], # top saddle 3

    [-9.4, 4.4, 0.3], # TL min 4
    [-3.7, 6.4, 0.2], # TL min 5

    [-8.75, -6.4, 0.3], # BL min 6
    [-9.1, -4.5, 0.2], # BL min 7 

    [6.7, -8.11, 0.2], # BR min 8
    [2.2, -8.2, 0.2], # BR min 9

    [5.7, 4.8, 0.3], # TR min 10
    [2.5, 5.3, 0.3]] # TR min 11
E = -200



#=================================================================================================

# defining V(x,y) function
def V(x,y):

    xterms = xco[0]*x**2 + xco[1]*x**3 + xco[2]*x**4 + xco[3]*x**5 + xco[4]*x**6
    yterms = yco[0]*y**2 + yco[1]*y**3 + yco[2]*y**4 + yco[3]*y**5 + yco[4]*y**6
    xyterms = xyco[0]*x*y + xyco[1]*(x**2)*y + xyco[2]*x*y**2 + xyco[3]*(x**2)*y**2 + xyco[4]*(x**3)*y + xyco[5]*x*y**3 + xyco[6]*(x**3)*y**2 + xyco[7]*(x**2)*y**3 + xyco[8]*(x**3)*y**3

    return (xterms + yterms + xyterms)

#print(V(3.35,3.1))

# negative derivative function for V(x,y)
def derivative(x,y):
    
    dvdx = -np.dot(xco, [2*x, 3*x**2, 4*x**3, 5*x**4, 6*x**5])-np.dot(xyco,[y, 2*x*y, y**2, 2*x*y**2, 3*(x**2)*y, y**3, 3*(x**2)*y**2, 2*x*y**3, 3*(x**2)*y**3])
    dvdy = -np.dot(yco, [2*y, 3*y**2, 4*y**3, 5*y**4, 6*y**5] )-np.dot(xyco, [x, x**2, 2*x*y, 2*(x**2)*y, x**3, 2*x*y**2, 2*(x**3)*y, 3*(x**2)*y**2, 3*(x**3)*y**2])

    return [dvdx, dvdy]

# setting up differential equations
def vdp_derivatives(t,a):

    [x,y,px,py] = a

    return [px, py, derivative(x,y)[0], derivative(x,y)[1]]

#=================================================================================================

# function that used to find periodic orbits
def func(approx_xyp):
    x,y,p = approx_xyp # can ony have one input so we put all vars into one

    t = np.linspace(0,p,1000)
    soly = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=[x,y,0,0], t_eval = t)
    
    coords = soly.y[:2]
    momentum = soly.y[2:]

    energy0 = V(coords[0,-1],coords[1,-1])
    energy1 = V(coords[0,0],coords[1,0])

    # outputs the last value for momentum and energy (E-V(x,y)) looking for all to be 0
    return [momentum[0,-1], momentum[1,-1], energy1+energy0-2*E]

# solves for periodic orbits and plots them given initial conditions
def periodic_orbits(approx_xyp,col):
    initial_x,initial_y,p = fsolve(func,approx_xyp, xtol=1e-12)

    #print(fsolve(func,approx_xyp, xtol=1e-12))

    t = np.linspace(0,2*p,1000)
    soly = solve_ivp(fun = vdp_derivatives, t_span = [t[0], t[-1]], y0=[initial_x,initial_y,0,0], t_eval = t)
    coords = soly.y[:2]
    momentum = soly.y[2:]

    #print(momentum[:,-1], V(coords[0,-1],coords[1,-1]), V(coords[0,0],coords[1,0]))

    #plt.plot(coords[0],coords[1], col, linewidth = 1, alpha = 1)
    return [initial_x,initial_y,p], coords

#fig, ((ax1, ax2), (ax3, ax4))  = plt.subplots(2, 2)

#fig.suptitle('Sharing x per column, y per row')

def contours(limits):
    xmax, xmin, ymax, ymin, zlim = limits
    # defining and plotting contours
    x1 = np.linspace(xmin,xmax,1000)
    y1 = np.linspace(ymin,ymax,1000)

    X, Y = np.meshgrid(x1, y1)
    Z = V(X, Y)

    # limit height of conotour lines
    if zlim != "none":
        for i in range(len(Z)):
            for j in range(len(Z[0])):
                    if Z[i,j]>zlim:
                        Z[i,j] = zlim

    levels = 20

    cs = ax.contour(X, Y, Z, levels)
    plt.clabel(cs, cs.levels[:], inline=True, fontsize=5)   
    '''
    cs2 = ax2.contour(X, Y, Z, levels)
    plt.clabel(cs2, cs2.levels[:], inline=True, fontsize=5)
    cs3 = ax3.contour(X, Y, Z, levels)
    plt.clabel(cs3, cs3.levels[:], inline=True, fontsize=5)
    cs4 = ax4.contour(X, Y, Z, levels)
    plt.clabel(cs4, cs4.levels[:], inline=True, fontsize=5)
    cs5 = ax5.contour(X, Y, Z, levels)
    plt.clabel(cs5, cs5.levels[:], inline=True, fontsize=5)
    cs6 = ax6.contour(X, Y, Z, levels)
    plt.clabel(cs6, cs6.levels[:], inline=True, fontsize=5)
    cs7 = ax7.contour(X, Y, Z, levels)
    plt.clabel(cs7, cs7.levels[:], inline=True, fontsize=5) 
    cs8 = ax8.contour(X, Y, Z, levels)
    plt.clabel(cs8, cs8.levels[:], inline=True, fontsize=5)
    cs9 = ax9.contour(X, Y, Z, levels)
    plt.clabel(cs9, cs9.levels[:], inline=True, fontsize=5)
    cs10 = ax10.contour(X, Y, Z, levels)
    plt.clabel(cs10, cs10.levels[:], inline=True, fontsize=5)
    cs11 = ax11.contour(X, Y, Z, levels)
    plt.clabel(cs11, cs11.levels[:], inline=True, fontsize=5)
    cs12 = ax12.contour(X, Y, Z, levels)
    plt.clabel(cs12, cs12.levels[:], inline=True, fontsize=5)
    cs13 = ax13.contour(X, Y, Z, levels)
    plt.clabel(cs13, cs13.levels[:], inline=True, fontsize=5)
    cs14 = ax14.contour(X, Y, Z, levels)
    plt.clabel(cs14, cs14.levels[:], inline=True, fontsize=5)
    cs15 = ax15.contour(X, Y, Z, levels)
    plt.clabel(cs15, cs15.levels[:], inline=True, fontsize=5)
    
    cs16 = ax16.contour(X, Y, Z, levels)
    plt.clabel(cs16, cs16.levels[:], inline=True, fontsize=5)
    cs17 = ax17.contour(X, Y, Z, levels)
    plt.clabel(cs17, cs17.levels[:], inline=True, fontsize=5)
    cs18 = ax18.contour(X, Y, Z, levels)
    plt.clabel(cs18, cs18.levels[:], inline=True, fontsize=5)
    
    cs19 = ax19.contour(X, Y, Z, levels)
    plt.clabel(cs19, cs19.levels[:], inline=True, fontsize=5)
    cs20 = ax20.contour(X, Y, Z, levels)
    plt.clabel(cs20, cs20.levels[:], inline=True, fontsize=5)
    cs21 = ax21.contour(X, Y, Z, levels)
    plt.clabel(cs21, cs21.levels[:], inline=True, fontsize=5)
    '''
    return

contours(limits)




count = 1
while E<=-200:

    for i in range(len(approx_xyp)):
        xi,yi,_ = approx_xyp[i]
        #print(V(xi,yi), i)
    
        approx_xyp[i], coords = periodic_orbits(approx_xyp[i], 'r-')
        print(approx_xyp[i], i)
        if count==1:
            ax.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            #fig.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200)
        '''      
        elif count == 2:
            ax2.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig2.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200)
        elif count == 3:
            ax3.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig3.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200) 
        elif count == 4:
            ax4.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig4.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200) 
        elif count == 5:
            ax5.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig5.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200)
        elif count == 6:
            ax6.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig6.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200) 
        elif count == 7:
            ax7.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig7.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200) 
        elif count == 8:
            ax8.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig8.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200)
        elif count == 9:
            ax9.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig9.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200) 
        elif count == 10:
            ax10.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig10.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200) 
        elif count == 11:
            ax11.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig11.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200)
        elif count == 12:
            ax12.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig12.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200) 
        elif count == 13:
            ax13.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig13.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200) 
        elif count == 14:
            ax14.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig14.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200) 
        elif count == 15:
            ax15.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig15.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200)
        
        elif count == 16:
            ax16.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig16.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200) 
        elif count == 17:
            ax17.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig17.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200)
        elif count == 18:
            ax18.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig18.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200) 
            
        elif count == 19:
            ax19.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig19.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200)
        elif count == 20:
            ax20.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig20.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200) 
        elif count == 21:
            ax21.plot(coords[0], coords[1], 'r-', linewidth = 0.8, alpha = 1)
            fig21.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200)
        '''
    count += 1
    E += 100    



#=================================================================================================




plt.xlabel('x')
plt.ylabel('y')
#plt.legend(loc = 'best')

#plt.savefig(r'[6] images/third plot/E=%i.png' %(E), format = 'png', dpi = 1200)

plt.show()

