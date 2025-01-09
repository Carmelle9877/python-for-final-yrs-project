# my attempts to find turning points but probably not useful

def turning_points(x0,y0):

    coords, momentum = ODE_solver(x0,y0,py) # all points on trajectory 

    # finding periodic orbits
    for i in range(20,len(coords[1])):
        
        pxi = momentum[0,i]
        pyi = momentum[1,i]

        if abs(pxi) < 0.1 and abs(pyi) < 0.1:
            print(i, pxi, pyi)
            return True
    return False

def find_turning(xlow, xhigh, ylow, yhigh):
    xstep = int((xhigh-xlow)/10)
    ystep = int((yhigh-ylow)/10)
    for i in range(10):
        x = xlow + i*xstep
        for j in range(10):
            y = ylow +j*ystep
            if turning_points(x,y) == True: 
                print(x,y)

    print("no turning points found")
    return


# This works well but Bartsch shat on it ;-;
def turningpoints(x,y,py):

    coords, momentum = ODE_solver(x,y,py)
    x0 = coords[0,:]
    y0 = coords[1,:]

    dx = np.diff(x0)
    dy = np.diff(y0)

    for i in range(1,points-2):
        dxi = dx[1:][i] * dx[:-1][i]
        dyi = dy[1:][i] * dy[:-1][i]

        if dxi <= 0 and dyi <= 0:
            print(coords[:,i]) 

    return 


# number of turning points in a time length
def numturningpoints(lst):
    dx = np.diff(lst)

    return np.sum(dx[1:] * dx[:-1] <= 0)


def Hessian(x,y):

    dvdxx = np.dot(xco, [2, 6*x, 12*x**2, 20*x**3, 30*x**4]) + np.dot(xyco, [0, 2*y, 0, 2*y**2, 6*x*y, 0, 6*x*y**2, 2*y**3, 6*x*y**3])
    dvdxy = np.dot(xyco, [1, 2*x, 2*y, 4*x*y, 3*x**2, 3*y**2, 6*(x**2)*y, 6*x*y**2, 9*(x**2)*y**2])
    dvdyy = np.dot(yco, [2, 6*y, 12*y**2, 20*y**3, 30*y**4]) + np.dot(xyco, [0, 0, 2*x, 2*x**2, 0, 6*x*y, 2*x**3, 6*(x**2)*y, 6*(x**3)*y])

    return -[[dvdxx, dvdxy], [dvdxy, dvdyy]]

def Jacobian(x,y):

    J = np.zeros((4,4))
    J[0,2] = 1
    J[1,3] = 1
    J[2:, :2] = Hessian(x,y)

    return J


def second_derivatives(x,y):

    dvdxx = np.dot(xco, [2, 6*x, 12*x**2, 20*x**3, 30*x**4]) + np.dot(xyco, [0, 2*y, 0, 2*y**2, 6*x*y, 0, 6*x*y**2, 2*y**3, 6*x*y**3])
    dvdxy = np.dot(xyco, [1, 2*x, 2*y, 4*x*y, 3*x**2, 3*y**2, 6*(x**2)*y, 6*x*y**2, 9*(x**2)*y**2])
    dvdyy = np.dot(yco, [2, 6*y, 12*y**2, 20*y**3, 30*y**4]) + np.dot(xyco, [0, 0, 2*x, 2*x**2, 0, 6*x*y, 2*x**3, 6*(x**2)*y, 6*(x**3)*y])

    return dvdxx, dvdxy, dvdyy


'''
approx_xyp1, coords1 = periodic_orbits(approx_xyp1, 'r-')
ax.plot(coords1[0], coords1[1], 'r-', linewidth = 0.8, alpha = 1)
'''
'''
approx_xyp2, coords2 = periodic_orbits(approx_xyp2, 'r-')
ax.plot(coords2[0], coords2[1], 'r-', linewidth = 0.8, alpha = 1)
approx_xyp3, coords3 = periodic_orbits(approx_xyp3, 'r-')
ax.plot(coords3[0], coords3[1], 'r-', linewidth = 0.8, alpha = 1)


approx_xyp4, coords4 = periodic_orbits(approx_xyp4, 'r-')
ax.plot(coords4[0], coords4[1], 'r-', linewidth = 0.8, alpha = 1)

approx_xyp5, coords5 = periodic_orbits(approx_xyp5, 'r-')
ax.plot(coords5[0], coords5[1], 'r-', linewidth = 0.8, alpha = 1)

approx_xyp6, coords6 = periodic_orbits(approx_xyp6, 'r-')
ax.plot(coords6[0], coords6[1], 'r-', linewidth = 0.8, alpha = 1)
approx_xyp7, coords7 = periodic_orbits(approx_xyp7, 'r-')
ax.plot(coords7[0], coords7[1], 'r-', linewidth = 0.8, alpha = 1)
approx_xyp8, coords8 = periodic_orbits(approx_xyp8, 'r-')
ax.plot(coords8[0], coords8[1], 'r-', linewidth = 0.8, alpha = 1)
'''
'''
approx_xyp1, coords1 = periodic_orbits(approx_xyp1, 'r-')
ax1.plot(coords1[0], coords1[1], 'r-', linewidth = 0.8, alpha = 1)
approx_xyp2, coords2 = periodic_orbits(approx_xyp2, 'r-')
ax1.plot(coords2[0], coords2[1], 'r-', linewidth = 0.8, alpha = 1)
approx_xyp3, coords3 = periodic_orbits(approx_xyp3, 'r-')
ax1.plot(coords3[0], coords3[1], 'r-', linewidth = 0.8, alpha = 1)
approx_xyp4, coords4 = periodic_orbits(approx_xyp4, 'r-')
ax1.plot(coords4[0], coords4[1], 'r-', linewidth = 0.8, alpha = 1)

approx_xyp5, coords5 = periodic_orbits(approx_xyp5, 'r-')
ax1.plot(coords5[0], coords5[1], 'r-', linewidth = 0.8, alpha = 1)
approx_xyp6, coords6 = periodic_orbits(approx_xyp6, 'r-')
ax1.plot(coords6[0], coords6[1], 'r-', linewidth = 0.8, alpha = 1)
approx_xyp7, coords7 = periodic_orbits(approx_xyp7, 'r-')
ax1.plot(coords7[0], coords7[1], 'r-', linewidth = 0.8, alpha = 1)
approx_xyp8, coords8 = periodic_orbits(approx_xyp8, 'r-')
ax1.plot(coords8[0], coords8[1], 'r-', linewidth = 0.8, alpha = 1)


periodic_orbits(approx_xyp5, 'r-')
periodic_orbits(approx_xyp6, 'r-') #-900
periodic_orbits(approx_xyp7, 'r-') #-1200
periodic_orbits(approx_xyp8, 'r-') #-900
 


periodic_orbits(approx_xyp9, 'b-')
periodic_orbits(approx_xyp10, 'g-')
periodic_orbits(approx_xyp11, 'r-')
#periodic_orbits(approx_xyp12, 'r-')

need to plot periodic orbits at multiple different energy levels to see how they merge when they get high enough
need to save each plot
need to use the xyp from the previous for the next one
dont plot them for ones that are too low or dont work

'''