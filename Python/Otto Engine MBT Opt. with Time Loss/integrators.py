### Roderick Landreth
### MER-371 Spring 2020
### for Cycle Integration Assignment
### Purpose: Integration method definitions

import numpy as np

# Takes a function and returns its integral. For multiple dimensions, input 
# initial conditions as an array, similar to variables
def rk_4(F,x_not,t0,dt,N):
    time_vals = np.arange(float(t0),(N+1)*dt + float(t0),dt)
    # make array, change values later for the sake of time savings
    f_vals = np.array([float(x_not)]*(N+1))
    for p in range(N):
        k1 = dt * F(f_vals[p],time_vals[p])
        k2 = dt * F(f_vals[p]+(k1/2),time_vals[p]+(dt/2))
        k3 = dt * F(f_vals[p]+(k2/2),time_vals[p]+(dt/2))
        k4 = dt * F(f_vals[p]+k3,time_vals[p]+dt)
        f_vals[p+1] = f_vals[p] + ( k1 + 2*k2 + 2*k3 + k4)/6.0
    return time_vals, f_vals


# Takes a function and returns its integral. For multiple dimensions, input 
# initial conditions as an array, similar to variables
# interchangebale with rk_4, but have to change a few indicies in the onecycle function
def euler(F,x_not,t0,dt,N):
    time_vals = np.arange(t0,(N+1)*dt + t0,dt)
    # make array, change values later for the sake of time savings
    f_vals = np.array([x_not]*(N+1))
    for p in range(N):
        f_vals[p+1] = dt * F(f_vals[p],time_vals[p])
    return time_vals, f_vals


# integrate the cycle as a closed loop of points with trapedoid integration, assuming first point is at maximum x
def integrate_cycle_trapezoid(xvar,yvar): 
    N = len(yvar)     
    # settign an average change in x makes trapezoidal integration a bit easier and much faster
    minx = np.array([min(xvar),np.where(xvar == min(xvar))[0][0]])
    data_range = max(xvar) - minx[0]
    dX = data_range/N
    # splitting the closed loop into top and bottom to integrate seperately
    xOneSide = np.array( xvar[ : int((minx[1]+1)) ] )
    yOneSide = np.array( yvar[ : int((minx[1]+1)) ] )
    sum1 = [ (xOneSide[x+1] - xOneSide[x]) * yOneSide[x] for x in range(1, (len(xOneSide)-1) )]
    xOtherSide = np.array( xvar[ int((minx[1]+1)) : ] )
    yOtherSide = np.array( yvar[ int((minx[1]+1)) : ] )
    sum2 = [yOtherSide[i]*(xOtherSide[i+1] - xOtherSide[i]) for i in range(1,len(yOtherSide)-1)]
    # the constants added to the end are the only difference between the square box and the 
    # trapeziod method, kaing into account end conditions at the same computation time & error buildup.
    return abs(abs(sum(sum1)+ (0.5 * dX * (yOneSide[-1] - yOneSide[0]))) - abs(sum(sum2)+ 0.5 * dX * (yOtherSide[-1] - yOtherSide[0]))) 
