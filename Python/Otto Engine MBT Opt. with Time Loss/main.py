### Roderick Landreth
### MER-371 Spring 2020
### for Cycle Integration Assignment
### Purpose: Creat PV diagram, find Area (work), plot the results, find a max theta to spark 

import math
#import cmath
import numpy as np
import time
import matplotlib.pyplot as plt
#import random
import integrators
import functions as f

# Does one cycle and outputs pressure vs total displacement of a single cylinder. 
# Note: Am able to change n and a values, but they have defaults indicated by a=x in the function definition.
def oneCycle(N,P1,T1,thetaSpark,thetaDuration,Rcomp,Qh,B,S,L,n=3,a=5):
    V1 = (math.pi*S*B**2)/4
    important_data=int(0.99*N)
    combust_duration=int(thetaDuration*important_data/(2*math.pi)) #number of datapoints evaluating just the combustion
    # angle range for one cycle of a four stroke engine, leaving out intake and exhaust
    angles = np.array(np.arange(0,2*math.pi,2*math.pi/(important_data+1)))                               #explore making this N for sime time loss
    # initial values that will be updated, takes less computation time to make the arrays now
    volumes = np.array([V1]*(N+1))
    pressures = np.array([float(P1)]*(N+1))
    for i in range(important_data+1):
        # always know the volume in this approximation
        # note: can make more accurate by including residuals in volume function
        Vol_previous = volumes[i]
        Vol_current = f.voftheta(angles[i],Rcomp,B,S,L)
        volumes[i+1] = Vol_current
        if angles[i] < thetaSpark:
            # compression Stroke, 1-2, Isentropic relations
            pressures[i+1] = f.isoPressure(pressures[i],Vol_previous,Vol_current)

        elif abs(thetaSpark - angles[i]) < 2*math.pi/important_data:
            # combustion, 2-3
            burn = combustionStroke(combust_duration,pressures[i],thetaSpark,thetaDuration,Qh,Rcomp,B,S,L,n,a)
            pressures[i+1 : int(i + len(burn))+1 ] = burn

        elif angles[i] - thetaDuration - thetaSpark >= 2*math.pi/important_data :#i == start-1 :
            # power stroke, 3-4, isentropicrelations down to V initial
            pressures[i] = f.isoPressure(pressures[i-1],Vol_previous,Vol_current)
    # This is just making a more even set of data, reducing the pressure/temperature in steps.
    # Note: Could be improved if exhaust angle and duration is specified, includes more time loss
    p_difference = pressures[important_data] - P1
    for t in range(int(N)-important_data):
        volumes[important_data+t+1] = volumes[important_data+t]
        pressures[important_data+t+1] = pressures[important_data+t] - p_difference/(N-important_data)
    return pressures, volumes, angles


# integrates over the combustion stroke, seperated for testing. Satisfying to see this work.
def combustionStroke(N,P1,thetaSpark,thetaduration,QtotReleased,compRatio,B,S,L,n=3,a=5):
    heat_addition = f.pressureDerivative(thetaSpark,thetaduration,QtotReleased,compRatio,B,S,L,n,a)
    return integrators.rk_4(heat_addition,P1,thetaSpark,thetaduration/N,N)[1]

# easy plotter for debugging
def plotCoords(xArray,yArray,color="green"):
    plt.figure(figsize=(10,10))
    #plot actual data
    plt.plot(xArray,yArray,color,marker='.',linestyle='none')
    #label plot
    plt.title('Pressure vs. Volume Approximation for Otto Cycle')
    plt.ylabel('Pressure (kPa)')
    plt.xlabel('Volume (m^3)')
    plt.legend(['Data for 1 full turn of the Crank Shaft'])
    plt.show()

# Exhaustively going through anlges to find the highest work value
# only altering spark start angles- 'ssas'
# oh god this is going to be a slow endeavor
def maximize(function,angle_range,othervars):
    ssas_array = np.array(np.arange(*angle_range)) * math.pi/180
    f_vals_ = np.array([0.0]*len(ssas_array))
    sass_best = 0.0
    for i in range(len(ssas_array)):
        f_vals_[i] = function(*othervars,ssas_array[i])
    fmax = max(f_vals_)
    for t in range(len(f_vals_)):
        if f_vals_[t] == fmax:
            sass_best = ssas_array[t]
    #plotCoords(ssas_array,f_vals_)          # Plotting this shows a quadratic looking set with the maximum work at the optimum angle
    return fmax, sass_best*180/math.pi


# formats the information for the maximization function (mainly cause I'm lazy)
def format_cycle(N,T1,P1,r,Qh,B,S,L,sd,ts):
    cycle = oneCycle(N,T1,P1,ts,sd,r,Qh,B,S,L)         
    volume = cycle[1]
    press = cycle[0]
    return integrators.integrate_cycle_trapezoid(volume,press)


# compiles the plots and the values created into a quality informational plot and saces the image in "Out".
# works for up to 3 entries of ignition durations
def data_compiling(N,P1,T1,three_ignition_durations,rComp,Qh,B,S,L):
    cycle_labels = np.array([])
    colors = np.array(["orange","blue","green"])
    plt.figure(figsize=(10,6))
    plt.rcParams.update({'font.size': 12})
    for i in range(len(three_ignition_durations)):
        val_set = [N,T1,P1,rComp,Qh,B,S,L,three_ignition_durations[i]*math.pi/180]
        answer = maximize(format_cycle,[130.0,200.0,1.0],val_set)
        cycle = oneCycle(N,P1,T1,answer[1]*math.pi/180,three_ignition_durations[i]*math.pi/180,rComp,Qh,B,S,L)
        xArray = cycle[1]
        yArray = cycle[0]
        # find and record efficiency, Wnet, Opt spark angle, and MEP in a data label
        cycle_labels = np.append(cycle_labels,['{} deg. duration at {} deg, \n {}% eff. at {}J, and {}kPa MEP'.format(
            three_ignition_durations[i],round(answer[1],1),round(answer[0]/Qh*100,2),round(answer[0],1),round(0.001*answer[0]/(max(xArray)-min(xArray)),1))])
        # find peak pressure and Temp, label that point
        maxPress = max(yArray)
        volMaxPress = xArray[np.where(yArray == maxPress)[0][0]]
        maxTemp = round(f.temp_pvrt(volMaxPress, maxPress, xArray[0],P1,T1),2)
        maxPress = round(maxPress/1000000,2)
        volMaxPress = round(volMaxPress*1000,1)
        # plot cycle and peak pressure point
        plt.annotate("{}MPa, {}K".format(maxPress,maxTemp),[volMaxPress,maxPress*1000])
        plt.plot(xArray*1000,yArray/1000,color=colors[i])
    plt.title('Pressure vs. Volume Approximations for One Cylinder Otto Cycle')
    plt.ylabel('Pressure (kPa)')
    plt.xlabel('Volume (L)')
    plt.legend([*cycle_labels])
    ax = plt.gca()
    plt.text(0.7*ax.get_xlim()[1],0.4*ax.get_ylim()[1],'N = {}'.format(N))
    plt.savefig('Out/NewCycle.png')
    plt.show()


# parameters for a 1979 fors 2.3L I-4, output is W(air standard cycle), not W(break)
# Entering combustion durations, this returns a plot of the cycle and the optimal spark
# start time for the most work output (MBT)
N = int(2*math.pi*10000) #  datapoints
B = 3.78*0.0254     #   m   Bore
S = 3.126*0.0254    #   m   Stroke
I = 5.2*0.0254      #   m   connecting rod length
p_init = 101325     #   Pa  initial pressure
t_init = 300        #   K   initial temperature
three_durations = [40.0,25.0,10.0]  # degrees combustion duration
r_compression = 9   #   unitless compression ratio
heat_in = 2000.0    #   J (from stoichiometric relations and 87(?) octane)
data_compiling(N,p_init,t_init,three_durations,r_compression,heat_in,B,S,I)
