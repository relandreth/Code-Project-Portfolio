#!/usr/bin/env python
# coding: utf-8

# # Week 8: Fourier Transforms

# ## Library Imports go here

# In[117]:


import math
import cmath
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import scipy.sparse as hungry
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import eigsh
from scipy.optimize import curve_fit as fit


# ## A Basic Fourier Transform Function

# ### The Code

# <font color = blue>
# Construct a function that will implement the basic Fourier transform of a discrete data set:
# 
# $$
# P_k = \sum_{j = 0}^{N-1} p_je^{2\pi ijk/N}, \hspace{.75in} k = 0, 1, 2, \dots, N-1
# $$
# 
# Your function should take as input a list of discrete values $[p_0, p_1, \dots, p_{N-1}]$ and a time step $\Delta t$.  It should output both a list of frequencies $[f_k]$ and a list of transform values $[P_k]$, where the frequency is given by
# 
# $$
# f_k = \left\{\begin{array}{rcl} k\Delta f = \frac{k}{N\Delta t} & \mbox{for} & k = 0, 1, 2, \dots, \frac{N}{2} \\ \\ -(N-k)\Delta f = \frac{k - N}{N\Delta t} & \mbox{for} & k = \frac{N}{2} + 1, \dots, N-1 \end{array}\right.
# $$
# 
# This assumes $N$ is an even integer.  If the input list has an odd number of elements, have your code simply discard the last element of the input.

# In[118]:


def fiveier(p_lst,time_step):
    N = len(p_lst)
    delta_f = 1/(N*time_step)
    #making the dataset have even entries
    if N % 2 ==1:
        N-=1
    #setting up the arrays
    freq = np.array([0.0]*N)
    transform_vals = np.array([0.0]*N)
    #calculating the frequency values and the next transform entry
    for k in range(N-1):
        if k <= N/2:
            freq[k] = k * delta_f
        else:
            freq[k] = (k - N) * delta_f
        transform_vals[k] = sum([p_lst[t]*cmath.exp(complex(0,2*math.pi*t*k/N)) for t in range(N)])
    
    return freq,transform_vals


# ### A Simple Test

# <font color = blue>
# To test out your function, use
# 
# $$
# p(t) = \cos 40\pi t
# $$
# 
# .
# 
# Start by creating a list of discrete values from this function, working with the time interval $[0, 10]$, and the time step $\Delta t = 0.01$.  Input this into your Fourier function, and plot the results.
# 
# Confirm that the peaks of the Fourier transform appear where they should.  What sets the height of the peaks?  Are they correct?

# In[119]:


discrete = np.array([math.cos(40*math.pi*t) for t in np.arange(0,10,0.01)],dtype=float)
data = fiveier(discrete,0.01)
#print(data)


# In[120]:


plt.plot(*data)
print(sum(data[1]))


# The peaks do appear at the correct points, as the base form of $p(t)$ is $aCos(2\pi f t)$, and in this $p(t)$, the f is 20. The peaks should appear at $\pm f$, as they do above. As for the peaks, the area of each should be $\frac{1}{df}$, which is $\frac{0.2}{2}*500$. Hmmmm. 

# ### Timing

# <font color = blue>
# The Fourier transform function as we have created it should require a time to compute that grows *quadratically* with $N$.  Why is that? Construct a function that takes as input the value $N$ and outputs the time required to find the Fourier transform of the function $p(t) = \cos 40\pi t$ on the interval $[0, 10]$, with $\Delta t = \frac{T}{N}$.  Use this function to create lists of the form $[N]$ and $[T_N]$, for $N = 50, 100, 150, 200, \dots, 1000$.  Confirm the quadratic behavior.

# In[121]:


def timing(N):
    #setup sample data
    dt=10/N
    timeRange = np.arange(0,10,dt)
    discrete_vals = np.array([math.cos(40*math.pi*t) for t in timeRange],dtype=float)
    #take time
    timeOne = time.time()
    #calculate data
    duration = fiveier(discrete_vals,dt)
    #take time again, and return the time differences.
    timeTwo = time.time()
    return timeTwo - timeOne


# In[122]:


N_range = np.array(range(50,1000,50))
data = np.array([ timing(n) for n in N_range])


# In[124]:


#Polynomial fit curve
def curve(x,a,b):
    return a*x**2.0 + b*x

fig, ax = plt.subplots(1,1,figsize=(14, 8))
#fit the data tot eh olynomial fit curve
popt, pcov = fit(curve,N_range,data)
#plot actual data
plt.plot(N_range,data)
#plot the fit function, change its appreaange from the other curve
ax.plot(N_range,np.array([curve(n,*popt) for n in N_range]),color='red',marker='.',linestyle='none')
#label plot
plt.title('Computation time for fourrier transform of X elements')
plt.ylabel('Seconds')
plt.xlabel('Length of Fourrier Transform Array')
plt.legend(['Collected Data','Polynomial Fit Curve'])

plt.show()


# ## * Another Look at the Relativistic Harmonic Oscillator

# <font color = blue>
# The fact that the relativistic harmonic oscillator has sinusoidal oscillations for small amplitudes, but roughly triangular oscillations for large amplitudes makes it particularly interesting to examine using a Fourier transform.  Recall that the relativistic oscillator satisfies the differential equation (after appropriate non-dimensionalization)
# 
# $$
# \ddot{x} = -x(1 - \dot{x}^2)^{3/2}
# $$
# 
# Use either RK4 or Verlet (as you did originally) to generate solutions to this equation, with 1000 time steps of size $\Delta t = 0.1$, where the object is released from rest with initial positions {0.1, 0.5, 1.0, and 5.0}.  Take the Fourier transform of each solution, and create a separate plot of the power spectrum for each solution.  (It may be best to use a logarithmic scale so as to see peaks of very different sizes clearly.)  Spend some time augmenting the plots: for the smaller amplitude cases show where the non-relativistic frequency peaks would be, and what heights they would be.  For the larger amplitude cases (where comparison with the SHO is less useful), label where the peaks actually are.  Discuss the results.

# In[8]:


def RK_four(F,x_not,dt,N):
    time_vals = np.arange(0,(N+1)*dt,dt)
    #make array, change values later for the sake of time savings
    f_vals = np.array([x_not]*(N+1))
    for i in range(N):
        k1 = dt * F(f_vals[i],time_vals[i])
        k2 = dt * F(f_vals[i]+(k1/2),time_vals[i]+(dt/2))
        k3 = dt * F(f_vals[i]+(k2/2),time_vals[i]+(dt/2))
        k4 = dt * F(f_vals[i]+k3,time_vals[i]+dt)
        f_vals[i+1] = f_vals[i] + ( k1 + 2*k2 + 2*k3 + k4)/6
    return time_vals, f_vals


# In[9]:


N=1000
dt=0.1

#takes a 1x2 array as f, containing x and vx, return 1x2 array constining vx and ax
def F(f,t):
    return np.array([f[1], - f[0]*(1-f[1]**2)**(3/2)])

#four datasets, each with different initial conditions
data1 = np.array([v[0] for v in RK_four(F,np.array([0.1,0.0]),dt,N)[1]])
data2 = np.array([v[0] for v in RK_four(F,np.array([0.5,0.0]),dt,N)[1]])
data3 = np.array([v[0] for v in RK_four(F,np.array([1.0,0.0]),dt,N)[1]])
data4 = np.array([v[0] for v in RK_four(F,np.array([5.0,0.0]),dt,N)[1]])
times = RK_four(F,np.array([0.1,0.0]),dt,N)[0]

fig, ax = plt.subplots(1,1,figsize=(14, 8))
plt.plot(times,data1)
plt.plot(times,data2)
plt.plot(times,data3)
plt.plot(times,data4)
plt.legend(['IC of x = 0.1','IC of x = 0.5','IC of x = 1.0','IC of x = 5.0'])
plt.title('Function Plots of Different Initial Conditions')
plt.show()


# In[10]:


data1f = fiveier(data1,dt)
data2f = fiveier(data2,dt)
data3f = fiveier(data3,dt)
data4f = fiveier(data4,dt)


# In[128]:


fig, ax = plt.subplots(1,1,figsize=(14, 8))
plt.semilogy(*data1f)
plt.legend(['IC of x = 0.1'])
plt.title('Function Plots of Different Initial Conditions')
plt.show()


# In[131]:


fig, ax = plt.subplots(1,1,figsize=(14, 8))
plt.semilogy(*data2f)
plt.legend(['IC of x = 0.5'])
plt.title('Function Plots of Different Initial Conditions')
plt.show()


# In[132]:


fig, ax = plt.subplots(1,1,figsize=(14, 8))
plt.semilogy(*data3f)
plt.legend(['IC of x = 1.0'])
plt.title('Function Plots of Different Initial Conditions')
plt.show()


# In[133]:


fig, ax = plt.subplots(1,1,figsize=(14, 8))
plt.semilogy(*data4f)
plt.legend(['IC of x = 5.0'])
plt.title('Function Plots of Different Initial Conditions')
plt.show()


# The higher the initial condition, the more Harmonics are seen because the function is more relativistic and closer to a triangle wave! Neat!

# ## The FFT Functions

# ### The FFT

# <font color = blue>
# Write a function that implements the FFT algorithm.  It should take as its input a list of positions $[p_j]$, which it should assume has length $2^n$ for some integer $n$, and it should output a list $[P_k]$.  (Because of its recursive nature, it's better to leave this one very simple in terms of the input and output formats.)

# In[12]:


#assumes that len(p_lst) is of length 2**q
def recursiveBad(p_lst):
    N=len(p_lst)
    omega = cmath.exp(complex(0,2*math.pi/N))
    Finput = np.array([0.0]*N,dtype=complex)
    evens = np.array([0.0]*(int(N/2)),dtype=complex)
    odds = np.array([0.0]*(int(N/2)),dtype=complex)
    
    if N == 1:
        #test if length 1
        return p_lst
    else:
        #split evens and odds
        for i in range(int(N/2)):
            evens[i] = p_lst[2*i]
            odds[i] = p_lst[(2*i) +1]
        #print(evens,odds)
        fevens = recursiveBad(evens) #where is recursive step? Probably here.
        fodds = recursiveBad(odds)
        #reassembles after magic
        for k in range(int(N/2)):
            Finput[k] = fevens[k] + omega**k*fodds[k]
            Finput[int(N/2 + k)] = fevens[k] - omega**k*fodds[k]
        return Finput
    


# In[13]:


lst = [8,4,5,6]

recursiveBad(lst)


# ### Fourier Wrapper

# <font color = blue>
# Now write a function structured the same way your first Fourier code (from last time) was set up: it should take as input the list $[p_k]$ and time step $\Delta t$, automatically truncate the input list until it has $2^n$ elements for some integer $n$, then feed that into your FFT code, then generate two lists as output: a list of frequencies $[f_k]$ and the list $[P_k]$.
# 
# Test your function on the same simple cosine data you used last time, to be sure it is working properly.

# In[134]:


def fffttt(p_lst,time_step):
    #setting up constants
    N=len(p_lst)
    p_lst = p_lst[0:2**(math.floor(math.log(N,2)))]
    N=len(p_lst)
    delta_f = 1/(N*time_step)
    freq = np.array([0.0]*N)
    
    bigP = recursiveBad(p_lst)
    #still need to find frequencies, just don't need that sum in there every time
    for k in range(N-1):
        if k <= N/2:
            freq[k] = k * delta_f
        else:
            freq[k] = (k - N) * delta_f
    return freq,bigP
    


# In[15]:


plt.plot(*fffttt(discrete,0.01))


# ### Power Spectrum

# <font color = blue>
# It's  also handy to have a module that works the same way the previous one did, but outputs the power spectrum instead of the Fourier transform.  Create a variation on the module you just wrote that does so.

# In[135]:


def ffftttPOWER(p_lst,time_step):
    data = fffttt(p_lst,time_step)
    #the square of the absolute value of each datapoint
    for i in range(len(data[1])):
         data[1][i] = abs(data[1][i])**2
    return data 


# ## Timing of the FFT

# <font color = blue>
# Create a function that takes as input an integer $N$ and outputs the time required to find the Fourier transform (using the FFT) of the function $p(t) = \cos 40\pi t$ on the interval $[0, 10]$ with $\Delta t = T/N$.  Use this function to create lists whose elements are of the form $[N]$ and $[T_N]$, with $N = 2, 4, 8, 16, 32, \dots, 2^{16}$.  Plot the results, and compare them with the timing of the original discrete Fourier transform code.  

# In[136]:


def timing2(N):
    #setup data
    dt=10/N
    time_vals = np.arange(0,10,dt)
    discrete_vals = np.array([math.cos(40*math.pi*t) for t in time_vals],dtype=float)
    #record time 1
    firstTime = time.time()
    duration = fffttt(discrete_vals,dt)
    #record time 2 after data is calculated, and return teh time stamp's difference
    secdondTime = time.time()
    return secdondTime - firstTime


# In[20]:


#Checkin the data
N_Range2 = np.array([2**n for n in range(1,17)])
log_time_data = np.array([ timing2(n) for n in N_Range2])
plt.plot(N_Range2,stuff)


# In[78]:


def log_function(K,H):
    return H*K#*math.log10(K) does not accept log, linear is close in shape to nlog(n) in this range

fig, ax = plt.subplots(1,1,figsize=(14, 8))
#Using the last fit compared to the new fit line
ax.plot(N_range,np.array([curve(n,*popt) for n in N_range]),color='red',marker='.',linestyle='none')

#plot new data
plt.plot(N_Range2,log_time_data)
#new fit
popt2, pcov2 = fit(log_function,N_Range2,log_time_data)
#new fit line
ax.plot(N_Range2,np.array([log_function(n,*popt2) for n in N_Range2]),color='green',marker='.',linestyle='none')
#labeling the plot
plt.title('Computation time for fourrier transform of X elements')
plt.ylabel('Seconds')
plt.xlabel('Length of Fourrier Transform Array')
plt.legend(['Slow Transform Method Curve','Collected Data','n*log(n) Fit Curve'])

plt.show()


# The red is the other method, and the blue (fit to green) is the FFT. Thats a remarkable difference, in my mind the difference between $n^2$ and n*log(n) seemed large, but the magnitude of the time difference is nonintuitive.

# ## Data Analysis: Pulse Periodicity

# <font color = blue>
# Consider the (artificially created) data set stored on Nexus titled "pulse.csv."  It represents a possible data set of the some signal that pulses at regular intervals.  These pulses are not completely identical, and riding on top of the signal is a significant amount of noise, such that without the tools of Fourier transform it might be impossible to tell that there is any sort of regularity.  With these tools, on the other hand, it is possible to determine with significant accuracy the periodicity of the small, regular signal hiding in all of the noise.

# ### Original Data

# <font color = blue>
# Begin by importing in the original data set, converting it to the proper format (it is originally a list of ordered pairs $[t, p]$), and plotting it.  Try various time ranges.  Make an initial guess as to what you think the period of the pulse signal is.  (If it appears to be impossible to tell, don't worry, that's the point.)  

# In[43]:


import pandas as pd

#You'll have to change the path name
df = pd.read_csv(r"C:\Users\relan\OneDrive\4 Senior year\Computational Physics\pulse.csv")

#making the data plottable, cuts off the first value
realish_time_data = np.array(df['0.1'].values)
realish_data = np.array(df['0.06838987216926336'].values)

#plotting the data
fig, ax = plt.subplots(1,1,figsize=(18, 8))
plt.plot(realish_time_data,realish_data)
plt.show()


# This looks like it hast no period I'm aware of, possibly a peak at ~20 and 820 seconds, a trough in between? Some good visually nebulous data. 

# ### Pre-Processing the Data

# #### Determining the Time Step

# <font color = blue>
# To perform the analysis, it is necessary to work out what the time step $\Delta t$ of the original data is.  Do so

# From the data file, the time step is 0.1 seconds. I'm not sure how else ytou would have liked me doing this, it could have been through a loop checking indicies, or a thing going through the time array above, but I wasn't sure what the difference was if I were only using this file and this data.

# #### Subtracting an Offset

# <font color = blue>
# You should notice that the signal provided is always positive.  If we are thinking of this data as representing oscillations, they are oscilations around some positive offset, rather than around zero.  Why does an offset present a problem for Fourier analysis?  (What would happen if you looked at the Fourier transofrm of a constant?)  Find a way to subtract this offset off of your data, so that it is centered around zero.

# The fourrier transform of a single constant is tjat constant, as used in the algorithm after breaking the list into the smallest even and odd lists possible. 
# 
# What I did was subtract the average from the dataset. 

# In[44]:


centered = realish_data - sum(realish_data)/len(realish_data)
plt.plot(realish_time_data,centered)


# ### * Power Spectrum

# <font color = blue>
# Now use the function you created earlier to generate the power spectrum of the data.  Plot the results, and use *this* to work out what the period of th pulsing signal was.

# In[45]:


#making the data
power_series = ffftttPOWER(centered,0.1)
#Dont need the below cause the function does that, too
#realish_time_data = realish_time_data[0:2**(math.floor(math.log(len(realish_time_data),2)))]

fig, ax = plt.subplots(1,1,figsize=(18, 8))
plt.plot(*power_series)
plt.show()


# In[46]:


#sectioning out the pulse frequewncy data (the slow way)
pulse = [0]
freq = [0]
for n in range(len(power_series[0])):
    if power_series[1][n] > 300:
        pulse.append(power_series[1][n])
        freq.append(power_series[0][n])


# In[137]:


fig, ax = plt.subplots(1,1,figsize=(10, 8))
plt.plot(freq,pulse,marker='o',linestyle='none')
plt.title('Highest nodes in Dataset Power Series, Corresponding to Signal Frequencies')
plt.show()

print(math.sqrt(freq[pulse.index(max(pulse))]))


# The most significant data here is signified by the middle two datapoints, pretaining to a frequency ($f_0$) of 0.29/s, a period of 3.45 (s). This corresponds to a function similar to $aCos(2 \pi f_0 t + \phi)$, though we have lost data about the phase shift. The other higher values also change what the curve would look like, so a plot could be constructed of the highest several datapoints without phase shift, meaning the origional function would be approximated little accuracy. 
# 
# Or, the inverse fourrier transform can be taken of the dataset.

# In[48]:


def inverse_fourrier(p_lst,time_step):
    N = len(p_lst)
    delta_f = 1/(N*time_step)
    if N % 2 ==1:
        N-=1
    freq = np.array([0.0]*N)
    transform_vals = np.array([0.0]*N)
    for k in range(N-1):
        #if k <= N/2:
        #    freq[k] = k * delta_f
        #else:
        #    freq[k] = (k - N) * delta_f
        transform_vals[k] = sum([p_lst[t]*cmath.exp(complex(0,-2*math.pi*t*k/N)) for t in range(N)])
    timelst = np.arange(0,len(p_lst),time_step)
    return timelst, transform_vals


# In[70]:


p_of_t = inverse_fourrier(power_series[1],0.1)


# In[138]:


time = np.arange(0,0.1*len(p_of_t[1]),0.1)

fig, ax = plt.subplots(1,1,figsize=(18, 8))
plt.plot(time[1:],p_of_t[1][1:])
#labeling
plt.title('Origional Function, Inverse Fourrier Transform')
plt.ylabel('Signal Strength')
plt.xlabel('Time (s)')
#add vertical lines to match up with signal peaks
plt.axvline(24,-10000,10000,color='purple')
plt.axvline(36,-10000,10000,color='purple')
plt.axvline(48,-10000,10000,color='purple')
plt.show


# I believe there is a faster way to do this, possibly using the FFT and just altering the recombination laws, as the inverse fourrier transform of a single element list is that element, same as a forwards transform.
# 
# This does have a signal period, visible to the eye, of about 10 or 12 seconds. Its amazing how different the before and after the information is completely hidden visually. The dataset starts out with a very large signal at t=0, but I cut it off to better see the rest. I have also seen this in signal interpretation using laplace transforms, a direc delta at t=0, though I forget the reasoning for this, I'll see if I can find it for class to ask and see if its similar resoning.

# This week I'll have both last week's benchmark for final project and this week's in there, labeled accordingly. I'm really just submitting different jupyter checkpoints, but I apologize for the delay.

# In[ ]:




