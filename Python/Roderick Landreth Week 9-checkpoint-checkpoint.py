#!/usr/bin/env python
# coding: utf-8

# # Week 9: Linear Algebra

# In[1]:


import math
import cmath
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import scipy.sparse as hungry
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import eigsh


# ## Basic Inversion Functions

# ### QR Decomposition

# <font color = blue>
# Write a function that takes as input a square $N \times B$ matrix $\mathbb{M}$ and outputs a pair of matrices $[\mathbb{Q}, \mathbb{R}]$, where $\mathbb{Q}$ is orthogonal, $\mathbb{R}$ is upper triangular, and $\mathbb{M} = \mathbb{Q}\mathbb{R}$.  Test your function on the matrix
# 
# .
# 
# $$
# \mathbb{M} = \left[\begin{array}{cccc} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 0 & 12 \\ 13 & 14 & 15 & 0 \end{array}\right]
# $$
# 
# .
# 
# Confirm that the function is working by checking all three features of your results: that $\mathbb{Q}$ is orthogonal, that $\mathbb{R}$ is upper triangular, and that $\mathbb{M} = \mathbb{Q}\mathbb{R}$.

# In[2]:


def decomp(matrixM):
    N=int(len(matrixM[0]))
    matrixQ = np.array([np.array([0.0]*N)]*N)
    matrixR = np.array([np.array([0.0]*N)]*N)
    
    #set up the first step
    MjVector = np.array([matrixM[0][i] for i in range(N)])
    #first = np.array([matrixM[0][i] for i in range(N)])
    matrixR[0][0] = np.linalg.norm(MjVector)
    matrixQ[0] = MjVector/matrixR[0][0]
    
    #iterate through steps
    for i in range(N):
        #find current M sub j
        MjVector = np.array([matrixM[j][i] for j in range(N)])
        #create a placeholder value of the next R sub i,i
        nextR = np.array([0.0]*N)
        
        for j in range(i):
            #calculate the previous R values for i < j
            matrixR[j][i] = np.dot(matrixQ[j],MjVector)
            #populate this placeholder with the dot products of the current Q and Mj multiplied by the current Q
            nextR -= np.dot(matrixQ[j],MjVector) * matrixQ[j]
        
        #the placeholder is a vector, so add the first M vector and then use it to create the next R_i,i and Q_i
        nextR += MjVector
        matrixQ[i] = nextR/np.linalg.norm(nextR)
        matrixR[i][i] = np.linalg.norm(nextR)
        
    return matrixQ.transpose() , matrixR
        


# In[3]:


nn = np.array([[1,2],[3,0]])
mmm = np.array([[1,2,3,4],[5,6,7,8],[9,10,0,12],[13,14,15,0]])

this = decomp(nn)
that = decomp(mmm)

#print(np.matmul(this[0], this[1]),"\n","\n",this[1],"\n","\n",np.matmul(this[0],this[0].transpose()),"\n","\n")

print(np.matmul(that[0], that[1]),"\n","\n",that[1],"\n","\n",np.matmul(that[0],that[0].transpose()))


# ### Failure Mode

# <font color = blue>
# Now test your function on the matrix
# 
# .
# 
# $$
# \mathbb{M} = \left[\begin{array}{cccc} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \\ 13 & 14 & 15 & 16 \end{array}\right]
# $$
# 
# .
# 
# What goes wrong?  Explain why the matrix above is unsuitable for QR decomposition

# This is not a tranposable matrix, so the basic solution and what our algorithm effectively does, $x = M^{-1}b$, doesn't work.

# ### Inverting an Upper Triangular Matrix

# <font color = blue>
# Now write a function that solves the linear equation
# 
# $$
# \mathbb{R}\vec{x} = \vec{b}
# $$
# 
# It should take as inputs an upper triangular matrix $\mathbb{R}$ and a vector $\vec{b}$, and output a vector $\vec{x}$.  Test your function by using the upper triangular matrix generated in the first part of this assignment, and the vector
# 
# .
# 
# $$
# \left[\begin{array}{c} 1 \\ 1 \\ 1 \\ 1 \end{array}\right]
# $$
# 
# .
# 
# Find a matrix $\mathbb{R}$ for which your function will not work, and explain what goes wrong.  Is there a choice of vector $\vec{b}$ that will create problems?

# In[4]:


def invertR(R,b):
    if abs(np.linalg.det(R)) < 10**-10:
        print('Error: R is non-invertable')
        return
    return np.matmul(np.linalg.inv(R),b)


# In[5]:


b=np.array([[1.0],[1.0],[1.0],[1.0]])

R_bad = np.array([[1., 0., 0.,  0.],
                 [ 0.,  1., 0., 0.],
                 [ 0.,  0., 0., 0.],
                 [ 0.,  0., 0., 1.]])

print(invertR(that[1],b))
print('\n',np.linalg.det(R_bad),invertR(R_bad,b))


# There be daemons if the R vector is not invertable, but concerning the b vector, as long as it's dimensions are correct then there should be no problem.

# ### Matrix Inverter

# <font color = blue>
# Now use the pieces you have created to solve the general problem
# 
# $$
# \mathbb{M}\vec{x} = \vec{b}
# $$
# 
# by first decomposing $\mathbb{M} = \mathbb{Q}\mathbb{R}$, and then solving
# 
# $$
# \mathbb{R}\vec{x} = \mathbb{Q}^{T}\vec{b}
# $$
# 
# Your function should take as inputs an $N \times N$ matrix $\mathbb{M}$, and an $N$-component vector $\vec{b}$.  Have your function print an error message and return a non-sensical result in the even that $\mathbb{M}$ is not square, that $\vec{b}$ does not have the correct number of components, or that $\mathbb{M}$ is not invertible.
# 
# Check your code using the matrices and vector we used earlier.

# In[6]:


def invertQ(M,b):
    N=len(b)
    #check if M is square matrix
    if len(M[0]) != len(M[1]):
        print("Error: M Matrix Not Square")
        return
    #check square matrix vs size of b vector
    elif len(M[0]) != N:
        print("Error: B matrix does not match M matrix")
        return
    #if the determinant is effectively 0, then it is not invertable
    elif abs(np.linalg.det(M)) < 10**(-10):
        print("Error: M matrix is not Invertable")
        return
    else:
        Q , R = decomp(M)
        return np.matmul(Q.transpose(),b)


# In[7]:


matrixM = np.array([[1,2,3,4],[5,6,7,8],[9,10,0,12],[13,14,15,0]])
b=np.array([[1.0],[1.0],[1.0],[1.0]])
breakDown = decomp(matrixM)
Q = breakDown[0]
R = breakDown[1]

#short cheety way
print(np.matmul( np.linalg.inv(matrixM), b) ,'\n')

#Algorithmic Method
Qb = invertQ(matrixM,b)
#print(Qb)
x = invertR(R,Qb)
print(x)


# For all in tents and for poses, These produce the same resule. Yay!

# ## Circuit Analysis

# ### * Circuit 1

# In[8]:


from IPython.display import IFrame, display
filepath = "http://beesbeesbees.com" # works with websites too!
#filepath = "circuit.pdf"
IFrame(filepath, width=100, height=50)


# <font color = blue>
# Use your matrix inverter to find the currents in the circuit above, analyzed in class.  Choose at least four different sets of values for the resistors and the input voltage, and discuss whether or not the results make physical sense.

# Equations:
#     $$ i_1 - i_2 - i_3 - i_4 = 0 $$\
#     $$ i_2 + i_3 + i_4 - i_5 = 0 $$\
#     $$ i_1R_1 + i_2R_2 + i_3R_5 = V_0 $$\
#     $$ i_1R_1 + i_2R_3 + i_3R_5 = V_0 $$\
#     $$ i_1R_1 + i_2R_4 + i_3R_5 = V_0 $$
#     

# In[19]:


#setup: constants and initial matricies
#Case 1: Equivalent circut including Vo, R1, R2, R5, adn ground
R1 = 100. #ohms
R2 = 100. #ohms
R3 = 100000000. #ohms
R4 = 100000000. #ohms
R5 = 100. #ohms
Vo = 900. #volts

matrixCurcuit1 = np.array([[1.,-1.,-1.,-1.,0.],
                           [0.,1.,1.,1.,-1.],
                           [R1,R2,0.,0.,R5],
                           [R1,0.,R3,0.,R5],
                           [R1,0.,0.,R4,R5]])
b=np.array([[0.0],[0.0],[Vo],[Vo],[Vo]])
cirOne = decomp(matrixCurcuit1)
Q = cirOne[0]
R = cirOne[1]


# In[20]:


#Algorithmic Method, produces matrtix containing values of current 1 thorugh 5
Qb = invertQ(matrixCurcuit1,b)
#print(Qb)
x = invertR(R,Qb)
print(x)


# This is testing the circuit analysis by increacing resistances of resitors 3 and 4 to make the circuit effectively one resistor connected between ground and the input voltage, with equivalent resistance equal to the sum of resistors 1, 2, and 5. This would mean through each of these resistors the current it eual to $\frac{V_o}{R_{1+2+5}} = \frac{900}{100 + 100 + 100} = 3$, as is seen in the reedout. This case is correct and makes physical sense.

# In[21]:


#Case 2:
R1 = 100. #ohms
R2 = 100000000. #ohms
R3 = 100. #ohms
R4 = 100000000. #ohms
R5 = 100. #ohms
Vo = 9. #volts

matrixCurcuit1 = np.array([[1.,-1.,-1.,-1.,0.],
                           [0.,1.,1.,1.,-1.],
                           [R1,R2,0.,0.,R5],
                           [R1,0.,R3,0.,R5],
                           [R1,0.,0.,R4,R5]])
b=np.array([[0.],[0.0],[0.0],[Vo],[Vo]])
cirOne = decomp(matrixCurcuit1)
Q = cirOne[0]
R = cirOne[1]
#Algorithmic Method, produces matrtix containing values of current 1 thorugh 5
Qb = invertQ(matrixCurcuit1,b)
#print(Qb)
x = invertR(R,Qb)
print(x)


# This is testing the circuit analysis by increacing resistances of resitors 2 and 4 to make the circuit effectively one resistor connected between ground and the input voltage, and shows that like the previous test, making the circuit simpler diplays more readily that the solutions agree with the physical equivalent.

# In[25]:


#Case 3: 
R1 = 100. #ohms
R2 = 300. #ohms
R3 = 300. #ohms
R4 = 300. #ohms
R5 = 100. #ohms
Vo = 900. #volts

matrixCurcuit1 = np.array([[1.,-1.,-1.,-1.,0.],
                           [0.,1.,1.,1.,-1.],
                           [R1,R2,0.,0.,R5],
                           [R1,0.,R3,0.,R5],
                           [R1,0.,0.,R4,R5]])
b=np.array([[0.],[0.0],[Vo],[Vo],[Vo]])
cirOne = decomp(matrixCurcuit1)
Q = cirOne[0]
R = cirOne[1]
#Algorithmic Method, produces matrtix containing values of current 1 thorugh 5
Qb = invertQ(matrixCurcuit1,b)
#print(Qb)
x = invertR(R,Qb)
print(x)


# This case is not simplifying the circuit at all, and uses equivalent resistance relations to predict that with three resistors in parallel, each with resistance 300ohms, their equivalent resistance is 100, and so like the previous circuits, the total circuit can be simplified into a voltage source, a 300ohm resistor, and ground.

# In[30]:


#Case 4:
R1 = 100. #ohms
R2 = 300. #ohms
R3 = 400. #ohms
R4 = 300. #ohms
R5 = 100. #ohms
Vo = 900. #volts

matrixCurcuit1 = np.array([[1.,-1.,-1.,-1.,0.],
                           [0.,1.,1.,1.,-1.],
                           [R1,R2,0.,0.,R5],
                           [R1,0.,R3,0.,R5],
                           [R1,0.,0.,R4,R5]])
b=np.array([[0.],[0.0],[Vo],[Vo],[Vo]])
cirOne = decomp(matrixCurcuit1)
Q = cirOne[0]
R = cirOne[1]
#Algorithmic Method, produces matrtix containing values of current 1 thorugh 5
Qb = invertQ(matrixCurcuit1,b)
#print(Qb)
x = invertR(R,Qb)
print(x)


# This circuit changes the value of one of the center resistances, and should show an increase in current flow through the two other resistors relative tot eh last case. This makes physical sense, as more resistance in one plane means the current will tend more towards a path with less resistance if thats available, which in thic case it is. These cases show that the ideal physical system matches the analytical model.

# ### * Circuit 2

# <font color = blue>
# Now analyze the circuit below in a similar manner.  This time, you will have to convert the problem into matrix form yourself.  (Include your work on that in the notebook.)

# In[14]:


from IPython.display import IFrame, display
filepath = "https://www.tylervigen.com/spurious-correlations" # Just a fun website for you to paruse
#filepath = "circuit2.pdf"
IFrame(filepath, width=700, height=400)


# Using the second example circuit found in lecture notes:
# Equations: $$ i_2 - i_4 - i_7 = 0 $$\
#             $$ i_4 - i_6 - i_8 = 0 $$\
#             $$ i_1 - i_3 - i_7 = 0 $$\
#             $$ i_3 - i_5 - i_8 = 0 $$\
#             $$ i_7R_7 + i_4R_4 + i_8R_8 + i_3R_3 = 0 $$\
#             $$ i_8R_8 + i_6R_6 + i_5R_5 = 0 $$\
#             $$ i_1R_1 + i_3R_3 + i_5R_5 = V_1 $$\
#             $$ i_2R_2 + i_7R_7 + i_1R_1 = V_2-V_1 $$
#     

# In[15]:


#setup: constants and initial matricies
#Case 1: Reducing Circuit to V1, R1, R3, R5 and Ground
R1 = 100. #ohms
R2 = 100000000. #ohms
R3 = 100. #ohms
R4 = 100000000. #ohms
R5 = 100. #ohms
R6 = 100000000. #ohms
R7 = 100. #ohms
R8 = 100. #ohms
V1 = 9. #volts
V2 = 9. #volts

matrixCircuit2 = np.array([[0.,1.,0.,-1.,0.,0.,-1.,0.],
                    [0.,0.,0.,1.,0.,-1.,0.,-1.],
                    [1.,0.,-1.,0.,0.,0.,-1.,0.],
                    [0.,0.,1.,0.,-1.,0.,0.,-1.],
                    [0.,0.,R3,R4,0.,0.,R7,R8],
                    [0.,0.,0.,0.,R5,R6,0.,R8],
                    [R1,0.,R3,0.,R5,0.,0.,0.],
                    [R1,R2,0.,0.,0.,0.,R7,0.]])
b=np.array([[0.],[0.],[0.],[0.],[0.],[0.],[V1],[V2-V1]])
cirOne = decomp(matrixCircuit2)
Q = cirOne[0]
R = cirOne[1]


# In[16]:


#Algorithmic Method, produces matrtix containing values of current 1 thorugh 8 
Qb = invertQ(matrixCircuit2,b)
#print(Qb)
x = invertR(R,Qb)
print(x)


# In this case, the circuit can be reduced to the sum of R1, 2, and 3 experiencing V1. So, $i_{1,3,and\; 5} = \frac{V_1}{the \; sum \;  of \; R1, 3, 5} = \frac{9}{300} = 0.03$, as is seen at each of the specified values of current. 

# In[17]:


#Case 2: Reducing Circuit to V2, R2, R4, R6 and Ground
R1 = 100000000. #ohms
R2 = 100. #ohms
R3 = 100000000. #ohms
R4 = 100. #ohms
R5 = 100000000. #ohms
R6 = 100. #ohms
R7 = 100. #ohms
R8 = 100. #ohms
V1 = 9. #volts
V2 = 9. #volts

matrixCircuit2 = np.array([[0.,1.,0.,-1.,0.,0.,-1.,0.],
                    [0.,0.,0.,1.,0.,-1.,0.,-1.],
                    [1.,0.,-1.,0.,0.,0.,-1.,0.],
                    [0.,0.,1.,0.,-1.,0.,0.,-1.],
                    [0.,0.,R3,R4,0.,0.,R7,R8],
                    [0.,0.,0.,0.,R5,R6,0.,R8],
                    [R1,0.,R3,0.,R5,0.,0.,0.],
                    [R1,R2,0.,0.,0.,0.,R7,0.]])
b=np.array([[0.],[0.],[0.],[0.],[0.],[0.],[V1],[V2-V1]])
cirOne = decomp(matrixCircuit2)
Q = cirOne[0]
R = cirOne[1]
#Algorithmic Method, produces matrtix containing values of current 1 thorugh 8 
Qb = invertQ(matrixCircuit2,b)
#print(Qb)
x = invertR(R,Qb)
print(x)


# Again, this is reducing the circuit to a path from $V_2$ this time to ground, through R2, R4, and R6. Again the value of 0.03 amps shoudl be found in $i_{2,4,6}$ as is produced in this model. Both of these sanple cases show sensible reactions, and the values of current in resistors I'm not interested in behave as expected, being small with respect to the high resistance of their resistors. 

# In[31]:


#Case 3: Reducing Circuit to V2, R2, R7, R1 and V1 as Ground
R1 = 100. #ohms
R2 = 100. #ohms
R3 = 100000000. #ohms
R4 = 100000000. #ohms
R5 = 100. #ohms
R6 = 100. #ohms
R7 = 100. #ohms
R8 = 100. #ohms

V1 = 91. #volts
V2 = 100. #volts

matrixCircuit2 = np.array([[0.,1.,0.,-1.,0.,0.,-1.,0.],
                    [0.,0.,0.,1.,0.,-1.,0.,-1.],
                    [1.,0.,-1.,0.,0.,0.,-1.,0.],
                    [0.,0.,1.,0.,-1.,0.,0.,-1.],
                    [0.,0.,R3,R4,0.,0.,R7,R8],
                    [0.,0.,0.,0.,R5,R6,0.,R8],
                    [R1,0.,R3,0.,R5,0.,0.,0.],
                    [R1,R2,0.,0.,0.,0.,R7,0.]])
b=np.array([[0.],[0.],[0.],[0.],[0.],[0.],[V1],[V2-V1]])
cirOne = decomp(matrixCircuit2)
Q = cirOne[0]
R = cirOne[1]
#Algorithmic Method, produces matrtix containing values of current 1 thorugh 8 
Qb = invertQ(matrixCircuit2,b)
#print(Qb)

x = invertR(R,Qb)
print(x)
R1 = 100. #ohms
R2 = 100. #ohms
R3 = 100000000. #ohms
R4 = 100000000. #ohms
R5 = 100. #ohms
R6 = 100. #ohms
R7 = 100. #ohms
R8 = 100. #ohms

V1 = 0. #volts
V2 = 9. #volts

matrixCircuit2 = np.array([[0.,1.,0.,-1.,0.,0.,-1.,0.],
                    [0.,0.,0.,1.,0.,-1.,0.,-1.],
                    [1.,0.,-1.,0.,0.,0.,-1.,0.],
                    [0.,0.,1.,0.,-1.,0.,0.,-1.],
                    [0.,0.,R3,R4,0.,0.,R7,R8],
                    [0.,0.,0.,0.,R5,R6,0.,R8],
                    [R1,0.,R3,0.,R5,0.,0.,0.],
                    [R1,R2,0.,0.,0.,0.,R7,0.]])
b=np.array([[0.],[0.],[0.],[0.],[0.],[0.],[V1],[V2-V1]])
cirOne = decomp(matrixCircuit2)
Q = cirOne[0]
R = cirOne[1]
#Algorithmic Method, produces matrtix containing values of current 1 thorugh 8 
Qb = invertQ(matrixCircuit2,b)
#print(Qb)
x = invertR(R,Qb)
print("\n",x)


# This final case illustrates the interaction between voltage sources, focussing ont eh loop from V2 through resistors R2m R7, and R1, with the voltage difference between V2 and V1 making V1 act as ground. This is equivalent to V2 being 9 and V1 being ground, as seen in the comparison from the first output to the second. This is confirmed by the equivalent values of resistors not including R3,R4,R5,R6 and R8, effictively removing the actual ground from the circuit. Currents through R1,R2, and R7 are the same in both instances, as expected- this gives me faith that this system is an accurate/ideal representation of this circuit

# ## Final Project Work

# <font color = blue>
# Turn in your "final project" notebook as well, updated to include your work from this week.
# 
# At the end, include a brief discussion of what you *intended* to accomplish as compared with what you *did* accomplish, as well as a plan for the rest of the project.

# In[ ]:




