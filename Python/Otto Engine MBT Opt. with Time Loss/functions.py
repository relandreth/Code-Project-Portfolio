### Roderick Landreth
### MER-371 Spring 2020
### for Cycle Integration  Assignment
### Purpose: Funtion definitions

import math

# Gas Constants (AT STP, can have these be functions to improve model):
# note: can be improved by making this module a class and allow different gasses
Cv = 718
Cp = 1005
k = Cp / Cv
R = Cp - Cv

# Heat Release (for r between 8 and 10)
def qoftheta(theta,thetaSpark,thetaduration,QtotReleased,n=3,a=5):
    return QtotReleased*( 1 - math.exp(-a*((theta-thetaSpark)/thetaduration)**n) )


# Heat Release Rate with respect to Crank Angle (for r between 8 and 10)
def qdtheta(theta,thetaSpark,thetaduration,QtotReleased,n=3,a=5):
    return QtotReleased * n * a * (1 - qoftheta(theta,thetaSpark,thetaduration,QtotReleased)/QtotReleased) * (((theta-thetaSpark)/thetaduration)**(n-1)) / thetaduration


# Derivative of Volume with respect to Crank Angle
def vdtheta(theta,B,S,I):
    displacement = 0.25*math.pi*S*B**2
    differentR = 2*I/S 
    # to make the zero angle the start of the cycle, maximum culinder volume
    theta += math.pi
    return (displacement/2) * math.sin(theta) * (1 + math.cos(theta) * (differentR**2 - math.sin(theta)**2 )**(-1/2) )
    

# Volume with respect to Crank Angle
def voftheta(theta,compRatio,B,S,I):
    displacement = 0.25*math.pi*S*B**2
    differentR = 2*I/S 
    # to make the zero angle the start of the cycle, maximum culinder volume
    theta += math.pi
    term1 = displacement/(compRatio - 1)
    term2 = (displacement/2)*(differentR + 1 - math.cos(theta) - (differentR**2 - math.sin(theta)**2 )**0.5 )
    return term1 + term2


# This structure allows me to set the constants in the first layer, and then continuously alter only theta and surrent pressure for a new answer
def pressureDerivative(thetaSpark,thetaduration,QtotReleased,compRatio,B,S,I,n=3,a=5):
    # Derivative of Pressure with respect to Crank Angle, the Differential Equation
    def pdtheta(Pcurrent, theta):
        Gtheta = (k - 1)*qdtheta(theta,thetaSpark,thetaduration,QtotReleased,n,a)/voftheta(theta,compRatio,B,S,I)
        Ftheta = k * vdtheta(theta,B,S,I) / voftheta(theta,compRatio,B,S,I) 
        return Gtheta - Pcurrent*Ftheta
    return pdtheta


# Isentropic relations, here for ease of reading and information hiding purposes
def isoTemperature(temp,v1,v2):
    return temp*(v1/v2)**(k - 1)

def isoPressure(press,v1,v2):
    return press*(v1/v2)**k

# Temperature PVNRT, not temperary pervert -use for finding temp at max presure.
# Note: can make more accureate by incorperating heat loss to the engine and head gain from friction
def temp_pvrt(volume, presure, V1,P1,T1):
    mAir = (P1*V1)/(R*T1)
    return (presure*volume)/(R*mAir)
