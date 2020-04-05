import numpy as np
import math
import vector as v

def twoPosition(P2x, P2y, theta1, theta2, thetaA, phiPm, betaA, Rho, ThetaBP, Gamma):
    p21 = math.sqrt((P2x)**2 + (P2y)**2)
    alpha2 = theta2-theta1
    delta2 = math.atan2(P2y,P2x)

    chosenMatrix = [[math.cos(thetaA + betaA) - math.cos(thetaA), math.cos(phiPm + alpha2) - math.cos(phiPm)],
                    [math.sin(thetaA + betaA) - math.sin(thetaA), math.sin(phiPm + alpha2) - math.sin(phiPm)]]
    knownMatrix = [p21 * math.cos(delta2), p21 * math.sin(delta2)]
    invA = np.inv(chosenMatrix)
    unknown = knownMatrix * invA

    wMagnitude = unknown[0]
    zMagnitude = unknown[1]

    P21 = v.Vector(p21, delta2)
    W1 = v.Vector(wMagnitude,thetaA)
    W2 = v.Vector(wMagnitude, thetaA + betaA)
    Z1 = v.Vector(zMagnitude,  phiPm)
    Z2 = v.Vector(zMagnitude, phiPm + alpha2)

    chosenMatrix = [[math.cos(Rho + Gamma) - math.cos(Rho), math.cos(ThetaBP + alpha2) - math.cos(ThetaBP)],
                    [math.sin(Rho + Gamma) - math.sin(Rho), math.sin(ThetaBP + alpha2) - math.sin(ThetaBP)]]
    knownMatrix = [p21 * math.cos(delta2), p21 * math.sin(delta2)]
    invA = np.inv(chosenMatrix)
    unknown = knownMatrix * invA

    uMagnitude = unknown[0]
    sMagnitude = unknown[1]

    U1 = v.Vector(uMagnitude, Rho)
    U2 = v.Vector(uMagnitude, Rho + Gamma)
    S1 = v.Vector(sMagnitude, ThetaBP)
    S2 = v.Vector(sMagnitude, ThetaBP + alpha2)


def twoPositionB(p21, delta2, alpha2, betaA, z, phi,gamma2,s,psi):
    delta2*= 1/v.RADIANDEGREE
    alpha2*=1/v.RADIANDEGREE
    betaA *= 1/v.RADIANDEGREE
    phi *= 1/v.RADIANDEGREE
    gamma2 *= 1/v.RADIANDEGREE
    psi *= 1/v.RADIANDEGREE

    chosenMatrix = np.matrix([[math.cos(betaA)-1, -math.sin(betaA)],[math.sin(betaA), math.cos(betaA)-1]])
    knownMatrix = np.matrix([[(p21 * math.cos(delta2)) - (z*(math.cos(phi+alpha2) - math.cos(phi)))],
                   [(p21 * math.sin(delta2)) - (z*(math.sin(phi + alpha2) - math.sin(phi)))]])
    invA = np.linalg.inv(chosenMatrix)
    unknown = invA * knownMatrix

    wx = unknown[0,0]
    wy = unknown[1,0]
    P21 = v.Vector(p21, delta2)
    W1 = v.Vector(wx,wy,True)
    W2 = v.Vector(W1.magnitude, W1.angle + (betaA*v.RADIANDEGREE))
    Z1 = v.Vector(z,phi*v.RADIANDEGREE)
    Z2 = v.Vector(Z1.magnitude, Z1.angle + (alpha2*v.RADIANDEGREE))

    mat1 ="{},{},{},{}\n{},{}\n".format(chosenMatrix[0,0],chosenMatrix[0,1],chosenMatrix[1,0],chosenMatrix[1,1],knownMatrix[0,0],knownMatrix[1,0])
    sol1 = "W1:, {}, {}\nW2:, {}, {}\nZ1:, {}, {}\nZ2:, {}, {}\n".format(W1.xCoord, W1.yCoord, W2.xCoord, W2.yCoord, Z1.xCoord, Z1.yCoord, Z2.xCoord, Z2.yCoord)
    print(mat1 + sol1)

    chosenMatrix2 = np.matrix([[math.cos(gamma2) - 1, -math.sin(gamma2)], [math.sin(gamma2), math.cos(gamma2) - 1]])
    knownMatrix2 = np.matrix([[(p21 * math.cos(delta2)) - (z * (math.cos(psi + alpha2) - math.cos(psi)))],
                             [(p21 * math.sin(delta2)) - (z * (math.sin(psi + alpha2) - math.sin(psi)))]])
    invB = np.linalg.inv(chosenMatrix2)
    unknown = invB * knownMatrix2

    ux = unknown[0, 0]
    uy = unknown[1, 0]
    U1 = v.Vector(ux, uy, True)
    U2 = v.Vector(U1.magnitude, U1.angle + (gamma2 * v.RADIANDEGREE))
    S1 = v.Vector(s, psi * v.RADIANDEGREE)
    S2 = v.Vector(S1.magnitude, S1.angle + (alpha2 * v.RADIANDEGREE))
    mat2="{},{},{},{}\n{},{}\n".format(chosenMatrix2[0,0],chosenMatrix2[0,1],chosenMatrix2[1,0],chosenMatrix2[1,1],knownMatrix2[0,0],knownMatrix2[1,0])
    sol2 = "U1:, {}, {}\nU2:, {}, {}\nS1:, {}, {}\nS2:, {}, {}".format(U1.xCoord, U1.yCoord, U2.xCoord, U2.yCoord,
                                                                     S1.xCoord, S1.yCoord, S2.xCoord, S2.yCoord)
    print()
    print(mat2 + sol2)
    twoPositionSolution = open("2positionSolution.txt","w")
    twoPositionSolution.write(mat1 + sol1 + mat2 + sol2)
    twoPositionSolution.close()

twoPositionB(1.974,21,57,50,1.721,-45,50,1.75,90)

