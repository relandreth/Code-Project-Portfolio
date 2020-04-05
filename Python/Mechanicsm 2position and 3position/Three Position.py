import numpy as np
import math
import vector as v

def threePositionB(p21, p31, delta2, delta3, alpha2, alpha3, beta2, beta3, gamma2, gamma3):
    delta2*= 1/v.RADIANDEGREE
    alpha2*=1/v.RADIANDEGREE
    beta2 *= 1/v.RADIANDEGREE
    gamma2 *= 1 / v.RADIANDEGREE
    delta3 *= 1/v.RADIANDEGREE
    alpha3 *= 1/v.RADIANDEGREE
    beta3 *= 1/v.RADIANDEGREE
    gamma3 *= 1 / v.RADIANDEGREE

    chosenMatrix = np.matrix([[math.cos(beta2)-1, -math.sin(beta2), math.cos(alpha2)-1, -math.sin(alpha2)],
                              [math.sin(beta2), math.cos(beta2)-1,math.sin(alpha2), math.cos(alpha2)-1],
                              [math.cos(beta3) - 1, -math.sin(beta3), math.cos(alpha3) - 1, -math.sin(alpha3)],
                              [math.sin(beta3), math.cos(beta3) - 1, math.sin(alpha3), math.cos(alpha3) - 1]])
    knownMatrix = np.matrix([[p21*math.cos(delta2)],
                            [p21*math.sin(delta2)],
                             [p31 * math.cos(delta3)],
                             [p31 * math.sin(delta3)]])
    invA = np.linalg.inv(chosenMatrix)
    unknown = invA * knownMatrix

    wx = unknown[0,0]
    wy = unknown[1,0]
    zx = unknown[2,0]
    zy = unknown[3,0]
    P21 = v.Vector(p21, delta2)
    W1 = v.Vector(wx,wy,True)
    W2 = v.Vector(W1.magnitude, W1.angle + (beta2*v.RADIANDEGREE))
    W3 = v.Vector(W1.magnitude, W1.angle + (beta3*v.RADIANDEGREE))
    Z1 = v.Vector(zx,zy,True)
    Z2 = v.Vector(Z1.magnitude, Z1.angle + (alpha2*v.RADIANDEGREE))
    Z3 = v.Vector(Z1.magnitude, Z1.angle + (alpha3*v.RADIANDEGREE))

    mat1 ="{},{},{},{}\n{},{},{},{}\n{},{},{},{}\n{},{},{},{}\n".format(
        chosenMatrix[0, 0], chosenMatrix[0, 1], chosenMatrix[0, 2], chosenMatrix[0, 3],
        chosenMatrix[1, 0], chosenMatrix[1, 1], chosenMatrix[1, 2], chosenMatrix[1, 3],
        chosenMatrix[2, 0], chosenMatrix[2, 1], chosenMatrix[2, 2], chosenMatrix[2, 3],
        chosenMatrix[3, 0], chosenMatrix[3, 1], chosenMatrix[3, 2], chosenMatrix[3, 3])
    mat2 = "{},{},{},{}\n".format(knownMatrix[0,0],
                                  knownMatrix[1,0],
                                  knownMatrix[2,0],
                                  knownMatrix[3,0])
    sol1 = "W1:, {}, {}\nW2:, {}, {}\nW3:, {}, {}\nZ1:, {}, {}\nZ2:, {}, {}\nZ1:, {}, {}\n".format(
        W1.xCoord, W1.yCoord, W2.xCoord, W2.yCoord, W3.xCoord, W3.yCoord,
        Z1.xCoord, Z1.yCoord, Z2.xCoord, Z2.yCoord, Z3.xCoord, Z3.yCoord)
    print(mat1 + mat2 + sol1)

    chosenMatrix2 = np.matrix([[math.cos(gamma2) - 1, -math.sin(gamma2), math.cos(alpha2) - 1, -math.sin(alpha2)],
                              [math.sin(gamma2), math.cos(gamma2) - 1, math.sin(alpha2), math.cos(alpha2) - 1],
                              [math.cos(gamma3) - 1, -math.sin(gamma3), math.cos(alpha3) - 1, -math.sin(alpha3)],
                              [math.sin(gamma3), math.cos(gamma3) - 1, math.sin(alpha3), math.cos(alpha3) - 1]])
    knownMatrix2 = np.matrix([[p21 * math.cos(delta2)],
                             [p21 * math.sin(delta2)],
                             [p31 * math.cos(delta3)],
                             [p31 * math.sin(delta3)]])
    invB = np.linalg.inv(chosenMatrix2)
    unknown2 = invB * knownMatrix2

    ux = unknown2[0, 0]
    uy = unknown2[1, 0]
    sx = unknown2[2, 0]
    sy = unknown2[3, 0]
    P31 = v.Vector(p31, delta3)
    U1 = v.Vector(ux, uy, True)
    U2 = v.Vector(U1.magnitude, U1.angle + (beta2 * v.RADIANDEGREE))
    U3 = v.Vector(U1.magnitude, U1.angle + (beta3 * v.RADIANDEGREE))
    S1 = v.Vector(sx, sy, True)
    S2 = v.Vector(S1.magnitude, S1.angle + (alpha2 * v.RADIANDEGREE))
    S3 = v.Vector(S1.magnitude, S1.angle + (alpha3 * v.RADIANDEGREE))

    mat3 = "{},{},{},{}\n{},{},{},{}\n{},{},{},{}\n{},{},{},{}\n".format(
        chosenMatrix2[0, 0], chosenMatrix2[0, 1], chosenMatrix2[0, 2], chosenMatrix2[0, 3],
        chosenMatrix2[1, 0], chosenMatrix2[1, 1], chosenMatrix2[1, 2], chosenMatrix2[1, 3],
        chosenMatrix2[2, 0], chosenMatrix2[2, 1], chosenMatrix2[2, 2], chosenMatrix2[2, 3],
        chosenMatrix2[3, 0], chosenMatrix2[3, 1], chosenMatrix2[3, 2], chosenMatrix2[3, 3])
    mat4 = "{},{},{},{}\n".format(knownMatrix2[0, 0],
                                  knownMatrix2[1, 0],
                                  knownMatrix2[2, 0],
                                  knownMatrix2[3, 0])
    sol2 = "U1:, {}, {}\nU2:, {}, {}\nU3:, {}, {}\nS1:, {}, {}\nS2:, {}, {}\nS1:, {}, {}\n".format(
        U1.xCoord, U1.yCoord, U2.xCoord, U2.yCoord, U3.xCoord, U3.yCoord,
        S1.xCoord, S1.yCoord, S2.xCoord, S2.yCoord, S3.xCoord, S3.yCoord)
    print()
    print(mat3 + mat4 + sol2)
    threePositionSolution = open("3positionSolution.txt","w")
    threePositionSolution.write(mat1 + mat2 + sol1 + mat3 + mat4 + sol2)
    threePositionSolution.close()

#threePositionB(1.5706,3.589,28.52,55.56,43,109,34.5,89.0,40.0,94.0)

AAA = np.matrix([[1,1,1],
                 [3,4,5],
                 [6,12,20]])
CCC = np.matrix([[-1.5],
                [0],
                [0]])
Amo = np.linalg.inv(AAA)
ans = Amo * CCC
print(ans)