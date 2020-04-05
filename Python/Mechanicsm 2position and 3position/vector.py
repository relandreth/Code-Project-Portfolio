import math

RADIANDEGREE = 180/math.pi

class Vector():
    xCoord = 0
    yCoord = 0
    magnitude = 0
    angle = 0
    xDirCos = 0
    yDirCos = 0
    coords = False
    magAng = False
    dirCos = False

    def __init__(self, xCoord, yCoord, id=False):
        if id:
            self.xCoord = xCoord
            self.yCoord = yCoord
            self.coords = True
            self.calc_DirCoss()
            self.calc_angle()
        if not id:
            self.angle = yCoord
            self.magnitude = xCoord
            self.magAng = True
            self.calc_coords()
            self.calc_DirCoss()


    def __str__(self):
        return "{:^.5f},{:^.5f},{:^.5f},{:^.5f},{:^.5f},{:^.5f},{:^.5f},{:^.5f}".format\
            (self.xCoord, self.yCoord, self.magnitude, self.angle, self.xDirCos, self.yDirCos, self.negate(self.yDirCos), self.xDirCos )


    def negate(self, number):
        return 0-number


    def calc_magnitude(self):
        self.magnitude = math.sqrt((self.xCoord**2) + (self.yCoord**2))


    def calc_coords(self):
        if self.magAng:
            self.xCoord = self.magnitude * math.cos(self.angle/RADIANDEGREE)
            self.yCoord = self.magnitude * math.sin(self.angle/RADIANDEGREE)


    def calc_DirCoss(self):
        if self.coords:
            self.calc_magnitude()
            self.xDirCos = self.xCoord/self.magnitude
            self.yDirCos = self.yCoord/self.magnitude
        if self.magAng:
            self.xDirCos = math.cos(self.angle/RADIANDEGREE)
            self.yDirCos = math.sin(self.angle/RADIANDEGREE)

    def calc_angle(self):
        if self.coords:
            self.angle = math.atan2(self.yCoord, self.xCoord) *RADIANDEGREE

#potentially add a dot product and cross product methods