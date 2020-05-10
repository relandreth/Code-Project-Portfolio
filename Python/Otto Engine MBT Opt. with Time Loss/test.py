### Roderick Landreth
### MER-371 Spring 2020
### for Cycle Integration Assignment
### Purpose: Testing functions and outputs, not the final product. run 'main' for final.


import math
import numpy as np
import main as m
import functions as f
import integrators as inte


N = int(2*math.pi*1000)
B = 3.78*0.0254
S = 3.126*0.0254
I = 5.2*0.0254

#ts = 160*math.pi/180
#thing=f.pressureDerivative(ts,ts/5,2000,1.85,9)
#print(thing(101325,ts))                  GOOD
#print(f.qdtheta(,ts,ts/5,2000.0))        GOOD
#print(f.voftheta(2*ts,1.85,9))           GOOD
#print(f.vdtheta(2*ts,1.85))              GOOD

#data = m.combustionStroke(1000,101325,ts,sd,2000,B,S,I)
#m.plotCoords(range(len(data)),data)                     


# cycle = m.oneCycle(N,300,101325,ts,sd,9,2000,B,S,I)
# volume = cycle[1]
# press = cycle[0]/1000
# anlge = cycle[2]


# answer = m.combustionStroke(95, 1280230.1563158624, 2.792526803190927, 0.4363323129985824, 2000.0, 9, 0.09601199999999999, 0.0794004, 0.13208)
# m.plotCoords(range(len(answer)),answer)

# duration = 40
# val_set = [N,300,101325,9,2000.0,B,S,I,duration*math.pi/180]
# answer = m.maximize(m.format_cycle,[140.0,190.0,1.0],val_set)
# print(answer)


cycle = m.oneCycle(N,101325,300,175*math.pi/180,10*math.pi/180,9,2000.0,B,S,I)           
volume = cycle[1]
press = cycle[0]
angle = cycle[2]
print(inte.integrate_cycle_trapezoid(volume,press))
m.plotCoords(volume,press)

m.data_compiling(N,101325,300,[40.0,25.0,10.0],9,2000.0,B,S,I)

#run 'main' to get the final output
