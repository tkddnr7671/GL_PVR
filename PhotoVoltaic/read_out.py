import numpy as np
import os

fpath = 'training_490.out'

f = open(fpath, 'r')
while True:
    line = f.readline()
    if not line:
        break
    print(line)

    

f.close()
