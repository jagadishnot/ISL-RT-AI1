import numpy as np
import os

files = os.listdir("data/landmarks")

zero = 0
valid = 0

for f in files:

    x = np.load("data/landmarks/" + f)

    if x.std() == 0:
        zero += 1
    else:
        valid += 1

print("Valid videos:", valid)
print("Zero videos:", zero)