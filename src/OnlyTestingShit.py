import numpy as np
n = 6 

l = np.empty( (3, 2*(n - 1), n//2), dtype=str) # day - week - matchOfDay

print(l)

l[0][0][0] = "String"

print(
    l[0][0][0]
)