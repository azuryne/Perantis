import numpy as np 

A = np.array([[2,1,1], [1,1,0], [0,1,-3]])
b = np.array([[2],[2],[1]])
x = np.linalg.solve(A,b)

for i in range(len(x)):
    print(f'x{i+1} = {x[i]}')