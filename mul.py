from keras import backend as k
import numpy as np
matrix_a=np.array([[1,2],[3,4]])
matrix_b=np.array([[5,6],[7,8]])
tensor_a=k.variable(matrix_a)
tensor_b=k.variable(matrix_b)
res=k.dot(tensor_a,tensor_b)
print(k.eval(res))
