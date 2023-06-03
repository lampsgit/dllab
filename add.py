from keras import backend
print("Enter two numbers")
data=[float(i) for i in list(input().split())]
res=backend.sum(data)
print(backend.eval(res))
