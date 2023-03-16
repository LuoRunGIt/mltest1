import numpy

#这里计算一个矩阵相乘
matrix1=numpy.mat([[-1,0.5,1],[1,-0.5,-1]])
print(matrix1)
matrix2=matrix1.T
print(matrix2)

matrix3=matrix2*matrix1
print(matrix3)

for i in  range(10):
    print(i)