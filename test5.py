# Implementation of matplotlib function
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

Z = np.random.rand(3, 3)
Z2 = np.random.rand(3, 3)
Z3 = np.random.rand(3, 3)
Z = np.array([[0.16,0.57,0.06],[0.2,1.1,2.3],[0,1,2]])
Z2 = np.array([[0.3,0.4,0.5],[0.5,0.555,0.7],[1.3,1.4,1.5]])
Z3 = np.array([[1,1,1],[2,2,0.0],[0,1,1]])
print(Z,Z2,Z3)
#颜色数量和z有关
#print(Z)绘制伪彩图
plt.pcolormesh(Z,Z2,Z3,cmap=plt.cm.Paired)

plt.title('matplotlib.pyplot.pcolormesh() function Example', fontweight="bold")
plt.show()