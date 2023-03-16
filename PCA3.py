import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

X=np.array([[-1,2,66,-1],[2,-6,58,-1],[1,9,36,1],
            [2,10,62,1],[3,5,83,2],[-3,8,45,-2]])
pca=PCA(n_components=2)#降到2维度
pca.fit(X)#训练
newX=pca.fit_transform(X)#降维后的
print(pca.explained_variance_ratio_)#贡献率
print(newX)
