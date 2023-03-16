#这个例子直接用numpy进行协方差矩阵的计算

import numpy as np
#注意类的写法
class PCA():
    def __init__(self,n_components):
        self.n_components=n_components

    def fit_transform(self,X):
        X=X-X.mean(axis=0)
        #这个是协方差矩阵
        self.covariance=np.dot(X.T,X)/X.shape[0]
        #这里有个问题 shape 0是什么
        #根据公式是1/m
        #求协方差矩阵的特征值和特征向量
        eig_vals,eig_vextors=np.linalg.eig(self.covariance)
        # 获得降序排列特征值的序号
        idx=np.argsort(-eig_vals)
        #降维矩阵
        self.components_=eig_vextors[:,idx[:self.n_components]]
        #对x进行降维
        return np.dot(X,self.components_)

#往2维进行降维
pca=PCA(n_components=2)
X=np.array([[-1,2,66,-1],[2,-6,58,-1],[1,9,36,1],
            [2,10,62,1],[3,5,83,2],[-3,8,45,-2]])
newX=pca.fit_transform(X)
print(newX)


