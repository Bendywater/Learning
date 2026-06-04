from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 加载数据
iris = load_iris()
X = iris.data

# 聚类
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# 可视化（先降维）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
plt.title('KMeans on Iris Dataset (PCA-Reduced)')
plt.xlabel('PC1')
plt.ylabel('PC2')
# plt.show()
plt.savefig('Unsupervised_Learning/outputs/PCA/kmeans_iris_pca.png')