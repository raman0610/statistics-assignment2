# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# Step 1: Load Dataset
df = pd.read_csv("Mall_Customers.csv")

print("Initial Data:")
print(df.head())


# Step 2: Preprocessing
numeric_cols = df.select_dtypes(include=np.number)

imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(numeric_cols)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)


# Step 3: Apply PCA
pca = PCA()
pca_data = pca.fit_transform(data_scaled)


explained_variance = pca.explained_variance_ratio_


# Step 4: Determine Optimal Components
plt.figure()
plt.plot(np.cumsum(explained_variance), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs Components')
plt.grid()
plt.show()


n_components = np.argmax(np.cumsum(explained_variance) >= 0.95) + 1
print(f"Optimal number of components: {n_components}")

pca_final = PCA(n_components=n_components)
reduced_data = pca_final.fit_transform(data_scaled)


# Step 5: Visualization (2D)
pca_2 = PCA(n_components=2)
data_2d = pca_2.fit_transform(data_scaled)

plt.figure()
plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.6)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA Projection')
plt.grid()
plt.show()


# 3D Visualization
pca_3 = PCA(n_components=3)
data_3d = pca_3.fit_transform(data_scaled)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], alpha=0.6)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D PCA Projection')

plt.show()