import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar los datos desde el archivo CSV
data = pd.read_csv('final_test_cleaned.csv')

# Separar las características (X) y las etiquetas (y)
X = data[['weight', 'age', 'height']]
y = data['size']

# Codificar las etiquetas usando LabelEncoder si son categóricas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA para reducir la dimensión a 2 para visualización
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Dividir el conjunto de datos
X_train, X_temp, y_train, y_temp = train_test_split(X_pca, y_encoded, test_size=0.3, random_state=42)
X_validation, X_testing, y_validation, y_testing = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Crear y entrenar el modelo K-Nearest Neighbors
n_classes = 7
knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, algorithm='auto')  
knn_model.fit(X_train, y_train)

# Predecir las etiquetas para todos los puntos en un rango específico
h = .02
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])

# Visualizar los resultados
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=label_encoder.inverse_transform(y_encoded), palette='viridis', alpha=0.7)
plt.contourf(xx, yy, Z.reshape(xx.shape), cmap='viridis', alpha=0.3)
plt.title('K-Nearest Neighbors Classifier with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
