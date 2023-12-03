import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

# Dividir el conjunto de datos
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
X_validation, X_testing, y_validation, y_testing = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Crear y entrenar el modelo SVM
svm_model = SVC(kernel='linear', C=100)
svm_model.fit(X_train, y_train)

# Graficar el modelo SVM en 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Graficar puntos
ax.scatter(X_testing[:, 0], X_testing[:, 1], X_testing[:, 2], c=y_testing, cmap='viridis', marker='o', label='Datos de prueba')

# Graficar hiperplano de decisión
xx, yy = np.meshgrid(np.linspace(X_testing[:, 0].min(), X_testing[:, 0].max(), 100),
                     np.linspace(X_testing[:, 1].min(), X_testing[:, 1].max(), 100))
zz = (-svm_model.intercept_[0] - svm_model.coef_[0, 0] * xx - svm_model.coef_[0, 1] * yy) / svm_model.coef_[0, 2]
ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray', antialiased=True, label='Hiperplano de decisión')

# Configuración del gráfico
ax.set_xlabel('Peso')
ax.set_ylabel('Edad')
ax.set_zlabel('Altura')
ax.set_title('Modelo SVM en 3D')

# Crear manualmente la leyenda
legend_labels = {'Datos de prueba': 'scatter', 'Hiperplano de decisión': 'surface'}
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=7, label=key) if value == 'scatter' 
           else plt.Line2D([0], [0], color='gray', alpha=0.3, linewidth=2, label=key) for key, value in legend_labels.items()]

# Agregar leyenda al gráfico
ax.legend(handles=handles)

# Mostrar el gráfico
plt.show()
