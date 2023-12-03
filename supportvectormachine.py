import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
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
svm_model = SVC(kernel='linear', C=10.0)
svm_model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = svm_model.predict(X_testing)

# Calcular la precisión
accuracy = accuracy_score(y_testing, y_pred)
print(f'Precisión en el conjunto de prueba: {accuracy}')

# Obtener la matriz de confusión
conf_matrix = confusion_matrix(y_testing, y_pred)

# Normalizar la matriz de confusión para mostrar porcentajes
conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

# Mostrar la matriz de confusión
disp = ConfusionMatrixDisplay(conf_matrix_percent, display_labels=label_encoder.classes_)
disp.plot(values_format=".2f")

# Mostrar la figura
plt.show()
