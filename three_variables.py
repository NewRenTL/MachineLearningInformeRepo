import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV (asegúrate de tener pandas instalado)
data = pd.read_csv('final_test_cleaned.csv')

# Separar las características (X) y las etiquetas (y)
X = data[['weight','age','height']]
y = data['size']

# Codificar las etiquetas usando LabelEncoder si son categóricas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalizar las características para asegurarte de que tengan media cero y desviación estándar uno
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir el conjunto de datos en entrenamiento (70%), prueba (15%) y validación (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
# X_TEMP guarda los datos x1, x2, x3 del 30%, y_temp guarda las etiquetas del 30%
X_validation, X_testing, y_validation, y_testing = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Codificar las etiquetas en formato one-hot si es necesario (para tallas categóricas)
# Esto se hace después de dividir los datos para evitar fugas de información
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(label_encoder.classes_))
y_val = tf.keras.utils.to_categorical(y_validation, num_classes=len(label_encoder.classes_))
y_test = tf.keras.utils.to_categorical(y_testing, num_classes=len(label_encoder.classes_))

# Verificar las formas de los conjuntos de datos
print("Forma de X_train:", X_train.shape)
print("Forma de X_val:", X_validation.shape)
print("Forma de X_test:", X_testing.shape)
print("Forma de y_train:", y_train.shape)
print("Forma de y_val:", y_val.shape)
print("Forma de y_test:", y_test.shape)

# Crear el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation="relu", input_shape=(3,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Guardar el mejor modelo durante el entrenamiento
checkpoint_path = "model_checkpoint_3_Variables.h5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,  # Guardar solo el modelo con la mejor métrica de validación
    monitor='val_accuracy',  # Métrica a monitorear (puedes cambiarla a tu métrica de interés)
    mode='max',  # 'max' si deseas maximizar la métrica, 'min' si deseas minimizar la pérdida
    verbose=1
)

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=200, batch_size=500, validation_data=(X_validation, y_val), callbacks=[model_checkpoint_callback])

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(X_testing, y_test)
print(f'Precisión en el conjunto de prueba: {test_accuracy}')

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_testing)
y_pred_classes = np.argmax(y_pred, axis=1)

# Obtener la matriz de confusión
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)

# Calcular los porcentajes en cada celda
conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

# Mostrar la matriz de confusión
disp = ConfusionMatrixDisplay(conf_matrix_percent, display_labels=label_encoder.classes_)
disp.plot(values_format=".2f")

# Mostrar la figura
plt.show()
