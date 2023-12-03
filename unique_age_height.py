import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import tensorflow as tf
import numpy as np

data = pd.read_csv("final_test_cleaned.csv")

#Solo altura y edad
X = data[["height","age"]]
y = data["size"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scale = StandardScaler()
X_scaled = scale.fit_transform(X)

X_train,X_temp,y_train,y_temp = train_test_split(X_scaled,y_encoded,test_size=0.3,random_state=42)

X_validation,X_testing,y_validation,y_testing = train_test_split(X_temp,y_temp,test_size=0.5,random_state=42)
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

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512,activation="relu",input_shape=(2,)),
    tf.keras.layers.Dense(256,activation="relu"),
    tf.keras.layers.Dense(len(label_encoder.classes_),activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


checkpoint_path = "model_checkpoint_age_height.h5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,  # Guardar solo el modelo con la mejor métrica de validación
    monitor='val_accuracy',  # Métrica a monitorear (puedes cambiarla a tu métrica de interés)
    mode='max',  # 'max' si deseas maximizar la métrica, 'min' si deseas minimizar la pérdida
    verbose=1
)

history = model.fit(X_train,y_train,epochs=200,batch_size=500,validation_data=(X_validation,y_val),callbacks=[model_checkpoint_callback])


# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(X_testing, y_test)
print(f'Precisión en el conjunto de prueba: {test_accuracy}')

