# mlp_employee_satisfaction.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# Cargar datos
df = pd.read_csv('Extended_Employee_Performance_and_Productivity_Data.csv')

# Filtrar columnas numéricas
numeric_columns = df.select_dtypes(include=['number']).drop('Employee_ID', axis=1)
cols = list(numeric_columns)

# Visualización de distribuciones
fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
for i, col in enumerate(cols):
    axes[i].hist(df[col], bins=20, color='skyblue', edgecolor='black')
    axes[i].set_title(col)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# Preparación de variables
X = numeric_columns.drop('Employee_Satisfaction_Score', axis=1)
y = numeric_columns['Employee_Satisfaction_Score']
y = y.apply(lambda x: round(x)-1)  # 5 categorías

# Estandarización
scaler = StandardScaler()
X_standar = scaler.fit_transform(X)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_standar, y, test_size=0.33, random_state=42)

# Codificación one-hot
y_onehot_train = tf.keras.utils.to_categorical(y_train, 5)
y_onehot_test = tf.keras.utils.to_categorical(y_test, 5)

# Implementación de tres MLP

# MLP 1
mlp1 = models.Sequential(name="MLP_1")
mlp1.add(layers.Input(shape=(X_train.shape[1],)))
mlp1.add(layers.Dense(32, activation='relu'))
mlp1.add(layers.Dense(5, activation='softmax'))

# MLP 2
mlp2 = models.Sequential(name="MLP_2")
mlp2.add(layers.Input(shape=(X_train.shape[1],)))
mlp2.add(layers.Dense(64, activation='relu'))
mlp2.add(layers.Dense(32, activation='relu'))
mlp2.add(layers.Dense(5, activation='softmax'))

# MLP 3
mlp3 = models.Sequential(name="MLP_3")
mlp3.add(layers.Input(shape=(X_train.shape[1],)))
mlp3.add(layers.Dense(128, activation='relu'))
mlp3.add(layers.Dense(64, activation='relu'))
mlp3.add(layers.Dense(32, activation='relu'))
mlp3.add(layers.Dense(5, activation='softmax'))

# Función para compilar y entrenar
def compile_and_fit(model, X_train, y_train, X_test, y_test, epochs=50):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test), 
                        epochs=epochs, batch_size=32)
    return history

# Entrenamiento de modelos
history1 = compile_and_fit(mlp1, X_train, y_onehot_train, X_test, y_onehot_test)
history2 = compile_and_fit(mlp2, X_train, y_onehot_train, X_test, y_onehot_test)
history3 = compile_and_fit(mlp3, X_train, y_onehot_train, X_test, y_onehot_test)

# Graficar curvas de pérdida y precisión
plt.figure(figsize=(14,5))

# Pérdida
plt.subplot(1,2,1)
plt.plot(history1.history['loss'], label='MLP1 Loss')
plt.plot(history2.history['loss'], label='MLP2 Loss')
plt.plot(history3.history['loss'], label='MLP3 Loss')
plt.plot(history1.history['val_loss'], '--', label='MLP1 Val Loss')
plt.plot(history2.history['val_loss'], '--', label='MLP2 Val Loss')
plt.plot(history3.history['val_loss'], '--', label='MLP3 Val Loss')
plt.title('Pérdida por época')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()

# Precisión
plt.subplot(1,2,2)
plt.plot(history1.history['accuracy'], label='MLP1 Acc')
plt.plot(history2.history['accuracy'], label='MLP2 Acc')
plt.plot(history3.history['accuracy'], label='MLP3 Acc')
plt.plot(history1.history['val_accuracy'], '--', label='MLP1 Val Acc')
plt.plot(history2.history['val_accuracy'], '--', label='MLP2 Val Acc')
plt.plot(history3.history['val_accuracy'], '--', label='MLP3 Val Acc')
plt.title('Precisión por época')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
