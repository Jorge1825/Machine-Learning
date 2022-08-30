import tensorflow as tf
import numpy as np

celcius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float) 
fharenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1]) # La capa Dense es una capa que conecta una capa de entrada con una capa de salida
modelo = tf.keras.Sequential([capa]) # El modelo es una secuencia de capas

modelo.compile(
    loss='mean_squared_error', # Función de pérdida para el modelo 
    optimizer=tf.keras.optimizers.Adam(0.1)) # Compila el modelo con el optimizador Adam y el error mean_squared_error

print("Inicio del entrenamiento")
historial = modelo.fit(celcius, fharenheit, epochs=1000, verbose=False) # Entrena el modelo con los datos de entrada y los datos de salida, epochs es el número de iteraciones, verbose es para mostrar el progreso
print("Modelo entrenado")

import matplotlib.pyplot as plt
plt.xlabel("Numero de vueltas")
plt.ylabel("Longitud de perdida")
plt.plot(historial.history['loss'])

print("Hagamos una predicción")
resultado = modelo.predict([[38]]) # Hace una predicción con el modelo

print("El resultado es: {}".format(resultado))