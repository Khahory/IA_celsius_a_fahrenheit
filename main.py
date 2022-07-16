import tensorflow as tf
import numpy as np

# ponemos entradas y salidas correcta, esto es mi file de aprendizaje
celsius = np.array([-40, 10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 50, 32, 46.4, 59, 71.6, 100.4], dtype=float)

# solo tenemos 2 neuronas, units es TOTAL de las neuronas de salida y input_shape para entrada
# capa = tf.keras.layers.Dense(units=1, input_shape=[1])
# modelo = tf.keras.Sequential([capa])

# agregare 2 capas con 3 neuronas
capa1 = tf.keras.layers.Dense(units=3, input_shape=[1])
capa2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([capa1, capa2, salida])


modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print('comenzando entrenamiento...')
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print('modelo entrenado')

import matplotlib.pyplot as plt

plt.xlabel('# Epoca')
plt.ylabel('Magnitud de perdida')
plt.plot(historial.history['loss'])

resultado = modelo.predict([100.0])
print(str(resultado))