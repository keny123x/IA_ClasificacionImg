import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import layers, models
from keras import datasets

# Carga los datos
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255.0, testing_images / 255.0

class_names = ['Avión', 'Auto', 'Pájaro', 'Gato', 'Ciervo', 'Perro', 'Rana', 'Caballo', 'Lancha', 'Camión']

# Muestra las primeras 16 imágenes del conjunto de entrenamiento
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

# Reducción de tamaño de los conjuntos de datos (opcional, para reducir tiempos de entrenamiento)
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:10000]  # Debería ser 10000, no 40000
testing_labels = testing_labels[:10000]  # Debería ser 10000, no 40000

# Construcción del modelo
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compilación del modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"perdido : {loss}")
print(f"exactitud: {accuracy}")

model.save('image_classifier.model')

model= models.load_model()
