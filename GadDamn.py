import os
import numpy as np
import cv2
from tensorflow import keras

# Cargar el modelo de Keras
print("[INFO] Cargando el modelo...")
PATH_TO_MODEL = "TiempoReal\Modelos\Bache v1\keras_model.h5"
model = keras.models.load_model(PATH_TO_MODEL)

# Cargar las etiquetas
labels_path = "TiempoReal\Modelos\Bache v1\labels.txt"
with open(labels_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

# Directorio donde se encuentran las imágenes
images_dir = 'Imagenes'

# Abrir archivo de texto para guardar los resultados de las detecciones
detections_file_path = "detections.txt"
detections_file = open(detections_file_path, "w")

# Función para realizar la detección
def detect_image(image, model, labels):
    # Preprocesar la imagen para el modelo
    # Esto es solo un ejemplo y deberá ajustarse a las necesidades específicas de tu modelo
    input_image = cv2.resize(image, (224, 224))  # Asumiendo que el modelo espera imágenes de 224x224
    input_image = input_image.astype('float32')
    input_image /= 255
    input_image = np.expand_dims(input_image, axis=0)
    
    # Realizar la detección usando el modelo
    predictions = model.predict(input_image)

    # Interpretar los resultados (esto también dependerá de cómo tu modelo fue entrenado)
    # Asumiendo que el modelo devuelve un array con la probabilidad de cada clase
    # y que estás interesado en la clase con la máxima probabilidad
    max_index = np.argmax(predictions[0])
    score = predictions[0][max_index]
    label = labels[max_index]
    
    # Esto es solo un ejemplo, necesitarás ajustar esto para que coincida con tu modelo
    box = [0, 0, 1, 1]  # Ejemplo: usar todo el frame como cuadro delimitador

    return [(label, score, box)]

# Procesar todas las imágenes en el directorio
for image_name in os.listdir(images_dir):
    image_path = os.path.join(images_dir, image_name)
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        color_image = cv2.imread(image_path)

        # Realizar la detección en la imagen
        predictions = detect_image(color_image, model, labels)

        # Asumiendo que 'predictions' es una lista de tuplas (label, score, box)
        for label, score, box in predictions:
            if score > 0.5 and label == "0 Baches":
                
                # Escribir en el archivo de texto
                detections_file.write(f"Image: {image_name} - Class: {label}, Score: {score}, Box: {box}\n")

# Cerrar el archivo de texto
detections_file.close()

print("[INFO] Detecciones completadas y guardadas en 'detections.txt'.")
