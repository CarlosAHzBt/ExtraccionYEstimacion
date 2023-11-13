import os
import pyrealsense2 as rs
import numpy as np
import cv2
from tensorflow import keras

# Carga el modelo de Keras y las etiquetas
print("[INFO] Cargando el modelo...")
PATH_TO_MODEL = "TiempoReal/Modelos/Bache v1/keras_model.h5"
model = keras.models.load_model(PATH_TO_MODEL)

labels_path = "TiempoReal/Modelos/Bache v1/labels.txt"
with open(labels_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

# Define la ruta al archivo .bag
bag_file = r'D:\Intel bags\20231015_175838.bag'

# Configura la transmisión desde el archivo .bag
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)

# Inicia el pipeline de RealSense
pipeline.start(config)

# Función para realizar la detección
def detect_image(image, model, labels):
    # Preprocesar la imagen para el modelo
    input_image = cv2.resize(image, (224, 224))  # Asumiendo que el modelo espera imágenes de 224x224
    input_image = input_image.astype('float32')
    input_image /= 255
    input_image = np.expand_dims(input_image, axis=0)
    
    # Realizar la detección usando el modelo
    predictions = model.predict(input_image)
    max_index = np.argmax(predictions[0])
    score = predictions[0][max_index]
    label = labels[max_index]
    
    return (label, score)

# Abrir archivo de texto para guardar los resultados de las detecciones
detections_file_path = "detections.txt"
detections_file = open(detections_file_path, "w")

try:
    frame_number = 0
    while True:
        # Espera a que llegue un par coherente de frames: profundidad y color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convierte las imágenes a arrays de numpy
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Realizar la detección en el frame actual
        label, score = detect_image(color_image, model, labels)

        # Si se detecta algo de interés, escribe en el archivo de texto
        if score > 0.5 and label == "0 Baches":  # Asume un umbral de confianza de 0.5, ajusta según sea necesario
            detections_file.write(f"Frame: {frame_number:05d}, Label: {label}, Score: {score:.2f}\n")

        frame_number += 1

except Exception as e:
    print(f"Se ha producido una excepción: {e}")

finally:
    # Detiene el pipeline y cierra el archivo de texto
    pipeline.stop()
    detections_file.close()

print("[INFO] Detecciones completadas y guardadas en 'detections.txt'.")
