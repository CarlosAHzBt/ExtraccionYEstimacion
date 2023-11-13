import os
import pyrealsense2 as rs
import numpy as np
import cv2
from tensorflow import keras

# Carga el modelo de Keras y las etiquetas
print("[INFO] Cargando el modelo...")
model_path = "TiempoReal/Modelos/Bache v1/keras_model.h5"
model = keras.models.load_model(model_path)

labels_path = "TiempoReal/Modelos/Bache v1/labels.txt"
with open(labels_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

# Define la ruta al archivo .bag y la carpeta donde se guardarán las imágenes
bag_file = r'D:\Intel bags\90cm bache.bag'
images_folder = 'Imagenes'

# Crea la carpeta si no existe
if not os.path.exists(images_folder):
    os.makedirs(images_folder)

# Configura la transmisión desde el archivo .bag
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)

# Inicia el pipeline de RealSense
pipeline.start(config)

# Función para realizar la detección
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
Ply_folder = "Ply/Extraidos"
# Función para guardar la nube de puntos como archivo .ply
def save_to_ply(color_frame, depth_frame, frame_number):
    ply_filename = f"{Ply_folder}/frame_{frame_number:05d}.ply"
    pc = rs.pointcloud()
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    points.export_to_ply(ply_filename, color_frame)
    print(f"[INFO] Frame {frame_number:05d} guardado como {ply_filename}")

# Abrir archivo de texto para guardar los resultados de las detecciones
detections_file_path = "detections.txt"
detections_file = open(detections_file_path, "w")

try:
    # Contador de frames
    frame_number = 0
    # Conjunto para almacenar los números de frames detectados
    detection_frames = set()

    while True:
        # Espera a que llegue un par coherente de frames: profundidad y color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convierte las imágenes a arrays de numpy
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Guarda cada frame como una imagen en la carpeta especificada
        cv2.imwrite(f'{images_folder}/frame_{frame_number:05d}.png', color_image)

        # Realizar la detección en el frame actual
        label, score = detect_image(color_image, model, labels)

        # Si se detecta algo de interés, escribe en el archivo de texto y añade al conjunto
        if score > 0.85 and label == "0 Baches":  # Asume un umbral de confianza de 0.5
            detections_file.write(f"Frame: {frame_number:05d}, Label: {label}, Score: {score:.2f}\n")
            detection_frames.add(frame_number)

        frame_number += 1

except Exception as e:
    print(f"Se ha producido una excepción: {e}")

finally:
    # Detiene el pipeline y cierra el archivo de texto
    pipeline.stop()
    detections_file.close()

# Reinicia la pipeline para la extracción de archivos .ply
pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, bag_file, repeat_playback=True)
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

try:
    # Contador de frames
    current_frame_number = 0
    while True:
        try:
            frames = pipeline.wait_for_frames()
        except RuntimeError as e:
            print(f"Error al recibir el fotograma: {e}")
            continue  # Salta al siguiente ciclo del bucle

        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            continue

        current_frame_number += 1
        
        if current_frame_number in detection_frames:
            # Guardar el frame actual como archivo .ply
            save_to_ply(color_frame, aligned_depth_frame, current_frame_number)
            
        # Si hemos procesado todos los frames detectados, podemos salir del bucle
        if current_frame_number > max(detection_frames):
            break

finally:
    pipeline.stop()

print("[INFO] Detecciones completadas y guardadas en 'detections.txt'.")
