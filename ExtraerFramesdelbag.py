import os
import pyrealsense2 as rs
import numpy as np
import cv2

# Define la ruta al archivo .bag y la carpeta donde se guardarán las imágenes
bag_file = r'D:\Intel bags\20231015_175838.bag'
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

        # Guarda cada frame como una imagen en la carpeta especificada
        
        cv2.imwrite(f'{images_folder}/frame_{frame_number:05d}.png', color_image)
        frame_number += 1

except Exception as e:
    print(f"Se ha producido una excepción: {e}")

finally:
    # Detiene el pipeline
    pipeline.stop()

print(f"Frames extraídos guardados en la carpeta: {images_folder}")
