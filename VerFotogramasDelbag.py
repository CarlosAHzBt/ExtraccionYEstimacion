import pyrealsense2 as rs
import cv2
import sys
import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf
import os

# Función para mostrar un frame específico
def show_frame_from_bag(bag_file, frame_number_to_show):
    # Configurar el objeto pipeline para leer desde el archivo .bag
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file)

    # Empezar a procesar los frames
    pipeline.start(config)

    # Crear objeto de alineación para alinear los frames de profundidad al espacio de color
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Contador de frames
    frame_count = 0

    try:
        # Iterar sobre los frames hasta llegar al frame deseado
        while True:
            # Esperar por un conjunto de frames y alinearlos
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            # Incrementar contador de frames
            frame_count += 1

            # Verificar si el frame actual es el que queremos mostrar
            if frame_count == frame_number_to_show:
                # Obtener los frames alineados
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                # Validar que ambos frames estén disponibles
                if not aligned_depth_frame or not color_frame:
                    raise RuntimeError("Could not acquire depth or color frames.")

                # Convertir imágenes a numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                depth_image = np.asanyarray(aligned_depth_frame.get_data())

                # Aplicar un colormap en el frame de profundidad (opcional)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                # Mostrar ambos frames
                cv2.imshow('Color Frame', color_image)
                cv2.imshow('Depth Frame', depth_colormap)

                # Esperar una tecla para cerrar las ventanas
                cv2.waitKey(0)
                break

            # Si ya pasamos el frame deseado, no hay necesidad de seguir procesando
            if frame_count > frame_number_to_show:
                break

    finally:
        # Detener el pipeline y cerrar todas las ventanas de OpenCV
        pipeline.stop()
        cv2.destroyAllWindows()

# Nombre del archivo .bag y número del frame a mostrar
bag_file = "BAG/deteccion.bag"  # Reemplaza con la ruta al archivo .bag
frame_number_to_show = 32  # Reemplaza con el número de frame que deseas mostrar

# Verificar si el archivo existe
if os.path.exists(bag_file):
    show_frame_from_bag(bag_file, frame_number_to_show)
else:
    print(f"El archivo {bag_file} no existe.")
    sys.exit(1)
