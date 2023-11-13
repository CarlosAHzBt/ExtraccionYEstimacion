import pyrealsense2 as rs
import numpy as np
import os

# Asegurarse de que la carpeta exista, si no, crearla
carpeta = 'NubesdelBAG'
if not os.path.exists(carpeta):
    os.makedirs(carpeta)

# Cargar el archivo bag
cfg = rs.config()
cfg.enable_device_from_file('grabacion.bag', repeat_playback=False)  # Asegurarse de que no se repita el archivo
pipeline = rs.pipeline()
profile = pipeline.start(cfg)

# Obtener el objeto de reproducción y deshabilitar la repetición
playback = profile.get_device().as_playback()
playback.set_real_time(False)

# Crear un objeto pointcloud
pc = rs.pointcloud()

frame_number = 0

try:
    while True:
        # Obtener un conjunto de frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Verificar si los frames son válidos
        if not depth_frame or not color_frame:
            print("Frames no válidos, finalizando...")
            break
        
        # Mapear el frame de color al de profundidad y calcular la nube de puntos
        pc.map_to(color_frame)
        pointcloud = pc.calculate(depth_frame)

        # Guardar la nube de puntos como un archivo ply en la carpeta especificada
        filename = os.path.join(carpeta, f'nube_de_puntos_{frame_number}.ply')
        pointcloud.export_to_ply(filename, color_frame)
        print(f'Nube de puntos guardada en {filename}')

        frame_number += 1

except RuntimeError as e:
    # Si el error es por el final del archivo .bag, terminar el bucle
    if "Frame didn't arrive" in str(e):
        print("Fin del archivo .bag, finalizando...")
    else:
        print(str(e))
finally:
    # Detener la pipeline
    pipeline.stop()
