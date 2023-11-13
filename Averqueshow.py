import pyrealsense2 as rs
import numpy as np
import os

# Función para guardar la nube de puntos como archivo .ply
def save_to_ply(color_frame, depth_frame, frame_number):
    ply_filename = f"frame_{frame_number:05d}.ply"
    pc = rs.pointcloud()
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    points.export_to_ply(ply_filename, color_frame)
    print(f"[INFO] Frame {frame_number:05d} guardado como {ply_filename}")

# Función para leer el archivo de detecciones y extraer los números de frame
def read_detection_frames(detections_file_path):
    detection_frames = set()
    with open(detections_file_path, "r") as file:
        for line in file:
            if "Image:" in line:
                # The frame number is now extracted from a part of the string that looks like 'frame_00000.png'
                frame_number_str = line.split(" ")[1].split(".")[0]  # This will get 'frame_00000'
                frame_number = int(frame_number_str.split("_")[1])  # This will extract '00000' as an integer
                detection_frames.add(frame_number)
    return detection_frames


# Leer el archivo de detecciones y extraer los números de frame
detections_file_path = "detections.txt"
detection_frames = read_detection_frames(detections_file_path)

# Extraer y guardar .ply para los frames detectados
def extract_ply_from_bag(bag_file, detection_frames):
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

# Nombre del archivo .bag de donde extraer los frames
bag_file = r"D:\Intel bags\20231015_175838.bag"

# Asegúrate de que el archivo .bag existe
if os.path.exists(bag_file):
    extract_ply_from_bag(bag_file, detection_frames)
else:
    print(f"El archivo {bag_file} no existe.")
