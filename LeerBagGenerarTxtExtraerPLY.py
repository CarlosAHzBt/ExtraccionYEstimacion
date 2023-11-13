import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf

# Cargar el modelo de TensorFlow
print("[INFO] Cargando el modelo...")
PATH_TO_CKPT = "TiempoReal/model.savedmodel/saved_model.pb"

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)

# Definir tensores de entrada y salida para TensorFlow
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Abrir archivo de texto para guardar los resultados de las detecciones
detections_file_path = "detections.txt"
detections_file = open(detections_file_path, "w")

# Iniciar el pipeline de RealSense
pipeline = rs.pipeline()
config = rs.config()
bag_file = r'D:\Intel bags\20231015_175838.bag.bag'
config.enable_device_from_file(bag_file, repeat_playback=False)
profile = pipeline.start(config)

# Contador de frames
current_frame_number = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # Convertir a array de numpy
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Realizar la detección usando el modelo
        image_expanded = np.expand_dims(color_image, axis=0)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        # Incrementar el contador de frames
        current_frame_number += 1
        print(f"Procesando frame: {current_frame_number}")

        # Almacenar resultados si hay detecciones con un score mayor al umbral
        for idx in range(int(num[0])):
            if scores[0][idx] > 0.9:
                # Guardar en el archivo de texto la información de detección
                detections_file.write(f"Frame: {current_frame_number}, "
                                      f"Class: {classes[0][idx]}, "
                                      f"Score: {scores[0][idx]}, "
                                      f"Box: {boxes[0][idx].tolist()}\n")
except RuntimeError as e:
    print(e)

finally:
    # Cerrar el pipeline y el archivo de texto
    print("[INFO] Terminando el streaming...")
    pipeline.stop()
    sess.close()
    detections_file.close()

# Función para guardar la nube de puntos como archivo .ply
def save_to_ply(color_frame, depth_frame, frame_number, pc):
    ply_filename = f"frame_{frame_number}.ply"
    points = pc.calculate(depth_frame)
    pc.map_to(color_frame)
    points.export_to_ply(ply_filename, color_frame)
    print(f"[INFO] Frame {frame_number} guardado como {ply_filename}")

# Función para extraer y guardar los frames como archivos .ply
def extract_ply_from_frames(bag_file, detection_frames):
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file, repeat_playback=True)
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    pc = rs.pointcloud()

    try:
        # Contador de frames
        current_frame_number = 0
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue

            current_frame_number += 1
            if current_frame_number in detection_frames:
                save_to_ply(color_frame, aligned_depth_frame, current_frame_number, pc)
    finally:
        pipeline.stop()

# Leer el archivo de detecciones y extraer los números de frame y guardarlos en la carpeta
detection_frames = set()
with open(detections_file_path, "r") as file:
    for line in file:
        if "Frame:" in line:
            frame_number = int(line.split(",")[0].split(":")[1])
            detection_frames.add(frame_number)

# Extraer y guardar .ply para los frames detectados
extract_ply_from_frames(bag_file, detection_frames)
