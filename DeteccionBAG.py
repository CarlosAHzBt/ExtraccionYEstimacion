import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf

# Función para reproducir el video del archivo .bag
def play_bag_video(path_to_bagfile):
    # Configure depth and color streams from bag file
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(path_to_bagfile, repeat_playback=False)

    # Start RealSense pipeline
    pipeline.start(config)
    frame_number = 0
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            frame_number += 1
            print(f"Frame: {frame_number}")
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Mostrar la imagen
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cerrar el pipeline
        print("[INFO] Terminando la reproducción del video...")
        pipeline.stop()
        cv2.destroyAllWindows()


# Cargar el modelo de TensorFlow
print("[INFO] Cargando el modelo...")
PATH_TO_CKPT = "TiempoReal\model.savedmodel\saved_model.pb"

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)

# Definir tensores de entrada y salida
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Hash de colores para las clases detectadas
colors_hash = {}

# Abrir archivo de texto para guardar los resultados de las detecciones
detections_file = open("detections.txt", "w")

frame_number = 0
# Iniciar el pipeline de RealSense
# Configuración de la entrada desde el archivo .bag
pipeline = rs.pipeline()
config = rs.config()

config.enable_device_from_file('BAG/40KM.bag', repeat_playback=False)

pipeline.start(config)
try:
    while True:
        # Esperar por los frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convertir a array de numpy
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Realizar la detección usando el modelo
        image_expanded = np.expand_dims(color_image, axis=0)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)

        # Almacenar resultados si hay detecciones con un score mayor al umbral
        for idx in range(int(num)):
            if scores[idx] > 0.9:
                class_ = classes[idx]
                score = scores[idx]
                box = boxes[idx]

                # Escribir en el archivo de texto
                detections_file.write(f"Frame: {frame_number}, Class: {class_}, Score: {score}, Box: {box}\n")

        print(f"Frame: {frame_number}")
        frame_number += 1

except RuntimeError as e:
    print(e)

finally:
    # Cerrar el pipeline y el archivo de texto
    print("[INFO] Terminando el streaming...")
    pipeline.stop()
    sess.close()
    detections_file.close()
    play_bag_video("BAG/deteccion.bag")