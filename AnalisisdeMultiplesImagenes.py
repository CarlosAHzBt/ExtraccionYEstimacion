import os
import numpy as np
import cv2
import tensorflow as tf

# Cargar el modelo de TensorFlow
print("[INFO] Cargando el modelo...")
PATH_TO_CKPT = "TiempoReal/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"

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

# Directorio donde se encuentran las imágenes
images_dir = 'Imagenes'

# Abrir archivo de texto para guardar los resultados de las detecciones
detections_file = open("detections.txt", "w")

# Procesar todas las imágenes en el directorio
for image_name in os.listdir(images_dir):
    image_path = os.path.join(images_dir, image_name)
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        color_image = cv2.imread(image_path)

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
                detections_file.write(f"Image: {image_name} - Class: {class_}, Score: {score}, Box: {box}\n")

# Cerrar el archivo de texto y la sesión de TensorFlow
detections_file.close()
sess.close()
