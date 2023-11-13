import open3d as o3d
import numpy as np
import os

# Definición de las funciones de procesamiento
def cargar_ply(nombre_archivo):
    pcd = o3d.io.read_point_cloud(nombre_archivo)
    return pcd

def eliminar_outliers(pcd, nb_neighbors=1000, std_ratio=0.5):
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd

def segmentar_entorno(pcd, punto_central, radio):
    puntos = np.asarray(pcd.points)
    diferencias = puntos - punto_central
    distancias = np.linalg.norm(diferencias, axis=1)
    indices_cercanos = np.where(distancias < radio)[0]
    return pcd.select_by_index(indices_cercanos)

def filtrar_puntos(pcd, z_min, z_max):
    puntos = np.asarray(pcd.points)
    filtrados = puntos[(puntos[:, 2] > z_min) & (puntos[:, 2] < z_max)]
    pcd_filtrado = o3d.geometry.PointCloud()
    pcd_filtrado.points = o3d.utility.Vector3dVector(filtrados)
    return pcd_filtrado

def segmentar_terreno(pcd, distancia_thresh=0.05):
    plano, inliers = pcd.segment_plane(distance_threshold=distancia_thresh,
                                        ransac_n=4,
                                        num_iterations=1000)
    inlier_cloud = pcd.select_by_index(inliers)
    return inlier_cloud, plano

def matriz_rotacion(v1, v2):
    v = np.cross(v1, v2)
    s = np.linalg.norm(v)
    c = np.dot(v1, v2)

    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
    return R

def nivelar_puntos(pcd, plano):
    A, B, C, D = plano
    norm = np.linalg.norm([A, B, C])
    vector = np.array([A, B, C]) / norm
    up = np.array([0, 0, 1])
    
    rot = matriz_rotacion(vector, up)
    transform = np.eye(4)
    transform[:3, :3] = rot
    return transform

# Procesamiento de cada archivo PLY
def procesar_ply(nombre_archivo_ply):
    pcd = cargar_ply(nombre_archivo_ply)
    pcd_filtrado = filtrar_puntos(pcd, z_min=-2, z_max=2)
    pcd_terreno, plano = segmentar_terreno(pcd_filtrado)
    transformacion = nivelar_puntos(pcd_terreno, plano)
    pcd_nivelado = pcd.transform(transformacion)
    pcd = eliminar_outliers(pcd_nivelado)
    superficie_fija = 0.95
    puntos = np.asarray(pcd.points)
    idx_punto_mas_profundo = np.argmin(puntos[:, 2])
    superficie_estimada = np.median(puntos[:, 2])
    profundidad_agujero = puntos[idx_punto_mas_profundo, 2] - superficie_estimada
    pcd_cercano = segmentar_entorno(pcd, puntos[idx_punto_mas_profundo], radio=0.2)
    #Nombre del archivo
    nombre_archivo = os.path.basename(nombre_archivo_ply)
    # Devuelve una cadena con la información requerida
    return f"Archivo: {nombre_archivo}\n" \
            f"Superficie estimada: {superficie_estimada:.4f} unidades\n" \
           f"Profundidad del agujero desde la superficie fija: {profundidad_agujero:.4f} unidades\n" \
           f"Posición del punto más profundo: {puntos[idx_punto_mas_profundo]}"

# Punto de entrada principal
if __name__ == "__main__":
    carpeta = "Ply/Extraidos"  # Define la carpeta donde están los archivos .ply
    resultados = []

    # Escanear la carpeta y procesar cada archivo .ply
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".ply"):
            nombre_archivo_ply = os.path.join(carpeta, archivo)
            resultado = procesar_ply(nombre_archivo_ply)
            resultados.append(resultado)

    # Escribir los resultados en un archivo txt
    with open('resultados_analisis.txt', 'w') as archivo_resultados:
        for resultado in resultados:
            archivo_resultados.write(resultado + '\n\n')
