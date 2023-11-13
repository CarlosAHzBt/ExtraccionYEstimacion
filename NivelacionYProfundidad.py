import open3d as o3d
import numpy as np

# Cargar el archivo PLY
def cargar_ply(nombre_archivo):
    pcd = o3d.io.read_point_cloud(nombre_archivo)
    return pcd

def eliminar_outliers(pcd, nb_neighbors=500, std_ratio=0.5):
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
                                        ransac_n=3,
                                        num_iterations=10000)
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


if __name__ == "__main__":
    nombre_archivo_ply = "frame_00077.ply"
    pcd = cargar_ply(nombre_archivo_ply)
    # Filtrado de puntos
    pcd_filtrado = filtrar_puntos(pcd, z_min=-2, z_max=2)  # Puedes ajustar estos valores según tus datos

    # Segmentación del terreno
    pcd_terreno, plano = segmentar_terreno(pcd_filtrado)

    # Nivelación de puntos
    transformacion = nivelar_puntos(pcd_terreno, plano)
    pcd_nivelado = pcd.transform(transformacion)

    # Eliminar outliers
    pcd = eliminar_outliers(pcd_nivelado)

    # Definir la superficie fija
    superficie_fija = 0.95 # 1 metro

    # Obtener las coordenadas de los puntos
    puntos = np.asarray(pcd.points)

    # Encontrar el índice del punto con la coordenada Z más baja (punto más profundo)
    idx_punto_mas_profundo = np.argmin(puntos[:, 2])
    # Calcular la superficie estimada (moda de alturas)
    superficie_estimada = np.median(puntos[:, 2])  # Puedes usar median o mean según tu necesidad

    print(f"Superficie estimada: {superficie_estimada:.4f} unidades")

    # Calcular la profundidad del agujero en base a la superficie fija
    profundidad_agujero = puntos[idx_punto_mas_profundo, 2] - superficie_estimada

    print(f"Profundidad del agujero desde la superficie fija: {profundidad_agujero:.4f} unidades")
    print(np.min(puntos[:, 0]), np.max(puntos[:,0]))
    print(f"Posición del punto más profundo: {puntos[idx_punto_mas_profundo]}")

    # Cambiar el color del punto más profundo a rosa
    rosa = [0, 0, 1]  # Color rosa (RGB)
    pcd.colors[idx_punto_mas_profundo] = rosa

    # Segmentar el entorno cercano al punto más profundo
    radio = 0.2 # Ajusta según tus necesidades
    pcd_cercano = segmentar_entorno(pcd, puntos[idx_punto_mas_profundo], radio)

    # Mostrar la sub-nube de puntos en una ventana
    o3d.visualization.draw_geometries([pcd_cercano], window_name="Entorno del punto más profundo", width=800, height=600)

    # Crear una ventana para visualizar toda la nube de puntos
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # Configurar la vista para centrarse en la nube de puntos
    vista = vis.get_view_control()
    vista.set_lookat(pcd.get_center())  # Enfocar la vista en el centro de la nube de puntos
    vista.set_up([0, -1, 0])            # Asegurar que el eje Y esté hacia arriba (ajusta según tu preferencia)

    vis.run()  # Iniciar la visualización
    vis.destroy_window()  # Cerrar la ventana cuando termines
