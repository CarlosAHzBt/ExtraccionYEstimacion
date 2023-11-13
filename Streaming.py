import pyrealsense2 as rs
import time

# Crear un objeto de configuración
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)  # Ajustar FPS a 60
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 60)  # Ajustar FPS a 60

# Habilitar la grabación a un archivo .bag
cfg.enable_record_to_file('grabacion.bag')

# Crear un objeto pipeline
pipeline = rs.pipeline()

# Configurar el perfil de streaming

# Iniciar la pipeline con la configuración
profile = pipeline.start(cfg)

# Obtener los dispositivos y sensores
device = profile.get_device()
depth_sensor = device.first_depth_sensor()

# Configurar la autoexposición
depth_sensor.set_option(rs.option.enable_auto_exposure, 1)

# Configurar la decimación
#decimation = rs.decimation_filter()
#decimation.set_option(rs.option.filter_magnitude, 2)  # Ajustar decimación a 2

# Grabar durante un tiempo determinado
time.sleep(10)  # Grabar durante 10 segundos

# Detener la grabación
pipeline.stop()
