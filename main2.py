import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import time
import logging
import traceback
import glob
import csv
import datetime
import shutil
import stat
import subprocess
import json

# Ensure stdout/stderr use UTF-8 encoding for unicode support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Importaciones para Greengrass
try:
    from awsiot.greengrasscoreipc.clientv2 import GreengrassCoreIPCClientV2
    from awsiot.greengrasscoreipc.model import SubscriptionResponseMessage, ServiceError, PublishMessage, JsonMessage, ServiceError
    greengrass_available = True
except ImportError:
    greengrass_available = False
    print("Modo sin Greengrass: La biblioteca de Greengrass Core IPC no está disponible")
# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Importar funciones de detección y análisis
from models.vertex_detector import VertexDetector
from models.defect_detector import DefectDetector
from common.zone_generator import visualize_zones
from common.defect_classifier import classify_defects_with_masks, visualize_results_with_masks
from utils.utils import order_points
import cv2
import numpy as np
from utils.utils import draw_arrow

# Importar procesadores de defectos específicos
from defects.diagonal_crack.processor import DiagonalCrackProcessor
from defects.midway_crack.processor import MidwayCrackProcessor
from defects.corner_crack.processor import CornerCrackProcessor
from defects.nucleo_esponjoso.processor import NucleoEsponjosoProcessor
from defects.inclusion_no_metalica.processor import InclusionNoMetalicaProcessor
from defects.rechupe.processor import RechupeProcessor
from defects.estrella.processor import EstrellaProcessor
from defects.sopladura.processor import SopladuraProcessor
from defects.abombamiento.processor import AbombamientoProcessor
from defects.romboidad.processor import RomboidadProcessor
from defects.etiqueta.label_extractor import LabelExtractor

# Configuración y variables de entorno para Greengrass
topic_read = os.getenv("TOPIC_READ", "PACC/Imagenes")
topic_write = os.getenv("TOPIC_WRITE", "PACC/Resultados")
logging_level = os.getenv("LOG_LEVEL", "INFO")
folder_output_ignorados = os.getenv("FOLDER_OUTPUT_IGNORADOS", "Detección PACC - Archivos ignorados")
folder_output_procesados = os.getenv("FOLDER_OUTPUT_PROCESADOS", "Detección PACC - Archivos procesados")

def generar_mascara_alternativa(image, log_path=None):
    """
    Genera una máscara alternativa cuando la detección con el modelo falla
    utilizando técnicas de procesamiento de imagen tradicionales
    
    Args:
        image: Imagen original
        
    Returns:
        mask: Máscara binaria de la palanquilla
    """
    try:
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Aplicar umbral adaptativo
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Método de Otsu
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Combinar ambos métodos
        thresh_combined = cv2.bitwise_or(thresh, thresh_otsu)
        
        # Operaciones morfológicas para limpiar
        kernel = np.ones((5, 5), np.uint8)
        thresh_cleaned = cv2.morphologyEx(thresh_combined, cv2.MORPH_CLOSE, kernel)
        thresh_cleaned = cv2.morphologyEx(thresh_cleaned, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Seleccionar el contorno de mayor área
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            
            # Crear máscara con solo el contorno principal
            mask = np.zeros_like(thresh_cleaned)
            cv2.drawContours(mask, [max_contour], 0, 255, -1)
            
            return mask
        else:
            # Si no hay contornos, usar umbral simple
            _, simple_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            return simple_mask
            
    except Exception as e:
        if log_path:
            actualizar_log(log_path, f"Error en generación de máscara alternativa: {e}", "ERROR")
        else:
            print(f"Error en generación de máscara alternativa: {e}")
        # Crear una máscara que cubra toda la imagen
        h, w = image.shape[:2]
        return np.ones((h, w), dtype=np.uint8) * 255

def extraer_vertices_de_mascara(mask, log_path=None):
    """
    Extrae los vértices de la máscara de forma robusta
    
    Args:
        mask: Máscara binaria de la palanquilla
        
    Returns:
        vertices: Array con las coordenadas de los 4 vértices
    """
    try:
        if mask is None:
            return None
            
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Seleccionar el contorno de mayor área
        max_contour = max(contours, key=cv2.contourArea)
        
        # Probar diferentes valores de epsilon para la aproximación
        for eps_factor in [0.02, 0.01, 0.03, 0.04, 0.05]:
            # Aproximar el contorno a un polígono
            perimeter = cv2.arcLength(max_contour, True)
            epsilon = eps_factor * perimeter
            approx = cv2.approxPolyDP(max_contour, epsilon, True)
            
            # Si obtenemos 4 vértices, usar esta aproximación
            if len(approx) == 4:
                vertices = approx.reshape(-1, 2)
                # Ordenar los vértices: [top-left, top-right, bottom-right, bottom-left]
                return order_points(vertices)
            else:
                if log_path:
                    actualizar_log(log_path, "Error: No se pudieron extraer 4 vértices...", "WARNING")
                else:
                    print("Error: No se pudieron extraer 4 vértices...")
        # Si no conseguimos 4 vértices con ningún epsilon, usar rectángulo mínimo
        return extraer_vertices_rectangulo_minimo(mask)
        
    except Exception as e:
        if log_path:
            actualizar_log(log_path, f"Error al extraer vértices de la máscara: {e}", "ERROR")
        else:
            print(f"Error al extraer vértices de la máscara: {e}")
        return None

def extraer_vertices_rectangulo_minimo(mask):
    """
    Extrae los vértices usando el rectángulo mínimo cuando la aproximación poligonal falla
    
    Args:
        mask: Máscara binaria de la palanquilla
        
    Returns:
        vertices: Array con las coordenadas de los 4 vértices
    """
    try:
        if mask is None:
            return None
            
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Si no hay contornos, usar los bordes de la imagen
            h, w = mask.shape[:2]
            return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.int32)
            
        # Seleccionar el contorno de mayor área
        max_contour = max(contours, key=cv2.contourArea)
        
        # Usar rectángulo mínimo orientado
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Ordenar los vértices
        return order_points(box)
        
    except Exception as e:
        print(f"Error al extraer vértices con rectángulo mínimo: {e}")
        # Si falla, usar los bordes de la imagen
        h, w = mask.shape[:2]
        return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.int32)
    
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def force_move(src_path: str, dest_folder: str):
    # Asegurarnos de que el destino final incluya el nombre de archivo
    filename = os.path.basename(src_path)
    dest_path = os.path.join(dest_folder, filename)
    # -- 1) Si ya existe en destino, quitar sólo-lectura y eliminarlo
    if os.path.exists(dest_path):
        # En Windows chmod sólo afecta al atributo readonly,
        # así que quitamos readonly antes de borrar:
        os.chmod(dest_path, stat.S_IWRITE)
        os.remove(dest_path)
    # -- 2) Mover (shutil.move hace rename o copia+borra según el caso)
    shutil.move(src_path, dest_path)
    # -- 3) Asegurar atributos de archivo normales (quitar readonly)
    os.chmod(dest_path, stat.S_IWRITE | stat.S_IREAD)
    # -- 4) Conceder control total a "Everyone" vía icacls (ACL de Windows)
    try:
        subprocess.run(
            ["icacls", dest_path, "/grant", "Everyone:F", "/C"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except subprocess.CalledProcessError as e:
        # Si falla icacls, al menos informamos del error
        print(f"Advertencia: no se pudo ajustar ACL con icacls: {e.stderr}")

def publish_message(ipc_client, topic, message):
    try:
        json_msg = JsonMessage(message=message)
        pub_msg = PublishMessage(json_message=json_msg)
        ipc_client.publish_to_topic(topic=topic, publish_message=pub_msg)
        logging.info(f"Publicado en topic {topic} payload: {message}")
    except ServiceError as se:
        logging.error(f"ServiceError al publicar en {topic}: {se}")
    except Exception:
        logging.exception(f"Error al publicar el mensaje: {message}")

def actualizar_log(log_path, message, level="INFO"):
    """
    Actualiza el archivo de log con un mensaje y nivel específico
    
    Args:
        log_path: Ruta al archivo de logs (puede ser None)
        message: Mensaje a registrar
        level: Nivel del log (INFO, WARNING, ERROR)
    """
    # Si log_path es None, solo imprimir en consola
    if log_path is None:
        print(f"{level}: {message}")
        return
    
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"{level}: {message}")
    
    try:
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Separar por líneas y escribir en el archivo
        if isinstance(message, str):
            message = message.split("\n")
        elif not isinstance(message, (list, tuple)):
            message = [str(message)]
            
        with open(log_path, 'a', encoding='utf-8') as f:
            for msg in message:
                f.write(f"{timestamp} - {level} - {msg}\n")
    except Exception as e:
        print(f"ERROR: No se pudo escribir en el log {log_path}: {e}")

def on_stream_event(event: SubscriptionResponseMessage, models, ipc_client) -> None:
    try:
        if event.json_message and event.json_message.message is not None:
            message = event.json_message.message
            logging.info(f"Mensaje recibido: {message}")
            ruta = message["ruta"]
            filename = os.path.basename(ruta)
            basename = filename.split(".")[0]
            workdir = os.getcwd()
            os.makedirs(workdir, exist_ok=True)
            os.makedirs(os.path.join(workdir, basename), exist_ok=True)
            dest_folder = ruta.replace(os.path.basename(os.path.dirname(ruta)), folder_output_procesados).rsplit(".", 1)[0]
            log_path = os.path.join(os.path.join(workdir, basename), 'logs.log')
            try:
                input_data = input_fn(ruta, log_path)
                prediction_results = predict_fn(input_data, models, workdir, log_path)
                output_fn(prediction_results, workdir, input_data)
                defect_json = generate_json(os.path.join(workdir, basename))
                msg_excel = {
                    "ruta_archivos": os.path.join(workdir, basename),
                    "ruta_excel": os.path.dirname(log_path),
                    "resultados": defect_json
                }
                publish_message(ipc_client, topic_write, msg_excel)
                actualizar_log(log_path, f"Publicado en topic {topic_write} payload: {msg_excel}")
            except Exception as e:
                dest_folder = ruta.replace(os.path.basename(os.path.dirname(ruta)), folder_output_ignorados).rsplit(".", 1)[0]
                actualizar_log(log_path, f"Error ejecutando inferencia: {filename} - {type(e)} - {e}")
                actualizar_log(log_path, traceback.format_exc())
            os.makedirs(dest_folder, exist_ok=True)
            force_move(log_path, dest_folder)
            print(f"Archivo de log movido a: {dest_folder}")

        else:
            logging.warning("Mensaje recibido no es JSON o está vacío")
    except Exception:
        logging.exception("Error al procesar el mensaje recibido")

def on_stream_error(error: Exception) -> bool:
    logging.error("Error en la transmisión de la suscripción")
    logging.error(error)
    return False  # Return True to close stream, False to keep stream open.

def on_stream_closed() -> None:
    logging.info("Transmisión de suscripción cerrada")
    
def model_fn(model_dir=None):
    """
    Carga los modelos necesarios para la detección de vértices y defectos
    
    Args:
        model_dir: Directorio donde se encuentran los modelos (opcional)
        
    Returns:
        Dictionary con los modelos cargados
    """
    
    # Rutas predeterminadas si no se especifica un directorio
    vertex_model_path = r"D:\Trabajo modelos\PACC\YOLOv12 - copia\Models\Vertex\model.pt"
    defect_model_path = r"D:\Trabajo modelos\PACC\YOLOv12 - copia\Models\Defect\model.pt"
    
    if model_dir:   
        # Si se proporciona un directorio, buscar los modelos allí
        vertex_path = os.path.join(model_dir, "modelo_1.pt")
        defect_path = os.path.join(model_dir, "modelo_2.pt")

        if os.path.exists(vertex_path):
            vertex_model_path = vertex_path
        if os.path.exists(defect_path):
            defect_model_path = defect_path
    
    # Inicializar los detectores
    print("Cargando modelo para detección de vértices/contornos...")
    vertex_detector = VertexDetector(vertex_model_path)
    
    print("Cargando modelo para detección de defectos...")
    defect_detector = DefectDetector(defect_model_path)
    
    # Inicializar los procesadores de defectos
    diagonal_processor = DiagonalCrackProcessor()
    midway_processor = MidwayCrackProcessor()
    corner_processor = CornerCrackProcessor()
    nucleo_processor = NucleoEsponjosoProcessor()
    inclusion_processor = InclusionNoMetalicaProcessor()
    rechupe_processor = RechupeProcessor()
    estrella_processor = EstrellaProcessor()
    sopladura_processor = SopladuraProcessor()
    abombamiento_processor = AbombamientoProcessor()
    romboidad_processor = RomboidadProcessor()
    
    # Nuevo: inicializar el procesador IMF
    imf_processor = IMFProcessor()
    
    # Nuevo: Inicializar el procesador de etiquetas
    print("Inicializando procesador de etiquetas...")
    label_extractor = LabelExtractor()
    
    # Verificar que la inicialización fue correcta
    try:
        # Test if OCR model is available
        print("Verificando disponibilidad de modelo OCR...")
        if hasattr(label_extractor, 'model_name'):
            print(f"Modelo OCR configurado: {label_extractor.model_name}")
        else:
            print("Advertencia: Modelo OCR no configurado correctamente")
    except Exception as e:
        print(f"Error inicializando extractor de etiquetas: {e}")
    
    return {
        'vertex_detector': vertex_detector,  # Para detectar contornos y vértices
        'defect_detector': defect_detector,  # Para detectar defectos
        'processors': {
            'grietas_diagonales': diagonal_processor,
            'grietas_medio_camino': midway_processor,
            'grietas_corner': corner_processor,
            'nucleo_esponjoso': nucleo_processor,
            'inclusion_no_metalica': inclusion_processor,
            'rechupe': rechupe_processor,
            'estrella': estrella_processor,
            'sopladura': sopladura_processor,
            'abombamiento': abombamiento_processor,
            'romboidad': romboidad_processor,
            'etiqueta': label_extractor,
            'imf': imf_processor  # Añadir el procesador IMF
        }
    }


def input_fn(image_path, log_path=None):
    """
    Procesa la entrada (ruta de imagen) y la prepara para el análisis
    
    Args:
        image_path: Ruta a la imagen a procesar
        log_path: Ruta al archivo de log (opcional)
        
    Returns:
        Dictionary con la información de entrada
    """
    # Verificar que la imagen existe
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"La imagen {image_path} no existe")
    
    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen desde {image_path}")
    
    # Obtener información básica de la imagen
    basename = os.path.basename(image_path)
    name, ext = os.path.splitext(basename)
    
    if log_path:
        actualizar_log(log_path, "Imagen leída exitosamente!")
    
    return {
        'image': image,
        'path': image_path,
        'basename': basename,
        'name': name,
        'ext': ext
    }

def predict_fn(input_data, models, output_dir=None, log_path=None):
    """
    Realiza la detección de vértices y defectos, y los clasifica
    
    Args:
        input_data: Diccionario con los datos de entrada
        models: Diccionario con los modelos cargados
        output_dir: Directorio donde guardar resultados
        
    Returns:
        Dictionary con los resultados del análisis
    """
    # Extraer los datos necesarios
    original_image = input_data['image'].copy()
    image = input_data['image']
    image_path = input_data['path']
    image_name = input_data['name']
    
    # Extraer los modelos
    vertex_detector = models['vertex_detector']
    defect_detector = models['defect_detector']
    
    # Verificar que el extractor de etiquetas esté disponible
    if 'etiqueta' not in models['processors']:
        print("Warning: Label extractor not initialized. Label orientation features will be disabled.")
        models['processors']['etiqueta'] = LabelExtractor()
    
    # 1. Pre-procesamiento de la imagen (opcional)
    # Redimensionar si la imagen es muy grande
    h, w = image.shape[:2]
    if h > 3000 or w > 3000:
        max_dim = 2000
        scale_factor = max_dim / max(h, w)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        print(f"Redimensionando imagen de {w}x{h} a {new_w}x{new_h}")
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 2. MEJORA: Obtener una máscara robusta de la palanquilla
    if log_path:
        actualizar_log(log_path, f"Generando máscara robusta de la palanquilla...")
    else:
        print(f"Generando máscara robusta de la palanquilla...")
    try:
        # Primero obtenemos resultados usando el modelo de detección
        vertex_result = vertex_detector.model.predict(
            image, 
            conf=vertex_detector.conf_threshold,
            device=vertex_detector.device
        )[0]
        
        # Verificar si hay máscaras en el resultado
        if hasattr(vertex_result, 'masks') and vertex_result.masks is not None:
            # Buscar la clase "palanquilla" para identificar la máscara correcta
            palanquilla_mask = None
            palanquilla_class_id = None
            
            # Encontrar el ID de la clase "palanquilla"
            if hasattr(vertex_detector.model, "names"):
                class_names = vertex_detector.model.names
                for id, name in class_names.items():
                    if isinstance(name, str) and (name.lower() == "palanquilla" or name.lower() == "class_1"):
                        palanquilla_class_id = id                        
                        if log_path:
                            actualizar_log(log_path, f"ID de clase para 'palanquilla' encontrado: {palanquilla_class_id}")
                        else:
                            print(f"ID de clase para 'palanquilla' encontrado: {palanquilla_class_id}")
                        break
            
            # Si encontramos el ID de clase, buscar la máscara correspondiente
            if palanquilla_class_id is not None:
                # Extraer las clases y confianzas
                boxes = vertex_result.boxes
                masks = vertex_result.masks
                
                # Buscar la máscara con mayor área para la clase palanquilla
                max_area = 0
                second_max_area = 0
                palanquilla_mask = None
                second_palanquilla_mask = None

                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    if cls_id == palanquilla_class_id:
                        # Obtener la máscara para esta detección
                        mask_data = masks[i].data.cpu().numpy()
                        mask = (mask_data > 0.98).astype(np.uint8) * 255
                        
                        # Asegurarse de que sea 2D
                        if len(mask.shape) > 2:
                            mask = mask[0]
                            
                        # Calcular el área
                        area = np.count_nonzero(mask)
                        
                        # Actualizar las máscaras según su tamaño
                        if area > max_area:
                            # La actual pasa a ser la más grande
                            # La que era más grande pasa a ser la segunda
                            second_max_area = max_area
                            second_palanquilla_mask = palanquilla_mask
                            max_area = area
                            palanquilla_mask = mask
                        elif area > second_max_area:
                            # La actual es mayor que la segunda pero menor que la más grande
                            second_max_area = area
                            second_palanquilla_mask = mask

                # Al final del bucle, usar second_palanquilla_mask en lugar de palanquilla_mask
                palanquilla_mask = second_palanquilla_mask
                            
                # Si no encontramos ninguna máscara, usar método alternativo
                if palanquilla_mask is None:
                    print("No se encontró máscara para la clase palanquilla")
                    palanquilla_mask = generar_mascara_alternativa(image)
            else:
                # Si no encontramos ID de clase, usar método alternativo
                print("No se encontró ID de clase para palanquilla")
                palanquilla_mask = generar_mascara_alternativa(image)
        else:
            # Si no hay máscaras, usar método alternativo
            print("No hay máscaras disponibles en el resultado")
            palanquilla_mask = generar_mascara_alternativa(image)
            
        # Verificar que la máscara tenga el tamaño correcto
        if palanquilla_mask is not None and palanquilla_mask.shape != (image.shape[0], image.shape[1]):
            print(f"Redimensionando máscara de {palanquilla_mask.shape} a {image.shape[:2]}")
            palanquilla_mask = cv2.resize(palanquilla_mask, (image.shape[1], image.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
            
        # Aplicar operaciones morfológicas para limpiar la máscara
        if palanquilla_mask is not None:
            kernel = np.ones((5, 5), np.uint8)
            palanquilla_mask = cv2.morphologyEx(palanquilla_mask, cv2.MORPH_CLOSE, kernel)
            palanquilla_mask = cv2.morphologyEx(palanquilla_mask, cv2.MORPH_OPEN, kernel)
            
    except Exception as e:
        # Para errores:
        if log_path:
            actualizar_log(log_path, f"Error al generar máscara robusta: {e}", "ERROR")
            actualizar_log(log_path, traceback.format_exc(), "ERROR")
        else:
            print(f"Error al generar máscara robusta: {e}")
        traceback.print_exc()
        # Crear una máscara por defecto
        palanquilla_mask = generar_mascara_alternativa(image)
    
    # 3. MEJORA: Extraer vértices directamente de la máscara
    print("Extrayendo vértices de la máscara...")
    vertices = extraer_vertices_de_mascara(palanquilla_mask)
    
    # Verificar que tengamos 4 vértices válidos
    if vertices is None or len(vertices) != 4:
        print("Error: No se pudieron extraer 4 vértices de la máscara. Usando método alternativo.")
        vertices = extraer_vertices_rectangulo_minimo(palanquilla_mask)
    
    # Verificar y corregir vértices si es necesario
    h, w = image.shape[:2]
    
    # Corrección de vértices fuera de los límites
    for i in range(len(vertices)):
        vertices[i][0] = max(0, min(w-1, vertices[i][0]))
        vertices[i][1] = max(0, min(h-1, vertices[i][1]))
    
    # 4. Guardar visualización de diagnóstico
    if output_dir:
        debug_dir = os.path.join(output_dir, image_name)
        os.makedirs(debug_dir, exist_ok=True)
    
    # MODIFICADO: NUEVO PUNTO 4.5 - Procesar abombamiento ANTES de la rotación
    print("Procesando abombamiento con imagen original (antes de rotación)...")
    try:
        # Obtener contorno principal de la máscara para el análisis de abombamiento
        contornos, _ = cv2.findContours(palanquilla_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contorno_principal = max(contornos, key=cv2.contourArea) if contornos else None
        
        # Realizar análisis de abombamiento
        abombamiento_processor = models['processors']['abombamiento']
        abombamiento_results_original = abombamiento_processor.process(
            image,  # Imagen original antes de rotar
            vertices,  # Vértices originales
            image_name=image_name,
            output_dir=output_dir,
            model=vertex_detector.model,
            conf_threshold=0.35,
            mask=palanquilla_mask
        )
        
        # Guardar los resultados para usarlos después
        abombamiento_data_original = abombamiento_results_original.get('processed_data', {})
        abombamiento_viz_original = abombamiento_results_original.get('visualizations', {})
        abombamiento_reports_original = abombamiento_results_original.get('report_paths', None)
        
        print(f"Abombamiento analizado exitosamente con la imagen original")
    except Exception as e:
        print(f"Error al procesar abombamiento con imagen original: {e}")
        import traceback
        traceback.print_exc()
        # Crear valores por defecto en caso de error
        abombamiento_data_original = {
            'lado_max_abombamiento': "Lado 1 (Top)",
            'max_abombamiento_porcentaje': 0.0,
            'max_abombamiento_pixeles': 0.0
        }
        abombamiento_viz_original = {}
        abombamiento_reports_original = None
    
    # 5. NUEVA SECCIÓN: ALINEAMIENTO DE LA PALANQUILLA
    print("Realizando alineamiento de la palanquilla...")
    try:
        # Obtener contorno principal de la máscara para el alineamiento
        contornos, _ = cv2.findContours(palanquilla_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contornos:
            contorno_principal = max(contornos, key=cv2.contourArea)
            
            # Detectar etiquetas para determinar orientación
            print("Detectando etiquetas para orientación...")
            etiqueta_detections = []
            try:
                vertex_result = vertex_detector.model.predict(image, conf=vertex_detector.conf_threshold, device=vertex_detector.device)[0]
                
                if hasattr(vertex_detector.model, "names") and vertex_result.boxes is not None:
                    class_names = vertex_detector.model.names
                    etiqueta_class_id = None
                    
                    # MEJORA: Imprimir todas las clases disponibles para diagnóstico
                    print(f"Clases disponibles en el modelo de vértices: {class_names}")
                    for id, name in class_names.items():
                        print(f"Clase ID {id}: {name}")
                    
                    # MODIFICACIÓN: Buscar explícitamente la clase etiqueta
                    for id, name in class_names.items():
                        if isinstance(name, str) and (name.lower() == "etiqueta" or 
                                                     name.lower() == "class_0" or 
                                                     name == "0"):
                            etiqueta_class_id = id
                            print(f"ID de clase para 'etiqueta' encontrado: {etiqueta_class_id}, nombre: {name}")
                            break
                    
                    # Si no encontramos "etiqueta", asumir que es la clase 0
                    if etiqueta_class_id is None and 0 in class_names:
                        etiqueta_class_id = 0
                        print(f"Asumiendo que la clase 0 es etiqueta: {class_names[0]}")
                    
                    if etiqueta_class_id is not None:
                        boxes = vertex_result.boxes
                        # Imprimir todas las detecciones para diagnóstico
                        print(f"Total de detecciones: {len(boxes)}")
                        for i, box in enumerate(boxes):
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            print(f"Detección #{i+1}: Clase {cls_id}, Confianza {conf:.2f}")
                            
                            if cls_id == etiqueta_class_id:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                
                                print(f"ETIQUETA detectada con confianza {conf:.2f} en bbox: ({x1}, {y1}, {x2}, {y2})")
                                
                                etiqueta_detections.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'conf': conf,
                                    'class': 'etiqueta',
                                    'cls_id': cls_id
                                })
            except Exception as e:
                print(f"Error al buscar etiquetas: {e}")
                import traceback
                traceback.print_exc()
            
            # Determinar el lado para rotación (basado en etiqueta o abombamiento)
            if etiqueta_detections:
                print("Determinando lado de rotación basado en etiqueta...")
                # Función para obtener centroide de la etiqueta
                def obtener_centroide_etiqueta(image, mask_img, bbox):
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    return (center_x, center_y)
                
                # Función para calcular la distancia a un segmento
                def distancia_a_segmento(p1, p2, punto):
                    p1, p2, punto = np.array(p1, dtype=float), np.array(p2, dtype=float), np.array(punto, dtype=float)
                    v = p2 - p1
                    u = punto - p1
                    t = np.dot(u, v) / np.dot(v, v) if np.dot(v, v) != 0 else 0
                    if t < 0:
                        return np.linalg.norm(punto - p1)
                    elif t > 1:
                        return np.linalg.norm(punto - p2)
                    else:
                        return np.abs(np.cross(v, u)) / np.linalg.norm(v)
                
                # Determinar lado más cercano a la etiqueta
                etiqueta_bbox = etiqueta_detections[0]['bbox']
                centroide = obtener_centroide_etiqueta(image, palanquilla_mask, etiqueta_bbox)
                
                lados = [
                    (vertices[0], vertices[1], "Lado 1 (Top)"),
                    (vertices[1], vertices[2], "Lado 2 (Right)"),
                    (vertices[2], vertices[3], "Lado 3 (Bottom)"),
                    (vertices[3], vertices[0], "Lado 4 (Left)")
                ]
                
                distancias = {}
                for a, b, nombre in lados:
                    d = distancia_a_segmento(a, b, centroide)
                    distancias[nombre] = d
                    print(f"Distancia al {nombre}: {d:.2f}")
                
                # Se obtiene el lado con la distancia mínima
                lado_rotacion = min(distancias, key=distancias.get)
                print("\nEl centroide está más cercano a:", lado_rotacion)
            else:
                print("No se encontró etiqueta, determinando lado basado en abombamiento...")
                # Función para calcular distancia de un punto a una línea
                def distancia_punto_a_linea(p1, p2, punto):
                    p1, p2, punto = np.array(p1, dtype=float), np.array(p2, dtype=float), np.array(punto, dtype=float)
                    return np.abs(np.cross(p2 - p1, punto - p1)) / np.linalg.norm(p2 - p1) if np.linalg.norm(p2 - p1) != 0 else 0
                
                # Definir los lados
                lados = [
                    (vertices[0], vertices[1], "Lado 1 (Top)"),
                    (vertices[1], vertices[2], "Lado 2 (Right)"),
                    (vertices[2], vertices[3], "Lado 3 (Bottom)"),
                    (vertices[3], vertices[0], "Lado 4 (Left)"),
                ]
                
                # Extraer los puntos del contorno
                contorno_pts = contorno_principal.reshape(-1, 2)
                puntos_por_lado = [[] for _ in range(4)]
                
                # Asignar cada punto del contorno al lado más cercano
                for punto in contorno_pts:
                    distancias = [distancia_a_segmento(p1, p2, punto) for p1, p2, _ in lados]
                    indice_min = np.argmin(distancias)
                    puntos_por_lado[indice_min].append(punto)
                
                # Calcular el abombamiento para cada lado
                abombamientos = []
                for i, (p1, p2, nombre) in enumerate(lados):
                    puntos_lado = np.array(puntos_por_lado[i]) if puntos_por_lado[i] else np.array([])
                    if len(puntos_lado) > 0:
                        distancias = np.array([distancia_punto_a_linea(p1, p2, pt) for pt in puntos_lado])
                        abombamiento = np.max(distancias) if len(distancias) > 0 else 0
                    else:
                        abombamiento = 0
                    abombamientos.append(abombamiento)
                    print(f"{nombre}: abombamiento = {abombamiento:.2f}")
                
                # Se determina el lado más recto (el de menor abombamiento)
                indice_recto = np.argmin(abombamientos)
                lado_rotacion = lados[indice_recto][2]
                print("El lado más recto es:", lado_rotacion)
            
            # Realizar la rotación de la imagen
            print(f"Rotando imagen según {lado_rotacion}...")
            
            # Definir los ángulos base de rotación para cada lado
            angulos_rotacion_base = {
                "Lado 1 (Top)": 180, 
                "Lado 2 (Right)": 270, 
                "Lado 3 (Bottom)": 180, 
                "Lado 4 (Left)": 270
            }
            
            # Calcular los ángulos de cada lado
            angles = {}
            # Lado 1 (Top): ángulo entre el lado (punto0 -> punto1) y la horizontal (eje X)
            vec_top = vertices[1] - vertices[0]
            angle_top = np.degrees(np.arctan2(float(vec_top[1]), float(vec_top[0])))
            angles["Lado 1 (Top)"] = angle_top
            
            # Lado 2 (Right): ángulo entre el lado (punto1 -> punto2) y la vertical (eje Y)
            vec_right = vertices[2] - vertices[1]
            angle_right = np.degrees(np.arctan2(float(vec_right[0]), float(vec_right[1])))
            angles["Lado 2 (Right)"] = angle_right
            
            # Lado 3 (Bottom): ángulo entre el lado (punto2 -> punto3) y la horizontal (eje X)
            vec_bottom = vertices[3] - vertices[2]
            angle_bottom = np.degrees(np.arctan2(float(vec_bottom[1]), float(vec_bottom[0])))
            angles["Lado 3 (Bottom)"] = angle_bottom
            
            # Lado 4 (Left): ángulo entre el lado (punto3 -> punto0) y la vertical (eje Y)
            vec_left = vertices[0] - vertices[3]
            angle_left = np.degrees(np.arctan2(float(vec_left[0]), float(vec_left[1])))
            angles["Lado 4 (Left)"] = angle_left
            
            # Mostrar los ángulos calculados
            for lado, ang in angles.items():
                print(f"{lado}: {ang:.2f} grados")
            
            # Calcular ángulo de rotación
            angulo_lado = angles[lado_rotacion]
            rotation_angle = -angulo_lado + angulos_rotacion_base[lado_rotacion]
            print(f"Ángulo de rotación necesario: {rotation_angle}°")
            
            # Rotar la imagen
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)
            
            # Rotar también la máscara
            rotated_mask = cv2.warpAffine(palanquilla_mask, rotation_matrix, (w, h), 
                                         flags=cv2.INTER_NEAREST, 
                                         borderMode=cv2.BORDER_CONSTANT, 
                                         borderValue=0)
            
            # Rotar los vértices
            vertices_homog = np.ones((len(vertices), 3))
            vertices_homog[:, :2] = vertices
            
            rotated_vertices = np.zeros((len(vertices), 2), dtype=np.int32)
            for i, vertex in enumerate(vertices_homog):
                x = rotation_matrix[0, 0] * vertex[0] + rotation_matrix[0, 1] * vertex[1] + rotation_matrix[0, 2]
                y = rotation_matrix[1, 0] * vertex[0] + rotation_matrix[1, 1] * vertex[1] + rotation_matrix[1, 2]
                rotated_vertices[i] = [int(x), int(y)]
            
            # Actualizar las variables para usar las versiones rotadas
            image = rotated_image
            palanquilla_mask = rotated_mask
            vertices = rotated_vertices
            
            # MODIFICACIÓN: Actualizar las coordenadas de las etiquetas después de la rotación
            if etiqueta_detections:
                print("Actualizando coordenadas de etiquetas después de la rotación...")
                rotated_etiquetas = []
                for etiqueta in etiqueta_detections:
                    x1, y1, x2, y2 = etiqueta['bbox']
                    
                    # Convertir las esquinas del bbox a coordenadas homogéneas
                    corners = np.array([
                        [x1, y1, 1],
                        [x2, y1, 1],
                        [x1, y2, 1],
                        [x2, y2, 1]
                    ])
                    
                    # Aplicar la matriz de rotación a cada esquina
                    rotated_corners = np.zeros((4, 2), dtype=np.int32)
                    for i, corner in enumerate(corners):
                        x = rotation_matrix[0, 0] * corner[0] + rotation_matrix[0, 1] * corner[1] + rotation_matrix[0, 2]
                        y = rotation_matrix[1, 0] * corner[0] + rotation_matrix[1, 1] * corner[1] + rotation_matrix[1, 2]
                        rotated_corners[i] = [int(x), int(y)]
                    
                    # Calcular el nuevo bounding box alineado con los ejes
                    x_min = np.min(rotated_corners[:, 0])
                    y_min = np.min(rotated_corners[:, 1])
                    x_max = np.max(rotated_corners[:, 0])
                    y_max = np.max(rotated_corners[:, 1])
                    
                    # Asegurar que las coordenadas estén dentro de los límites de la imagen
                    x_min = max(0, min(w-1, x_min))
                    y_min = max(0, min(h-1, y_min))
                    x_max = max(0, min(w-1, x_max))
                    y_max = max(0, min(h-1, y_max))
                    
                    # Crear la nueva etiqueta rotada
                    rotated_etiqueta = etiqueta.copy()
                    rotated_etiqueta['bbox'] = (x_min, y_min, x_max, y_max)
                    rotated_etiquetas.append(rotated_etiqueta)
                
                # Reemplazar las etiquetas originales con las rotadas
                etiqueta_detections = rotated_etiquetas
                
                # Debug - guardar imagen de las etiquetas rotadas
                if output_dir:
                    debug_etiquetas_img = rotated_image.copy()
                    for i, etiqueta in enumerate(etiqueta_detections):
                        x1, y1, x2, y2 = etiqueta['bbox']
                        cv2.rectangle(debug_etiquetas_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(debug_etiquetas_img, f"Etiqueta {i+1}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Extraer y guardar las imágenes de las etiquetas rotadas para verificación
                    etiq_dir = os.path.join(debug_dir, "etiqueta")
                    os.makedirs(etiq_dir, exist_ok=True)
                    for i, etiqueta in enumerate(etiqueta_detections):
                        x1, y1, x2, y2 = etiqueta['bbox']
                        etiqueta_img = rotated_image[y1:y2, x1:x2].copy()
                        cv2.imwrite(os.path.join(etiq_dir, f"{image_name}_etiqueta_{i+1}_rotada.jpg"), etiqueta_img)
            
            # Guardar información de rotación
            rotacion_info = {
                'angulo': rotation_angle,
                'lado_rotacion': lado_rotacion,
                'lado_etiqueta': lado_rotacion if etiqueta_detections else None,
                'etiqueta_bbox': etiqueta_detections[0]['bbox'] if etiqueta_detections else None
            }
            
        else:
            print("Error: No se encontraron contornos para el alineamiento")
            rotacion_info = {
                'angulo': 0,
                'lado_rotacion': None,
                'lado_etiqueta': None,
                'etiqueta_bbox': None
            }
    except Exception as e:
        print(f"Error en el proceso de alineamiento: {e}")
        import traceback
        traceback.print_exc()
        rotacion_info = {
            'angulo': 0,
            'lado_rotacion': None,
            'lado_etiqueta': None,
            'etiqueta_bbox': None
        }
    
    # 6. Generar máscaras de zona con los vértices detectados (ahora usando los vértices rotados)
    print(f"Generando máscaras de zonas")
    zones_img, zone_masks = visualize_zones(image, vertices)
    
    # 7. Detectar defectos con el detector de defectos
    print(f"Detectando defectos")
    # MODIFICACIÓN: Usar la imagen rotada para la detección de defectos
    detections, yolo_result = defect_detector.detect_defects(image)  # imagen ya está rotada
    
    if not detections:
        print("No se detectaron defectos en esta imagen.")
    
    # 8. Código para el mapeo de clases
    class_mapping = {}
    if defect_detector.class_names:
        print("Creando mapeo de clases basado en modelo de defectos:")
        print(f"Clases en el modelo: {defect_detector.class_names}")
        
        for idx, name in enumerate(defect_detector.class_names):
            name_lower = name.lower()
            if 'grieta' in name_lower:
                class_mapping[name] = 'grieta'
                print(f"  - '{name}' mapeado a 'grieta'")
            elif 'punto' in name_lower:
                class_mapping[name] = 'puntos'
                print(f"  - '{name}' mapeado a 'puntos'")
            elif 'rechup' in name_lower:
                class_mapping[name] = 'rechupe'
                print(f"  - '{name}' mapeado a 'rechupe'")
            elif 'sopladura' in name_lower:
                class_mapping[name] = 'sopladura'
                print(f"  - '{name}' mapeado a 'sopladura'")
            elif 'estrella' in name_lower:
                class_mapping[name] = 'estrella'
                print(f"  - '{name}' mapeado a 'estrella'")
            else:
                class_mapping[name] = name
                print(f"  - '{name}' mantenido como '{name}'")
    
    # 9. Clasificar los defectos según su posición en las zonas
    # MODIFICACIÓN: Usar la imagen rotada para la clasificación de defectos
    classified_detections = classify_defects_with_masks(detections, zone_masks, image, yolo_result, class_mapping)
    
    # 10. Procesar romboidad (el abombamiento ya se procesó antes de la rotación)
    # MODIFICACIÓN: No volver a procesar abombamiento, usar los resultados originales
    romboidad_processor = models['processors']['romboidad']
    romboidad_results = romboidad_processor.process(
        image,  # imagen ya está rotada
        vertices,  # vértices ya están rotados
        image_name=image_name,
        output_dir=output_dir
    )
    
    # 11. Procesar cada tipo de defecto
    results = {}
    
    # MODIFICADO: Usar los resultados de abombamiento obtenidos ANTES de la rotación
    print("Usando resultados de abombamiento pre-rotación...")
    results['abombamiento'] = {
        'processed_data': abombamiento_data_original,
        'visualizations': abombamiento_viz_original,
        'report_paths': abombamiento_reports_original
    }
    
    results['romboidad'] = romboidad_results
    
    # MODIFICACIÓN: Procesar todos los tipos de defectos con la imagen rotada
    for defect_type, defects in classified_detections.items():
        if defects and defect_type in models['processors']:
            # Corregimos el bug aquí: usamos defect_type directamente
            processor = models['processors'][defect_type]
            results[defect_type] = processor.process(
                defects, 
                image,  # imagen ya está rotada
                vertices,  # vértices ya están rotados
                zone_masks,
                image_name=image_name,
                output_dir=output_dir
            )
    
    # 12. Procesar etiquetas si se detectaron
    if etiqueta_detections and 'etiqueta' in models['processors']:
        print(f"Procesando {len(etiqueta_detections)} etiqueta(s) con OCR...")
        label_extractor = models['processors']['etiqueta']
        label_results = label_extractor.process(
            etiqueta_detections,
            image,  # imagen ya está rotada
            corners=vertices,  # vértices ya están rotados
            zone_masks=zone_masks,
            image_name=image_name,
            output_dir=output_dir
        )
        
        results['etiqueta'] = label_results
    else:
        print("No se detectaron etiquetas en la imagen o no está disponible el procesador de etiquetas.")
    
    return {
        'vertices': vertices,
        'zones_img': zones_img,
        'zone_masks': zone_masks,
        'detections': detections,
        'etiqueta_detections': etiqueta_detections,
        'yolo_result': yolo_result,
        'classified_detections': classified_detections,
        'processed_results': results,
        'palanquilla_mask': palanquilla_mask,
        'image_procesada': image,  # Esta es la imagen rotada
        'rotacion_info': rotacion_info,
        'original_image': original_image
    }

def draw_all_defects_on_original_image(original_image, defects, processed_results, defect_type, corners):
    """
    Dibuja todos los defectos del mismo tipo en la imagen original con sus análisis
    
    Args:
        original_image: Imagen original donde dibujar los defectos
        defects: Lista de defectos del mismo tipo
        processed_results: Resultados procesados con métricas
        defect_type: Tipo de defecto
        corners: Esquinas de la palanquilla
        
    Returns:
        Imagen original con todos los defectos visualizados
    """
    # Crear copia de la imagen original
    result_img = original_image.copy()
    
    # Dibujar el contorno verde de la palanquilla
    if corners is not None and len(corners) == 4:
        cv2.polylines(result_img, [np.array(corners)], True, (0, 255, 0), 2)
    
    # Título para la imagen
    title = defect_type.replace('_', ' ').title()
    cv2.putText(result_img, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(result_img, f"Total: {len(defects)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Procesar cada defecto según su tipo específico
    for i, defect in enumerate(processed_results):
        # Obtener bbox y dibujar rectángulo
        bbox = defects[i]['bbox']
        x1, y1, x2, y2 = bbox
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Dibujar número de defecto
        cv2.putText(result_img, f"#{i+1}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Dibujar métricas y vectores según el tipo de defecto
        if defect_type.startswith('grietas'):
            # Para grietas (diagonales, corner, medio_camino)
            L = defect.get('L', 0)
            e = defect.get('e', 0)
            D = defect.get('D', 0)
            
            # Dibujar puntos extremos y vector para longitud L
            if 'global_pt1' in defect and 'global_pt2' in defect:
                global_pt1 = defect['global_pt1']
                global_pt2 = defect['global_pt2']
                
                if isinstance(global_pt1, tuple) and isinstance(global_pt2, tuple):
                    # Convertir a enteros si no lo son
                    global_pt1 = (int(global_pt1[0]), int(global_pt1[1]))
                    global_pt2 = (int(global_pt2[0]), int(global_pt2[1]))
                    
                    # Dibujar puntos extremos en azul
                    cv2.circle(result_img, global_pt1, 5, (255, 0, 0), -1)
                    cv2.circle(result_img, global_pt2, 5, (255, 0, 0), -1)
                    
                    # Dibujar flecha L (amarilla) entre puntos extremos
                    draw_arrow(result_img, global_pt1, global_pt2, (0, 255, 255), 2, 10, f"L={L:.1f}px")
            
            # Dibujar puntos y vector para distancia D
            if 'contour_point' in defect and 'edge_point' in defect:
                contour_point = defect['contour_point']
                edge_point = defect['edge_point']
                
                if isinstance(contour_point, tuple) and isinstance(edge_point, tuple):
                    # Convertir a enteros si no lo son
                    contour_point = (int(contour_point[0]), int(contour_point[1]))
                    edge_point = (int(edge_point[0]), int(edge_point[1]))
                    
                    # Dibujar punto del contorno en rojo
                    cv2.circle(result_img, contour_point, 5, (0, 0, 255), -1)
                    
                    # Dibujar punto del borde en amarillo
                    cv2.circle(result_img, edge_point, 5, (255, 255, 0), -1)
                    
                    # Dibujar flecha D (naranja) desde el punto del contorno hasta el borde
                    draw_arrow(result_img, contour_point, edge_point, (0, 165, 255), 2, 10, f"D={D:.1f}px")
            
            # Información adicional en cuadro de texto
            text_x = x1
            text_y = y2 + 20
            cv2.putText(result_img, f"L={L:.1f}px, e={e:.1f}px, D={D:.1f}px", 
                      (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        elif defect_type == 'estrella' or defect_type == 'rechupe':
            # Para estrellas y rechupes
            diametro = defect.get('diametro', 0)
            
            # Dibujar puntos extremos y vector para diámetro
            if 'global_pt1' in defect and 'global_pt2' in defect:
                global_pt1 = defect['global_pt1']
                global_pt2 = defect['global_pt2']
                
                if isinstance(global_pt1, tuple) and isinstance(global_pt2, tuple):
                    # Convertir a enteros si no lo son
                    global_pt1 = (int(global_pt1[0]), int(global_pt1[1]))
                    global_pt2 = (int(global_pt2[0]), int(global_pt2[1]))
                    
                    # Dibujar puntos extremos en azul
                    cv2.circle(result_img, global_pt1, 5, (255, 0, 0), -1)
                    cv2.circle(result_img, global_pt2, 5, (255, 0, 0), -1)
                    
                    # Dibujar flecha para diámetro
                    draw_arrow(result_img, global_pt1, global_pt2, (0, 255, 255), 2, 10, f"D={diametro:.1f}px")
            
            # Información adicional en cuadro de texto
            text_x = x1
            text_y = y2 + 20
            cv2.putText(result_img, f"Diámetro={diametro:.1f}px", 
                      (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        elif defect_type == 'nucleo_esponjoso':
            # Para núcleos esponjosos
            diametro = defect.get('diametro', 0)
            area = defect.get('area_nucleo', 0)
            porcentaje = defect.get('porcentaje_area', 0)
            
            # Dibujar puntos extremos y vector para diámetro
            if 'global_pt1' in defect and 'global_pt2' in defect:
                global_pt1 = defect['global_pt1']
                global_pt2 = defect['global_pt2']
                
                if isinstance(global_pt1, tuple) and isinstance(global_pt2, tuple):
                    # Convertir a enteros si no lo son
                    global_pt1 = (int(global_pt1[0]), int(global_pt1[1]))
                    global_pt2 = (int(global_pt2[0]), int(global_pt2[1]))
                    
                    # Dibujar puntos extremos en azul
                    cv2.circle(result_img, global_pt1, 5, (255, 0, 0), -1)
                    cv2.circle(result_img, global_pt2, 5, (255, 0, 0), -1)
                    
                    # Dibujar flecha para diámetro
                    draw_arrow(result_img, global_pt1, global_pt2, (0, 255, 255), 2, 10, f"D={diametro:.1f}px")
            
            # Información adicional en cuadro de texto
            text_x = x1
            text_y = y2 + 20
            cv2.putText(result_img, f"Diam={diametro:.1f}px, Área={area:.0f}px², %={porcentaje:.2f}%", 
                      (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        elif defect_type == 'sopladura':
            # Para sopladuras
            L = defect.get('L', 0)
            D = defect.get('D', 0)
            area = defect.get('area', 0)
            direccion = defect.get('direccion', '')
            
            # Dibujar información en cuadro de texto
            text_x = x1
            text_y = y2 + 20
            cv2.putText(result_img, f"L={L:.1f}px, D={D:.1f}px, Dir={direccion}", 
                      (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        elif defect_type == 'inclusion_no_metalica':
            # Para inclusiones no metálicas
            num_inclusiones = defect.get('num_inclusiones', 0)
            area = defect.get('area_interseccion', 0)
            concentracion = defect.get('metrica_C', 0)
            
            # Si hay contornos de inclusiones disponibles, mostrarlos en la imagen
            if 'contours' in defect and defect['contours'] is not None:
                contours = defect['contours']
                
                # Ajustar las coordenadas de los contornos al ROI
                adjusted_contours = []
                for contour in contours:
                    adjusted_contour = contour.copy()
                    adjusted_contour[:,:,0] += x1
                    adjusted_contour[:,:,1] += y1
                    adjusted_contours.append(adjusted_contour)
                
                # Dibujar contornos y centros de inclusiones
                for j, contour in enumerate(adjusted_contours):
                    # Dibujar contorno en verde
                    cv2.drawContours(result_img, [contour], 0, (0, 255, 0), 1)
                    
                    # Calcular centro del contorno
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Dibujar punto rojo en el centro
                        cv2.circle(result_img, (cx, cy), 3, (0, 0, 255), -1)
                        # Numerar las inclusiones
                        cv2.putText(result_img, str(j+1), (cx+4, cy+4), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Dibujar cuadrado de análisis si está disponible
            if 'square' in defect:
                sq_x1, sq_y1, sq_x2, sq_y2 = defect['square']
                # Ajustar al ROI
                sq_x1 += x1
                sq_y1 += y1
                sq_x2 += x1
                sq_y2 += y1
                cv2.rectangle(result_img, (sq_x1, sq_y1), (sq_x2, sq_y2), (0, 255, 0), 2)
            
            # Información adicional en cuadro de texto
            text_x = x1
            text_y = y2 + 20
            cv2.putText(result_img, f"Incl.={num_inclusiones}, Área={area:.0f}px², C={concentracion:.6f}", 
                      (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        elif defect_type == 'etiqueta':
            # Para etiquetas con OCR
            code = defect.get('code', 'UNKNOWN')
            quality = defect.get('quality', 'UNKNOWN')
            line = defect.get('line', 'UNKNOWN')
            
            # Información OCR en cuadro de texto
            text_x = x1
            text_y = y2 + 20
            cv2.putText(result_img, f"Code: {code}", (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(result_img, f"Quality: {quality}", (text_x, text_y+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(result_img, f"Line: {line}", (text_x, text_y+40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result_img

def overlay_sopladura_visualizations(original_image, processed_results, vertices):
    """
    Superpone las visualizaciones específicas de sopladura sobre la imagen original
    
    Args:
        original_image: Imagen original de la palanquilla
        processed_results: Resultados procesados de sopladuras
        vertices: Vértices de la palanquilla
        
    Returns:
        Una imagen con las visualizaciones de sopladuras superpuestas
    """
    # Crear copia de la imagen original
    result_img = original_image.copy()
    
    # Dibujar contorno de la palanquilla
    if vertices is not None and len(vertices) == 4:
        cv2.polylines(result_img, [np.array(vertices)], True, (0, 255, 0), 2)
    
    # Título para la imagen
    cv2.putText(result_img, "Sopladuras", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Verificar que existe la información necesaria
    if 'processed_data' not in processed_results or 'visualizations' not in processed_results:
        return result_img
    
    sopladuras = processed_results['processed_data']
    visualizations = processed_results['visualizations']
    
    # Procesar cada sopladura
    for i, sopladura in enumerate(sopladuras):
        # Obtener datos básicos
        bbox = sopladura.get('bbox', None)
        if bbox is None:
            continue
            
        x1, y1, x2, y2 = bbox
        lado = sopladura.get('lado', 'desconocido')
        L = sopladura.get('L', 0)
        D = sopladura.get('D', 0)
        
        # Buscar la visualización específica para esta sopladura
        visualization_key = f"sopladura_{lado}"
        if visualization_key in visualizations:
            # Obtener la visualización con mapa de calor
            viz_image = visualizations[visualization_key]
            
            # Redimensionar la visualización al tamaño del bounding box
            viz_resized = cv2.resize(viz_image, (x2-x1, y2-y1))
            
            # Crear una máscara para la visualización
            # Esta máscara nos permite superponer solo la parte relevante
            # y mantener la transparencia adecuada
            gray = cv2.cvtColor(viz_resized, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            
            # Crear una región de interés (ROI) en la imagen original
            roi = result_img[y1:y2, x1:x2]
            
            # Superponer la visualización en la ROI con transparencia
            # Para cada pixel donde la máscara es no-cero
            for c in range(0, 3):
                roi[:, :, c] = np.where(mask > 0, 
                                        viz_resized[:, :, c] * 0.7 + roi[:, :, c] * 0.3, 
                                        roi[:, :, c])
            
            # Opcional: dibujar un borde alrededor del área de sopladura
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 255), 1)
        else:
            # Si no hay visualización específica, dibujar solo el rectángulo
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Etiqueta con información sobre la sopladura
        label = f"#{i+1} {lado}: L={L}px, D={D}px"
        
        # Posicionar la etiqueta según el lado para evitar solapamientos
        text_y = y1 - 10
        if y1 < 30:  # Si está muy arriba, ponerla abajo
            text_y = y2 + 20
        
        # Añadir un fondo negro para mejor visibilidad del texto
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result_img, 
                    (x1, text_y - text_size[1] - 2), 
                    (x1 + text_size[0], text_y + 2), 
                    (0, 0, 0), -1)
        
        # Dibujar la etiqueta
        cv2.putText(result_img, label, (x1, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Información general sobre el número de sopladuras
    cv2.putText(result_img, f"Total: {len(sopladuras)} (una por lado)", (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return result_img
def overlay_analysis_on_image(original_image, analysis_image, alpha=0.7):
    """
    Superpone una imagen de análisis (con fondo negro) sobre la imagen original
    
    Args:
        original_image: Imagen original de la palanquilla
        analysis_image: Imagen de análisis (con fondo negro y elementos gráficos)
        alpha: Factor de transparencia para la superposición (0-1)
        
    Returns:
        Una imagen con el análisis superpuesto sobre la original
    """
    # Verificar que ambas imágenes existen
    if original_image is None or analysis_image is None:
        return original_image if original_image is not None else analysis_image
    
    # Redimensionar la imagen de análisis para que coincida con la original
    analysis_resized = cv2.resize(analysis_image, (original_image.shape[1], original_image.shape[0]))
    
    # Crear una máscara para los elementos no negros en la imagen de análisis
    # Esto nos permite superponer solo los elementos importantes (líneas, círculos, etc.)
    # y no el fondo negro
    gray = cv2.cvtColor(analysis_resized, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Crear imagen resultado que comienza como una copia de la original
    result = original_image.copy()
    
    # Para cada pixel donde la máscara es blanca (elementos gráficos en la imagen de análisis)
    # mezclamos los colores de la imagen original y la de análisis según el factor alpha
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    result = cv2.addWeighted(result, 1.0, analysis_resized, alpha, 0)
    
    # Donde la máscara es negra (fondo negro en la imagen de análisis)
    # mantenemos el color original sin cambios
    inverted_mask = 1.0 - mask_3channel
    result = result * mask_3channel + original_image * inverted_mask
    
    return result.astype(np.uint8)

def output_fn(prediction_results, output_dir, input_data):
    """
    Guarda los resultados del análisis con visualizaciones mejoradas:
    - Una imagen consolidada por tipo de defecto que muestra los análisis sobre la imagen final rotada
    - Reportes completos con todas las métricas
    - Máscara de la palanquilla
    """
    from utils.utils import safe_write_file, draw_arrow
    import json
    import cv2
    import numpy as np
    import os
    from pathlib import Path
    
    # Extraer información básica
    name = input_data['name']
    ext = input_data['ext']
    
    # Obtener datos clave para el procesamiento
    original_image = prediction_results.get('original_image', None)
    processed_image = prediction_results.get('image_procesada', original_image)  # Imagen rotada
    vertices = prediction_results.get('vertices', None)
    classified_detections = prediction_results.get('classified_detections', {})
    processed_results = prediction_results.get('processed_results', {})
    
    # NUEVO: Extraer la máscara de la palanquilla
    palanquilla_mask = prediction_results.get('palanquilla_mask', None)
    
    # Crear carpeta principal para esta imagen
    image_output_dir = os.path.join(output_dir, name)
    os.makedirs(image_output_dir, exist_ok=True)
    
    output_paths = {}
    
    # NUEVO: Guardar la máscara de la palanquilla
    if palanquilla_mask is not None:
        # Guardar la máscara binaria
        mask_path = os.path.join(image_output_dir, f"{name}_mascara{ext}")
        cv2.imwrite(mask_path, palanquilla_mask)
        output_paths['palanquilla_mask'] = mask_path
        print(f"Máscara de la palanquilla guardada en: {mask_path}")
        
        # Crear una versión coloreada de la máscara para mejor visualización
        # Convertir la máscara binaria a una imagen en color
        mask_colored = np.zeros_like(processed_image)
        # Usar un color verde para la máscara
        mask_colored[palanquilla_mask > 0] = [0, 255, 0]
        
        # Guardar también la versión coloreada
        colored_mask_path = os.path.join(image_output_dir, f"{name}_mascara_color{ext}")
        cv2.imwrite(colored_mask_path, mask_colored)
        output_paths['colored_mask'] = colored_mask_path
        
        # Crear una versión superpuesta sobre la imagen original
        alpha = 0.3  # Factor de transparencia
        mask_overlay = processed_image.copy()
        # Aplicar una superposición semi-transparente
        cv2.addWeighted(processed_image, 0.7, mask_colored, 0.3, 0, mask_overlay)
        
        # Dibujar también el contorno en la superposición para mayor claridad
        contours, _ = cv2.findContours(palanquilla_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_overlay, contours, -1, (0, 255, 0), 2)
        
        # Guardar la superposición
        overlay_path = os.path.join(image_output_dir, f"{name}_mascara_overlay{ext}")
        cv2.imwrite(overlay_path, mask_overlay)
        output_paths['mask_overlay'] = overlay_path
        print(f"Visualización superpuesta de la máscara guardada en: {overlay_path}")
    # 1. Guardar la comparación antes y después de la rotación si hay rotación
    if 'rotacion_info' in prediction_results and prediction_results['rotacion_info']['angulo'] != 0:
        # Get original and rotated images
        original_image = prediction_results['original_image']
        rotated_image = prediction_results['image_procesada']
        
        # Create comparison visualization
        h, w = original_image.shape[:2]
        h_comp = min(h, 800)
        w_comp = int(w * (h_comp / h))
        
        original_resized = cv2.resize(original_image, (w_comp, h_comp))
        rotated_resized = cv2.resize(rotated_image, (w_comp, h_comp))
        
        # Side-by-side comparison
        comparison = np.zeros((h_comp, w_comp*2, 3), dtype=np.uint8)
        comparison[:, :w_comp] = original_resized
        comparison[:, w_comp:] = rotated_resized
        
        # Add labels
        angulo = prediction_results['rotacion_info']['angulo']
        cv2.putText(comparison, "ORIGINAL", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, f"ROTADA {angulo:.1f}°", (w_comp+10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save comparison
        comp_path = os.path.join(image_output_dir, f"{name}_rotacion_comparacion{ext}")
        cv2.imwrite(comp_path, comparison)
        output_paths['rotacion_comparacion'] = comp_path
    
    # 2. Guardar imagen con todos los defectos detectados (visualización general)
    from common.defect_classifier import visualize_results_with_masks
    result_image = visualize_results_with_masks(
        processed_image,  # Usar imagen procesada (rotada)
        prediction_results['classified_detections']
    )
    result_path = os.path.join(image_output_dir, f"{name}_resultado{ext}")
    cv2.imwrite(result_path, result_image)
    output_paths['result_img'] = result_path
    
    # 3. Crear una visualización consolidada para cada tipo de defecto
    for defect_type, defects in classified_detections.items():
        if not defects:  # Saltar si no hay defectos de este tipo
            continue
            
        # Crear directorio para este tipo de defecto si no existe
        defect_dir = os.path.join(image_output_dir, defect_type)
        os.makedirs(defect_dir, exist_ok=True)
        
        # Tratamiento especial para defectos de análisis geométrico que no requieren consolidación
        if defect_type in ['abombamiento', 'romboidad']:
            # Mantener comportamiento existente para estos tipos
            if defect_type in processed_results and 'visualizations' in processed_results[defect_type]:
                visualizations = processed_results[defect_type]['visualizations']
                for viz_name, viz_img in visualizations.items():
                    # Usar overlay_analysis_on_image para superponer el análisis sobre la imagen original
                    overlaid_img = overlay_analysis_on_image(processed_image, viz_img)
                    
                    # Guardar tanto la imagen original de análisis como la superpuesta
                    viz_path = os.path.join(defect_dir, f"{name}_{viz_name}{ext}")
                    cv2.imwrite(viz_path, viz_img)
                    output_paths[f'{defect_type}_{viz_name}'] = viz_path
                    
                    # Guardar la versión superpuesta
                    overlay_path = os.path.join(defect_dir, f"{name}_{viz_name}_overlay{ext}")
                    cv2.imwrite(overlay_path, overlaid_img)
                    output_paths[f'{defect_type}_{viz_name}_overlay'] = overlay_path
            continue
        
        # Tratamiento especial para sopladuras
        if defect_type == 'sopladura' and defect_type in processed_results:
            # Generar visualización específica para sopladuras
            sopladura_img = overlay_sopladura_visualizations(
                processed_image,
                processed_results[defect_type],
                vertices
            )
            
            # Guardar la visualización
            sopladura_path = os.path.join(defect_dir, f"{name}_sopladura_consolidado{ext}")
            cv2.imwrite(sopladura_path, sopladura_img)
            output_paths['sopladura_visualizacion'] = sopladura_path
            
            # Aplicar también overlay para visualizaciones específicas de sopladuras
            if 'visualizations' in processed_results[defect_type]:
                for viz_name, viz_img in processed_results[defect_type]['visualizations'].items():
                    # Solo para imágenes (no máscaras)
                    if viz_name.startswith('sopladura_') and not viz_name.endswith('_mask'):
                        overlaid_img = overlay_analysis_on_image(processed_image, viz_img)
                        
                        overlay_path = os.path.join(defect_dir, f"{name}_{viz_name}_overlay{ext}")
                        cv2.imwrite(overlay_path, overlaid_img)
                        output_paths[f'{defect_type}_{viz_name}_overlay'] = overlay_path
            continue

        # CONSOLIDACIÓN PRINCIPAL: Para todos los demás tipos de defectos
        if defect_type in processed_results and 'processed_data' in processed_results[defect_type]:
            # Obtener los resultados procesados para este tipo de defecto
            processed_data = processed_results[defect_type]['processed_data']
            
            # Crear imagen consolidada con todos los defectos de este tipo
            consolidated_image = draw_all_defects_on_original_image(
                processed_image,  # Usar imagen ROTADA
                defects,
                processed_data,
                defect_type,
                vertices
            )
            
            # Guardar la imagen consolidada
            consolidated_path = os.path.join(defect_dir, f"{name}_{defect_type}_consolidado{ext}")
            cv2.imwrite(consolidated_path, consolidated_image)
            output_paths[f'{defect_type}_consolidated'] = consolidated_path
            
            # Aplicar overlay para visualizaciones individuales de este tipo de defecto
            if 'visualizations' in processed_results[defect_type]:
                for viz_name, viz_img in processed_results[defect_type]['visualizations'].items():
                    # Para cada visualización, crear una versión superpuesta
                    if isinstance(viz_img, np.ndarray) and len(viz_img.shape) == 3:  # Verificar que es una imagen color
                        try:
                            overlaid_img = overlay_analysis_on_image(processed_image, viz_img)
                            
                            # Guardar la visualización original
                            orig_viz_path = os.path.join(defect_dir, f"{name}_{viz_name}{ext}")
                            cv2.imwrite(orig_viz_path, viz_img)
                            
                            # Guardar la versión superpuesta
                            overlay_path = os.path.join(defect_dir, f"{name}_{viz_name}_overlay{ext}")
                            cv2.imwrite(overlay_path, overlaid_img)
                            output_paths[f'{defect_type}_{viz_name}_overlay'] = overlay_path
                        except Exception as e:
                            print(f"Error al crear overlay para {viz_name}: {e}")
    
    return output_paths


def process(self, etiqueta_detections, image, corners=None, zone_masks=None, image_name=None, output_dir=None):
    """
    Process all detected labels
    
    Args:
        etiqueta_detections: List of label detections from vertex detector
        image: Original image
        corners: Corners of the palanquilla (optional)
        zone_masks: Zone masks (optional)
        image_name: Image name (without extension)
        output_dir: Output directory for saving reports
        
    Returns:
        processed_data: Dictionary with processing results
    """
    results = []
    visualizations = {}
    
    # Process each detected label
    for i, detection in enumerate(etiqueta_detections):
        x1, y1, x2, y2 = detection['bbox']
        conf = detection.get('conf', 0)
        
        # Extract the label region
        label_image = image[y1:y2, x1:x2].copy()
        
        # Process the label with OCR
        print(f"Procesando etiqueta #{i+1} con OCR...")
        ocr_data = self.extract_label_content(label_image)
        
        # Combine data
        label_data = {
            'id': i+1,
            'code': ocr_data.get('code', 'UNKNOWN'),
            'quality': ocr_data.get('quality', 'UNKNOWN'),
            'line': ocr_data.get('line', 'UNKNOWN'),
            'conf': conf,
            'bbox': (x1, y1, x2, y2)
        }
        
        # Create visualization
        viz_img = label_image.copy()
        # Add text overlay on the visualization
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(viz_img, f"Code: {ocr_data.get('code', 'UNKNOWN')}", (10, 20), font, 0.5, (0, 0, 255), 2)
        cv2.putText(viz_img, f"Quality: {ocr_data.get('quality', 'UNKNOWN')}", (10, 40), font, 0.5, (0, 0, 255), 2)
        cv2.putText(viz_img, f"Line: {ocr_data.get('line', 'UNKNOWN')}", (10, 60), font, 0.5, (0, 0, 255), 2)
        
        # Save visualization
        visualization_key = f"etiqueta_{i+1}"
        visualizations[visualization_key] = viz_img
        
        # Also save the original label image for reference
        visualization_key_orig = f"etiqueta_orig_{i+1}"
        visualizations[visualization_key_orig] = label_image
        
        results.append(label_data)
    
    # If there are results and we have image name and output directory, generate a report
    report_paths = None
    if results and image_name and output_dir:
        # Create directory for this type - ensure it's named "etiqueta"
        etiqueta_dir = os.path.join(output_dir, image_name, "etiqueta")
        os.makedirs(etiqueta_dir, exist_ok=True)
        
        # Generate the report in the specific folder
        report_paths = self.generate_report(image_name, results, etiqueta_dir)
        print(f"Reporte de etiquetas generado en: {report_paths[0]}")
        print(f"Reporte de texto de etiquetas generado en: {report_paths[1]}")
        
        # Guardar también en formato JSON
        json_path = os.path.join(etiqueta_dir, f"{image_name}_etiqueta_ocr.json")
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"JSON de etiquetas guardado en: {json_path}")
            
            # Añadir el path del JSON a los report_paths
            if isinstance(report_paths, tuple):
                report_paths = report_paths + (json_path,)
            else:
                report_paths = (report_paths, json_path)
        except Exception as e:
            print(f"Error al guardar JSON de etiquetas: {e}")
    
    return {
        'processed_data': results,
        'visualizations': visualizations,
        'report_paths': report_paths
    }


def process_directory(directory_path, output_dir=None, model_dir=None, global_log_path=None):
    """
    Procesa todas las imágenes en un directorio
    
    Args:
        directory_path: Ruta al directorio
        output_dir: Directorio donde guardar los resultados
        model_dir: Directorio donde están los modelos
        
    Returns:
        success_count: Número de imágenes procesadas exitosamente
    """
    if global_log_path:
        actualizar_log(global_log_path, f"Iniciando procesamiento del directorio: {directory_path}")
    
    # Verificar que el directorio exista
    if not os.path.exists(directory_path):
        print(f"El directorio {directory_path} no existe")
        return 0
    
    # Valor predeterminado para el directorio de salida
    if output_dir is None:
        output_dir = r"D:\Trabajo modelos\PACC\YOLOv12 - copia\Clasificacion_por_zonas"
    
    # Extensiones de imagen válidas
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # Encontrar todas las imágenes en el directorio
    image_files = []
    for ext in valid_extensions:
        image_files.extend(list(Path(directory_path).glob(f'*{ext}')))
        image_files.extend(list(Path(directory_path).glob(f'*{ext.upper()}')))
    
    # Evitar cargar los modelos múltiples veces
    models = model_fn(model_dir)
    
    # Procesar cada imagen
    success_count = 0
    for img_path in image_files:
        try:
            # 1. Procesar la entrada
            input_data = input_fn(str(img_path))
            
            # Crear carpeta específica para esta imagen
            image_name = input_data['name']
            image_dir = os.path.join(output_dir, image_name)
            os.makedirs(image_dir, exist_ok=True)
            
            # 2. Realizar predicciones
            prediction_results = predict_fn(input_data, models, output_dir)
            
            # 3. Generar y guardar salidas
            output_fn(prediction_results, output_dir, input_data)
            
            success_count += 1
            if global_log_path:
                actualizar_log(global_log_path, f"Procesada imagen {success_count}/{len(image_files)}: {img_path.name}")
            
        except Exception as e:
            if global_log_path:
                actualizar_log(global_log_path, f"Error al procesar {img_path}: {e}", "ERROR")
    
    print(f"Procesadas {success_count} de {len(image_files)} imágenes")
    return success_count
def read_csv_file(csv_path):
    """Lee un archivo CSV y devuelve una lista de diccionarios"""
    if not os.path.exists(csv_path):
        return []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            return list(reader)
    except Exception as e:
        print(f"Error leyendo CSV {csv_path}: {str(e)}")
        return []

def get_parameters_from_csv(defect_type, csv_data):
    """Extrae los parámetros relevantes para un tipo de defecto desde el CSV y selecciona el registro con valores más altos"""
    if not csv_data or len(csv_data) == 0:
        return {}
    
    # Para muchos tipos de defectos, queremos encontrar el caso más severo
    if defect_type in ['grietas_diagonales', 'grietas_corner', 'grietas_medio_camino']:
        # Buscar la grieta más larga
        max_length_row = csv_data[0]
        max_length = float(max_length_row.get('L', 0)) if 'L' in max_length_row else 0
        max_index = 0  # Índice de la grieta más relevante
        
        for i, row in enumerate(csv_data):
            current_length = float(row.get('L', 0)) if 'L' in row else 0
            if current_length > max_length:
                max_length = current_length
                max_length_row = row
                max_index = i
        
        row = max_length_row
        return {
            "longitud_mm": float(row.get('L', 0)) if 'L' in row else 0,
            "distancia_a_superficie_mm": float(row.get('D', 0)) if 'D' in row else 0,
            "espesor_mm": float(row.get('e', 0)) if 'e' in row else 0,
            "_id_relevante": max_index + 1  # Guardamos el índice relevante (+1 porque los índices empiezan en 0)
        }
    elif defect_type == 'inclusion_no_metalica':
        # Buscar la inclusión con mayor concentración
        max_conc_row = csv_data[0]
        max_conc = float(max_conc_row.get('concentracion', 0)) if 'concentracion' in max_conc_row else 0
        max_index = 0  # Índice de la inclusión más relevante
        
        for i, row in enumerate(csv_data):
            current_conc = float(row.get('concentracion', 0)) if 'concentracion' in row else 0
            if current_conc > max_conc:
                max_conc = current_conc
                max_conc_row = row
                max_index = i
        
        row = max_conc_row
        return {
            "num_inclusiones": int(row.get('num_inclusiones', 0)) if 'num_inclusiones' in row else 0,
            "concentracion": float(row.get('concentracion', 0)) if 'concentracion' in row else 0,
            "_id_relevante": max_index + 1  # Guardamos el índice relevante
        }
    elif defect_type == 'nucleo_esponjoso':
        # CASO ESPECIAL: Buscar el núcleo con MENOR porcentaje de área (cambiado según requerimiento)
        min_area_row = csv_data[0]
        min_area = float(min_area_row.get('porcentaje_area', float('inf'))) if 'porcentaje_area' in min_area_row else float('inf')
        min_index = 0  # Índice del núcleo más pequeño
        
        for i, row in enumerate(csv_data):
            current_area = float(row.get('porcentaje_area', 0)) if 'porcentaje_area' in row else 0
            if current_area > 0 and current_area < min_area:  # Solo considerar valores positivos
                min_area = current_area
                min_area_row = row
                min_index = i
        
        row = min_area_row
        return {
            "diametro": float(row.get('diametro', 0)) if 'diametro' in row else 0,
            "porcentaje_area": float(row.get('porcentaje_area', 0)) if 'porcentaje_area' in row else 0,
            "_id_relevante": min_index + 1  # Guardamos el índice relevante
        }
    elif defect_type == 'rechupe' or defect_type == 'estrella':
        # Buscar el rechupe o estrella con mayor diámetro
        max_diam_row = csv_data[0]
        max_diam = float(max_diam_row.get('diametro', 0)) if 'diametro' in max_diam_row else 0
        max_index = 0  # Índice del defecto más relevante
        
        for i, row in enumerate(csv_data):
            current_diam = float(row.get('diametro', 0)) if 'diametro' in row else 0
            if current_diam > max_diam:
                max_diam = current_diam
                max_diam_row = row
                max_index = i
        
        row = max_diam_row
        return {
            "diametro": float(row.get('diametro', 0)) if 'diametro' in row else 0,
            "_id_relevante": max_index + 1  # Guardamos el índice relevante
        }
    elif defect_type == 'romboidad':
        # Para romboidad, normalmente solo hay un registro
        row = csv_data[0]
        return {
            "diagonal_mayor": float(row.get('Diagonal_Mayor_D(px)', 0)) if 'Diagonal_Mayor_D(px)' in row else 0,
            "diagonal_menor": float(row.get('Diagonal_Menor_d(px)', 0)) if 'Diagonal_Menor_d(px)' in row else 0,
            "diferencia": float(row.get('Diferencia(px)', 0)) if 'Diferencia(px)' in row else 0,
            "_id_relevante": 1  # Siempre el primero para romboidad
        }
    elif defect_type == 'abombamiento':
        # Para abombamiento, procesar los valores de I, M, T
        abomb_values = {}
        
        # Extraer valores para I, M, T de los registros disponibles
        if len(csv_data) >= 3:  # Si hay al menos 3 registros, asumimos que son I, M, T
            abomb_values['I'] = float(csv_data[0].get('Abombamiento_Porcentaje', 0)) if 'Abombamiento_Porcentaje' in csv_data[0] else 0
            abomb_values['M'] = float(csv_data[1].get('Abombamiento_Porcentaje', 0)) if 'Abombamiento_Porcentaje' in csv_data[1] else 0
            abomb_values['T'] = float(csv_data[2].get('Abombamiento_Porcentaje', 0)) if 'Abombamiento_Porcentaje' in csv_data[2] else 0
        else:
            # Si no hay suficientes registros, usar los disponibles o valores por defecto
            for idx, r in enumerate(csv_data):
                if idx == 0:
                    abomb_values['I'] = float(r.get('Abombamiento_Porcentaje', 0)) if 'Abombamiento_Porcentaje' in r else 0
                elif idx == 1:
                    abomb_values['M'] = float(r.get('Abombamiento_Porcentaje', 0)) if 'Abombamiento_Porcentaje' in r else 0
                elif idx == 2:
                    abomb_values['T'] = float(r.get('Abombamiento_Porcentaje', 0)) if 'Abombamiento_Porcentaje' in r else 0
        
        # También incluimos información de lado si está disponible
        for idx, r in enumerate(csv_data):
            lado = r.get('Lado', f'Lado{idx+1}')
            porcentaje = float(r.get('Abombamiento_Porcentaje', 0)) if 'Abombamiento_Porcentaje' in r else 0
            abomb_values[lado] = porcentaje
        
        # Identificar el lado con mayor abombamiento para la imagen relevante
        max_abomb = 0
        max_lado_idx = 0
        for idx, r in enumerate(csv_data):
            porcentaje = float(r.get('Abombamiento_Porcentaje', 0)) if 'Abombamiento_Porcentaje' in r else 0
            if porcentaje > max_abomb:
                max_abomb = porcentaje
                max_lado_idx = idx
        
        abomb_values['_id_relevante'] = max_lado_idx + 1
        return abomb_values
    
    elif defect_type == 'sopladura':
        # Buscar la sopladura con mayor área o longitud
        max_severity_row = csv_data[0]
        max_severity = 0
        max_index = 0  # Índice de la sopladura más relevante
        
        for i, row in enumerate(csv_data):
            # Calcular severidad basada en área y longitud
            area = float(row.get('area', 0)) if 'area' in row else 0
            length = float(row.get('L', 0)) if 'L' in row else 0
            
            # Priorizar por área, pero también considerar longitud
            current_severity = area * 0.7 + length * 0.3
            
            if current_severity > max_severity:
                max_severity = current_severity
                max_severity_row = row
                max_index = i
        
        row = max_severity_row
        return {
            "longitud_mm": float(row.get('L', 0)) if 'L' in row else 0,
            "distancia_a_superficie_mm": float(row.get('D', 0)) if 'D' in row else 0,
            "area": float(row.get('area', 0)) if 'area' in row else 0,
            "lado": row.get('lado', '') if 'lado' in row else '',
            "direccion": row.get('direccion', '') if 'direccion' in row else '',
            "_id_relevante": max_index + 1  # Guardamos el índice relevante
        }
    
    # Si no hay mapeo específico, devolver todo el primer registro
    return {k: v for k, v in csv_data[0].items() if k != 'id' and k != 'bbox' and k != 'conf'}

def get_etiqueta_info(root_dir):
    """Extrae información de la etiqueta (código, calidad, línea) si está disponible"""
    etiqueta_dir = os.path.join(root_dir, 'etiqueta')
    if not os.path.exists(etiqueta_dir):
        return {"codigo": "", "calidad": "", "linea": ""}
    
    etiqueta_csvs = glob.glob(os.path.join(etiqueta_dir, '*etiqueta_report.csv'))
    if not etiqueta_csvs:
        return {"codigo": "", "calidad": "", "linea": ""}
    
    etiqueta_data = read_csv_file(etiqueta_csvs[0])
    if not etiqueta_data or len(etiqueta_data) == 0:
        return {"codigo": "", "calidad": "", "linea": ""}
    
    row = etiqueta_data[0]
    return {
        "codigo": row.get('code', '') if 'code' in row else '',
        "calidad": row.get('quality', '') if 'quality' in row else '',
        "linea": row.get('line', '') if 'line' in row else ''
    }

def generate_json(root_dir):
    """Genera el JSON con la información de defectos"""
    # Obtener información de la etiqueta
    etiqueta_info = get_etiqueta_info(root_dir)
    base_name = os.path.basename(root_dir)  # Nombre base para buscar imágenes (p.ej. DSC00002)
    
    # Estructura básica del JSON
    result = {
        "general": {
            "seccion": "160x160 mm",
            "fecha": f"{datetime.datetime.now().day}-{datetime.datetime.now().month}-{datetime.datetime.now().year}",
            "calidad": etiqueta_info["calidad"],
            "nombre": "BUZA SUMERGIDA VD"  # Nombre por defecto
        },
        "defectos": []
    }
    
    # Extraer código como nombre si está disponible
    if etiqueta_info["codigo"]:
        result["general"]["nombre"] = f"PALANQUILLA {etiqueta_info['codigo']}"
    
    # Tipos de defectos a buscar
    defect_types = [
        'grietas_diagonales', 'grietas_corner', 'grietas_medio_camino', 
        'rechupe', 'nucleo_esponjoso', 'inclusion_no_metalica', 
        'romboidad', 'estrella', 'abombamiento', 'sopladura'
    ]
    
    # Conteo de defectos por tipo
    defect_counts = {}
    
    # Procesar cada tipo de defecto
    for defect_type in defect_types:
        defect_dir = os.path.join(root_dir, defect_type)
        if not os.path.exists(defect_dir):
            continue
        
        # Buscar archivos CSV
        csv_pattern = f"*{defect_type}_report.csv"
        csv_files = glob.glob(os.path.join(defect_dir, csv_pattern))
        
        # Procesar cada archivo CSV encontrado
        for csv_file in csv_files:
            csv_data = read_csv_file(csv_file)
            if not csv_data:
                continue
            
            # Contar el número de filas (defectos) en el CSV
            num_defects = len(csv_data)
            
            # Actualizar contador de defectos
            defect_counts[defect_type] = num_defects
            
            # Obtener parámetros del defecto más relevante
            params = get_parameters_from_csv(defect_type, csv_data)
            
            # Obtener el ID del defecto más relevante
            id_relevante = params.pop('_id_relevante', 1) if isinstance(params, dict) else 1
            
            # Buscar la imagen específica según el tipo de defecto
            image_path = ""
            
            # Casos especiales para romboidad y abombamiento
            if defect_type == 'romboidad':
                # Para romboidad, buscar imágenes con "visualization" o similares
                romboidad_files = glob.glob(os.path.join(defect_dir, f"{base_name}_romboidad_visualization*.jpg")) + \
                                 glob.glob(os.path.join(defect_dir, f"{base_name}_romboidad_visualization*.JPG"))
                if romboidad_files:
                    image_path = romboidad_files[0]
                else:
                    # Buscar cualquier imagen de romboidad
                    all_files = glob.glob(os.path.join(defect_dir, f"{base_name}_romboidad*.jpg")) + \
                               glob.glob(os.path.join(defect_dir, f"{base_name}_romboidad*.JPG"))
                    if all_files:
                        image_path = all_files[0]
            
            elif defect_type == 'abombamiento':
                # Para abombamiento, buscar imágenes con "enhanced" o "visualization"
                abomb_files = glob.glob(os.path.join(defect_dir, f"{base_name}_abombamiento*enhanced*.jpg")) + \
                             glob.glob(os.path.join(defect_dir, f"{base_name}_abombamiento*enhanced*.JPG"))
                if abomb_files:
                    image_path = abomb_files[0]
                else:
                    # Alternativa: buscar visualization
                    abomb_viz_files = glob.glob(os.path.join(defect_dir, f"{base_name}_abombamiento*visualization*.jpg")) + \
                                     glob.glob(os.path.join(defect_dir, f"{base_name}_abombamiento*visualization*.JPG"))
                    if abomb_viz_files:
                        image_path = abomb_viz_files[0]
            
            else:
                # Para todos los demás tipos, buscar imágenes con el patrón específico del ID relevante
                
                # Transformar el tipo de defecto al nombre singular para búsqueda
                singular_name = defect_type
                if defect_type.startswith('grietas_'):
                    # Convertir "grietas_XXX" a "grieta"
                    singular_name = 'grieta'
                elif defect_type == 'inclusion_no_metalica':
                    singular_name = 'inclusion'
                elif defect_type == 'nucleo_esponjoso':
                    singular_name = 'nucleo'
                
                # Buscar patrón específico con singular_name_id_global
                specific_pattern = f"{singular_name}_{id_relevante}_global"
                specific_files = glob.glob(os.path.join(defect_dir, f"{base_name}_{specific_pattern}*.jpg")) + \
                               glob.glob(os.path.join(defect_dir, f"{base_name}_{specific_pattern}*.JPG"))
                
                if specific_files:
                    image_path = specific_files[0]
                else:
                    # Alternativa: buscar con el formato defect_type_id_global
                    alternative_pattern = f"{defect_type}_{id_relevante}_global"
                    alt_files = glob.glob(os.path.join(defect_dir, f"{base_name}_{alternative_pattern}*.jpg")) + \
                               glob.glob(os.path.join(defect_dir, f"{base_name}_{alternative_pattern}*.JPG"))
                    
                    if alt_files:
                        image_path = alt_files[0]
                    else:
                        # Última opción: buscar cualquier archivo con _id_global
                        global_pattern = f"_{id_relevante}_global"
                        global_files = glob.glob(os.path.join(defect_dir, f"*{global_pattern}*.jpg")) + \
                                      glob.glob(os.path.join(defect_dir, f"*{global_pattern}*.JPG"))
                        
                        if global_files:
                            image_path = global_files[0]
                        else:
                            # Si todo falla, buscar cualquier imagen en la carpeta
                            any_files = glob.glob(os.path.join(defect_dir, "*.jpg")) + \
                                     glob.glob(os.path.join(defect_dir, "*.JPG"))
                            if any_files:
                                image_path = any_files[0]
            
            # Nombre formal del defecto
            defect_name = defect_type.replace('_', ' ').title()
            
            # Crear objeto de defecto
            defect_obj = {
                "nombre": defect_name,
                "parametros": params,
                "imagen_ruta": image_path,
                "codigo": etiqueta_info["codigo"],
                "calidad": etiqueta_info["calidad"],
                "linea": etiqueta_info["linea"],
                "unidad_medida": "mm",
                "cantidad": num_defects
            }
            
            # Añadir log para depuración - puedes eliminar esto en producción
            print(f"Defecto: {defect_type}, ID relevante: {id_relevante}, Imagen: {os.path.basename(image_path) if image_path else 'No encontrada'}")
            
            result["defectos"].append(defect_obj)
            
            # Solo incluimos un defecto por tipo
            break
    
    # Añadir conteo total de defectos a la información general
    result["general"]["conteo_defectos"] = defect_counts
    
    return result

def main():
    log_level = getattr(logging, logging_level, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout
    )
    
    logging.info("Cargando modelos...")
    models = model_fn()
    logging.info("Modelos cargados exitosamente!")
    logging.info(f"Suscribiéndose al tópico: {topic_read}...")
    ipc_client_subscribe = GreengrassCoreIPCClientV2()
    ipc_client_publish = GreengrassCoreIPCClientV2()
    operation = None
    try:
        # Usar la firma correcta del SDK
        _, operation = ipc_client_subscribe.subscribe_to_topic(
            topic=topic_read,
            on_stream_event=lambda x: on_stream_event(x, models, ipc_client_publish),
            on_stream_error=on_stream_error,
            on_stream_closed=on_stream_closed
        )
        logging.info(f"Suscripción exitosa al tópico: {topic_read}")
        while True:
            time.sleep(1)
    except ServiceError as se:
        logging.error(f"ServiceError al suscribirse a {topic_read}: {se}")
    except KeyboardInterrupt:
        logging.info("Suscripción detenida por el usuario")
    except Exception:
        logging.exception("Error en la suscripción")
    finally:
        if operation:
            operation.close()
        logging.info("Suscripción finalizada")

# Código para prueba con imagen específica - mantiene la prueba original de paste-2.txt
if __name__ == "__main__":
    # Si se ejecuta directamente sin argumentos, pero quieres probar con la imagen específica
    # en lugar de entrar al modo interactivo, descomentar y ejecutar estas líneas:
    
    # Cargar modelos
    models = model_fn()
    
    # Usar la misma ruta que en paste-2.txt
    ruta = r"D:\Trabajo modelos\PACC\YOLOv12 - copia\pruebas diagonales\DSC00002.JPG"
    filename = os.path.basename(ruta)
    basename = filename.split(".")[0]
    workdir = os.getcwd()
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, basename), exist_ok=True)
    dest_folder = ruta.replace(os.path.basename(os.path.dirname(ruta)), folder_output_procesados).rsplit(".", 1)[0]
    log_path = os.path.join(os.path.join(workdir, basename), 'logs.log')
    
    try:
        # Ejecutar el flujo de procesamiento
        input_data = input_fn(ruta, log_path)
        prediction_results = predict_fn(input_data, models, workdir, log_path)
        output_fn(prediction_results, workdir, input_data)
        defect_json = generate_json(os.path.join(workdir, basename))
        
        # Crear el mensaje de resultados (como en paste-2.txt)
        msg_excel = {
            "ruta_archivos": os.path.join(workdir, basename),
            "ruta_excel": os.path.dirname(log_path),
            "resultados": defect_json
        }
        
        print("Resultados generados exitosamente:")
        print(json.dumps(msg_excel, indent=2, cls=NpEncoder))
        
        # Guardar también como JSON
        json_path = os.path.join(os.path.join(workdir, basename), f"{basename}_resultados.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(defect_json, f, ensure_ascii=False, indent=4, cls=NpEncoder)
        
        print(f"\nResultados JSON guardados en: {json_path}")
        actualizar_log(log_path, f"Procesamiento exitoso. Resultado guardado en {json_path}")
        
        # En un entorno real con Greengrass, aquí se publicaría el mensaje
        # La siguiente línea simula la publicación pero no envía realmente el mensaje
        actualizar_log(log_path, f"Simulación: Publicado en topic {topic_write} payload: {msg_excel}")
        
    except Exception as e:
        dest_folder = ruta.replace(os.path.basename(os.path.dirname(ruta)), folder_output_ignorados).rsplit(".", 1)[0]
        print(f"Error al procesar la imagen: {e}")
        actualizar_log(log_path, f"Error ejecutando inferencia: {filename} - {type(e)} - {e}")
        actualizar_log(log_path, traceback.format_exc())