import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Importar funciones de detección y análisis
from models.vertex_detector import VertexDetector
from models.defect_detector import DefectDetector
from common.zone_generator import visualize_zones
from common.defect_classifier import classify_defects_with_masks, visualize_results_with_masks
from utils.utils import order_points
from utils.contorno import obtener_contorno_imagen, rotar_imagen_lado, redimensionar_recortar_palanquilla

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
from defects.etiqueta.label_extractor import LabelExtractor,determinar_orientacion_etiqueta  # Nueva importación para etiquetas

def model_fn(model_dir=None):
    """
    Carga los modelos necesarios para la detección de vértices y defectos
    
    Args:
        model_dir: Directorio donde se encuentran los modelos (opcional)
        
    Returns:
        Dictionary con los modelos cargados
    """
    # Rutas predeterminadas si no se especifica un directorio
    vertex_model_path = r"D:\Trabajo modelos\PACC\YOLOv12 - copia\Models\Vertex\modelo_1.pt"
    defect_model_path = r"D:\Trabajo modelos\PACC\YOLOv12 - copia\Models\Defect\modelo_2.pt"
    
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
    
    # Nuevo: Inicializar el procesador de etiquetas
    label_extractor = LabelExtractor()
    
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
            'etiqueta': label_extractor  # Nuevo: Procesador de etiquetas
        }
    }


def input_fn(image_path):
    """
    Procesa la entrada (ruta de imagen) y la prepara para el análisis
    
    Args:
        image_path: Ruta a la imagen a procesar
        
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
    
    return {
        'image': image,
        'path': image_path,
        'basename': basename,
        'name': name,
        'ext': ext
    }


def predict_fn(input_data, models, output_dir=None):
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
    image = input_data['image']
    image_path = input_data['path']
    image_name = input_data['name']
    
    # Extraer los modelos
    vertex_detector = models['vertex_detector']
    defect_detector = models['defect_detector']
    
    # 1. Pre-procesamiento de la imagen (opcional)
    # Redimensionar si la imagen es muy grande
    h, w = image.shape[:2]
    if h > 3000 or w > 3000:
        max_dim = 2000
        scale_factor = max_dim / max(h, w)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        print(f"Redimensionando imagen de {w}x{h} a {new_w}x{new_h}")
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # NUEVA PARTE: Detectar etiquetas primero para orientación
    print("Detectando etiquetas para orientación...")
    # Detectar etiquetas con el vertex_detector
    etiqueta_detections = []
    
    try:
        # Obtener resultados del vertex detector para la imagen actual
        vertex_result = vertex_detector.model.predict(image, conf=vertex_detector.conf_threshold, device=vertex_detector.device)[0]
        
        # Buscar la clase "etiqueta" en los resultados del vertex detector
        if hasattr(vertex_detector.model, "names") and vertex_result.boxes is not None:
            class_names = vertex_detector.model.names
            etiqueta_class_id = None
            
            # Mostrar todas las clases disponibles y sus IDs para diagnóstico
            print(f"Clases disponibles en el modelo de vértices: {class_names}")
            
            # Encontrar el ID de clase para "etiqueta"
            for id, name in class_names.items():
                if isinstance(name, str) and name.lower() == "etiqueta":
                    etiqueta_class_id = id
                    print(f"ID de clase para 'etiqueta' encontrado: {etiqueta_class_id}")
                    break
            
            # Si no encontramos "etiqueta", buscar "class_0" que podría ser la etiqueta
            if etiqueta_class_id is None and 0 in class_names:
                etiqueta_class_id = 0
                print(f"No se encontró 'etiqueta' explícitamente. Usando class_0 como etiqueta, nombre: {class_names[0]}")
            
            if etiqueta_class_id is not None:
                # Procesar todas las cajas y encontrar las de clase "etiqueta"
                boxes = vertex_result.boxes
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    if cls_id == etiqueta_class_id:
                        # Esto es una etiqueta
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0].item())
                        
                        print(f"Etiqueta detectada con confianza {conf:.2f} en bbox: ({x1}, {y1}, {x2}, {y2})")
                        
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
    
    # Hacer una detección inicial rápida de vértices para tener referencia de la palanquilla
    initial_vertices = None
    try:
        initial_vertices, _, _ = obtener_contorno_imagen(
            image, vertex_detector.model, vertex_detector.conf_threshold, 
            target_class=vertex_detector.target_class)
    except Exception as e:
        print(f"Error en detección inicial de vértices: {e}")
    
    # Si se detectaron etiquetas, alinear la imagen según la orientación de la etiqueta
    rotated_image = image.copy()  # Por defecto, usar la imagen original
    if etiqueta_detections:
        # Ordenar etiquetas por confianza descendente
        etiqueta_detections.sort(key=lambda x: x['conf'], reverse=True)
        mejor_etiqueta = etiqueta_detections[0]  # Usar la etiqueta de mayor confianza
        
        # Determinar ángulo de rotación basado en la etiqueta
        angulo_rotacion, lado_etiqueta = determinar_orientacion_etiqueta(
            mejor_etiqueta, image, initial_vertices)
        
        # Rotar la imagen
        if angulo_rotacion != 0:
            print(f"Rotando imagen {angulo_rotacion}° para alinear etiqueta...")
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angulo_rotacion, 1.0)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)
            
            # Guardar información sobre la rotación para el resultado final
            rotacion_info = {
                'angulo': angulo_rotacion,
                'lado_etiqueta': lado_etiqueta,
                'etiqueta_bbox': mejor_etiqueta['bbox']
            }
        else:
            print("No es necesario rotar la imagen, etiqueta ya está en posición correcta.")
            rotacion_info = {
                'angulo': 0,
                'lado_etiqueta': lado_etiqueta,
                'etiqueta_bbox': mejor_etiqueta['bbox']
            }
    else:
        print("No se detectaron etiquetas para orientar la imagen.")
        rotacion_info = {
            'angulo': 0,
            'lado_etiqueta': 'no_detectada',
            'etiqueta_bbox': None
        }
    
    # Usar la imagen rotada para el resto del procesamiento
    image = rotated_image
    
    # 2. Detectar vértices de la palanquilla (con la imagen ya rotada)
    print(f"Detectando vértices en imagen rotada: {image_path}")
    try:
        # Usar la función mejorada directamente desde utils.contorno
        # ahora pasando la clase objetivo desde el detector de vértices
        from utils.contorno import obtener_contorno_imagen
        vertices, contorno_principal, palanquilla_mask = obtener_contorno_imagen(
            image, vertex_detector.model, vertex_detector.conf_threshold, 
            target_class=vertex_detector.target_class)
        
        success = vertices is not None and len(vertices) == 4
        
        if not success:
            print("Error: No se pudieron detectar los vértices correctamente. Usando método alternativo.")
            # Usar método alternativo del detector de vértices
            vertices, success, palanquilla_mask = vertex_detector.detect_vertices_alternative(image)
            
            if not success or vertices is None:
                print("Error: También falló el método alternativo. Usando toda la imagen.")
                # Si todo falla, asegurar que palanquilla_mask sea None (será creado más tarde según los vértices)
                h, w = image.shape[:2]
                vertices = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
                # Crear una máscara que cubra toda la imagen
                palanquilla_mask = np.ones((h, w), dtype=np.uint8) * 255
    except Exception as e:
        print(f"Error al detectar vértices: {e}")
        import traceback
        traceback.print_exc()
        h, w = image.shape[:2]
        vertices = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        # Crear una máscara que cubra toda la imagen
        palanquilla_mask = np.ones((h, w), dtype=np.uint8) * 255
        success = False

    # 3. Verificar y corregir vértices si es necesario
    # Asegurar que los vértices están dentro de los límites de la imagen
    h, w = image.shape[:2]
    
    # Corrección de vértices fuera de los límites
    for i in range(len(vertices)):
        vertices[i][0] = max(0, min(w-1, vertices[i][0]))
        vertices[i][1] = max(0, min(h-1, vertices[i][1]))
    
    # Verificar que el área del cuadrilátero sea razonable
    contorno_np = vertices.reshape(-1, 1, 2)
    area = cv2.contourArea(contorno_np)
    
    if area < 10000 or area > 0.95 * w * h:  # Muy pequeño o casi toda la imagen
        print(f"Advertencia: Área del cuadrilátero anormal ({area} píxeles). Ajustando a toda la imagen.")
        vertices = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        contorno_np = vertices.reshape(-1, 1, 2)
        # Recrear la máscara
        palanquilla_mask = np.ones((h, w), dtype=np.uint8) * 255

    # 5. Generar máscaras de zona con los vértices detectados
    print(f"Generando máscaras de zonas")
    zones_img, zone_masks = visualize_zones(image, vertices)
    
    # 6. Detectar defectos con el detector de defectos
    print(f"Detectando defectos")
    detections, yolo_result = defect_detector.detect_defects(image)
    
    if not detections:
        print("No se detectaron defectos en esta imagen.")
    
    # 7. Crear mapeo de clases basado en los nombres de clase del modelo de defectos
    class_mapping = {}
    if defect_detector.class_names:
        print("Creando mapeo de clases basado en modelo de defectos:")
        # Mostrar los nombres de clase disponibles
        print(f"Clases en el modelo: {defect_detector.class_names}")
        
        # Mapear clases del modelo a las categorías esperadas en el código
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
                # Si no hay coincidencia, usar el nombre original
                class_mapping[name] = name
                print(f"  - '{name}' mantenido como '{name}'")
    
    # 8. Clasificar los defectos según su posición en las zonas
    classified_detections = classify_defects_with_masks(detections, zone_masks, image, yolo_result, class_mapping)
    
    # 9. Analizar abombamiento (siempre se ejecuta, no depende de detecciones)
    # Para el abombamiento usar el modelo de vértices/máscara de palanquilla, NO el de defectos
    abombamiento_processor = models['processors']['abombamiento']
    abombamiento_results = abombamiento_processor.process(
        image,
        vertices,
        image_name=image_name,
        output_dir=output_dir,
        model=vertex_detector.model,  # Usando el modelo de vértices/máscara
        conf_threshold=0.35,  # Valor reducido para mejorar detección
        mask=palanquilla_mask
    )
    
    # 10. Analizar romboidad (siempre se ejecuta, no depende de detecciones)
    romboidad_processor = models['processors']['romboidad']
    romboidad_results = romboidad_processor.process(
        image,
        vertices,
        image_name=image_name,
        output_dir=output_dir
    )
    
    # Procesar cada tipo de defecto con su procesador específico
    results = {}  # Inicializar el diccionario de resultados
    results['abombamiento'] = abombamiento_results
    results['romboidad'] = romboidad_results
    
    for defect_type, defects in classified_detections.items():
        if defects and defect_type in models['processors']:
            processor = models['processors'][defect_type]
            # Pasar el nombre de la imagen y el directorio de salida
            results[defect_type] = processor.process(
                defects, 
                image, 
                vertices, 
                zone_masks,
                image_name=image_name,
                output_dir=output_dir
            )
    
    # Si se detectaron etiquetas, procesarlas con OCR
    if etiqueta_detections and 'etiqueta' in models['processors']:
        print(f"Procesando {len(etiqueta_detections)} etiqueta(s) con OCR...")
        label_extractor = models['processors']['etiqueta']
        label_results = label_extractor.process(
            etiqueta_detections,
            image,
            corners=vertices,
            zone_masks=zone_masks,
            image_name=image_name,
            output_dir=output_dir
        )
        
        # Añadir los resultados de las etiquetas a los resultados
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
        'image_procesada': image,
        'rotacion_info': rotacion_info  # Nueva clave con información de rotación
    }

def output_fn(prediction_results, output_dir, input_data):
    """
    Guarda los resultados del análisis en la estructura de carpetas adecuada
    
    Args:
        prediction_results: Resultados del análisis
        output_dir: Directorio base donde guardar los resultados
        input_data: Información de la imagen de entrada
        
    Returns:
        Dictionary con las rutas donde se guardaron los resultados
    """
    # Importar función para escribir archivos con codificación UTF-8
    from utils.utils import safe_write_file
    
    # Extraer información básica
    name = input_data['name']
    ext = input_data['ext']
    image = input_data['image']
    
    # Crear carpeta principal para esta imagen
    image_output_dir = os.path.join(output_dir, name)
    os.makedirs(image_output_dir, exist_ok=True)
    
    output_paths = {}
    
    # Guardar la imagen de zonas
    zones_img = prediction_results['zones_img']
    zones_path = os.path.join(image_output_dir, f"{name}_zonas{ext}")
    cv2.imwrite(zones_path, zones_img)
    output_paths['zones_img'] = zones_path
    
    # Guardar visualización de vértices
    vertices = prediction_results['vertices']
    vertices_img = prediction_results['image_procesada'].copy()
    
    # Dibujar el polígono formado por los vértices
    cv2.polylines(vertices_img, [vertices], True, (0, 255, 0), 3)
    
    # Dibujar y numerar los vértices
    for i, vertice in enumerate(vertices):
        cv2.circle(vertices_img, tuple(vertice), 10, (0, 0, 255), -1)
        cv2.putText(vertices_img, str(i+1), tuple(vertice), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2, cv2.LINE_AA)
    
    vertices_path = os.path.join(image_output_dir, f"{name}_vertices{ext}")
    cv2.imwrite(vertices_path, vertices_img)
    output_paths['vertices_img'] = vertices_path
    
    # Guardar la máscara de la palanquilla si existe
    if 'palanquilla_mask' in prediction_results and prediction_results['palanquilla_mask'] is not None:
        mask_path = os.path.join(image_output_dir, f"{name}_palanquilla_mask.png")
        cv2.imwrite(mask_path, prediction_results['palanquilla_mask'])
        output_paths['palanquilla_mask'] = mask_path
    
    # Guardar imagen con todos los defectos detectados
    result_image = visualize_results_with_masks(
        prediction_results['image_procesada'], 
        prediction_results['classified_detections']
    )
    result_path = os.path.join(image_output_dir, f"{name}_resultado{ext}")
    cv2.imwrite(result_path, result_image)
    output_paths['result_img'] = result_path
    
    # Guardar información de rotación si existe
    if 'rotacion_info' in prediction_results:
        rotacion_info = prediction_results['rotacion_info']
        rotacion_path = os.path.join(image_output_dir, f"{name}_rotacion_info.txt")
        
        with open(rotacion_path, 'w', encoding='utf-8') as f:
            f.write(f"INFORMACIÓN DE ROTACIÓN - {name}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Ángulo de rotación: {rotacion_info['angulo']}°\n")
            f.write(f"Lado de la etiqueta: {rotacion_info['lado_etiqueta']}\n")
            
            if rotacion_info['etiqueta_bbox']:
                x1, y1, x2, y2 = rotacion_info['etiqueta_bbox']
                f.write(f"Bounding box de etiqueta: ({x1}, {y1}, {x2}, {y2})\n")
                f.write(f"Ancho etiqueta: {x2-x1} píxeles\n")
                f.write(f"Alto etiqueta: {y2-y1} píxeles\n")
        
        output_paths['rotacion_info'] = rotacion_path
    
    # Guardar los resultados de cada tipo de defecto
    classified_detections = prediction_results['classified_detections']
    
    # Llamar a cada procesador específico para guardar sus resultados
    for defect_type, defects in classified_detections.items():
        if defects and defect_type in prediction_results['processed_results']:
            # Crear directorio para este tipo de defecto
            defect_dir = os.path.join(image_output_dir, defect_type)
            os.makedirs(defect_dir, exist_ok=True)
            
            # Guardar los resultados específicos de este tipo de defecto
            processed_results = prediction_results['processed_results'][defect_type]
            
            # Los reportes ya deberían estar guardados en la ubicación correcta
            if 'report_paths' in processed_results:
                output_paths[f'{defect_type}_reports'] = processed_results['report_paths']
            
            if 'visualizations' in processed_results:
                for viz_name, viz_img in processed_results['visualizations'].items():
                    viz_path = os.path.join(defect_dir, f"{name}_{defect_type}_{viz_name}{ext}")
                    cv2.imwrite(viz_path, viz_img)
                    
                    if defect_type not in output_paths:
                        output_paths[defect_type] = {}
                    
                    if 'visualizations' not in output_paths[defect_type]:
                        output_paths[defect_type]['visualizations'] = {}
                    
                    output_paths[defect_type]['visualizations'][viz_name] = viz_path
    
    # Guardar resultados de análisis de propiedades geométricas (abombamiento y romboidad)
    for property_type in ['abombamiento', 'romboidad']:
        if property_type in prediction_results['processed_results']:
            processed_results = prediction_results['processed_results'][property_type]
            
            # Los reportes ya deberían estar guardados en la ubicación correcta
            if 'report_paths' in processed_results:
                output_paths[f'{property_type}_reports'] = processed_results['report_paths']
            
            if 'visualizations' in processed_results:
                # Crear directorio para este tipo de propiedad
                property_dir = os.path.join(image_output_dir, property_type)
                os.makedirs(property_dir, exist_ok=True)
                
                for viz_name, viz_img in processed_results['visualizations'].items():
                    viz_path = os.path.join(property_dir, f"{name}_{property_type}_{viz_name}{ext}")
                    cv2.imwrite(viz_path, viz_img)
                    
                    if property_type not in output_paths:
                        output_paths[property_type] = {}
                    
                    if 'visualizations' not in output_paths[property_type]:
                        output_paths[property_type]['visualizations'] = {}
                    
                    output_paths[property_type]['visualizations'][viz_name] = viz_path
    
    # Guardar resultados de etiquetas específicamente
    if 'etiqueta' in prediction_results['processed_results']:
        etiqueta_results = prediction_results['processed_results']['etiqueta']
        
        # Crear directorio para etiquetas
        etiqueta_dir = os.path.join(image_output_dir, "etiqueta")
        os.makedirs(etiqueta_dir, exist_ok=True)
        
        # Guardar rutas de reportes
        if 'report_paths' in etiqueta_results:
            output_paths['etiqueta_reports'] = etiqueta_results['report_paths']
        
        # Guardar visualizaciones
        if 'visualizations' in etiqueta_results:
            if 'etiqueta' not in output_paths:
                output_paths['etiqueta'] = {}
            
            output_paths['etiqueta']['visualizations'] = {}
            
            for viz_name, viz_img in etiqueta_results['visualizations'].items():
                viz_path = os.path.join(etiqueta_dir, f"{name}_{viz_name}{ext}")
                cv2.imwrite(viz_path, viz_img)
                output_paths['etiqueta']['visualizations'][viz_name] = viz_path
    
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


def process_directory(directory_path, output_dir=None, model_dir=None):
    """
    Procesa todas las imágenes en un directorio
    
    Args:
        directory_path: Ruta al directorio
        output_dir: Directorio donde guardar los resultados
        model_dir: Directorio donde están los modelos
        
    Returns:
        success_count: Número de imágenes procesadas exitosamente
    """
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
            print(f"Procesada imagen {success_count}/{len(image_files)}: {img_path.name}")
            
        except Exception as e:
            print(f"Error al procesar {img_path}: {e}")
    
    print(f"Procesadas {success_count} de {len(image_files)} imágenes")
    return success_count


def main():
    """
    Función principal para ejecutar desde línea de comandos
    """
    parser = argparse.ArgumentParser(description="Procesador de Palanquillas")
    
    parser.add_argument("--mode", type=str, choices=["image", "directory"], default="image",
                        help="Modo de procesamiento: 'image' para una imagen, 'directory' para un directorio")
    
    parser.add_argument("--input", type=str, required=True,
                       help="Ruta a la imagen o directorio a procesar")
    
    parser.add_argument("--output", type=str, 
                       default=r"D:\Trabajo modelos\PACC\YOLOv12 - copia\Clasificacion_por_zonas",
                       help="Directorio donde guardar los resultados")
    
    
    parser.add_argument("--model-dir", type=str, default=None,
                       help="Directorio donde se encuentran los modelos (opcional)")
    
    args = parser.parse_args()
    
    # Configuración según el modo
    if args.mode == "image":
        success = process_image(args.input, args.output, args.model_dir)
        if success:
            print("\nProcesamiento completado exitosamente.")
            print(f"Resultados guardados en: {os.path.join(args.output, os.path.splitext(os.path.basename(args.input))[0])}")
        else:
            print("\nHubo errores durante el procesamiento.")
    
    elif args.mode == "directory":
        success_count = process_directory(args.input, args.output, args.model_dir)
        if success_count > 0:
            print(f"\nProcesamiento de directorio completado.")
            print(f"Se procesaron correctamente {success_count} imágenes.")
            print(f"Resultados guardados en: {args.output}")
        else:
            print("\nNo se pudo procesar ninguna imagen del directorio.")


if __name__ == "__main__":
    # Ejecutar el modo interactivo si no se proporcionan argumentos
    if len(sys.argv) == 1:
        print("=== PROCESADOR DE PALANQUILLAS CON ANÁLISIS DE DEFECTOS ===")
        
        print("\nModo de procesamiento:")
        print("1. Procesar una imagen")
        print("2. Procesar un directorio completo") 
        
        
        
        # Procesar un directorio
        dir_path = r"D:\Trabajo modelos\PACC\YOLOv12 - copia\pruebas diagonales"
        output_dir = r"D:\Trabajo modelos\PACC\YOLOv12 - copia\Clasificacion_por_zonas"
        
        
        success_count = process_directory(dir_path, output_dir)
        if success_count > 0:
            print(f"\nProcesamiento de directorio completado.")
            print(f"Se procesaron correctamente {success_count} imágenes.")
            print(f"Resultados guardados en: {output_dir}")
        else:
            print("\nNo se pudo procesar ninguna imagen del directorio.")
        
        
    else:
        main()