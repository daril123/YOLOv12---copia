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


def model_fn(model_dir=None):
    """
    Carga los modelos necesarios para la detección de vértices y defectos
    
    Args:
        model_dir: Directorio donde se encuentran los modelos (opcional)
        
    Returns:
        Dictionary con los modelos cargados
    """
    # Rutas predeterminadas si no se especifica un directorio
    vertex_model_path = r"D:\Trabajo modelos\PACC\YOLOv12 - copia\vertices\best.pt"
    defect_model_path = r"D:\Trabajo modelos\PACC\YOLOv12 - copia\PACC_results\defectos_tipoB_yolov11\weights\best.pt"
    
    if model_dir:
        # Si se proporciona un directorio, buscar los modelos allí
        vertex_path = os.path.join(model_dir, "vertex_model.pt")
        defect_path = os.path.join(model_dir, "defect_model.pt")
        
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
            'romboidad': romboidad_processor
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
    
    

    # 2. Detectar vértices de la palanquilla
    print(f"Detectando vértices en: {image_path}")
    try:
        vertices, success, palanquilla_mask = obtener_contorno_imagen(image, vertex_detector.model, vertex_detector.conf_threshold)
        
        if not success or vertices is None:
            print("Error: No se pudieron detectar los vértices correctamente. Usando método alternativo.")
            # Usar un método alternativo para detectar los vértices
            from common.detector import detect_palanquilla
            vertices, success, palanquilla_mask = detect_palanquilla(image)
            
            if not success or vertices is None:
                print("Error: También falló el método alternativo. Usando toda la imagen.")
                # Si todo falla, asegurar que palanquilla_mask sea None (será creado más tarde según los vértices)
                h, w = image.shape[:2]
                vertices = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
                palanquilla_mask = None
    except Exception as e:
        print(f"Error al detectar vértices: {e}")
        import traceback
        traceback.print_exc()
        h, w = image.shape[:2]
        vertices = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        palanquilla_mask = None
        success = False

    # 3. Rotar imagen para alinear con el lado más recto
    print(f"Rotando imagen para alinear palanquilla")
    try:
        # Determinar lado más recto para rotación
        if vertices is not None and success:
            try:
                contorno_aux, contorno_principal, _ = obtener_contorno_imagen(image, vertex_detector.model, 0.5)
                
                if contorno_aux is not None and contorno_principal is not None:
                    # Importar módulo de abombamiento para determinar el lado más recto
                    from defects.abombamiento.abombamiento import obtener_lado_rotacion_abombamiento
                    lado_recto = obtener_lado_rotacion_abombamiento(vertices, contorno_principal)
                    
                    # Rotar la imagen
                    image_rotada = rotar_imagen_lado(image, lado_recto, vertices)
                    
                    # Actualizar la imagen y detectar los nuevos vértices
                    image = image_rotada
                    vertices_aux, success_aux, palanquilla_mask_aux = obtener_contorno_imagen(image, vertex_detector.model, 0.5)
                    
                    if success_aux and vertices_aux is not None:
                        vertices = vertices_aux
                        palanquilla_mask = palanquilla_mask_aux
                    else:
                        print("Error: No se pudieron detectar los vértices después de la rotación. Manteniendo los vértices originales.")
            except Exception as inner_e:
                print(f"Error al determinar lado de rotación: {inner_e}")
        else:
            print("No se pudo rotar la imagen: vértices no disponibles.")
    except Exception as e:
        print(f"Error al rotar la imagen: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Generar máscaras de zona con los vértices detectados
    print(f"Generando máscaras de zonas")
    zones_img, zone_masks = visualize_zones(image, vertices)
    
    # 5. Detectar defectos con el detector de defectos
    print(f"Detectando defectos")
    detections, yolo_result = defect_detector.detect_defects(image)
    
    if not detections:
        print("No se detectaron defectos en esta imagen.")
    
    # 6. Clasificar los defectos según su posición en las zonas
    classified_detections = classify_defects_with_masks(detections, zone_masks, image, yolo_result)
    
    # 7. Analizar abombamiento (siempre se ejecuta, no depende de detecciones)
    # Para el abombamiento usar el modelo de vértices/máscara de palanquilla, NO el de defectos
    abombamiento_processor = models['processors']['abombamiento']
    abombamiento_results = abombamiento_processor.process(
        image,
        vertices,
        image_name=image_name,
        output_dir=output_dir,
        model=vertex_detector.model,  # Usando el modelo de vértices/máscara
        conf_threshold=0.5,  # Valor típico para detección de máscaras
        mask=palanquilla_mask
    )
    
    # 8. Analizar romboidad (siempre se ejecuta, no depende de detecciones)
    romboidad_processor = models['processors']['romboidad']
    romboidad_results = romboidad_processor.process(
        image,
        vertices,
        image_name=image_name,
        output_dir=output_dir
    )
    
    # Procesar cada tipo de defecto con su procesador específico
    results = {
        'abombamiento': abombamiento_results,
        'romboidad': romboidad_results
    }
    
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
    
    return {
        'vertices': vertices,
        'zones_img': zones_img,
        'zone_masks': zone_masks,
        'detections': detections,
        'yolo_result': yolo_result,
        'classified_detections': classified_detections,
        'processed_results': results,
        'palanquilla_mask': palanquilla_mask,
        'image_procesada': image
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
    
    return output_paths


def process_image(image_path, output_dir=None, model_dir=None):
    """
    Procesa una imagen con el flujo completo
    
    Args:
        image_path: Ruta a la imagen a procesar
        output_dir: Directorio donde guardar los resultados
        model_dir: Directorio donde están los modelos
        
    Returns:
        output_paths: Rutas donde se guardaron los resultados
    """
    # Valor predeterminado para el directorio de salida
    if output_dir is None:
        output_dir = r"D:\Trabajo modelos\PACC\YOLOv12 - copia\Clasificacion_por_zonas"
    
    try:
        # 1. Cargar los modelos
        models = model_fn(model_dir)
        
        # 2. Procesar la entrada
        input_data = input_fn(image_path)
        
        # Crear carpeta específica para esta imagen
        image_name = input_data['name']
        image_dir = os.path.join(output_dir, image_name)
        os.makedirs(image_dir, exist_ok=True)
        
        # 3. Realizar predicciones
        prediction_results = predict_fn(input_data, models, output_dir)
        
        # 4. Generar y guardar salidas
        output_paths = output_fn(prediction_results, output_dir, input_data)
        
        return output_paths
    
    except Exception as e:
        print(f"Error al procesar la imagen {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


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
        mode = input("Selecciona una opción (1/2): ")
        
        if mode == "1":
            # Procesar una imagen
            image_path = input("Ruta a la imagen: ")
            output_dir = input("Directorio de salida (dejar en blanco para usar predeterminado): ")
            
            if not output_dir.strip():
                output_dir = r"D:\Trabajo modelos\PACC\YOLOv12 - copia\Clasificacion_por_zonas"
            
            success = process_image(image_path, output_dir)
            if success:
                print("\nProcesamiento completado exitosamente.")
                print(f"Resultados guardados en: {os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0])}")
            else:
                print("\nHubo errores durante el procesamiento.")
        
        elif mode == "2":
            # Procesar un directorio
            dir_path = input("Ruta al directorio: ")
            output_dir = input("Directorio de salida (dejar en blanco para usar predeterminado): ")
            
            if not output_dir.strip():
                output_dir = r"D:\Trabajo modelos\PACC\YOLOv12 - copia\Clasificacion_por_zonas"
            
            success_count = process_directory(dir_path, output_dir)
            if success_count > 0:
                print(f"\nProcesamiento de directorio completado.")
                print(f"Se procesaron correctamente {success_count} imágenes.")
                print(f"Resultados guardados en: {output_dir}")
            else:
                print("\nNo se pudo procesar ninguna imagen del directorio.")
        
        else:
            print("Opción no válida")
    else:
        main()