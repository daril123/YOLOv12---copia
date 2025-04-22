import cv2
import numpy as np
import os

from utils.utils import draw_arrow
from common.detector import analizar_direccion_grieta_mask  # Cambio importante aquí

def classify_defects_with_masks(detections, zone_masks, image, yolo_result):
    """
    Clasifica los defectos según su tipo y ubicación utilizando las máscaras de segmentación
    en lugar de bounding boxes para mayor precisión.
    
    Args:
        detections: Diccionario con detecciones por clase
        zone_masks: Diccionario con máscaras para cada zona
        image: Imagen original donde se detectaron los defectos
        yolo_result: Resultado original de YOLO con máscaras de segmentación
    
    Returns:
        classified_detections: Diccionario con defectos clasificados
    """
    # Inicializar diccionario de clasificaciones
    classified_detections = {
        'grietas_diagonales': [],
        'grietas_medio_camino': [],
        'grietas_corner': [],
        'inclusion_no_metalica': [],
        'nucleo_esponjoso': [],
        'estrella': [],
        'rechupe': [],
        'sopladura': []
    }
    
    # Obtener máscaras de YOLO si están disponibles
    masks = None
    if hasattr(yolo_result, 'masks') and yolo_result.masks is not None:
        masks = yolo_result.masks
    
    # PASO 1: PROCESAMIENTO DE GRIETAS
    if 'grieta' in detections:
        # Colección temporal de grietas normales (no diagonales)
        grietas_normales = []
        
        # Clasificar grietas
        for detection in detections['grieta']:
            x1, y1, x2, y2 = detection['bbox']
            
            # Obtener la máscara si está disponible
            mask = None
            if masks is not None and 'mask_index' in detection:
                mask_idx = detection['mask_index']
                mask_data = masks[mask_idx].data.cpu().numpy()
                mask = (mask_data > 0.5).astype(np.uint8) * 255
                
                # Asegurarse de que la máscara tiene el tamaño correcto
                if len(mask.shape) > 2:
                    mask = mask[0]
                if mask.shape != (image.shape[0], image.shape[1]):
                    mask_resized = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    mask_resized[y1:y2, x1:x2] = cv2.resize(mask, (x2-x1, y2-y1))
                    mask = mask_resized
            else:
                # Si no hay máscara disponible, crear una a partir del ROI
                roi = image[y1:y2, x1:x2].copy()
                
                # Convertir a escala de grises
                if len(roi.shape) == 3:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    roi_gray = roi
                
                # Binarizar la imagen para obtener una máscara
                _, roi_mask = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Crear máscara del tamaño de la imagen
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                mask[y1:y2, x1:x2] = roi_mask
            
            # Guardar la máscara en la detección
            detection['mask'] = mask
            
            # Analizar dirección usando la máscara - Usar la importación correcta
            angulo, box, direccion_texto, es_diagonal = analizar_direccion_grieta_mask(mask[y1:y2, x1:x2])
            
            # Almacenar información de dirección
            detection['angulo'] = angulo
            detection['direccion'] = direccion_texto
            detection['es_diagonal'] = es_diagonal
            
            # Almacenar box si existe
            if box is not None:
                box_global = box.copy()
                box_global[:, 0] += x1
                box_global[:, 1] += y1
                detection['box_points'] = box_global
            else:
                detection['box_points'] = None
            
            # Clasificar según dirección (primero separar diagonales)
            if es_diagonal:
                # Si es diagonal, clasificarla como tal (independientemente de la zona)
                classified_detections['grietas_diagonales'].append(detection)
            else:
                # Si no es diagonal, la añadimos a la lista temporal para posterior
                # clasificación por zona
                grietas_normales.append(detection)
        
        # PASO 2: CLASIFICAR LAS GRIETAS NORMALES POR ZONA
        for detection in grietas_normales:
            mask = detection['mask']
            
            # Verificar en qué zona está la grieta usando la máscara
            in_yellow = cv2.countNonZero(cv2.bitwise_and(mask, zone_masks['amarillo'])) > 0
            in_red = cv2.countNonZero(cv2.bitwise_and(mask, zone_masks['rojo'])) > 0
            in_purple = cv2.countNonZero(cv2.bitwise_and(mask, zone_masks['morado'])) > 0
            
            # Clasificar según ubicación
            if in_purple:
                # Si está en extremos medios, es grieta corner
                classified_detections['grietas_corner'].append(detection)
            elif in_yellow and not in_red:
                # Si está en la zona amarilla pero no en la roja, es grieta de medio camino
                classified_detections['grietas_medio_camino'].append(detection)
    
    # PASO 3: PROCESAR PUNTOS
    if 'puntos' in detections:
        for detection in detections['puntos']:
            x1, y1, x2, y2 = detection['bbox']
            
            # Obtener la máscara si está disponible
            mask = None
            if masks is not None and 'mask_index' in detection:
                mask_idx = detection['mask_index']
                mask_data = masks[mask_idx].data.cpu().numpy()
                mask = (mask_data > 0.5).astype(np.uint8) * 255
                
                # Asegurarse de que la máscara tiene el tamaño correcto
                if len(mask.shape) > 2:
                    mask = mask[0]
                if mask.shape != (image.shape[0], image.shape[1]):
                    mask_resized = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    mask_resized[y1:y2, x1:x2] = cv2.resize(mask, (x2-x1, y2-y1))
                    mask = mask_resized
            else:
                # Si no hay máscara disponible, crear una a partir del ROI
                roi = image[y1:y2, x1:x2].copy()
                
                # Convertir a escala de grises
                if len(roi.shape) == 3:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    roi_gray = roi
                
                # Binarizar la imagen para obtener una máscara
                _, roi_mask = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Crear máscara del tamaño de la imagen
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                mask[y1:y2, x1:x2] = roi_mask
            
            # Guardar la máscara en la detección
            detection['mask'] = mask
            
            # Verificar en qué zona está el punto usando la máscara
            in_red = cv2.countNonZero(cv2.bitwise_and(mask, zone_masks['rojo'])) > 0
            in_yellow = cv2.countNonZero(cv2.bitwise_and(mask, zone_masks['amarillo'])) > 0
            in_green = cv2.countNonZero(cv2.bitwise_and(mask, zone_masks['verde'])) > 0
            
            # Clasificar según ubicación
            if in_red and in_yellow:
                classified_detections['nucleo_esponjoso'].append(detection)
            elif in_green and in_yellow:
                classified_detections['inclusion_no_metalica'].append(detection)
    
    # PASO 4: PROCESAR ESTRELLAS (siempre en zona roja)
    if 'estrella' in detections:
        for detection in detections['estrella']:
            x1, y1, x2, y2 = detection['bbox']
            
            # Obtener la máscara si está disponible
            mask = None
            if masks is not None and 'mask_index' in detection:
                mask_idx = detection['mask_index']
                mask_data = masks[mask_idx].data.cpu().numpy()
                mask = (mask_data > 0.5).astype(np.uint8) * 255
                
                # Asegurarse de que la máscara tiene el tamaño correcto
                if len(mask.shape) > 2:
                    mask = mask[0]
                if mask.shape != (image.shape[0], image.shape[1]):
                    mask_resized = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    mask_resized[y1:y2, x1:x2] = cv2.resize(mask, (x2-x1, y2-y1))
                    mask = mask_resized
            else:
                # Si no hay máscara disponible, crear una a partir del ROI
                roi = image[y1:y2, x1:x2].copy()
                
                # Convertir a escala de grises
                if len(roi.shape) == 3:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    roi_gray = roi
                
                # Binarizar la imagen para obtener una máscara
                _, roi_mask = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Crear máscara del tamaño de la imagen
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                mask[y1:y2, x1:x2] = roi_mask
            
            # Guardar la máscara en la detección
            detection['mask'] = mask
            
            # Las estrellas siempre van a la lista de estrellas (típicamente en zona roja)
            classified_detections['estrella'].append(detection)
    
    # PASO 5: PROCESAR RECHUPES (siempre en zona roja)
    if 'rechupe' in detections:
        for detection in detections['rechupe']:
            x1, y1, x2, y2 = detection['bbox']
            
            # Obtener la máscara si está disponible
            mask = None
            if masks is not None and 'mask_index' in detection:
                mask_idx = detection['mask_index']
                mask_data = masks[mask_idx].data.cpu().numpy()
                mask = (mask_data > 0.5).astype(np.uint8) * 255
                
                # Asegurarse de que la máscara tiene el tamaño correcto
                if len(mask.shape) > 2:
                    mask = mask[0]
                if mask.shape != (image.shape[0], image.shape[1]):
                    mask_resized = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    mask_resized[y1:y2, x1:x2] = cv2.resize(mask, (x2-x1, y2-y1))
                    mask = mask_resized
            else:
                # Si no hay máscara disponible, crear una a partir del ROI
                roi = image[y1:y2, x1:x2].copy()
                
                # Convertir a escala de grises
                if len(roi.shape) == 3:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    roi_gray = roi
                
                # Binarizar la imagen para obtener una máscara
                _, roi_mask = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Crear máscara del tamaño de la imagen
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                mask[y1:y2, x1:x2] = roi_mask
            
            # Guardar la máscara en la detección
            detection['mask'] = mask
            
            # Los rechupes siempre van a la lista de rechupes (típicamente en zona roja)
            classified_detections['rechupe'].append(detection)
    
    # PASO 6: PROCESAR SOPLADURAS (pueden estar en cualquier zona)
    if 'sopladura' in detections:
        for detection in detections['sopladura']:
            x1, y1, x2, y2 = detection['bbox']
            
            # Obtener la máscara si está disponible
            mask = None
            if masks is not None and 'mask_index' in detection:
                mask_idx = detection['mask_index']
                mask_data = masks[mask_idx].data.cpu().numpy()
                mask = (mask_data > 0.5).astype(np.uint8) * 255
                
                # Asegurarse de que la máscara tiene el tamaño correcto
                if len(mask.shape) > 2:
                    mask = mask[0]
                if mask.shape != (image.shape[0], image.shape[1]):
                    mask_resized = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    mask_resized[y1:y2, x1:x2] = cv2.resize(mask, (x2-x1, y2-y1))
                    mask = mask_resized
            else:
                # Si no hay máscara disponible, crear una a partir del ROI
                roi = image[y1:y2, x1:x2].copy()
                
                # Convertir a escala de grises
                if len(roi.shape) == 3:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    roi_gray = roi
                
                # Binarizar la imagen para obtener una máscara
                _, roi_mask = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Crear máscara del tamaño de la imagen
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                mask[y1:y2, x1:x2] = roi_mask
            
            # Guardar la máscara en la detección
            detection['mask'] = mask
            
            # Las sopladuras se clasifican como un tipo específico
            classified_detections['sopladura'].append(detection)
    
    return classified_detections

def visualize_results_with_masks(image, classified_detections):
    """
    Visualiza los resultados de clasificación utilizando las máscaras de segmentación
    y agrega vectores de medición para las grietas
    
    Args:
        image: Imagen original
        classified_detections: Diccionario con defectos clasificados
    
    Returns:
        result_image: Imagen con defectos destacados
    """
    # Hacer una copia de la imagen
    result_image = image.copy()
    
    # Colores para diferentes tipos de defectos (BGR)
    color_map = {
        'grietas_medio_camino': (0, 165, 255),    # Naranja
        'grietas_corner': (255, 0, 0),           # Azul
        'grietas_diagonales': (0, 0, 128),       # Rojo oscuro
        'inclusion_no_metalica': (0, 255, 255),  # Amarillo
        'nucleo_esponjoso': (0, 0, 255),         # Rojo
        'estrella': (255, 0, 255),               # Magenta
        'rechupe': (128, 0, 128),                # Morado
        'sopladura': (0, 128, 128),              # Verde-azulado
        'romboidad': (200, 200, 200),            # Gris claro
        'abombamiento': (100, 100, 255)          # Rosa
    }
    
    # Dibujar cada tipo de defecto
    for defect_type, detections in classified_detections.items():
        color = color_map.get(defect_type, (255, 255, 255))
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Obtener la confianza (con valor predeterminado si no existe)
            conf = detection.get('conf', 1.0)
            
            # Obtener la clase (con valor predeterminado si no existe)
            cls = detection.get('class', defect_type)
            
            # Si hay máscara, dibujar el contorno en lugar del rectángulo
            if 'mask' in detection and detection['mask'] is not None:
                mask = detection['mask']
                
                # Encontrar contornos en la máscara
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Dibujar los contornos
                cv2.drawContours(result_image, contours, -1, color, 2)
                
                # Calcular el centro del contorno para las grietas
                if defect_type in ['grietas_diagonales', 'grietas_medio_camino', 'grietas_corner']:
                    # Encontrar el contorno principal
                    if contours:
                        main_contour = max(contours, key=cv2.contourArea)
                        
                        # Calcular momentos y centro
                        M = cv2.moments(main_contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            center = (cx, cy)
                            
                            # Dibujar el centro
                            cv2.circle(result_image, center, 5, (255, 255, 255), -1)
                            
                            # Si hay medidas, mostrar vectores
                            if 'detection_metrics' in detection:
                                metrics = detection['detection_metrics']
                                
                                # Extraer las medidas
                                L = metrics.get('L', 0)
                                e = metrics.get('e', 0)
                                D = metrics.get('D', 0)
                                
                                # Si hay información de puntos extremos, dibujar L
                                if 'global_pt1' in metrics and 'global_pt2' in metrics:
                                    global_pt1 = metrics['global_pt1']
                                    global_pt2 = metrics['global_pt2']
                                    
                                    # Dibujar los puntos extremos
                                    cv2.circle(result_image, global_pt1, 5, (255, 0, 0), -1)
                                    cv2.circle(result_image, global_pt2, 5, (255, 0, 0), -1)
                                    
                                    # Dibujar flecha para L
                                    draw_arrow(result_image, global_pt1, global_pt2, (0, 255, 255), 2, 10, f"L={L:.1f}px")
                                
                                # Si hay información de punto del contorno y punto del borde, dibujar D
                                if 'contour_point' in metrics and 'edge_point' in metrics:
                                    contour_point = metrics['contour_point']
                                    edge_point = metrics['edge_point']
                                    
                                    # Dibujar un rectángulo rojo en el punto del contorno
                                    cv2.rectangle(result_image, 
                                                 (contour_point[0]-5, contour_point[1]-5), 
                                                 (contour_point[0]+5, contour_point[1]+5), 
                                                 (0, 0, 255), 2)
                                    
                                    # Dibujar el punto del borde
                                    cv2.circle(result_image, edge_point, 5, (255, 255, 0), -1)
                                    
                                    # Dibujar flecha para D
                                    draw_arrow(result_image, contour_point, edge_point, (0, 165, 255), 2, 15, f"D={D:.1f}px")
            else:
                # Si no hay máscara, usar el rectángulo
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Añadir etiqueta con confianza y dirección si existe
            label = ""
            
            # Para análisis geométrico, mostrar etiqueta especial
            if defect_type in ['romboidad', 'abombamiento']:
                label = f"{defect_type.capitalize()} (Análisis Geométrico)"
            # Para grietas, incluir dirección si está disponible
            elif (defect_type in ['grietas_diagonales', 'grietas_medio_camino', 'grietas_corner']) and 'direccion' in detection:
                label = f"{defect_type.replace('_', ' ').title()} ({conf:.2f}) - {detection['direccion']}"
            # Para el resto de defectos, mostrar tipo y confianza
            else:
                label = f"{defect_type.replace('_', ' ').title()} ({conf:.2f})"
            
            y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(result_image, label, (x1, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Para grietas, dibuja el ángulo y el rectángulo de orientación
            if (defect_type in ['grietas_diagonales', 'grietas_medio_camino', 'grietas_corner']) and 'angulo' in detection and detection['angulo'] is not None:
                angulo_txt = f"Ángulo: {detection['angulo']:.1f}°"
                cv2.putText(result_image, angulo_txt, (x1, y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Dibujar el rectángulo orientado que muestra la dirección de la grieta
                if 'box_points' in detection and detection['box_points'] is not None:
                    cv2.drawContours(result_image, [detection['box_points']], 0, (0, 255, 0), 2)
    
    return result_image