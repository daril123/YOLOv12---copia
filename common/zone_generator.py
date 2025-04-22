import cv2
import numpy as np
from utils.utils import interpolate_point, interpolate_quadrilateral, get_mask_center

def generate_zone_masks(image_shape, corners):
    """
    Genera máscaras binarias para cada zona adaptadas a la forma de la palanquilla
    utilizando interpolación para mantener proporciones correctas
    
    Args:
        image_shape: Tupla (altura, ancho) de la imagen original
        corners: Esquinas de la palanquilla [top-left, top-right, bottom-right, bottom-left]
    
    Returns:
        zone_masks: Diccionario con máscaras para cada zona
    """
    # Crear una imagen en blanco del tamaño de la imagen original
    h, w = image_shape[:2]
    blank_image = np.zeros((h, w), dtype=np.uint8)
    
    # Definir proporciones para las zonas (como porcentajes)
    lateral_width_pct = 0.15
    top_bottom_height_pct = 0.25
    center_size_pct = 0.20
    
    # Crear una máscara completa de la palanquilla
    palanquilla_mask = blank_image.copy()
    cv2.fillPoly(palanquilla_mask, [corners], 255)
    
    # Inicializar el diccionario de máscaras de zona
    zone_masks = {
        'rojo': blank_image.copy(),
        'amarillo': blank_image.copy(),
        'verde': blank_image.copy(),
        'morado': blank_image.copy()
    }
    
    # Extraer las esquinas para mayor claridad
    tl, tr, br, bl = corners
    
    # Calcular puntos internos importantes usando interpolación
    
    # Puntos para la zona amarilla (intermedia)
    tl_yellow = interpolate_quadrilateral(corners, lateral_width_pct, top_bottom_height_pct)
    tr_yellow = interpolate_quadrilateral(corners, 1-lateral_width_pct, top_bottom_height_pct)
    br_yellow = interpolate_quadrilateral(corners, 1-lateral_width_pct, 1-top_bottom_height_pct)
    bl_yellow = interpolate_quadrilateral(corners, lateral_width_pct, 1-top_bottom_height_pct)
    
    # Puntos para la zona verde (intermedia exterior)
    # Superior centro
    tl_green_top = interpolate_quadrilateral(corners, lateral_width_pct*2, 0)
    tr_green_top = interpolate_quadrilateral(corners, 1-lateral_width_pct*2, 0)
    bl_green_top = interpolate_quadrilateral(corners, lateral_width_pct*2, top_bottom_height_pct)
    br_green_top = interpolate_quadrilateral(corners, 1-lateral_width_pct*2, top_bottom_height_pct)
    
    # Inferior centro
    tl_green_bottom = interpolate_quadrilateral(corners, lateral_width_pct*2, 1-top_bottom_height_pct)
    tr_green_bottom = interpolate_quadrilateral(corners, 1-lateral_width_pct*2, 1-top_bottom_height_pct)
    bl_green_bottom = interpolate_quadrilateral(corners, lateral_width_pct*2, 1)
    br_green_bottom = interpolate_quadrilateral(corners, 1-lateral_width_pct*2, 1)
    
    # Izquierda
    tl_green_left = interpolate_quadrilateral(corners, 0, top_bottom_height_pct)
    tr_green_left = interpolate_quadrilateral(corners, lateral_width_pct, top_bottom_height_pct)
    bl_green_left = interpolate_quadrilateral(corners, 0, 1-top_bottom_height_pct)
    br_green_left = interpolate_quadrilateral(corners, lateral_width_pct, 1-top_bottom_height_pct)
    
    # Derecha
    tl_green_right = interpolate_quadrilateral(corners, 1-lateral_width_pct, top_bottom_height_pct)
    tr_green_right = interpolate_quadrilateral(corners, 1, top_bottom_height_pct)
    bl_green_right = interpolate_quadrilateral(corners, 1-lateral_width_pct, 1-top_bottom_height_pct)
    br_green_right = interpolate_quadrilateral(corners, 1, 1-top_bottom_height_pct)
    
    # Puntos para zona roja (centro)
    # Calcular el centro de la palanquilla
    center = interpolate_quadrilateral(corners, 0.5, 0.5)
    # Calcular el tamaño promedio de la palanquilla
    width = np.linalg.norm(np.array(tr) - np.array(tl))
    height = np.linalg.norm(np.array(bl) - np.array(tl))
    avg_size = (width + height) / 2
    half_center = int(avg_size * center_size_pct / 2)
    
    # En lugar de usar un rectángulo simple, usaremos interpolación desde el centro
    # para mantener la perspectiva correcta
    center_point = np.array(center)
    # Calcular vectores de dirección para mantener la perspectiva
    vec_top = (np.array(interpolate_quadrilateral(corners, 0.5, 0)) - center_point) / np.linalg.norm(np.array(interpolate_quadrilateral(corners, 0.5, 0)) - center_point)
    vec_right = (np.array(interpolate_quadrilateral(corners, 1, 0.5)) - center_point) / np.linalg.norm(np.array(interpolate_quadrilateral(corners, 1, 0.5)) - center_point)
    vec_bottom = (np.array(interpolate_quadrilateral(corners, 0.5, 1)) - center_point) / np.linalg.norm(np.array(interpolate_quadrilateral(corners, 0.5, 1)) - center_point)
    vec_left = (np.array(interpolate_quadrilateral(corners, 0, 0.5)) - center_point) / np.linalg.norm(np.array(interpolate_quadrilateral(corners, 0, 0.5)) - center_point)
    
    # Calcular las cuatro esquinas del centro usando los vectores de dirección
    tl_red = (center_point - vec_left * half_center - vec_top * half_center).astype(int)
    tr_red = (center_point + vec_right * half_center - vec_top * half_center).astype(int)
    br_red = (center_point + vec_right * half_center + vec_bottom * half_center).astype(int)
    bl_red = (center_point - vec_left * half_center + vec_bottom * half_center).astype(int)
    
    # CREAR MÁSCARAS DE ZONA
    
    # Zona roja (centro)
    red_roi = blank_image.copy()
    cv2.fillPoly(red_roi, [np.array([tl_red, tr_red, br_red, bl_red])], 255)
    zone_masks['rojo'] = cv2.bitwise_and(red_roi, palanquilla_mask)
    
    # Zona amarilla (intermedia)
    yellow_roi = blank_image.copy()
    cv2.fillPoly(yellow_roi, [np.array([tl_yellow, tr_yellow, br_yellow, bl_yellow])], 255)
    # Quitar la zona roja de la zona amarilla
    yellow_mask = cv2.bitwise_and(yellow_roi, palanquilla_mask)
    yellow_mask = cv2.bitwise_and(yellow_mask, cv2.bitwise_not(zone_masks['rojo']))
    zone_masks['amarillo'] = yellow_mask
    
    # Zona verde (intermedio exterior)
    green_roi = blank_image.copy()
    
    # Superior centro
    cv2.fillPoly(green_roi, [np.array([tl_green_top, tr_green_top, br_green_top, bl_green_top])], 255)
    # Inferior centro
    cv2.fillPoly(green_roi, [np.array([tl_green_bottom, tr_green_bottom, br_green_bottom, bl_green_bottom])], 255)
    # Izquierda
    cv2.fillPoly(green_roi, [np.array([tl_green_left, tr_green_left, br_green_left, bl_green_left])], 255)
    # Derecha
    cv2.fillPoly(green_roi, [np.array([tl_green_right, tr_green_right, br_green_right, bl_green_right])], 255)
    
    # Aplicar la máscara de la palanquilla y eliminar superposiciones con otras zonas
    green_mask = cv2.bitwise_and(green_roi, palanquilla_mask)
    green_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(zone_masks['rojo']))
    green_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(zone_masks['amarillo']))
    zone_masks['verde'] = green_mask
    
    # Zona morada (extremos medios) - todo lo que queda
    purple_mask = cv2.bitwise_and(
        palanquilla_mask,
        cv2.bitwise_not(cv2.bitwise_or(
            cv2.bitwise_or(zone_masks['rojo'], zone_masks['amarillo']),
            zone_masks['verde']
        ))
    )
    zone_masks['morado'] = purple_mask
    
    return zone_masks

def visualize_zones(image, corners):
    """
    Dibuja las zonas en la imagen original
    
    Args:
        image: Imagen original
        corners: Esquinas de la palanquilla
    
    Returns:
        imagen_con_zonas: Imagen con zonas dibujadas
        zone_masks: Diccionario con máscaras para cada zona
    """
    # Generar las máscaras de zona adaptadas a la palanquilla
    zone_masks = generate_zone_masks(image.shape, corners)
    
    # Colores en formato BGR (OpenCV)
    colors = {
        'rojo': (0, 0, 255),      # Medio (centro)
        'amarillo': (0, 255, 255), # Intermedio
        'verde': (0, 255, 0),     # Intermedio exterior
        'morado': (255, 0, 255),  # Extremos medios
        'negro': (0, 0, 0)        # Bordes
    }
    
    # Hacer una copia de la imagen
    imagen_con_zonas = image.copy()
    
    # Crear capa de overlay para transparencia
    overlay = imagen_con_zonas.copy()
    
    # Colorear cada zona con su color correspondiente
    for zone_name, color in colors.items():
        if zone_name == 'negro':
            continue  # Saltamos el color negro que es para los bordes
            
        # Obtener la máscara de esta zona
        zone_mask = zone_masks[zone_name]
        
        # Crear una versión a color de la máscara
        colored_mask = np.zeros_like(imagen_con_zonas)
        colored_mask[zone_mask > 0] = color
        
        # Superponer la máscara coloreada en la imagen de overlay
        overlay = np.where(zone_mask[:, :, np.newaxis] > 0, colored_mask, overlay)
    
    # Aplicar transparencia
    alpha = 0.3  # Factor de transparencia
    cv2.addWeighted(overlay, alpha, imagen_con_zonas, 1 - alpha, 0, imagen_con_zonas)
    
    # Dibujar bordes de las zonas (usando las máscaras)
    for zone_name, mask in zone_masks.items():
        # Encontrar contornos de la máscara, incluyendo los internos
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imagen_con_zonas, contours, -1, colors['negro'], 2)
    
    # Añadir etiquetas para cada zona
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    # Etiquetas para cada zona
    zone_labels = {
        'rojo': "medio",
        'amarillo': "intermedio",
        'verde': "intermedio exterior",
        'morado': "extremos medios"
    }
    
    # Añadir etiquetas en los centros de las máscaras
    for zone_name, label in zone_labels.items():
        center = get_mask_center(zone_masks[zone_name])
        if center:
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = center[0] - text_size[0] // 2
            text_y = center[1] + text_size[1] // 2
            
            # Añadir fondo blanco
            cv2.rectangle(imagen_con_zonas, 
                         (text_x - 2, text_y - text_size[1] - 2),
                         (text_x + text_size[0] + 2, text_y + 2),
                         (255, 255, 255), -1)
            
            # Añadir texto
            cv2.putText(imagen_con_zonas, label, (text_x, text_y), 
                       font, font_scale, (0, 0, 0), font_thickness)
    
    return imagen_con_zonas, zone_masks