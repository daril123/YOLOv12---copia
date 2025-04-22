import cv2
import numpy as np
from utils.utils import order_points

def detect_palanquilla(image):
    """
    Detecta los vértices de la palanquilla en la imagen
    
    Args:
        image: Imagen de entrada (BGR)
    
    Returns:
        corners: Lista de esquinas [top-left, top-right, bottom-right, bottom-left]
        success: True si se detectó correctamente, False en caso contrario
    """
    try:
        # Hacer una copia de la imagen
        img_copy = image.copy()
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        
        # Aplicar umbral para destacar la palanquilla del fondo
        # Podemos usar umbral adaptativo para manejar variaciones en iluminación
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # También podemos probar con el método de Otsu que es bueno para imágenes bimodales
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Combinar ambos métodos para mayor robustez
        thresh_combined = cv2.bitwise_or(thresh, thresh_otsu)
        
        # Operaciones morfológicas para limpiar la imagen
        kernel = np.ones((5, 5), np.uint8)
        thresh_cleaned = cv2.morphologyEx(thresh_combined, cv2.MORPH_CLOSE, kernel)
        thresh_cleaned = cv2.morphologyEx(thresh_cleaned, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Si no hay contornos, probar con otro enfoque
        if not contours:
            # Probar con detección de bordes de Canny
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Ordenar contornos por área
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Tomar el contorno más grande (debería ser la palanquilla)
        if contours:
            largest_contour = contours[0]
            
            # Aproximar el contorno a un polígono
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Si la aproximación tiene 4 puntos, asumimos que es un rectángulo
            if len(approx) == 4:
                corners = approx.reshape(4, 2)
                
                # Ordenar las esquinas [top-left, top-right, bottom-right, bottom-left]
                corners = order_points(corners)
                return corners, True
            
            # Si no se obtuvo un cuadrilátero, usar el rectángulo mínimo
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="int")
            
            # Ordenar las esquinas [top-left, top-right, bottom-right, bottom-left]
            corners = order_points(box)
            return corners, True
        
        # Si todo falla, usar los bordes de la imagen
        h, w = image.shape[:2]
        corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="int")
        return corners, False
        
    except Exception as e:
        print(f"Error en detect_palanquilla: {e}")
        # Retornar los bordes de la imagen
        h, w = image.shape[:2]
        corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="int")
        return corners, False

def analizar_direccion_grieta_mask(mask):
    """
    Implementación exacta de la función original para analizar dirección
    
    Args:
        mask: Máscara binaria de la grieta
        
    Returns:
        angulo: Ángulo de la grieta
        box: Puntos del rectángulo que marcan la dirección
        direccion_texto: Descripción textual de la dirección
        es_diagonal: True si la grieta es diagonal, False en caso contrario
    """
    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None, False
    
    # Encontrar el contorno más grande (la grieta principal)
    contorno_principal = max(contours, key=cv2.contourArea)
    
    # Calcular el rectángulo de área mínima
    rect = cv2.minAreaRect(contorno_principal)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Obtener el centro, tamaño y ángulo del rectángulo
    (cx, cy), (ancho, alto), angulo = rect
    
    # Corregir el ángulo para que represente la dirección
    # Si el ancho es menor que la altura, ajustar el ángulo
    if ancho < alto:
        angulo += 90
    
    # Normalizar el ángulo entre 0 y 180
    angulo = angulo % 180
    
    # Determinar si es diagonal
    es_diagonal = False
    direccion_texto = ""
    
    if 0 <= angulo < 22.5 or 157.5 <= angulo < 180:
        direccion_texto = "horizontal"
        es_diagonal = False
    elif 22.5 <= angulo < 67.5:
        direccion_texto = "diagonal descendente"
        es_diagonal = True
    elif 67.5 <= angulo < 112.5:
        direccion_texto = "vertical"
        es_diagonal = False
    else:  # 112.5 <= angulo < 157.5
        direccion_texto = "diagonal ascendente"
        es_diagonal = True
    
    return angulo, box, direccion_texto, es_diagonal

def analizar_direccion_grieta(detection, image):
    """
    Analiza la dirección de una grieta para determinar si es diagonal
    
    Args:
        detection: Diccionario con información de la detección (bbox, conf, class)
        image: Imagen original donde se detectó la grieta
        
    Returns:
        es_diagonal: True si la grieta es diagonal, False en caso contrario
        angulo: Ángulo de la grieta
        direccion_texto: Descripción textual de la dirección
        box_points: Puntos del rectángulo orientado que muestra la dirección
    """
    x1, y1, x2, y2 = detection['bbox']
    
    # Extraer la región de la grieta
    roi = image[y1:y2, x1:x2].copy()
    
    # Convertir a escala de grises
    if len(roi.shape) == 3:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi
    
    # Binarizar la imagen para obtener una máscara
    _, mask = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Analizar la dirección usando la máscara
    angulo, box, direccion_texto, es_diagonal = analizar_direccion_grieta_mask(mask)
    
    if angulo is None:
        return False, None, "indeterminada", None
    
    # Ajustar los puntos del rectángulo a las coordenadas globales de la imagen
    if box is not None:
        box_global = box.copy()
        box_global[:, 0] += x1
        box_global[:, 1] += y1
    else:
        box_global = None
    
    return es_diagonal, angulo, direccion_texto, box_global