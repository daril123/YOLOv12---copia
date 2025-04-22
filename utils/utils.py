import cv2
import numpy as np

def order_points(pts):
    """
    Ordena los puntos [top-left, top-right, bottom-right, bottom-left]
    
    Args:
        pts: Array de puntos (4 puntos)
    
    Returns:
        rect: Puntos ordenados
    """
    # Inicializar rect ordenado
    rect = np.zeros((4, 2), dtype="float32")
    
    # La suma de coordenadas es mínima en top-left y máxima en bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # La diferencia es mínima en top-right y máxima en bottom-left
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    # Retornar esquinas ordenadas
    return rect.astype(int)

def interpolate_point(p1, p2, ratio):
    """
    Interpola un punto entre dos puntos dados según un ratio
    
    Args:
        p1: Primer punto [x, y]
        p2: Segundo punto [x, y]
        ratio: Valor entre 0 y 1 para la interpolación
        
    Returns:
        Punto interpolado [x, y]
    """
    return [int(p1[0] + (p2[0] - p1[0]) * ratio), 
            int(p1[1] + (p2[1] - p1[1]) * ratio)]

def interpolate_quadrilateral(quad, u, v):
    """
    Interpola un punto dentro de un cuadrilátero usando coordenadas normalizadas.
    
    Args:
        quad: Lista de 4 puntos [top-left, top-right, bottom-right, bottom-left]
        u: Coordenada horizontal normalizada (0 a 1, de izquierda a derecha)
        v: Coordenada vertical normalizada (0 a 1, de arriba a abajo)
        
    Returns:
        Punto interpolado [x, y]
    """
    # Interpolar en el borde superior
    top = interpolate_point(quad[0], quad[1], u)
    # Interpolar en el borde inferior
    bottom = interpolate_point(quad[3], quad[2], u)
    # Interpolar entre los dos puntos para obtener el punto final
    return interpolate_point(top, bottom, v)

def get_mask_center(mask):
    """
    Encuentra el centro de una máscara usando momentos
    
    Args:
        mask: Máscara binaria
        
    Returns:
        tuple: Coordenadas (x, y) del centro o None si no hay píxeles
    """
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    return None

def draw_arrow(image, pt1, pt2, color, thickness=2, arrow_size=10, text="", text_offset=(0, 0)):
    """
    Dibuja una flecha con texto en la imagen
    
    Args:
        image: Imagen donde dibujar
        pt1: Punto inicial (x, y)
        pt2: Punto final (x, y)
        color: Color (B, G, R)
        thickness: Grosor de la línea
        arrow_size: Tamaño de la punta de flecha
        text: Texto a mostrar junto a la flecha
        text_offset: Desplazamiento del texto (dx, dy)
    """
    import math
    
    # Dibujar la línea principal
    cv2.line(image, pt1, pt2, color, thickness)
    
    # Calcular ángulo de la flecha
    angle = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    
    # Dibujar la punta de la flecha
    pt_arrow1 = (int(pt2[0] - arrow_size * math.cos(angle + math.pi/6)),
                int(pt2[1] - arrow_size * math.sin(angle + math.pi/6)))
    pt_arrow2 = (int(pt2[0] - arrow_size * math.cos(angle - math.pi/6)),
                int(pt2[1] - arrow_size * math.sin(angle - math.pi/6)))
    
    cv2.line(image, pt2, pt_arrow1, color, thickness)
    cv2.line(image, pt2, pt_arrow2, color, thickness)
    
    # Añadir texto si se proporciona
    if text:
        # Calcular punto medio para el texto
        mid_x = (pt1[0] + pt2[0]) // 2 + text_offset[0]
        mid_y = (pt1[1] + pt2[1]) // 2 + text_offset[1]
        
        # Añadir un fondo blanco para mejor visibilidad
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, 
                    (mid_x - 2, mid_y - text_size[1] - 2),
                    (mid_x + text_size[0] + 2, mid_y + 2),
                    (255, 255, 255), -1)
        
        # Dibujar el texto
        cv2.putText(image, text, (mid_x, mid_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def find_extreme_points(contour):
    """
    Encuentra los puntos más extremos en un contorno - los puntos más distantes entre sí
    
    Args:
        contour: Contorno de OpenCV
    
    Returns:
        point1, point2: Los dos puntos más extremos
        max_distance: La distancia máxima entre estos puntos
    """
    # Obtener todos los puntos del contorno
    points = contour.reshape(-1, 2)
    
    # Inicializar variables
    max_distance = 0
    extreme_points = (None, None)
    
    # Comparar cada par de puntos
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            # Calcular distancia euclidiana
            dist = np.sqrt((points[i][0] - points[j][0])**2 + 
                          (points[i][1] - points[j][1])**2)
            
            # Actualizar si encontramos una distancia mayor
            if dist > max_distance:
                max_distance = dist
                extreme_points = (tuple(points[i]), tuple(points[j]))
    
    return extreme_points[0], extreme_points[1], max_distance

def find_closest_edge_point(contour, edges):
    """
    Encuentra el punto del contorno que está más cerca de alguno de los bordes de la palanquilla
    
    Args:
        contour: Contorno de la grieta
        edges: Lista de bordes de la palanquilla, cada borde es un par de puntos [(x1,y1), (x2,y2)]
    
    Returns:
        contour_point: Punto del contorno más cercano a un borde
        edge_point: Punto en el borde más cercano al contorno
        min_distance: Distancia mínima encontrada
        closest_edge: El borde más cercano
    """
    # Obtener todos los puntos del contorno
    contour_points = contour.reshape(-1, 2)
    
    min_distance = float('inf')
    contour_point = None
    edge_point = None
    closest_edge = None
    
    # Para cada punto del contorno
    for point in contour_points:
        # Comprobar cada borde
        for edge in edges:
            p1, p2 = edge
            
            # Vector de la línea del borde
            line_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
            line_length = np.sqrt(line_vec[0]**2 + line_vec[1]**2)
            
            if line_length == 0:
                continue
            
            # Vector normalizado del borde
            line_unit_vec = line_vec / line_length
            
            # Vector desde p1 al punto del contorno
            p1_to_point = np.array([point[0] - p1[0], point[1] - p1[1]], dtype=float)
            
            # Proyección del vector p1_to_point sobre la línea del borde
            projection_length = np.dot(p1_to_point, line_unit_vec)
            
            # Encontrar el punto de proyección en la línea
            if projection_length <= 0:
                # El punto más cercano es p1
                closest_p = p1
                dist = np.sqrt((point[0] - p1[0])**2 + (point[1] - p1[1])**2)
            elif projection_length >= line_length:
                # El punto más cercano es p2
                closest_p = p2
                dist = np.sqrt((point[0] - p2[0])**2 + (point[1] - p2[1])**2)
            else:
                # El punto más cercano está en la línea entre p1 y p2
                closest_p = (int(p1[0] + line_unit_vec[0] * projection_length),
                            int(p1[1] + line_unit_vec[1] * projection_length))
                
                # Calcular la distancia perpendicular
                # Fórmula: ||(p1_to_point) - (projection_length * line_unit_vec)||
                projected_vec = line_unit_vec * projection_length
                dist_vec = p1_to_point - projected_vec
                dist = np.sqrt(dist_vec[0]**2 + dist_vec[1]**2)
            
            # Actualizar si encontramos una distancia menor
            if dist < min_distance:
                min_distance = dist
                contour_point = tuple(point)
                edge_point = closest_p
                closest_edge = edge
    
    return contour_point, edge_point, min_distance, closest_edge