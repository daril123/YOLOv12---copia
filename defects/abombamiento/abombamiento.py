import cv2
import numpy as np
import matplotlib.pyplot as plt

def distancia_a_segmento(p1, p2, punto):
    """
    Calcula la distancia mínima de un punto al segmento definido por p1 y p2.
    """
    p1, p2, punto = np.array(p1), np.array(p2), np.array(punto)
    v = p2 - p1
    u = punto - p1
    t = np.dot(u, v) / np.dot(v, v)
    if t < 0:
        return np.linalg.norm(punto - p1)
    elif t > 1:
        return np.linalg.norm(punto - p2)
    else:
        return np.abs(np.cross(v, u)) / np.linalg.norm(v)

def distancia_punto_a_linea(p1, p2, punto):
    """
    Calcula la distancia perpendicular de un punto a la línea infinita
    definida por p1 y p2 (para el cálculo del abombamiento).
    
    Args:
        p1, p2: Dos puntos que definen la línea
        punto: Punto del que calcular la distancia
        
    Returns:
        La distancia perpendicular a la línea
    """
    try:
        p1, p2, punto = np.array(p1), np.array(p2), np.array(punto)
        # Vector de la línea
        linea = p2 - p1
        # Magnitud del vector de la línea
        mag_linea = np.linalg.norm(linea)
        # Evitar división por cero
        if mag_linea == 0:
            return np.linalg.norm(punto - p1)
        # Cálculo de la distancia perpendicular
        return np.abs(np.cross(linea, punto - p1)) / mag_linea
    except Exception as e:
        print(f"Error al calcular distancia de punto a línea: {e}")
        return 0.0

def proyectar_punto_en_linea(p1, p2, p):
    """
    Devuelve el punto de proyección de 'p' sobre la línea infinita definida por p1 y p2.
    
    Args:
        p1, p2: Dos puntos que definen la línea
        p: Punto a proyectar
        
    Returns:
        El punto proyectado sobre la línea
    """
    if p is None:
        return None
        
    try:
        p = np.array(p, dtype=float)
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        
        # Vector de dirección de la línea
        line_vec = p2 - p1
        
        # Si la línea es un punto, retornar ese punto
        if np.all(line_vec == 0):
            return p1
            
        # Proyección escalar del vector (p - p1) sobre line_vec
        t = np.dot(p - p1, line_vec) / np.dot(line_vec, line_vec)
        
        # Punto proyectado
        proj = p1 + t * line_vec
        
        return tuple(int(x) for x in proj)
    except Exception as e:
        print(f"Error al proyectar punto en línea: {e}")
        return None

def obtener_lado_rotacion_abombamiento(contorno, contorno_principal):
    """
    Determina el lado más recto de la palanquilla para usar como referencia en la rotación.
    
    Args:
        contorno: Vértices de la palanquilla [top-left, top-right, bottom-right, bottom-left]
        contorno_principal: Contorno completo de la palanquilla
    
    Returns:
        El nombre del lado más recto ("Lado 1 (Top)", "Lado 2 (Right)", etc.)
    """
    # Verificar si los datos de entrada son válidos
    if contorno is None or contorno_principal is None:
        print("Error: Contornos no válidos para determinar el lado más recto.")
        return "Lado 1 (Top)"  # Valor predeterminado
    
    # Asegurar que contorno es un array numpy
    contorno = np.array(contorno)
        
    # Definir los lados de la caja (pares de vértices consecutivos)
    lados = [
        (contorno[0], contorno[1], "Lado 1 (Top)"),
        (contorno[1], contorno[2], "Lado 2 (Right)"),
        (contorno[2], contorno[3], "Lado 3 (Bottom)"),
        (contorno[3], contorno[0], "Lado 4 (Left)"),
    ]

    # Extraer los puntos del contorno
    try:
        contorno_pts = contorno_principal.reshape(-1, 2)
    except AttributeError:
        print("Error: El contorno principal no es un array válido.")
        return "Lado 1 (Top)"  # Valor predeterminado
        
    puntos_por_lado = [[] for _ in range(4)]

    # Asignar cada punto del contorno al lado más cercano
    for punto in contorno_pts:
        distancias = [distancia_a_segmento(p1, p2, punto) for p1, p2, _ in lados]
        indice_min = np.argmin(distancias)
        puntos_por_lado[indice_min].append(punto)
        
    # Calcular el abombamiento (distancia perpendicular máxima) para cada lado
    abombamientos = []
    punto_max_por_lado = []  # Punto que produce el máximo abombamiento en cada lado
    for i, (p1, p2, nombre) in enumerate(lados):
        puntos_lado = np.array(puntos_por_lado[i]) if puntos_por_lado[i] else np.array([])
        if len(puntos_lado) > 0:
            distancias = np.array([distancia_punto_a_linea(p1, p2, pt) for pt in puntos_lado])
            abombamiento = np.max(distancias) if len(distancias) > 0 else 0
            punto_max = puntos_lado[np.argmax(distancias)] if len(distancias) > 0 else None
        else:
            abombamiento = 0
            punto_max = None
        abombamientos.append(abombamiento)
        punto_max_por_lado.append(punto_max)
        print(f"{nombre}: abombamiento = {abombamiento:.2f}")

    # Se determina el lado más recto (el de menor abombamiento)
    if len(abombamientos) > 0:
        indice_recto = np.argmin(abombamientos)
        lado_recto_info = lados[indice_recto]
        print("El lado más recto es:", lado_recto_info[2])
        return lado_recto_info[2]
    else:
        return "Lado 1 (Top)"  # Valor predeterminado si no hay datos

def obtener_abombamiento(contorno, contorno_principal):
    """
    Calcula el abombamiento (desviación) para cada lado del cuadrilátero.
    
    Args:
        contorno: Vértices de la palanquilla [top-left, top-right, bottom-right, bottom-left]
        contorno_principal: Contorno completo de la palanquilla
    
    Returns:
        Una tupla con:
        - max_porcentaje: Porcentaje máximo de abombamiento
        - max_abombamiento_pix: Valor máximo de abombamiento en píxeles
        - lado_max: Lado donde se encuentra el máximo abombamiento
        - punto_max: Punto donde se encuentra el máximo abombamiento
        - abombamientos_por_lado: Diccionario con los valores de abombamiento por lado
    """
    # Verificar que los contornos no son None
    if contorno is None or contorno_principal is None:
        print("Error: Contornos inválidos para calcular abombamiento")
        # Retornar valores por defecto
        return 0.0, 0.0, "Lado 1 (Top)", None, {}
    
    # Convertir contorno a array numpy si no lo es ya
    contorno = np.array(contorno)
        
    try:
        lados = {
            "Lado 1 (Top)": (contorno[0], contorno[1]),
            "Lado 2 (Right)": (contorno[1], contorno[2]),
            "Lado 3 (Bottom)": (contorno[2], contorno[3]),
            "Lado 4 (Left)": (contorno[3], contorno[0])
        }
        # Se define si el lado es horizontal o vertical para usar el lado nominal correcto.
        lado_pos = {
            "Lado 1 (Top)": "Horizontal", 
            "Lado 2 (Right)": "Vertical", 
            "Lado 3 (Bottom)": "Horizontal", 
            "Lado 4 (Left)": "Vertical"
        }
        # Para cada orientación se elige el lado nominal (usaremos el opuesto al que se está midiendo para tener referencia)
        lados_pos = {
            "Horizontal": ["Lado 1 (Top)", "Lado 3 (Bottom)"],
            "Vertical": ["Lado 2 (Right)", "Lado 4 (Left)"]
        }
        
        # Extraer todos los puntos del contorno principal
        try:
            contorno_pts = contorno_principal.reshape(-1, 2)
        except Exception as e:
            print(f"Error al procesar contorno principal: {e}")
            contorno_pts = np.array([])
            
        puntos_por_lado = [[] for _ in range(4)]
        
        # Agrupar los puntos al lado más cercano
        i = 0
        for nombre, (p1, p2) in lados.items():
            for punto in contorno_pts:
                try:
                    distancia = distancia_a_segmento(p1, p2, punto)
                    if distancia < 20:  # Umbral para considerar puntos cercanos a un lado
                        puntos_por_lado[i].append(punto)
                except Exception as e:
                    print(f"Error al calcular distancia a segmento: {e}")
            i += 1
        
        abombamientos_pixeles = {}
        abombamientos_porcentuales = {}
        punto_max_por_lado = {}  # Se guarda para cada lado el punto que genera el máximo abombamiento
        proyeccion_max_por_lado = {}  # Proyección del punto con máximo abombamiento sobre la línea
        
        i = 0
        for nombre, (p1, p2) in lados.items():
            # Calcular la longitud nominal del lado
            try:
                longitud_nominal = np.linalg.norm(np.array(p1) - np.array(p2))
            except Exception as e:
                print(f"Error al calcular longitud nominal: {e}")
                longitud_nominal = 1.0  # Valor por defecto para evitar división por cero
            
            puntos_lado = np.array(puntos_por_lado[i]) if len(puntos_por_lado[i]) > 0 else np.array([])
            
            if len(puntos_lado) > 0:
                try:
                    # Calcular distancias perpendiculares de cada punto a la línea
                    dists = np.array([distancia_punto_a_linea(p1, p2, pt) for pt in puntos_lado])
                    abombamiento = np.max(dists) if len(dists) > 0 else 0
                    idx_max = np.argmax(dists) if len(dists) > 0 else 0
                    punto_max = puntos_lado[idx_max] if len(dists) > 0 else None
                    
                    # Calcular la proyección del punto con máximo abombamiento
                    proyeccion = proyectar_punto_en_linea(p1, p2, punto_max) if punto_max is not None else None
                except Exception as e:
                    print(f"Error al calcular distancias para {nombre}: {e}")
                    abombamiento = 0
                    punto_max = None
                    proyeccion = None
            else:
                abombamiento = 0
                punto_max = None
                proyeccion = None
            
            # Calcular el porcentaje de abombamiento respecto a la longitud nominal
            porcentaje = (abombamiento / longitud_nominal * 100) if longitud_nominal != 0 else 0
            
            abombamientos_pixeles[nombre] = abombamiento
            abombamientos_porcentuales[nombre] = porcentaje
            punto_max_por_lado[nombre] = punto_max
            proyeccion_max_por_lado[nombre] = proyeccion
            
            print(f"{nombre}: abombamiento = {abombamiento:.2f} px, {porcentaje:.2f}%")
            i += 1
        
        # Identificar el lado con mayor abombamiento (según porcentaje)
        if abombamientos_porcentuales:
            # Buscar el lado con el valor máximo de abombamiento porcentual
            lado_max = max(abombamientos_porcentuales.items(), key=lambda x: x[1])[0]
            max_abombamiento_pix = abombamientos_pixeles[lado_max]
            max_porcentaje = abombamientos_porcentuales[lado_max]
        else:
            lado_max = "Lado 1 (Top)"
            max_abombamiento_pix = 0
            max_porcentaje = 0
        
        print("El mayor abombamiento se encuentra en", lado_max, "con un valor de", 
              max_abombamiento_pix, "px, lo que equivale a", max_porcentaje, "%")
        
        # Crear diccionario con todos los resultados por lado
        abombamientos_por_lado = {
            lado: {
                'abombamiento_px': abombamientos_pixeles.get(lado, 0),
                'abombamiento_porcentaje': abombamientos_porcentuales.get(lado, 0),
                'punto_max': punto_max_por_lado.get(lado, None),
                'proyeccion': proyeccion_max_por_lado.get(lado, None),
                'vertices': lados.get(lado, (None, None))
            }
            for lado in lados
        }
        
    except Exception as e:
        print(f"Error general en obtener_abombamiento: {e}")
        return 0.0, 0.0, "Lado 1 (Top)", None, {}
    
    return max_porcentaje, max_abombamiento_pix, lado_max, punto_max_por_lado.get(lado_max), abombamientos_por_lado

def visualizar_abombamiento(image, contorno, contorno_principal, guardar_path=None):
    """
    Visualiza el abombamiento en todos los lados de la palanquilla.
    
    Args:
        image: Imagen original
        contorno: Vértices de la palanquilla [top-left, top-right, bottom-right, bottom-left]
        contorno_principal: Contorno completo de la palanquilla
        guardar_path: Ruta donde guardar la imagen resultante (opcional)
    
    Returns:
        Imagen con visualización del abombamiento
    """
    try:
        # Verificar entradas
        if image is None or contorno is None or contorno_principal is None:
            print("Error: Entrada inválida para visualizar abombamiento")
            if image is not None:
                return image.copy()
            return None
            
        # Convertir contorno a array numpy si no lo es ya
        contorno = np.array(contorno)
            
        # Obtener los datos de abombamiento
        _, _, _, _, abombamientos_por_lado = obtener_abombamiento(contorno, contorno_principal)
        
        # Crear una copia de la imagen para no modificar la original
        img_resultado = image.copy()
        
        # Dibujar el contorno de la palanquilla
        pts = contorno.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img_resultado, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Colores para cada lado (BGR)
        colores = {
            "Lado 1 (Top)": (255, 0, 0),      # Azul
            "Lado 2 (Right)": (0, 255, 0),    # Verde
            "Lado 3 (Bottom)": (0, 0, 255),   # Rojo
            "Lado 4 (Left)": (0, 255, 255)    # Amarillo
        }
        
        # Para cada lado, dibujar la línea nominal y la línea de abombamiento
        for lado, datos in abombamientos_por_lado.items():
            p1, p2 = datos['vertices']
            
            # Verificar que ambos puntos sean válidos
            if p1 is None or p2 is None:
                continue
                
            p1 = tuple(np.array(p1).astype(int))
            p2 = tuple(np.array(p2).astype(int))
            
            # Dibujar la línea nominal
            cv2.line(img_resultado, p1, p2, colores[lado], 2)
            
            # Si hay punto de máximo abombamiento, dibujarlo
            if datos['punto_max'] is not None and datos['proyeccion'] is not None:
                punto_max = tuple(np.array(datos['punto_max']).astype(int))
                proyeccion = tuple(np.array(datos['proyeccion']).astype(int))
                
                # Dibujar línea de abombamiento
                cv2.line(img_resultado, punto_max, proyeccion, (0, 0, 255), 2)
                
                # Dibujar puntos
                cv2.circle(img_resultado, punto_max, 5, (0, 0, 255), -1)
                cv2.circle(img_resultado, proyeccion, 5, (0, 255, 255), -1)
                
                # Añadir etiqueta con valor
                abombamiento_px = datos['abombamiento_px']
                abombamiento_porcentaje = datos['abombamiento_porcentaje']
                texto = f"{abombamiento_px:.1f}px ({abombamiento_porcentaje:.1f}%)"
                
                # Posición de la etiqueta (ajustar según la posición del punto)
                pos_x = (punto_max[0] + proyeccion[0]) // 2
                pos_y = (punto_max[1] + proyeccion[1]) // 2
                
                # Añadir fondo negro para mejor visibilidad
                (text_width, text_height), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img_resultado, 
                             (pos_x - 5, pos_y - text_height - 5), 
                             (pos_x + text_width + 5, pos_y + 5), 
                             (0, 0, 0), -1)
                
                # Añadir texto
                cv2.putText(img_resultado, texto, (pos_x, pos_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Guardar la imagen si se proporciona una ruta
        if guardar_path:
            cv2.imwrite(guardar_path, img_resultado)
            print(f"Imagen guardada en: {guardar_path}")
        
        return img_resultado
    except Exception as e:
        print(f"Error al visualizar abombamiento: {e}")
        # Devolver la imagen original en caso de error
        return image.copy() if image is not None else None

def calcular_abombamiento(image, contorno, contorno_principal, factor_mm_px=1.0):
    """
    Calcula el abombamiento en milímetros según la fórmula C = X / L * 100.
    
    Args:
        image: Imagen original
        contorno: Vértices de la palanquilla [top-left, top-right, bottom-right, bottom-left]
        contorno_principal: Contorno completo de la palanquilla
        factor_mm_px: Factor de conversión de píxeles a milímetros
    
    Returns:
        Un diccionario con los resultados del abombamiento para cada lado
    """
    try:
        # Verificar entradas
        if contorno is None or contorno_principal is None:
            print("Error: Entrada inválida para calcular abombamiento")
            return {}
            
        # Convertir contorno a array numpy si no lo es ya
        contorno = np.array(contorno)
            
        # Obtener datos de abombamiento
        _, _, _, _, abombamientos_por_lado = obtener_abombamiento(contorno, contorno_principal)
        
        # Crear diccionario de resultados
        resultados = {}
        
        for lado, datos in abombamientos_por_lado.items():
            # X es el valor de abombamiento en píxeles
            X_px = datos['abombamiento_px']
            X_mm = X_px * factor_mm_px
            
            # Longitud nominal del lado
            p1, p2 = datos['vertices']
            
            # Verificar que los vértices sean válidos
            if p1 is None or p2 is None:
                L_px = 1.0  # Evitar división por cero
                L_mm = 1.0
            else:
                L_px = np.linalg.norm(np.array(p1) - np.array(p2))
                L_mm = L_px * factor_mm_px
            
            # Calcular C según la fórmula C = X / L * 100
            C = X_mm / L_mm * 100 if L_mm > 0 else 0
            
            resultados[lado] = {
                'X_mm': X_mm,
                'L_mm': L_mm,
                'C_porcentaje': C,
                'X_px': X_px,
                'L_px': L_px
            }
        
        return resultados
    except Exception as e:
        print(f"Error al calcular abombamiento: {e}")
        return {}